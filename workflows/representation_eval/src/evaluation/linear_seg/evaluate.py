"""Main evaluation entry point for linear segmentation (YAML-driven)."""

import gc
import json
import logging
import os
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

from .dataset import load_dataset
from .segmenter import Segmenter

logger = logging.getLogger("ssl_eval")


def _default_test_batch_size(dataset_name: str) -> int:
    return 8 if dataset_name in {"ade20k", "cityscapes"} else 16


def _persist_results(cfg: Dict, results: Dict):
    experiment_cfg = cfg.get("experiment", {})
    output_dir = experiment_cfg.get("output_dir")
    if not output_dir:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "effective_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def evaluate_linear_seg(model, cfg: Dict) -> Dict:
    """
    Evaluate linear segmentation using native PyTorch DDP and YAML config.

    Args:
        model: Pre-loaded backbone model
        cfg: validated experiment config (from `load_linear_seg_config`)

    Returns:
        Nested dict: {dataset_name: {head_name: {best_lr, best_miou, best_acc, per_lr_results}}}
    """
    if not dist.is_available() or not dist.is_initialized():
        logger.warning("Distributed not initialized. Linear-seg evaluation requires torchrun/DDP context.")
        return {}

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(torch.cuda.current_device())))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    head_cfg = cfg["head_search"]
    distributed_cfg = cfg.get("distributed", {})

    dataset_names = data_cfg["datasets"]
    train_res = int(data_cfg["resolution"]["train"])
    test_res = int(data_cfg["resolution"]["val"])
    train_aug = data_cfg["augmentation"]["train"]
    val_aug = data_cfg["augmentation"]["val"]
    num_workers = int(distributed_cfg.get("num_workers", 4))
    persistent_workers = bool(distributed_cfg.get("persistent_workers", False))

    head_specs = head_cfg["head_specs"]
    head_names = [spec["name"] for spec in head_specs]
    test_lrs = [float(lr) for lr in head_cfg["lrs"]]
    max_steps = int(train_cfg["max_steps"])
    global_batch_size = int(train_cfg["global_batch_size"])
    optimizer_cfg = train_cfg["optimizer"]
    amp_cfg = train_cfg["amp"]
    feature_cfg = cfg["model"]["feature_source"]
    model_type = cfg["model"]["model_type"]

    if rank == 0:
        logger.info("Starting linear segmentation evaluation (YAML protocol)")
        logger.info(f"  Datasets: {dataset_names}")
        logger.info(f"  Heads: {head_names}")
        logger.info(f"  LRs: {test_lrs}")
        logger.info(f"  Max steps: {max_steps}, Global batch size: {global_batch_size}")
        logger.info(f"  Feature mode: {feature_cfg.get('mode', 'single')}")
        logger.info(f"  Train res: {train_res}, Val res: {test_res}")
        logger.info(f"  Num workers: {num_workers}, persistent_workers: {persistent_workers}, GPUs: {world_size}")

    all_results: Dict = {}
    for dataset_name in dataset_names:
        if rank == 0:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Evaluating dataset: {dataset_name}")
            logger.info(f"{'=' * 60}")

        try:
            train_ds, num_classes = load_dataset(
                dataset_name=dataset_name,
                root=data_cfg["root"],
                split="train",
                resolution=train_res,
                augmentations=train_aug,
            )
            val_ds, _ = load_dataset(
                dataset_name=dataset_name,
                root=data_cfg["root"],
                split="val",
                resolution=test_res,
                augmentations=val_aug,
            )

            if rank == 0:
                logger.info(
                    f"Loaded {dataset_name}: {num_classes} classes, "
                    f"train={len(train_ds)}, val={len(val_ds)}"
                )

            test_sampler = torch.utils.data.distributed.DistributedSampler(
                val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
            )
            test_batch_size = int(cfg.get("eval", {}).get("test_batch_size", _default_test_batch_size(dataset_name)))
            test_loader = DataLoader(
                val_ds,
                batch_size=test_batch_size,
                shuffle=False,
                sampler=test_sampler,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(num_workers > 0 and persistent_workers),
            )

            segmenter = Segmenter(
                model=model,
                model_type=model_type,
                num_classes=num_classes,
                dataset_name=dataset_name,
                train_dataset=train_ds,
                head_specs=head_specs,
                test_lrs=test_lrs,
                max_steps=max_steps,
                global_batch_size=global_batch_size,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                feature_cfg=feature_cfg,
                optimizer_cfg=optimizer_cfg,
                amp_cfg=amp_cfg,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
            )
            dataset_results = segmenter.run_test(test_loader)["per_head_results"]
            all_results[dataset_name] = dataset_results

            if rank == 0:
                for head_type, result in dataset_results.items():
                    logger.info(
                        f"Best for {dataset_name}/{head_type}: "
                        f"LR={result['best_lr']}, mIoU={result['best_miou']:.2f}, "
                        f"Acc={result['best_acc']:.2f}"
                    )

            del segmenter
            del train_ds, val_ds, test_loader
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as error:
            if rank == 0:
                logger.error(f"Evaluation failed for dataset={dataset_name}: {error}")
                import traceback

                traceback.print_exc()
            all_results[dataset_name] = {
                head_name: {"best_lr": None, "best_miou": 0.0, "best_acc": 0.0, "per_lr_results": {}}
                for head_name in head_names
            }
        finally:
            dist.barrier()

    if rank == 0:
        logger.info(f"\n{'=' * 60}")
        logger.info("SUMMARY")
        logger.info(f"{'=' * 60}")
        for dataset_name, head_results in all_results.items():
            for head_type, values in head_results.items():
                logger.info(
                    f"  {dataset_name}/{head_type}: mIoU={values['best_miou']:.2f}, "
                    f"Acc={values['best_acc']:.2f}, LR={values['best_lr']}"
                )
        _persist_results(cfg, all_results)

    return all_results
