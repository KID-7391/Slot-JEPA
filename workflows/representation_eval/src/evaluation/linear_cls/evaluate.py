"""DINOv2-style linear classification evaluation."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from src.evaluation.common.distributed import (
    autocast_context as _autocast_context,
    dist_enabled as _dist_enabled,
    is_main_process as _is_main_process,
    rank as _rank,
    remove_ddp_wrapper as _remove_ddp_wrapper,
    world_size as _world_size,
)
from .dataset import load_classification_datasets
from .features import create_linear_input, extract_embedding_lists, resolve_n_last_blocks


logger = logging.getLogger("ssl_eval.linear_cls")


def _scale_lr(learning_rate: float, batch_size: int, world_size: int) -> float:
    return float(learning_rate) * (float(batch_size) * float(world_size)) / 256.0


def _resolve_per_rank_batch_size(
    *,
    batch_size: Optional[int],
    global_batch_size: Optional[int],
    world_size: int,
    field_name: str,
) -> Tuple[int, int]:
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        if global_batch_size <= 0:
            raise ValueError(f"{field_name}.global_batch_size must be > 0, got {global_batch_size}")
        if global_batch_size % world_size != 0:
            raise ValueError(
                f"{field_name}.global_batch_size={global_batch_size} must be divisible by world_size={world_size}"
            )
        per_rank_batch_size = global_batch_size // world_size
    elif batch_size is not None:
        per_rank_batch_size = int(batch_size)
        if per_rank_batch_size <= 0:
            raise ValueError(f"{field_name}.batch_size must be > 0, got {per_rank_batch_size}")
        global_batch_size = per_rank_batch_size * world_size
    else:
        raise ValueError(f"{field_name}.global_batch_size or {field_name}.batch_size must be provided")

    if batch_size is not None and per_rank_batch_size != int(batch_size):
        raise ValueError(
            f"{field_name}.batch_size={batch_size} conflicts with "
            f"{field_name}.global_batch_size={global_batch_size} for world_size={world_size}"
        )

    return per_rank_batch_size, int(global_batch_size)


class LinearClassifier(nn.Module):
    """Linear layer trained on top of frozen embeddings."""

    def __init__(
        self,
        out_dim: int,
        use_n_blocks: int,
        use_avgpool: bool,
        num_classes: int,
        pooling_mode: str = "cls",
    ):
        super().__init__()
        self.use_n_blocks = int(use_n_blocks)
        self.use_avgpool = bool(use_avgpool)
        self.pooling_mode = str(pooling_mode)
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, embedding_output):
        linear_input = create_linear_input(
            embedding_output,
            use_n_blocks=self.use_n_blocks,
            use_avgpool=self.use_avgpool,
            pooling_mode=self.pooling_mode,
        )
        return self.linear(linear_input)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict: Dict[str, nn.Module]):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict(classifiers_dict)

    def forward(self, embedding_output):
        return {name: classifier(embedding_output) for name, classifier in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


class EmbeddingFeatureModel(nn.Module):
    """Frozen backbone wrapper returning per-layer cls/patch embeddings."""

    def __init__(
        self,
        model,
        model_type: str,
        feature_cfg: Dict,
        amp_cfg: Dict,
        device: torch.device,
        max_n_blocks: int,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.model_type = model_type
        self.feature_cfg = feature_cfg
        self.amp_cfg = amp_cfg
        self.device = device
        self.max_n_blocks = int(max_n_blocks)

    def forward(self, images: torch.Tensor):
        with torch.inference_mode():
            with _autocast_context(self.device, self.amp_cfg):
                return extract_embedding_lists(
                    self.model,
                    images,
                    model_type=self.model_type,
                    feature_cfg=self.feature_cfg,
                    max_n_blocks=self.max_n_blocks,
                )


class TopKAccuracyMeter:
    def __init__(self, ks: Sequence[int], device: torch.device):
        self.ks = tuple(sorted({int(k) for k in ks}))
        self.device = device
        self.correct = {k: torch.zeros(1, device=device, dtype=torch.float64) for k in self.ks}
        self.total = torch.zeros(1, device=device, dtype=torch.float64)

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        if logits.ndim != 2:
            raise ValueError(f"Expected logits with shape [B, C], got {tuple(logits.shape)}")
        if target.ndim != 1:
            target = target.view(-1)

        batch_size, num_classes = logits.shape
        max_k = min(max(self.ks), num_classes)
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.unsqueeze(1))

        for k in self.ks:
            effective_k = min(k, num_classes)
            self.correct[k] += correct[:, :effective_k].any(dim=1).sum()
        self.total += batch_size

    def compute(self) -> Dict[str, float]:
        totals = {k: value.clone() for k, value in self.correct.items()}
        total = self.total.clone()

        if _dist_enabled():
            for value in totals.values():
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

        denom = max(total.item(), 1.0)
        return {f"top-{k}": 100.0 * totals[k].item() / denom for k in self.ks}


def _setup_linear_classifiers(
    sample_output,
    n_last_blocks_list: Sequence[int],
    avgpool_options: Sequence[bool],
    learning_rates: Sequence[float],
    batch_size: int,
    num_classes: int,
    device: torch.device,
    pooling_mode: str,
) -> Tuple[nn.Module, List[Dict]]:
    world_size = _world_size()
    classifiers_dict: Dict[str, nn.Module] = {}
    optimizer_param_groups: List[Dict] = []

    for n_blocks in n_last_blocks_list:
        for use_avgpool in avgpool_options:
            out_dim = create_linear_input(
                sample_output,
                use_n_blocks=n_blocks,
                use_avgpool=use_avgpool,
                pooling_mode=pooling_mode,
            ).shape[1]
            for base_lr in learning_rates:
                scaled_lr = _scale_lr(base_lr, batch_size=batch_size, world_size=world_size)
                name = f"classifier_{n_blocks}_blocks_avgpool_{use_avgpool}_lr_{scaled_lr:.5f}".replace(".", "_")
                classifier = LinearClassifier(
                    out_dim=out_dim,
                    use_n_blocks=n_blocks,
                    use_avgpool=use_avgpool,
                    num_classes=num_classes,
                    pooling_mode=pooling_mode,
                ).to(device)
                classifiers_dict[name] = classifier
                optimizer_param_groups.append({"params": classifier.parameters(), "lr": scaled_lr})

    all_classifiers = AllClassifiers(classifiers_dict)
    if _dist_enabled():
        device_ids = [device.index] if device.type == "cuda" else None
        all_classifiers = DistributedDataParallel(all_classifiers, device_ids=device_ids)

    return all_classifiers, optimizer_param_groups


def _build_train_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
):
    if _dist_enabled():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=_world_size(),
            rank=_rank(),
            shuffle=True,
            drop_last=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0 and persistent_workers),
    )
    return loader, sampler


def _build_eval_loader(dataset, batch_size: int, num_workers: int):
    if _dist_enabled():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=_world_size(),
            rank=_rank(),
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    return loader


def _iterate_train_loader(train_loader, train_sampler, max_iter: int):
    epoch = 0
    loader_iterator = None

    for _ in range(max_iter):
        if loader_iterator is None:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            loader_iterator = iter(train_loader)
            epoch += 1

        try:
            yield next(loader_iterator)
        except StopIteration:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            loader_iterator = iter(train_loader)
            epoch += 1
            yield next(loader_iterator)


@torch.no_grad()
def _evaluate_linear_classifiers(
    feature_model: EmbeddingFeatureModel,
    linear_classifiers: nn.Module,
    data_loader,
    topk: Sequence[int],
    best_classifier_on_val: Optional[str] = None,
    prefix: str = "",
) -> Dict:
    module = _remove_ddp_wrapper(linear_classifiers)
    meters = {
        name: TopKAccuracyMeter(topk, device=feature_model.device)
        for name in module.classifiers_dict.keys()
    }

    linear_classifiers.eval()
    data_iterator = data_loader
    if _is_main_process():
        data_iterator = tqdm(data_loader, desc=f"Eval{prefix}", ascii=True, leave=False)

    for images, target in data_iterator:
        images = images.to(feature_model.device, non_blocking=True)
        target = target.to(feature_model.device, non_blocking=True)

        embedding_output = feature_model(images)
        outputs = linear_classifiers(embedding_output)
        for name, logits in outputs.items():
            meters[name].update(logits, target)

    classifier_results = {name: meter.compute() for name, meter in meters.items()}

    best_classifier_name = None
    best_top1 = -1.0
    for name, metrics in classifier_results.items():
        if best_classifier_on_val is not None:
            if name == best_classifier_on_val:
                best_classifier_name = name
                best_top1 = metrics.get("top-1", -1.0)
                break
        elif metrics.get("top-1", -1.0) > best_top1:
            best_classifier_name = name
            best_top1 = metrics.get("top-1", -1.0)

    if best_classifier_name is None:
        raise RuntimeError("Failed to select a best classifier from validation results.")

    best_metrics = classifier_results[best_classifier_name]
    return {
        "best_classifier": {
            "name": best_classifier_name,
            **best_metrics,
        },
        "per_classifier": classifier_results,
    }


def _persist_results(cfg: Dict, summary: Dict, evaluation_history: List[Dict], classifier_state: Dict[str, torch.Tensor]):
    output_dir = cfg.get("experiment", {}).get("output_dir")
    if not output_dir or not _is_main_process():
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "effective_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    history_path = output_path / "results_eval_linear.jsonl"
    with history_path.open("w", encoding="utf-8") as handle:
        for record in evaluation_history:
            handle.write(json.dumps(record) + "\n")

    torch.save(classifier_state, output_path / "linear_classifiers.pt")


def evaluate_linear_cls(model, cfg: Dict) -> Dict:
    rank = _rank()
    world_size = _world_size()
    device = torch.device(
        f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        if torch.cuda.is_available()
        else "cpu"
    )

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    head_cfg = cfg["head_search"]
    train_cfg = cfg["train"]
    eval_cfg = cfg["eval"]
    distributed_cfg = cfg.get("distributed", {})

    batch_size, global_batch_size = _resolve_per_rank_batch_size(
        batch_size=train_cfg.get("batch_size"),
        global_batch_size=train_cfg.get("global_batch_size"),
        world_size=world_size,
        field_name="train",
    )
    eval_batch_size, eval_global_batch_size = _resolve_per_rank_batch_size(
        batch_size=eval_cfg.get("batch_size"),
        global_batch_size=eval_cfg.get("global_batch_size"),
        world_size=world_size,
        field_name="eval",
    )
    num_workers = int(distributed_cfg.get("num_workers", 8))
    max_iter = int(train_cfg["epochs"]) * int(train_cfg["epoch_length"])
    eval_period = int(eval_cfg.get("eval_period_iterations", train_cfg["epoch_length"]))
    learning_rates = [float(lr) for lr in head_cfg["learning_rates"]]
    requested_n_last_blocks = [int(value) for value in head_cfg["n_last_blocks"]]
    avgpool_options = [bool(value) for value in head_cfg["use_avgpool"]]
    pooling_mode = str(head_cfg.get("pooling_mode", "cls"))
    topk = [int(k) for k in eval_cfg["topk"]]
    persistent_workers = bool(distributed_cfg.get("persistent_workers", True))

    dataset_bundle = load_classification_datasets(data_cfg)
    train_dataset = dataset_bundle.train_dataset
    val_dataset = dataset_bundle.val_dataset
    test_dataset = dataset_bundle.test_dataset
    num_classes = dataset_bundle.num_classes
    use_validation_split = val_dataset is not None

    if rank == 0:
        logger.info("Starting linear classification evaluation")
        logger.info(f"  Dataset: {data_cfg.get('dataset', 'imagenet')}")
        logger.info(f"  Train split: {dataset_bundle.train_descriptor}")
        if dataset_bundle.val_descriptor is not None:
            logger.info(f"  Val split: {dataset_bundle.val_descriptor}")
        else:
            logger.info("  Val split: None")
        if dataset_bundle.test_descriptor is not None:
            logger.info(f"  Test split: {dataset_bundle.test_descriptor}")
        logger.info(f"  Num classes: {num_classes}")
        logger.info(f"  Train batch size / GPU: {batch_size}")
        logger.info(f"  Train global batch size: {global_batch_size}")
        logger.info(f"  Eval batch size / GPU: {eval_batch_size}")
        logger.info(f"  Eval global batch size: {eval_global_batch_size}")
        logger.info(f"  Max iterations: {max_iter}")
        logger.info(f"  Learning rates: {learning_rates}")
        logger.info(f"  Requested n_last_blocks: {requested_n_last_blocks}")
        logger.info(f"  Avgpool options: {avgpool_options}")
        logger.info(f"  Pooling mode: {pooling_mode}")
        logger.info(f"  Optimizer: {train_cfg['optimizer'].get('type', 'sgd')}")
        logger.info(f"  Top-k metrics: {topk}")
        logger.info(f"  Num workers: {num_workers}, world_size: {world_size}")

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    max_requested_blocks = max(requested_n_last_blocks)
    feature_model = EmbeddingFeatureModel(
        model=model,
        model_type=model_cfg["model_type"],
        feature_cfg=model_cfg["feature_source"],
        amp_cfg=train_cfg["amp"],
        device=device,
        max_n_blocks=max_requested_blocks,
    )

    sample_image = train_dataset[0][0].unsqueeze(0).to(device)
    sample_output = feature_model(sample_image)
    resolved_n_last_blocks = resolve_n_last_blocks(
        available_blocks=len(sample_output[0]),
        requested_n_last_blocks=requested_n_last_blocks,
    )
    if rank == 0 and resolved_n_last_blocks != sorted(set(requested_n_last_blocks)):
        logger.warning(
            "Resolved n_last_blocks differs from requested values. "
            f"requested={requested_n_last_blocks}, resolved={resolved_n_last_blocks}"
        )

    linear_classifiers, optimizer_param_groups = _setup_linear_classifiers(
        sample_output=sample_output,
        n_last_blocks_list=resolved_n_last_blocks,
        avgpool_options=avgpool_options,
        learning_rates=learning_rates,
        batch_size=batch_size,
        num_classes=num_classes,
        device=device,
        pooling_mode=pooling_mode,
    )

    optimizer_cfg = train_cfg["optimizer"]
    optimizer = torch.optim.SGD(
        optimizer_param_groups,
        momentum=float(optimizer_cfg.get("momentum", 0.9)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=0.0)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, train_sampler = _build_train_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = _build_eval_loader(
            val_dataset,
            batch_size=eval_batch_size,
            num_workers=num_workers,
        )
    test_loader = None
    if test_dataset is not None:
        test_loader = _build_eval_loader(
            test_dataset,
            batch_size=eval_batch_size,
            num_workers=num_workers,
        )

    evaluation_history: List[Dict] = []
    progress = _iterate_train_loader(train_loader, train_sampler, max_iter=max_iter)
    if rank == 0:
        progress = tqdm(progress, total=max_iter, desc="Linear probe training", ascii=True)

    linear_classifiers.train()
    for iteration, batch in enumerate(progress):
        images, target = batch
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        embedding_output = feature_model(images)
        outputs = linear_classifiers(embedding_output)

        losses = [loss_fn(logits, target) for logits in outputs.values()]
        loss = torch.stack(losses).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if rank == 0 and iteration % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")

        if use_validation_split and eval_period > 0 and (iteration + 1) % eval_period == 0 and iteration != max_iter - 1:
            interim_results = _evaluate_linear_classifiers(
                feature_model=feature_model,
                linear_classifiers=linear_classifiers,
                data_loader=val_loader,
                topk=topk,
                prefix=f"@{iteration + 1}",
            )
            evaluation_history.append({"iteration": iteration + 1, **interim_results})
            linear_classifiers.train()
            if rank == 0:
                best_classifier = interim_results["best_classifier"]
                logger.info(
                    f"[iter {iteration + 1}] best={best_classifier['name']}, "
                    f"top-1={best_classifier.get('top-1', 0.0):.2f}, "
                    f"top-5={best_classifier.get('top-5', 0.0):.2f}"
                )

    final_val_results = None
    final_test_results = None
    selection_split = None

    if use_validation_split:
        final_val_results = _evaluate_linear_classifiers(
            feature_model=feature_model,
            linear_classifiers=linear_classifiers,
            data_loader=val_loader,
            topk=topk,
            prefix="@final",
        )
        evaluation_history.append({"iteration": max_iter, "split": "val", **final_val_results})
        selection_split = "val"

        if test_loader is not None:
            final_test_results = _evaluate_linear_classifiers(
                feature_model=feature_model,
                linear_classifiers=linear_classifiers,
                data_loader=test_loader,
                topk=topk,
                best_classifier_on_val=final_val_results["best_classifier"]["name"],
                prefix="@test",
            )
            evaluation_history.append({"iteration": max_iter, "split": "test", **final_test_results})
        best_results = final_val_results
    else:
        if test_loader is None:
            raise ValueError("No evaluation split is configured. Please provide data.test_split when data.val_split is null.")
        final_test_results = _evaluate_linear_classifiers(
            feature_model=feature_model,
            linear_classifiers=linear_classifiers,
            data_loader=test_loader,
            topk=topk,
            prefix="@test",
        )
        evaluation_history.append({"iteration": max_iter, "split": "test", **final_test_results})
        selection_split = "test"
        best_results = final_test_results

    summary = {
        "dataset": data_cfg.get("dataset", "imagenet"),
        "num_classes": num_classes,
        "data_splits": {
            "train": dataset_bundle.train_descriptor,
        },
        "selection_split": selection_split,
        "max_iter": max_iter,
        "batch_size_per_gpu": batch_size,
        "global_batch_size": global_batch_size,
        "eval_batch_size_per_gpu": eval_batch_size,
        "eval_global_batch_size": eval_global_batch_size,
        "world_size": world_size,
        "resolved_n_last_blocks": resolved_n_last_blocks,
        "avgpool_options": avgpool_options,
        "pooling_mode": pooling_mode,
        "optimizer": optimizer_cfg,
        "best_classifier": best_results["best_classifier"],
        "per_classifier": best_results["per_classifier"],
    }
    if dataset_bundle.val_descriptor is not None:
        summary["data_splits"]["val"] = dataset_bundle.val_descriptor
    if dataset_bundle.test_descriptor is not None:
        summary["data_splits"]["test"] = dataset_bundle.test_descriptor
    if final_val_results is not None:
        summary["val_results"] = final_val_results
    if final_test_results is not None:
        summary["test_results"] = final_test_results

    classifier_state = {
        "state_dict": _remove_ddp_wrapper(linear_classifiers).state_dict(),
        "best_classifier": best_results["best_classifier"]["name"],
        "summary": summary,
    }
    _persist_results(cfg, summary, evaluation_history, classifier_state)

    if rank == 0:
        best_classifier = best_results["best_classifier"]
        if selection_split is None:
            logger.info(
                f"Final test classifier: {best_classifier['name']} "
                f"(top-1={best_classifier.get('top-1', 0.0):.2f}, "
                f"top-5={best_classifier.get('top-5', 0.0):.2f})"
            )
        else:
            logger.info(
                f"Final best classifier: {best_classifier['name']} "
                f"(top-1={best_classifier.get('top-1', 0.0):.2f}, "
                f"top-5={best_classifier.get('top-5', 0.0):.2f})"
            )
        if final_test_results is not None and selection_split is not None:
            test_best_classifier = final_test_results["best_classifier"]
            logger.info(
                f"Test metrics for selected classifier: {test_best_classifier['name']} "
                f"(top-1={test_best_classifier.get('top-1', 0.0):.2f}, "
                f"top-5={test_best_classifier.get('top-5', 0.0):.2f})"
            )

    return summary
