"""Core segmenter module for linear segmentation evaluation."""

import os
import time
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.evaluation.common import autocast_context, extract_spatial_features

from .utils import EvaluationMetrics, LinearMappingMultiHead


class Segmenter(nn.Module):
    """Linear probing evaluator supporting multi-head and multi-LR search."""

    def __init__(
        self,
        model,
        model_type: str,
        num_classes: int,
        dataset_name: str,
        train_dataset,
        head_specs: List[Dict[str, Any]],
        test_lrs: List[float],
        max_steps: int,
        global_batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        feature_cfg: Dict[str, Any],
        optimizer_cfg: Dict[str, Any],
        amp_cfg: Dict[str, Any],
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        self.num_classes = int(num_classes)
        self.dataset_name = dataset_name
        self.train_dataset = train_dataset
        self.head_specs = [dict(spec) for spec in head_specs]
        self.head_names = [str(spec["name"]) for spec in self.head_specs]
        self.test_lrs = [float(lr) for lr in test_lrs]
        self.max_steps = int(max_steps)
        self.global_batch_size = int(global_batch_size)
        self.num_workers = int(num_workers)
        self.persistent_workers = bool(persistent_workers)
        self.feature_cfg = feature_cfg
        self.optimizer_cfg = optimizer_cfg
        self.amp_cfg = dict(amp_cfg)

        list_cfg = self.feature_cfg.get("list", {}) if self.feature_cfg.get("mode") == "list" else {}
        projection_cfg = list_cfg.get("projection", {})
        self.projection_dim = int(projection_cfg["out_dim"]) if projection_cfg.get("enabled", False) else None

        self.model.to(self.device)
        self.model.eval()

        self.linear_evaluator: LinearMappingMultiHead = None
        self.linear_metrics = {
            head_name: {
                lr: EvaluationMetrics(
                    f"valid/linear_seg/{dataset_name}/head_{head_name}/lr_{lr}/",
                    self.num_classes,
                    0,
                    False,
                    device=self.device,
                )
                for lr in self.test_lrs
            }
            for head_name in self.head_names
        }
        self.profile_steps = max(0, int(os.environ.get("LINEAR_SEG_PROFILE_STEPS", "0")))
        self.profile_only = self.profile_steps > 0 and bool(int(os.environ.get("LINEAR_SEG_PROFILE_ONLY", "1")))
        self.profile_complete = False

    def _sync_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _print(self, *args, **kwargs):
        if self.rank == 0:
            print(*args, **kwargs)

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return extract_spatial_features(
            self.model,
            images,
            model_type=self.model_type,
            feature_cfg=self.feature_cfg,
        )

    def _build_train_loader(self):
        per_gpu_batch = max(1, self.global_batch_size // max(1, self.world_size))
        if self.rank == 0 and per_gpu_batch * self.world_size != self.global_batch_size:
            self._print(
                f"[Warning] global_batch_size={self.global_batch_size} is not divisible by world_size={self.world_size}. "
                f"Using effective global batch {per_gpu_batch * self.world_size}."
            )

        sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset, num_replicas=self.world_size, rank=self.rank
        )
        loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=per_gpu_batch,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0 and self.persistent_workers),
        )
        return loader, sampler

    def setup_linear_head(self):
        """Initialize and train all (head, lr) combinations."""
        self._print(f"\nTraining heads {self.head_names} with LRs: {self.test_lrs} for {self.max_steps} steps...")

        train_loader, sampler = self._build_train_loader()
        epoch = 0
        loader_iterator = None
        iterator = range(self.max_steps)
        if self.rank == 0:
            iterator = tqdm(iterator, desc="Linear probe training", ascii=True)

        profile_stats = {
            "data_wait": 0.0,
            "h2d": 0.0,
            "feature_extract": 0.0,
            "head_update": 0.0,
        }
        profiled_steps = 0

        def _emit_profile():
            if self.rank == 0 and profiled_steps > 0:
                self._print("Linear-seg profiling (avg over first %d steps):" % profiled_steps)
                for key, total_time in profile_stats.items():
                    self._print(f"  {key}: {1000.0 * total_time / profiled_steps:.1f} ms")

        for step in iterator:
            batch_wait_start = time.perf_counter()
            if loader_iterator is None:
                sampler.set_epoch(epoch)
                loader_iterator = iter(train_loader)
                epoch += 1

            try:
                batch = next(loader_iterator)
            except StopIteration:
                sampler.set_epoch(epoch)
                loader_iterator = iter(train_loader)
                epoch += 1
                batch = next(loader_iterator)
            batch_wait_end = time.perf_counter()

            h2d_start = time.perf_counter()
            image = batch[0].to(self.device, non_blocking=True)
            label = batch[1].to(self.device, non_blocking=True)
            self._sync_device()
            h2d_end = time.perf_counter()

            feat_start = time.perf_counter()
            with torch.no_grad():
                with autocast_context(self.device, self.amp_cfg):
                    features = self._extract_features(image)
            self._sync_device()
            feat_end = time.perf_counter()

            if self.linear_evaluator is None:
                in_dim = int(features.shape[1])
                self.linear_evaluator = LinearMappingMultiHead(
                    num_classes=self.num_classes,
                    in_dim=in_dim,
                    local_rank=self.local_rank,
                    lr_list=self.test_lrs,
                    head_configs=self.head_specs,
                    optimizer_cfg=self.optimizer_cfg,
                    projection_dim=self.projection_dim,
                )

            head_start = time.perf_counter()
            self.linear_evaluator.update(features.float(), label)
            self._sync_device()
            head_end = time.perf_counter()

            if step < self.profile_steps:
                profile_stats["data_wait"] += batch_wait_end - batch_wait_start
                profile_stats["h2d"] += h2d_end - h2d_start
                profile_stats["feature_extract"] += feat_end - feat_start
                profile_stats["head_update"] += head_end - head_start
                profiled_steps += 1
                if profiled_steps >= self.profile_steps and self.profile_only:
                    _emit_profile()
                    self.profile_complete = True
                    break

        if self.linear_evaluator is None:
            raise RuntimeError("Linear evaluator was not initialized. Check train dataset/loader.")

        if profiled_steps > 0 and not self.profile_complete:
            _emit_profile()

        for head_metrics in self.linear_metrics.values():
            for metric in head_metrics.values():
                metric.reset()

        self._print("Linear probe training completed.")

    def test_step(self, batch):
        with torch.inference_mode():
            image = batch[0].to(self.device, non_blocking=True)
            label = batch[1].to(self.device, non_blocking=True)

            with autocast_context(self.device, self.amp_cfg):
                features = self._extract_features(image)
            pred_map = self.linear_evaluator.predict(features.float())

            for combo_key, head_name, _, lr in self.linear_evaluator.iter_combos():
                pred = pred_map[combo_key]
                pred_upsampled = F.interpolate(
                    pred, label.shape[-2:], mode="bilinear", align_corners=False
                ).argmax(1)
                self.linear_metrics[head_name][lr].update(pred_upsampled, label)

    def run_test(self, test_loader):
        """Run training and evaluation and return per-head results."""
        self.setup_linear_head()
        if self.profile_complete and self.profile_only:
            self._print("Profiling-only run: skipping validation.")
            return {"per_head_results": {}}

        self.model.eval()
        self.linear_evaluator.eval()

        test_iterator = test_loader
        if self.rank == 0:
            test_iterator = tqdm(test_loader, desc="Testing", ascii=True)
        for batch in test_iterator:
            self.test_step(batch)

        per_head_results = {}
        for head_name in self.head_names:
            best_miou, best_acc = -1.0, -1.0
            best_lr = None
            per_lr_results = {}

            for lr in self.test_lrs:
                metrics = self.linear_metrics[head_name][lr].compute()
                metric_prefix = f"valid/linear_seg/{self.dataset_name}/head_{head_name}/lr_{lr}/"
                miou = metrics.get(metric_prefix + "mIoU", 0.0)
                acc = metrics.get(metric_prefix + "Accuracy", 0.0)

                per_lr_results[lr] = {"mIoU": miou, "Accuracy": acc}
                if miou > best_miou:
                    best_miou = miou
                    best_acc = acc
                    best_lr = lr

            per_head_results[head_name] = {
                "best_lr": best_lr,
                "best_miou": best_miou,
                "best_acc": best_acc,
                "per_lr_results": per_lr_results,
            }

        if self.rank == 0:
            self._print(f"\n{'=' * 60}")
            self._print(f"Results for {self.dataset_name}:")
            for head_name in self.head_names:
                result = per_head_results[head_name]
                self._print(
                    f"  [{head_name}] best LR={result['best_lr']}, "
                    f"mIoU={result['best_miou']:.2f}, Acc={result['best_acc']:.2f}"
                )
                for lr, values in result["per_lr_results"].items():
                    marker = " <-- best" if lr == result["best_lr"] else ""
                    self._print(
                        f"    LR={lr}: mIoU={values['mIoU']:.2f}, Acc={values['Accuracy']:.2f}{marker}"
                    )
            self._print(f"{'=' * 60}")

        del self.linear_evaluator
        torch.cuda.empty_cache()
        return {"per_head_results": per_head_results}
