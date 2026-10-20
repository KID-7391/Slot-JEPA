"""Utility classes for linear segmentation evaluation."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from scipy.optimize import linear_sum_assignment


class EvaluationMetrics:
    """Compute segmentation metrics (mIoU, Accuracy) with distributed support."""

    def __init__(self, prefix: str, n_classes: int, extra_clusters: int = 0,
                 compute_hungarian: bool = False, device=None):
        self.n_classes = n_classes
        self.extra_clusters = extra_clusters if compute_hungarian else 0
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stats = torch.zeros(n_classes + self.extra_clusters, n_classes,
                                 dtype=torch.int64, device=device)

    def reset(self):
        self.stats.zero_()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1)
            preds = preds.reshape(-1)
            mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes)
            actual = actual[mask]
            preds = preds[mask]

            if actual.numel() == 0:
                return

            self.stats += torch.bincount(
                (self.n_classes + self.extra_clusters) * actual + preds,
                minlength=self.n_classes * (self.n_classes + self.extra_clusters)
            ).reshape(self.n_classes, self.n_classes + self.extra_clusters).t().to(self.stats.device)

    def compute(self):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.stats, op=dist.ReduceOp.SUM)

        if self.compute_hungarian:
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]
            else:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0]))
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)
        else:
            self.assignments = (torch.arange(self.n_classes).unsqueeze(1),
                                torch.arange(self.n_classes).unsqueeze(1))
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)

        metric_dict = {
            self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
            self.prefix + "Accuracy": opc.item()
        }
        return {k: 100 * v for k, v in metric_dict.items()}


class UpsampleBlock(nn.Module):
    """Upsample block for FCN head."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim // 2, 1)
        self.bn1 = nn.SyncBatchNorm(in_dim // 2)
        self.conv2 = nn.Conv2d(in_dim // 2, in_dim, 3, 1, 1)
        self.bn2 = nn.SyncBatchNorm(in_dim)
        self.conv3 = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x + input)
        x0 = self.conv3(x)
        x0 = F.interpolate(x0, scale_factor=2, mode='bilinear', align_corners=False)
        return x0


def _build_head(
    head_cfg: Dict[str, Any],
    in_dim: int,
    num_classes: int,
    default_projection_dim: Optional[int] = None,
) -> nn.Module:
    """Build a segmentation head model."""
    head_type = head_cfg["type"]
    projection_dim = head_cfg.get("projection_dim", default_projection_dim)
    bias = bool(head_cfg.get("bias", True))
    dropout = float(head_cfg.get("dropout", 0.0))
    sync_bn = bool(head_cfg.get("sync_bn", True))

    layers: List[nn.Module] = []
    current_dim = in_dim

    if projection_dim is not None and projection_dim != in_dim:
        layers.append(nn.Conv2d(in_dim, projection_dim, kernel_size=1, bias=bias))
        current_dim = projection_dim
    if dropout > 0:
        layers.append(nn.Dropout2d(p=dropout))

    if head_type == "linear":
        layers.append(nn.Conv2d(current_dim, num_classes, kernel_size=1, bias=bias))
    elif head_type == "bn_linear":
        use_sync_bn = sync_bn and dist.is_available() and dist.is_initialized()
        norm_layer = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm2d
        layers.append(norm_layer(current_dim))
        layers.append(nn.Conv2d(current_dim, num_classes, kernel_size=1, bias=bias))
    else:
        raise ValueError(f"Unknown head type: {head_type}")

    if len(layers) == 1:
        return layers[0]
    return nn.Sequential(*layers)


class LinearMappingMultiHead(nn.Module):
    """Train/search multiple (head_config, lr) combinations simultaneously."""

    def __init__(
        self,
        num_classes: int,
        in_dim: int,
        local_rank: int,
        lr_list: List[float],
        head_configs: List[Dict[str, Any]],
        optimizer_cfg: Dict[str, Any] = None,
        projection_dim: int = None,
        background: int = 0,
    ):
        super(LinearMappingMultiHead, self).__init__()
        self.num_classes = num_classes
        self.background = background
        self.in_dim = in_dim
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.lr_list = [float(lr) for lr in lr_list]
        self.head_configs = [dict(cfg) for cfg in head_configs]
        self.optimizer_cfg = optimizer_cfg or {"type": "adam", "weight_decay": 0.0, "betas": [0.9, 0.999]}
        self.projection_dim = projection_dim

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
        self.distributed = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.distributed else 1

        self.combo_specs: List[Tuple[Dict[str, Any], float]] = [
            (head_cfg, lr) for head_cfg in self.head_configs for lr in self.lr_list
        ]
        self.key_to_combo: Dict[str, Dict[str, Any]] = {}
        self.models = nn.ModuleDict()
        self.optims: Dict[str, torch.optim.Optimizer] = {}
        self.profile_steps = max(0, int(os.environ.get("LINEAR_SEG_HEAD_PROFILE_STEPS", "0")))
        self.profile_stats = {
            "validity_check": 0.0,
            "zero_grad": 0.0,
            "forward_interp_loss_backward": 0.0,
            "sync_gradients": 0.0,
            "optimizer_step": 0.0,
        }
        self.profiled_steps = 0
        self.profile_printed = False
        self.reset()

    @staticmethod
    def _combo_key(head_name: str, lr: float) -> str:
        return f"{head_name}__lr_{str(lr).replace('-', 'm').replace('.', '_')}"

    def _make_optimizer(self, model: nn.Module, lr: float) -> torch.optim.Optimizer:
        opt_cfg = self.optimizer_cfg
        opt_type = opt_cfg.get("type", "adam").lower()
        weight_decay = float(opt_cfg.get("weight_decay", 0.0))
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))

        if opt_type == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        if opt_type == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    def reset(self):
        self.models = nn.ModuleDict()
        self.optims = {}
        self.key_to_combo = {}

        for head_cfg, lr in self.combo_specs:
            head_name = str(head_cfg["name"])
            key = self._combo_key(head_name, lr)
            model = _build_head(
                head_cfg=head_cfg,
                in_dim=self.in_dim,
                num_classes=self.num_classes + self.background,
                default_projection_dim=self.projection_dim,
            )
            model.to(self.device)
            self.models[key] = model
            self.key_to_combo[key] = {"head_name": head_name, "head_cfg": dict(head_cfg), "lr": lr}
            self.optims[key] = self._make_optimizer(model, lr)

    def iter_combos(self) -> List[Tuple[str, str, Dict[str, Any], float]]:
        return [
            (key, spec["head_name"], spec["head_cfg"], float(spec["lr"]))
            for key, spec in self.key_to_combo.items()
        ]

    def _sync_gradients(self):
        """All-reduce gradients of all heads in a single fused operation."""
        if not self.distributed:
            return

        grads = []
        for model in self.models.values():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grads.append(parameter.grad.data.view(-1))

        if not grads:
            return

        flat = torch.cat(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world_size)

        offset = 0
        for model in self.models.values():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    numel = parameter.grad.numel()
                    parameter.grad.data.copy_(flat[offset : offset + numel].view_as(parameter.grad))
                    offset += numel

    def _sync_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _maybe_print_profile(self):
        if self.profile_printed or self.profiled_steps <= 0:
            return
        print("Linear-seg head profiling (avg over first %d steps):" % self.profiled_steps)
        for key, total_time in self.profile_stats.items():
            print(f"  {key}: {1000.0 * total_time / self.profiled_steps:.1f} ms")
        self.profile_printed = True

    @torch.enable_grad()
    def update(self, feats, label):
        validity_start = time.perf_counter()
        label_for_loss = label
        invalid_label = (label_for_loss < 0) | (label_for_loss >= self.num_classes)
        if invalid_label.any():
            label_for_loss = label_for_loss.clone()
            label_for_loss[invalid_label] = 255
        label_flat = label_for_loss.view(-1)
        mask = (label_flat >= 0) & (label_flat < self.num_classes)
        has_local_valid = bool(mask.any().item())

        if self.distributed:
            has_valid = torch.tensor([int(has_local_valid)], device=self.device)
            dist.all_reduce(has_valid, op=dist.ReduceOp.SUM)
            if has_valid.item() == 0:
                return
        elif not has_local_valid:
            return
        self._sync_device()
        validity_end = time.perf_counter()

        zero_grad_start = time.perf_counter()
        for optimizer in self.optims.values():
            optimizer.zero_grad(set_to_none=True)
        self._sync_device()
        zero_grad_end = time.perf_counter()

        compute_start = time.perf_counter()
        for model in self.models.values():
            logit = model(feats.data)
            logit = F.interpolate(logit, label_for_loss.shape[-2:], mode="bilinear", align_corners=False)
            if has_local_valid:
                loss = self.loss_fn(logit, label_for_loss)
                loss.backward()
            else:
                loss = (logit * 0).sum()
                loss.backward()
        self._sync_device()
        compute_end = time.perf_counter()

        sync_start = time.perf_counter()
        self._sync_gradients()
        self._sync_device()
        sync_end = time.perf_counter()

        step_start = time.perf_counter()
        for optimizer in self.optims.values():
            optimizer.step()
        self._sync_device()
        step_end = time.perf_counter()

        if self.profiled_steps < self.profile_steps:
            self.profile_stats["validity_check"] += validity_end - validity_start
            self.profile_stats["zero_grad"] += zero_grad_end - zero_grad_start
            self.profile_stats["forward_interp_loss_backward"] += compute_end - compute_start
            self.profile_stats["sync_gradients"] += sync_end - sync_start
            self.profile_stats["optimizer_step"] += step_end - step_start
            self.profiled_steps += 1
            if self.profiled_steps >= self.profile_steps:
                self._maybe_print_profile()

    @torch.no_grad()
    def predict(self, x) -> Dict[str, torch.Tensor]:
        return {key: model(x) for key, model in self.models.items()}
