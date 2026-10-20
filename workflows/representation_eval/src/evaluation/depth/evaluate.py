"""Main entrypoint for NYUv2 linear depth probing."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import yaml

from src.evaluation.common import extract_spatial_features
from src.evaluation.common.distributed import (
    autocast_context as _autocast_context,
    dist_enabled as _dist_enabled,
    is_main_process as _is_main_process,
    rank as _rank,
    remove_ddp_wrapper as _remove_ddp_wrapper,
    world_size as _world_size,
)

from .dataset import build_depth_dataset, build_depth_loader
from .loss import MultiLoss
from .metrics import DepthMetricsMeter, SUPPORTED_DEPTH_METRICS, calculate_depth_metrics
from .models import DepthHeadModel
from .schedulers import WarmupOneCycleLR, build_scheduler
from .utils import align_depth_least_square, make_valid_mask
from .visualization import save_depth_prediction


logger = logging.getLogger("ssl_eval.depth")


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
    return per_rank_batch_size, int(global_batch_size)


def _cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _center_pad_to_multiple(images: torch.Tensor, multiple: int) -> torch.Tensor:
    if multiple <= 1:
        return images
    _, _, height, width = images.shape
    new_height = ((height + multiple - 1) // multiple) * multiple
    new_width = ((width + multiple - 1) // multiple) * multiple
    pad_h = new_height - height
    pad_w = new_width - width
    if pad_h == 0 and pad_w == 0:
        return images
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return torch.nn.functional.pad(images, (pad_left, pad_right, pad_top, pad_bottom))


def _adapt_images_to_backbone(images: torch.Tensor, backbone) -> torch.Tensor:
    input_pad_size = getattr(backbone, "input_pad_size", None)
    if input_pad_size is None:
        input_pad_size = getattr(backbone, "patch_size", None)
    if input_pad_size is None and hasattr(backbone, "patch_embed"):
        input_pad_size = getattr(backbone.patch_embed, "patch_size", None)
    if input_pad_size is None:
        return images
    if isinstance(input_pad_size, (tuple, list)):
        input_pad_size = int(input_pad_size[0])
    return _center_pad_to_multiple(images, int(input_pad_size))


def _extract_features(backbone, images: torch.Tensor, cfg: Dict, device: torch.device):
    images = _adapt_images_to_backbone(images, backbone)
    with torch.no_grad():
        with _autocast_context(device, cfg["train"]["amp"]):
            return extract_spatial_features(
                backbone,
                images,
                model_type=cfg["model"]["model_type"],
                feature_cfg=cfg["model"]["feature_source"],
            ).detach()


def _build_optimizer(model: torch.nn.Module, train_cfg: Dict):
    optimizer_cfg = train_cfg["optimizer"]
    opt_type = optimizer_cfg["type"].lower()
    lr = float(optimizer_cfg["lr"])
    weight_decay = float(optimizer_cfg["weight_decay"])
    betas = tuple(float(v) for v in optimizer_cfg["betas"])
    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer type: {opt_type}")


def _build_optimizer_with_lr(model: torch.nn.Module, train_cfg: Dict, lr: float):
    optimizer_cfg = train_cfg["optimizer"]
    opt_type = optimizer_cfg["type"].lower()
    weight_decay = float(optimizer_cfg["weight_decay"])
    betas = tuple(float(v) for v in optimizer_cfg["betas"])
    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=float(lr), betas=betas, weight_decay=weight_decay)
    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=float(lr), betas=betas, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer type: {opt_type}")


def _build_scheduler_for_lr(optimizer: torch.optim.Optimizer, train_cfg: Dict):
    scheduler_cfg = train_cfg["scheduler"]
    scheduler_type = scheduler_cfg["type"]
    if scheduler_type != "warmup_onecycle":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    return WarmupOneCycleLR(
        optimizer=optimizer,
        total_steps=int(train_cfg["max_steps"]),
        warmup_iters=int(scheduler_cfg.get("warmup_iters", 0)),
        max_lr=None,
        base_momentum=float(scheduler_cfg.get("base_momentum", 0.85)),
        max_momentum=float(scheduler_cfg.get("max_momentum", 0.95)),
        final_div_factor=float(scheduler_cfg.get("final_div_factor", 1000.0)),
    )


def _lr_to_key(lr: float) -> str:
    return f"lr_{float(lr):.8g}".replace("-", "m").replace(".", "_")


def _lr_to_text(lr: float) -> str:
    return f"{float(lr):.8g}"


class DepthHeadLRSweep(nn.Module):
    """Train multiple depth heads with different LRs on shared frozen features."""

    def __init__(
        self,
        *,
        in_channels: int,
        head_cfg: Dict,
        loss_cfg: Dict,
        train_cfg: Dict,
        lr_list: List[float],
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.lr_list = [float(lr) for lr in lr_list]
        self.key_to_lr = {_lr_to_key(lr): float(lr) for lr in self.lr_list}
        self.models = nn.ModuleDict()
        self.criteria = nn.ModuleDict()
        self.optims: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {}
        self.distributed = _dist_enabled()
        self.world_size = _world_size()

        for lr in self.lr_list:
            key = _lr_to_key(lr)
            model = DepthHeadModel(in_channels=in_channels, head_cfg=deepcopy(head_cfg)).to(device)
            self.models[key] = model
            self.criteria[key] = MultiLoss(deepcopy(loss_cfg)).to(device)
            self.optims[key] = _build_optimizer_with_lr(model, train_cfg, lr=lr)
            self.schedulers[key] = _build_scheduler_for_lr(self.optims[key], train_cfg)
        self._broadcast_initial_state()

    def iter_heads(self):
        for key in self.models:
            yield key, self.key_to_lr[key], self.models[key]

    def _broadcast_initial_state(self):
        if not self.distributed:
            return
        for module in list(self.models.values()) + list(self.criteria.values()):
            for tensor in module.state_dict().values():
                if torch.is_tensor(tensor):
                    dist.broadcast(tensor, src=0)

    def _sync_gradients(self):
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

    def train_step(
        self,
        *,
        features: torch.Tensor,
        depth_gt: torch.Tensor,
        valid_mask: torch.Tensor,
        cfg: Dict,
        step: int,
        target_hw: Tuple[int, int],
    ) -> Dict[str, float]:
        for optimizer in self.optims.values():
            optimizer.zero_grad(set_to_none=True)

        loss_values: Dict[str, float] = {}
        for key, lr, model in self.iter_heads():
            with _autocast_context(self.device, cfg["train"]["amp"]):
                preds = model(features.detach())
            preds = _resize_depth_predictions(preds, target_hw)
            loss = self.criteria[key](preds, depth_gt, valid_mask)
            if not torch.isfinite(loss):
                if _is_main_process():
                    logger.error(
                        "Non-finite depth loss at step %s lr=%s. amp=%s valid_pixels=%s/%s pred_finite=%s/%s pred_%s depth_%s",
                        step + 1,
                        lr,
                        cfg["train"].get("amp", {}),
                        int(valid_mask.sum().item()),
                        int(valid_mask.numel()),
                        int(torch.isfinite(preds).sum().item()),
                        int(preds.numel()),
                        _finite_range_text(preds.detach()),
                        _finite_range_text(depth_gt.detach()),
                    )
                raise FloatingPointError(f"Non-finite depth loss at step {step + 1} lr={lr}")
            loss.backward()
            loss_values[key] = float(loss.detach().item())

        self._sync_gradients()
        for key, model in self.models.items():
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            self.optims[key].step()
            self.schedulers[key].step()

        return loss_values

    @torch.no_grad()
    def predict(self, features: torch.Tensor, cfg: Dict) -> Dict[str, torch.Tensor]:
        outputs = {}
        for key, _, model in self.iter_heads():
            with _autocast_context(self.device, cfg["train"]["amp"]):
                outputs[key] = model(features)
        return outputs

    def state_for_save(self, best_key: str | None = None) -> Dict:
        return {
            "best_lr": self.key_to_lr.get(best_key) if best_key is not None else None,
            "state_dicts": {
                key: {
                    "lr": self.key_to_lr[key],
                    "state_dict": model.state_dict(),
                }
                for key, model in self.models.items()
            },
        }


def _predict_depth(
    backbone,
    head_model: torch.nn.Module,
    images: torch.Tensor,
    cfg: Dict,
    device: torch.device,
    *,
    use_tta: bool = False,
):
    features = _extract_features(backbone, images, cfg, device=device).detach()
    with _autocast_context(device, cfg["train"]["amp"]):
        preds = head_model(features)

    if not use_tta:
        return preds

    flipped_images = torch.flip(images, dims=[-1])
    flipped_features = _extract_features(backbone, flipped_images, cfg, device=device).detach()
    with _autocast_context(device, cfg["train"]["amp"]):
        flipped_preds = head_model(flipped_features)
    flipped_preds = torch.flip(flipped_preds, dims=[-1])
    return (preds + flipped_preds) / 2.0


def _finite_range_text(tensor: torch.Tensor) -> str:
    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return "no finite values"
    return f"min={float(finite.min().item()):.6g} max={float(finite.max().item()):.6g}"


def _resize_depth_predictions(preds: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if preds.shape[-2:] == target_hw:
        return preds

    # PyTorch 1.13 does not implement bilinear upsampling for bf16 on this stack.
    resize_input = preds.float() if preds.dtype == torch.bfloat16 else preds
    resized = torch.nn.functional.interpolate(
        resize_input,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return resized


def _persist_results(cfg: Dict, summary: Dict, history: Iterable[Dict], head_state: Dict[str, torch.Tensor]):
    output_dir = cfg.get("experiment", {}).get("output_dir")
    if not output_dir or not _is_main_process():
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "effective_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    with (output_path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_path / "results_eval_depth.jsonl").open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(record) + "\n")
    torch.save(head_state, output_path / "depth_head.pt")


def _evaluate_loader(
    *,
    split_name: str,
    loader,
    backbone,
    head_model: torch.nn.Module,
    cfg: Dict,
    data_cfg: Dict,
    device: torch.device,
):
    eval_cfg = cfg["eval"]
    data_eval_cfg = data_cfg["eval"]
    metric_names = [name for name in eval_cfg["metrics"] if name in SUPPORTED_DEPTH_METRICS]
    meter = DepthMetricsMeter(metric_names, device=device)

    module = _remove_ddp_wrapper(head_model)
    module.eval()
    backbone.eval()

    seen_indices: set = set()

    for images, depth_gt, indices in loader:
        images = images.to(device, non_blocking=True)
        depth_gt = depth_gt.to(device, non_blocking=True)

        preds = _predict_depth(
            backbone,
            head_model=head_model,
            images=images,
            cfg=cfg,
            device=device,
            use_tta=bool(eval_cfg.get("use_tta", False)),
        )

        preds = _resize_depth_predictions(preds, depth_gt.shape[-2:])

        gt_eval = torch.where(
            torch.logical_or(depth_gt >= float(eval_cfg["max_depth"]), depth_gt <= float(eval_cfg["min_depth"])),
            torch.as_tensor(float(eval_cfg["ignored_value"]), device=device, dtype=depth_gt.dtype),
            depth_gt,
        )
        valid_mask = make_valid_mask(
            gt_eval,
            eval_mask=data_eval_cfg["eval_mask"],
            ignored_value=float(eval_cfg["ignored_value"]),
        )

        if bool(eval_cfg.get("align_least_squares", False)):
            aligned_preds = []
            for batch_idx in range(preds.shape[0]):
                aligned_pred, _, _ = align_depth_least_square(
                    gt_eval[batch_idx : batch_idx + 1],
                    preds[batch_idx : batch_idx + 1],
                    valid_mask[batch_idx : batch_idx + 1],
                )
                aligned_preds.append(aligned_pred)
            preds = torch.cat(aligned_preds, dim=0)

        preds = preds.clamp(min=float(eval_cfg["min_depth"]), max=float(eval_cfg["max_depth"]))

        for batch_idx in range(preds.shape[0]):
            sample_idx = int(indices[batch_idx])
            if sample_idx in seen_indices:
                continue
            seen_indices.add(sample_idx)
            metric_values = calculate_depth_metrics(
                gt_eval[batch_idx : batch_idx + 1],
                preds[batch_idx : batch_idx + 1],
                valid_mask[batch_idx : batch_idx + 1],
                metric_names=metric_names,
            )
            meter.update(metric_values)
            if bool(cfg["result"].get("save_results", False)) and _is_main_process():
                save_depth_prediction(
                    image=images[batch_idx].detach().cpu(),
                    pred=preds[batch_idx].detach().cpu(),
                    target=gt_eval[batch_idx].detach().cpu(),
                    save_index=sample_idx,
                    output_dir=cfg["experiment"]["output_dir"],
                    result_cfg=cfg["result"],
                    mean=data_cfg["mean"],
                    std=data_cfg["std"],
                )

    results = meter.compute()
    if _is_main_process():
        logger.info(
            f"[{split_name}] " + " ".join(f"{name}={value:.4f}" for name, value in results.items())
        )
    return results


@torch.no_grad()
def _evaluate_loader_lr_sweep(
    *,
    split_name: str,
    loader,
    backbone,
    head_sweep: DepthHeadLRSweep,
    cfg: Dict,
    data_cfg: Dict,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    eval_cfg = cfg["eval"]
    data_eval_cfg = data_cfg["eval"]
    metric_names = [name for name in eval_cfg["metrics"] if name in SUPPORTED_DEPTH_METRICS]
    meters = {
        key: DepthMetricsMeter(metric_names, device=device)
        for key, _, _ in head_sweep.iter_heads()
    }

    head_sweep.eval()
    backbone.eval()
    seen_indices: set = set()

    for images, depth_gt, indices in loader:
        images = images.to(device, non_blocking=True)
        depth_gt = depth_gt.to(device, non_blocking=True)

        features = _extract_features(backbone, images, cfg, device=device).detach()
        preds_by_key = head_sweep.predict(features, cfg)

        if bool(eval_cfg.get("use_tta", False)):
            flipped_images = torch.flip(images, dims=[-1])
            flipped_features = _extract_features(backbone, flipped_images, cfg, device=device).detach()
            flipped_preds_by_key = head_sweep.predict(flipped_features, cfg)
            for key in preds_by_key:
                flipped_preds = torch.flip(flipped_preds_by_key[key], dims=[-1])
                preds_by_key[key] = (preds_by_key[key] + flipped_preds) / 2.0

        gt_eval = torch.where(
            torch.logical_or(depth_gt >= float(eval_cfg["max_depth"]), depth_gt <= float(eval_cfg["min_depth"])),
            torch.as_tensor(float(eval_cfg["ignored_value"]), device=device, dtype=depth_gt.dtype),
            depth_gt,
        )
        valid_mask = make_valid_mask(
            gt_eval,
            eval_mask=data_eval_cfg["eval_mask"],
            ignored_value=float(eval_cfg["ignored_value"]),
        )

        for key, preds in preds_by_key.items():
            preds = _resize_depth_predictions(preds, depth_gt.shape[-2:])

            if bool(eval_cfg.get("align_least_squares", False)):
                aligned_preds = []
                for batch_idx in range(preds.shape[0]):
                    aligned_pred, _, _ = align_depth_least_square(
                        gt_eval[batch_idx : batch_idx + 1],
                        preds[batch_idx : batch_idx + 1],
                        valid_mask[batch_idx : batch_idx + 1],
                    )
                    aligned_preds.append(aligned_pred)
                preds = torch.cat(aligned_preds, dim=0)

            preds = preds.clamp(min=float(eval_cfg["min_depth"]), max=float(eval_cfg["max_depth"]))

            for batch_idx in range(preds.shape[0]):
                sample_idx = int(indices[batch_idx])
                if sample_idx in seen_indices:
                    continue
                metric_values = calculate_depth_metrics(
                    gt_eval[batch_idx : batch_idx + 1],
                    preds[batch_idx : batch_idx + 1],
                    valid_mask[batch_idx : batch_idx + 1],
                    metric_names=metric_names,
                )
                meters[key].update(metric_values)

        for batch_idx in range(depth_gt.shape[0]):
            seen_indices.add(int(indices[batch_idx]))

    results = {key: meter.compute() for key, meter in meters.items()}
    if _is_main_process():
        for key, metrics in results.items():
            lr = head_sweep.key_to_lr[key]
            logger.info(
                f"[{split_name}][lr={lr:g}] " + " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
            )
    return results


def _select_best_lr_key(val_results_by_key: Dict[str, Dict[str, float]]) -> str:
    if not val_results_by_key:
        raise ValueError("Cannot select best depth LR from empty validation results.")
    if all("rmse" in metrics for metrics in val_results_by_key.values()):
        return min(val_results_by_key, key=lambda key: float(val_results_by_key[key]["rmse"]))
    if all("abs_rel" in metrics for metrics in val_results_by_key.values()):
        return min(val_results_by_key, key=lambda key: float(val_results_by_key[key]["abs_rel"]))
    if all("a1" in metrics for metrics in val_results_by_key.values()):
        return max(val_results_by_key, key=lambda key: float(val_results_by_key[key]["a1"]))
    first_metric = next(iter(next(iter(val_results_by_key.values())).keys()))
    return min(val_results_by_key, key=lambda key: float(val_results_by_key[key][first_metric]))


def _evaluate_depth_lr_sweep(
    *,
    backbone,
    cfg: Dict,
    train_loader,
    train_sampler,
    val_loader,
    test_loader,
    train_data_cfg: Dict,
    test_data_cfg: Dict,
    train_batch_size: int,
    train_global_batch_size: int,
    eval_batch_size: int,
    eval_global_batch_size: int,
    in_channels: int,
    device: torch.device,
) -> Dict:
    lr_list = [float(lr) for lr in cfg["head_search"]["lrs"]]
    if _is_main_process():
        logger.info(f"  Depth head LR sweep: {lr_list}")
        logger.info("  Training all LR heads in parallel on shared frozen features.")
        logger.info("  Selecting best LR by final validation RMSE.")

    head_sweep = DepthHeadLRSweep(
        in_channels=in_channels,
        head_cfg=cfg["head"],
        loss_cfg=cfg["loss"],
        train_cfg=cfg["train"],
        lr_list=lr_list,
        device=device,
    )

    train_iterator = _cycle_loader(train_loader)
    max_steps = int(cfg["train"]["max_steps"])
    log_losses = {key: 0.0 for key, _, _ in head_sweep.iter_heads()}

    for step in range(max_steps):
        if train_sampler is not None and step % max(1, len(train_loader)) == 0:
            train_sampler.set_epoch(step // max(1, len(train_loader)))

        images, depth_gt, _ = next(train_iterator)
        images = images.to(device, non_blocking=True)
        depth_gt = depth_gt.to(device, non_blocking=True)

        features = _extract_features(backbone, images, cfg, device=device).detach()
        valid_mask = make_valid_mask(
            depth_gt,
            eval_mask=train_data_cfg["train"]["eval_mask"],
            ignored_value=float(cfg["eval"]["ignored_value"]),
        )
        depth_gt = depth_gt.clone()
        depth_gt[~valid_mask] = float(cfg["eval"]["ignored_value"])

        loss_values = head_sweep.train_step(
            features=features,
            depth_gt=depth_gt,
            valid_mask=valid_mask,
            cfg=cfg,
            step=step,
            target_hw=depth_gt.shape[-2:],
        )
        for key, value in loss_values.items():
            log_losses[key] = value

        if _is_main_process() and ((step + 1) % 50 == 0 or step == 0):
            loss_text = " ".join(
                f"lr={head_sweep.key_to_lr[key]:g}:{log_losses[key]:.6f}" for key in log_losses
            )
            logger.info(f"Step {step + 1}/{max_steps} {loss_text}")

    val_by_key = _evaluate_loader_lr_sweep(
        split_name="val",
        loader=val_loader,
        backbone=backbone,
        head_sweep=head_sweep,
        cfg=cfg,
        data_cfg=train_data_cfg,
        device=device,
    )
    best_key = _select_best_lr_key(val_by_key)
    test_by_key = _evaluate_loader_lr_sweep(
        split_name="test",
        loader=test_loader,
        backbone=backbone,
        head_sweep=head_sweep,
        cfg=cfg,
        data_cfg=test_data_cfg,
        device=device,
    )

    per_lr_results = {}
    history = []
    for key, lr in head_sweep.key_to_lr.items():
        lr_text = _lr_to_text(lr)
        per_lr_results[lr_text] = {
            "val": val_by_key[key],
            "test": test_by_key[key],
        }
        history.append({"iteration": max_steps, "split": "val", "lr": lr, **val_by_key[key]})
        history.append({"iteration": max_steps, "split": "test", "lr": lr, **test_by_key[key]})

    best_lr = head_sweep.key_to_lr[best_key]
    summary = {
        "train_batch_size_per_gpu": train_batch_size,
        "train_global_batch_size": train_global_batch_size,
        "eval_batch_size_per_gpu": eval_batch_size,
        "eval_global_batch_size": eval_global_batch_size,
        "feature_channels": in_channels,
        "head_search_lrs": lr_list,
        "best_lr": best_lr,
        "selection_metric": "val/rmse",
        "val": val_by_key[best_key],
        "test": test_by_key[best_key],
        "per_lr_results": per_lr_results,
    }
    head_state = head_sweep.state_for_save(best_key=best_key)
    head_state["head_config"] = cfg["head"]
    _persist_results(cfg, summary, history, head_state)
    return summary


def evaluate_depth(model, cfg: Dict) -> Dict:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    seed = int(cfg.get("experiment", {}).get("seed", 0)) + _rank()
    _set_seed(seed)

    world_size = _world_size()
    distributed_cfg = cfg.get("distributed", {})
    num_workers = int(distributed_cfg.get("num_workers", 4))
    persistent_workers = bool(distributed_cfg.get("persistent_workers", False))

    train_batch_size, train_global_batch_size = _resolve_per_rank_batch_size(
        batch_size=cfg["train"].get("batch_size"),
        global_batch_size=cfg["train"].get("global_batch_size"),
        world_size=world_size,
        field_name="train",
    )
    eval_batch_size, eval_global_batch_size = _resolve_per_rank_batch_size(
        batch_size=cfg["eval"].get("batch_size"),
        global_batch_size=cfg["eval"].get("global_batch_size"),
        world_size=world_size,
        field_name="eval",
    )

    if _is_main_process():
        train_dataset_name = cfg["data"].get("dataset", "depth")
        test_dataset_name = cfg.get("eval_data", cfg["data"]).get("dataset", train_dataset_name)
        if test_dataset_name == train_dataset_name:
            logger.info(f"Starting {train_dataset_name} depth evaluation")
        else:
            logger.info(f"Starting depth transfer evaluation: train={train_dataset_name}, test={test_dataset_name}")
        logger.info(f"  Train batch size / GPU: {train_batch_size}")
        logger.info(f"  Train global batch size: {train_global_batch_size}")
        logger.info(f"  Eval batch size / GPU: {eval_batch_size}")
        logger.info(f"  Eval global batch size: {eval_global_batch_size}")

    backbone = model.to(device)
    backbone.eval()
    backbone.requires_grad_(False)

    train_data_cfg = cfg["data"]
    test_data_cfg = cfg.get("eval_data", train_data_cfg)
    train_dataset = build_depth_dataset(cfg, split="train", data_cfg=train_data_cfg)
    val_dataset = build_depth_dataset(cfg, split=cfg["eval"]["val_split"], data_cfg=train_data_cfg)
    test_dataset = build_depth_dataset(cfg, split=cfg["eval"]["test_split"], data_cfg=test_data_cfg)

    train_loader, train_sampler = build_depth_loader(
        train_dataset,
        batch_size=train_batch_size,
        split="train",
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        distributed=_dist_enabled(),
        rank=_rank(),
        world_size=world_size,
        seed=seed,
    )
    val_loader, _ = build_depth_loader(
        val_dataset,
        batch_size=eval_batch_size,
        split="val",
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        distributed=_dist_enabled(),
        rank=_rank(),
        world_size=world_size,
    )
    test_loader, _ = build_depth_loader(
        test_dataset,
        batch_size=eval_batch_size,
        split="test",
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        distributed=_dist_enabled(),
        rank=_rank(),
        world_size=world_size,
    )

    sample_images, _, _ = next(iter(train_loader))
    sample_images = sample_images.to(device, non_blocking=True)
    sample_features = _extract_features(backbone, sample_images, cfg, device=device)
    in_channels = int(sample_features.shape[1])

    depth_lrs = [float(lr) for lr in cfg.get("head_search", {}).get("lrs", [cfg["train"]["optimizer"]["lr"]])]
    if len(depth_lrs) > 1:
        return _evaluate_depth_lr_sweep(
            backbone=backbone,
            cfg=cfg,
            train_loader=train_loader,
            train_sampler=train_sampler,
            val_loader=val_loader,
            test_loader=test_loader,
            train_data_cfg=train_data_cfg,
            test_data_cfg=test_data_cfg,
            train_batch_size=train_batch_size,
            train_global_batch_size=train_global_batch_size,
            eval_batch_size=eval_batch_size,
            eval_global_batch_size=eval_global_batch_size,
            in_channels=in_channels,
            device=device,
        )

    head_model = DepthHeadModel(in_channels=in_channels, head_cfg=cfg["head"]).to(device)
    if _dist_enabled():
        device_ids = [device.index] if device.type == "cuda" else None
        head_model = DistributedDataParallel(head_model, device_ids=device_ids)

    criterion = MultiLoss(cfg["loss"]).to(device)
    optimizer = _build_optimizer(_remove_ddp_wrapper(head_model), cfg["train"])
    scheduler = build_scheduler(optimizer, cfg["train"])

    history = []
    train_iterator = _cycle_loader(train_loader)
    eval_period = int(cfg["eval"].get("eval_period_iterations", 0))

    for step in range(int(cfg["train"]["max_steps"])):
        if train_sampler is not None and step % max(1, len(train_loader)) == 0:
            train_sampler.set_epoch(step // max(1, len(train_loader)))

        images, depth_gt, _ = next(train_iterator)
        images = images.to(device, non_blocking=True)
        depth_gt = depth_gt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = _predict_depth(backbone, head_model, images, cfg=cfg, device=device, use_tta=False)
        preds = _resize_depth_predictions(preds, depth_gt.shape[-2:])

        valid_mask = make_valid_mask(
            depth_gt,
            eval_mask=train_data_cfg["train"]["eval_mask"],
            ignored_value=float(cfg["eval"]["ignored_value"]),
        )
        depth_gt = depth_gt.clone()
        depth_gt[~valid_mask] = float(cfg["eval"]["ignored_value"])

        loss = criterion(preds, depth_gt, valid_mask)
        if not torch.isfinite(loss):
            if _is_main_process():
                logger.error(
                    "Non-finite depth loss at step %s. amp=%s valid_pixels=%s/%s pred_finite=%s/%s pred_%s depth_%s",
                    step + 1,
                    cfg["train"].get("amp", {}),
                    int(valid_mask.sum().item()),
                    int(valid_mask.numel()),
                    int(torch.isfinite(preds).sum().item()),
                    int(preds.numel()),
                    _finite_range_text(preds.detach()),
                    _finite_range_text(depth_gt.detach()),
                )
            raise FloatingPointError(f"Non-finite depth loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_remove_ddp_wrapper(head_model).parameters(), float(cfg["train"]["grad_clip"]))
        optimizer.step()
        scheduler.step()

        if _is_main_process() and ((step + 1) % 50 == 0 or step == 0):
            logger.info(f"Step {step + 1}/{cfg['train']['max_steps']} loss={float(loss.item()):.6f}")

        if eval_period > 0 and (step + 1) % eval_period == 0 and step + 1 < int(cfg["train"]["max_steps"]):
            val_results = _evaluate_loader(
                split_name="val",
                loader=val_loader,
                backbone=backbone,
                head_model=head_model,
                cfg=cfg,
                data_cfg=train_data_cfg,
                device=device,
            )
            history.append({"iteration": step + 1, "split": "val", **val_results})

    val_results = _evaluate_loader(
        split_name="val",
        loader=val_loader,
        backbone=backbone,
        head_model=head_model,
        cfg=cfg,
        data_cfg=train_data_cfg,
        device=device,
    )
    test_results = _evaluate_loader(
        split_name="test",
        loader=test_loader,
        backbone=backbone,
        head_model=head_model,
        cfg=cfg,
        data_cfg=test_data_cfg,
        device=device,
    )
    history.append({"iteration": int(cfg["train"]["max_steps"]), "split": "val", **val_results})
    history.append({"iteration": int(cfg["train"]["max_steps"]), "split": "test", **test_results})

    summary = {
        "train_batch_size_per_gpu": train_batch_size,
        "train_global_batch_size": train_global_batch_size,
        "eval_batch_size_per_gpu": eval_batch_size,
        "eval_global_batch_size": eval_global_batch_size,
        "feature_channels": in_channels,
        "val": val_results,
        "test": test_results,
    }
    head_state = {
        "state_dict": _remove_ddp_wrapper(head_model).state_dict(),
        "head_config": cfg["head"],
    }
    _persist_results(cfg, summary, history, head_state)
    return summary
