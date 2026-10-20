"""Depth evaluation metrics."""

from __future__ import annotations

from typing import Dict, Iterable

import math

import torch


SUPPORTED_DEPTH_METRICS = {
    "a1",
    "a2",
    "a3",
    "abs_rel",
    "rmse",
    "log_10",
    "rmse_log",
    "silog",
    "sq_rel",
    "mae",
}


def calculate_depth_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    valid_mask: torch.Tensor | None,
    metric_names: Iterable[str],
) -> Dict[str, float]:
    names = list(metric_names)
    if valid_mask is None:
        valid_mask = torch.ones_like(gt, dtype=bool)
    valid_mask = torch.logical_and(valid_mask, gt > 0)
    if valid_mask.sum() == 0:
        return {name: float("nan") for name in names}

    gt = gt[valid_mask]
    pred = pred[valid_mask]

    metrics: Dict[str, float] = {}
    thresh = torch.maximum(gt / pred, pred / gt)
    if "a1" in names:
        metrics["a1"] = float((thresh < 1.25).float().mean())
    if "a2" in names:
        metrics["a2"] = float((thresh < 1.25**2).float().mean())
    if "a3" in names:
        metrics["a3"] = float((thresh < 1.25**3).float().mean())

    error = gt - pred
    sq_error = error**2
    if "mae" in names:
        metrics["mae"] = float(torch.mean(torch.abs(error)))
    if "abs_rel" in names:
        metrics["abs_rel"] = float(torch.mean(torch.abs(error) / gt))
    if "sq_rel" in names:
        metrics["sq_rel"] = float(torch.mean(sq_error / gt))
    if "rmse" in names:
        metrics["rmse"] = float(torch.sqrt(sq_error.mean()))

    error_log = torch.log(gt) - torch.log(pred)
    sq_error_log = error_log**2
    if "rmse_log" in names:
        metrics["rmse_log"] = float(torch.sqrt(sq_error_log.mean()))
    if "silog" in names:
        silog = torch.sqrt(torch.mean(sq_error_log) - torch.mean(error_log) ** 2) * 100
        metrics["silog"] = float(torch.nan_to_num(silog, nan=0.0))
    if "log_10" in names:
        metrics["log_10"] = float(torch.abs(torch.log10(gt) - torch.log10(pred)).mean())

    return metrics


class DepthMetricsMeter:
    def __init__(self, metric_names: Iterable[str], device: torch.device):
        self.metric_names = [name for name in metric_names]
        self.device = device
        self.sums = {name: torch.zeros(1, device=device, dtype=torch.float64) for name in self.metric_names}
        self.counts = {name: torch.zeros(1, device=device, dtype=torch.float64) for name in self.metric_names}

    def update(self, metric_values: Dict[str, float]):
        for name, value in metric_values.items():
            if name not in self.sums or math.isnan(value):
                continue
            self.sums[name] += float(value)
            self.counts[name] += 1.0

    def compute(self) -> Dict[str, float]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for tensor in list(self.sums.values()) + list(self.counts.values()):
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        results = {}
        for name in self.metric_names:
            denom = self.counts[name].item()
            results[name] = float(self.sums[name].item() / denom) if denom > 0 else float("nan")
        return results
