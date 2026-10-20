"""Shared distributed training and AMP utilities for evaluation modules."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def dist_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if dist_enabled() else 0


def world_size() -> int:
    return dist.get_world_size() if dist_enabled() else 1


def is_main_process() -> bool:
    return rank() == 0


def remove_ddp_wrapper(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def parse_amp_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def autocast_context(device: torch.device, amp_cfg: Dict):
    enabled = bool(amp_cfg.get("enabled", True)) and device.type == "cuda"
    if not enabled:
        return nullcontext()
    dtype = parse_amp_dtype(amp_cfg.get("dtype", "float16"))
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)
