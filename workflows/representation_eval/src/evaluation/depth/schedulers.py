"""Learning-rate schedulers for depth probing."""

from __future__ import annotations

import math

from torch.optim import lr_scheduler as torch_schedulers
from torch.optim.optimizer import Optimizer


BaseLRScheduler = getattr(torch_schedulers, "LRScheduler", torch_schedulers._LRScheduler)


def annealing_cos(start, end, pct):
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out


class WarmupOneCycleLR(BaseLRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_iters: int = 0,
        max_lr: float | None = None,
        base_momentum: float = 0.85,
        max_momentum: float = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1000.0,
        last_epoch: int = -1,
    ):
        self.total_steps = int(total_steps)
        self.warmup_iters = int(warmup_iters)
        self.base_momentum = float(base_momentum)
        self.max_momentum = float(max_momentum)
        self.final_div_factor = float(final_div_factor)
        self.div_factor = float(div_factor)

        for group in optimizer.param_groups:
            if "initial_lr" not in group:
                peak_lr = float(max_lr if max_lr is not None else group["lr"])
                group["initial_lr"] = peak_lr / self.div_factor
                group["max_lr"] = peak_lr
                group["min_lr"] = group["initial_lr"] / self.final_div_factor
                group["lr"] = peak_lr / self.final_div_factor if self.warmup_iters > 0 else group["initial_lr"]
                if "betas" in group:
                    group["betas"] = (self.max_momentum, *group["betas"][1:])
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch
        if step_num > self.total_steps:
            raise ValueError(
                f"Tried to step {step_num} times. The specified number of total steps is {self.total_steps}"
            )
        lrs = []
        for group in self.optimizer.param_groups:
            if step_num < self.warmup_iters:
                cosine_peak = group["max_lr"] * (1 + math.cos(math.pi * step_num / float(self.total_steps))) / 2
                lr = group["max_lr"] / self.final_div_factor + cosine_peak * step_num / max(1, self.warmup_iters)
            else:
                pct = (step_num - self.warmup_iters) / max(1, self.total_steps - self.warmup_iters)
                lr = annealing_cos(group["max_lr"], group["min_lr"], pct)
                if "betas" in group:
                    momentum = annealing_cos(self.base_momentum, self.max_momentum, pct)
                    group["betas"] = (momentum, *group["betas"][1:])
            lrs.append(lr)
        return lrs


def build_scheduler(optimizer: Optimizer, train_cfg):
    scheduler_cfg = train_cfg["scheduler"]
    scheduler_type = scheduler_cfg["type"]
    if scheduler_type != "warmup_onecycle":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    return WarmupOneCycleLR(
        optimizer=optimizer,
        total_steps=int(train_cfg["max_steps"]),
        warmup_iters=int(scheduler_cfg.get("warmup_iters", 0)),
        max_lr=float(train_cfg["optimizer"]["lr"]),
        base_momentum=float(scheduler_cfg.get("base_momentum", 0.85)),
        max_momentum=float(scheduler_cfg.get("max_momentum", 0.95)),
        final_div_factor=float(scheduler_cfg.get("final_div_factor", 1000.0)),
    )
