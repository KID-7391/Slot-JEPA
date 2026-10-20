"""Depth losses."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class GradientLoss(nn.Module):
    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, valid_mask: torch.Tensor | None = None):
        input_downscaled = [input_tensor] + [input_tensor[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        target_downscaled = [target_tensor] + [target_tensor[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        if valid_mask is not None:
            mask_downscaled = [valid_mask] + [valid_mask[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        else:
            mask_downscaled = [torch.ones_like(target, dtype=bool) for target in target_downscaled]

        gradient_loss = 0.0
        for pred, target, mask in zip(input_downscaled, target_downscaled, mask_downscaled):
            n_valid = torch.sum(mask).clamp_min(1)
            diff = (pred - target) * mask
            v_gradient = torch.abs(diff[..., 0:-2, :] - diff[..., 2:, :])
            v_mask = mask[..., 0:-2, :] * mask[..., 2:, :]
            h_gradient = torch.abs(diff[..., :, 0:-2] - diff[..., :, 2:])
            h_mask = mask[..., :, 0:-2] * mask[..., :, 2:]
            gradient_loss += (torch.sum(v_gradient * v_mask) + torch.sum(h_gradient * h_mask)) / n_valid
        return gradient_loss


class GradientLogLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 0.001

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, valid_mask: torch.Tensor | None = None):
        input_downscaled = [input_tensor] + [input_tensor[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        target_downscaled = [target_tensor] + [target_tensor[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        if valid_mask is not None:
            mask_downscaled = [valid_mask] + [valid_mask[..., :: 2 * i, :: 2 * i] for i in range(1, 4)]
        else:
            mask_downscaled = [torch.ones_like(target, dtype=bool) for target in target_downscaled]

        gradient_loss = 0.0
        for pred, target, mask in zip(input_downscaled, target_downscaled, mask_downscaled):
            n_valid = torch.sum(mask).clamp_min(1)
            diff = (torch.log(pred + self.eps) - torch.log(target + self.eps)) * mask
            v_gradient = torch.abs(diff[..., 0:-2, :] - diff[..., 2:, :])
            v_mask = mask[..., 0:-2, :] * mask[..., 2:, :]
            h_gradient = torch.abs(diff[..., :, 0:-2] - diff[..., :, 2:])
            h_mask = mask[..., :, 0:-2] * mask[..., :, 2:]
            gradient_loss += (torch.sum(v_gradient * v_mask) + torch.sum(h_gradient * h_mask)) / n_valid
        return gradient_loss


class L1Loss(nn.Module):
    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor, valid_mask: torch.Tensor | None = None):
        loss = nn.functional.l1_loss(input_tensor, target_tensor, reduction="none")
        mask = valid_mask if valid_mask is not None else torch.ones_like(input_tensor, dtype=bool)
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-7)


class SigLoss(nn.Module):
    def __init__(self, warm_up: bool = True, warm_iter: int = 100):
        super().__init__()
        self.eps = 0.001
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.register_buffer("warm_up_counter", torch.tensor(0, dtype=torch.long), persistent=False)

    def forward(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor, valid_mask: torch.Tensor | None = None):
        if valid_mask is None:
            valid_mask = torch.ones_like(depth_gt, dtype=bool)
        pred = depth_pred[valid_mask]
        target = depth_gt[valid_mask]
        g = torch.log(pred + self.eps) - torch.log(target + self.eps)
        d_g = 0.15 * torch.pow(torch.mean(g), 2)
        if self.warm_up and self.warm_up_counter.item() < self.warm_iter:
            self.warm_up_counter += 1
        else:
            d_g += torch.var(g)
        return torch.sqrt(torch.abs(d_g))


class MultiLoss(nn.Module):
    LOSS_FACTORIES = {
        "sigloss": lambda: SigLoss(warm_up=True, warm_iter=100),
        "gradient_loss": GradientLoss,
        "gradient_log_loss": GradientLogLoss,
        "l1": L1Loss,
    }

    def __init__(self, loss_cfg: Dict[str, float]):
        super().__init__()
        self.losses = nn.ModuleDict({name: self.LOSS_FACTORIES[name]() for name in loss_cfg.keys()})
        self.weights = {name: float(weight) for name, weight in loss_cfg.items()}

    def forward(self, depth_pred: torch.Tensor, depth_gt: torch.Tensor, valid_mask: torch.Tensor | None = None):
        total_loss = 0.0
        for name, loss_module in self.losses.items():
            total_loss += self.weights[name] * loss_module(depth_pred, depth_gt, valid_mask)
        return total_loss
