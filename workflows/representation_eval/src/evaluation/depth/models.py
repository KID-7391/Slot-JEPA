"""Depth head models."""

from __future__ import annotations

from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn


class FeaturesToDepth(nn.Module):
    def __init__(self, min_depth: float, max_depth: float, bins_strategy: str = "linear", norm_strategy: str = "linear"):
        super().__init__()
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.bins_strategy = bins_strategy
        self.norm_strategy = norm_strategy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_bins = x.shape[1]
        if n_bins > 1:
            if self.bins_strategy == "linear":
                bins = torch.linspace(self.min_depth, self.max_depth, n_bins, device=x.device, dtype=x.dtype)
            elif self.bins_strategy == "log":
                bins = torch.linspace(
                    torch.log(torch.tensor(self.min_depth, device=x.device, dtype=x.dtype)),
                    torch.log(torch.tensor(self.max_depth, device=x.device, dtype=x.dtype)),
                    n_bins,
                    device=x.device,
                    dtype=x.dtype,
                ).exp()
            else:
                raise ValueError(f"Unsupported bins_strategy={self.bins_strategy}")

            if self.norm_strategy == "linear":
                logits = torch.relu(x) + 0.1
                logits = logits / logits.sum(dim=1, keepdim=True)
            elif self.norm_strategy == "softmax":
                logits = torch.softmax(x, dim=1)
            elif self.norm_strategy == "sigmoid":
                logits = torch.sigmoid(x)
                logits = logits / logits.sum(dim=1, keepdim=True)
            else:
                raise ValueError(f"Unsupported norm_strategy={self.norm_strategy}")
            output = torch.einsum("bkhw,k->bhw", logits, bins).unsqueeze(1)
        else:
            output = torch.relu(x) + self.min_depth
        return output


class LinearDepthHead(nn.Module):
    def __init__(self, in_channels: int, head_cfg: Dict):
        super().__init__()
        projection_dim = head_cfg.get("projection_dim")
        use_batchnorm = bool(head_cfg.get("use_batchnorm", True))
        use_sync_bn = use_batchnorm and dist.is_available() and dist.is_initialized()
        norm_cls = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm2d

        layers = []
        current_channels = in_channels
        if projection_dim is not None and int(projection_dim) != in_channels:
            projection_dim = int(projection_dim)
            layers.append(nn.Conv2d(in_channels, projection_dim, kernel_size=1, bias=bool(head_cfg.get("bias", True))))
            current_channels = projection_dim
        if use_batchnorm:
            layers.append(norm_cls(current_channels))
        dropout = float(head_cfg.get("dropout", 0.0))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.pre_depth = nn.Sequential(*layers) if layers else nn.Identity()
        self.conv_depth = nn.Conv2d(
            current_channels,
            int(head_cfg["n_output_channels"]),
            kernel_size=1,
            bias=bool(head_cfg.get("bias", True)),
        )
        nn.init.normal_(self.conv_depth.weight, mean=0.0, std=0.01)
        if self.conv_depth.bias is not None:
            nn.init.zeros_(self.conv_depth.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.pre_depth(features)
        return self.conv_depth(features)


class DepthHeadModel(nn.Module):
    def __init__(self, in_channels: int, head_cfg: Dict):
        super().__init__()
        self.decoder = LinearDepthHead(in_channels=in_channels, head_cfg=head_cfg)
        self.features_to_depth = FeaturesToDepth(
            min_depth=float(head_cfg["min_depth"]),
            max_depth=float(head_cfg["max_depth"]),
            bins_strategy=str(head_cfg["bins_strategy"]),
            norm_strategy=str(head_cfg["norm_strategy"]),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(features)
        return self.features_to_depth(logits)
