"""Visualization helpers for depth predictions."""

from __future__ import annotations

import os

import matplotlib
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def normalized_tensor_to_pil(tensor: torch.Tensor, mean, std) -> Image.Image:
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    image = torch.clamp(tensor * std_tensor + mean_tensor, 0.0, 1.0)
    return TF.to_pil_image(image.cpu())


def depth_tensor_to_colorized_pil(depth_tensor: torch.Tensor, cmap: str = "plasma") -> Image.Image:
    value = depth_tensor.detach().cpu().reshape(depth_tensor.shape[-2:]).numpy()
    vmin = float(value.min())
    vmax = float(value.max())
    if vmax > vmin:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.0
    colormap = matplotlib.cm.get_cmap(cmap)
    rgb = colormap(value, bytes=True)[..., :3]
    return Image.fromarray(rgb)


def save_depth_prediction(
    *,
    image: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    save_index: int,
    output_dir: str,
    result_cfg,
    mean,
    std,
):
    save_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)
    save_format = result_cfg["save_format"]
    if save_format == "pth":
        torch.save(
            {
                "image": image.cpu(),
                "pred": pred.cpu(),
                "target": target.cpu(),
            },
            os.path.join(save_dir, f"results_{int(save_index)}.pth"),
        )
        return

    image_pil = normalized_tensor_to_pil(image, mean=mean, std=std)
    pred_pil = depth_tensor_to_colorized_pil(pred)
    target_pil = depth_tensor_to_colorized_pil(target)

    if result_cfg.get("save_resolution") is not None:
        resolution = int(result_cfg["save_resolution"])
        image_pil = TF.resize(image_pil, resolution, interpolation=InterpolationMode.BILINEAR)
        pred_pil = TF.resize(pred_pil, resolution, interpolation=InterpolationMode.NEAREST)
        target_pil = TF.resize(target_pil, resolution, interpolation=InterpolationMode.NEAREST)

    band = np.concatenate([np.array(image_pil), np.array(pred_pil), np.array(target_pil)], axis=1)
    Image.fromarray(band).save(os.path.join(save_dir, f"results_{int(save_index)}.{save_format}"))
