"""Shared utilities for depth training and evaluation."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch


logger = logging.getLogger("ssl_eval.depth")


def make_valid_mask(input_tensor: torch.Tensor, eval_mask: str = "FULL", ignored_value: float = 0.0):
    batch_size, _, height, width = input_tensor.shape
    mask = torch.zeros_like(input_tensor, dtype=torch.bool)
    if eval_mask == "NYU_EIGEN":
        y1, y2, x1, x2 = 45, 471, 41, 601
        orig_h, orig_w = 480, 640
        y1_new = int((y1 / orig_h) * height)
        y2_new = int((y2 / orig_h) * height)
        x1_new = int((x1 / orig_w) * width)
        x2_new = int((x2 / orig_w) * width)
        mask[:, :, y1_new:y2_new, x1_new:x2_new] = True
    elif eval_mask in {"KITTI_GARG", "KITTI_EIGEN"}:
        if eval_mask == "KITTI_GARG":
            y1_ratio, y2_ratio = 0.40810811, 0.99189189
        else:
            y1_ratio, y2_ratio = 0.3324324, 0.91351351
        x1_ratio, x2_ratio = 0.03594771, 0.96405229
        y1_new = int(y1_ratio * height)
        y2_new = int(y2_ratio * height)
        x1_new = int(x1_ratio * width)
        x2_new = int(x2_ratio * width)
        mask[:, :, y1_new:y2_new, x1_new:x2_new] = True
    elif eval_mask == "FULL":
        mask.fill_(True)
    else:
        raise ValueError(f"Unsupported eval_mask={eval_mask}")

    ignored_mask = ~((input_tensor == ignored_value).all(dim=1, keepdim=True))
    return mask & ignored_mask.expand(batch_size, 1, height, width)


def align_depth_least_square(
    gt_arr: np.ndarray | torch.Tensor,
    pred_arr: np.ndarray | torch.Tensor,
    valid_mask_arr: np.ndarray | torch.Tensor,
    max_resolution: Tuple[int, int] | None = None,
):
    ori_shape = pred_arr.shape
    dtype = pred_arr.dtype
    device = None
    if isinstance(pred_arr, torch.Tensor):
        assert isinstance(gt_arr, torch.Tensor)
        assert isinstance(valid_mask_arr, torch.Tensor)
        device = pred_arr.device
        pred_arr = pred_arr.detach().float().cpu().numpy()
        gt_arr = gt_arr.detach().cpu().numpy()
        valid_mask_arr = valid_mask_arr.detach().cpu().numpy()

    gt = np.squeeze(gt_arr)
    pred = np.squeeze(pred_arr)
    valid_mask = np.squeeze(valid_mask_arr).astype(bool)

    if max_resolution is not None:
        scale_factor = np.min(np.array(max_resolution) / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=float(scale_factor), mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float()).bool().numpy()

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))
    ones = np.ones_like(pred_masked)
    matrix = np.concatenate([pred_masked, ones], axis=-1)

    try:
        solution = np.linalg.lstsq(matrix, gt_masked, rcond=None)[0]
        scale, shift = solution
    except np.linalg.LinAlgError:
        scale = 1.0
        shift = 0.0
        logger.info("Least-squares depth alignment failed; falling back to identity scale/shift.")

    aligned_pred = pred_arr * scale + shift
    aligned_pred = aligned_pred.reshape(ori_shape)
    if isinstance(aligned_pred, np.ndarray):
        aligned_pred = torch.as_tensor(aligned_pred, dtype=dtype, device=device)
    return aligned_pred, float(scale), float(shift)
