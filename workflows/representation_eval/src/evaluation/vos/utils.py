"""Shared I/O utilities for VOS evaluation."""

import glob
import math
import os

import cv2
import numpy as np
import torch
from PIL import Image

# Default alignment: LCM of common patch sizes (14, 16) = 112.
# This ensures the resized frame is divisible by any supported patch_size.
DEFAULT_ALIGN = 112


def _align(value, alignment, mode="floor"):
    """Align value to a multiple of alignment with the requested rounding mode."""
    ratio = float(value) / float(alignment)
    if mode == "floor":
        aligned = int(math.floor(ratio)) * alignment
    elif mode == "nearest":
        aligned = int(round(ratio)) * alignment
    else:
        raise ValueError(f"Unsupported align_mode={mode}")
    return max(int(aligned), int(alignment))


def color_normalize(x, mean=(0.485, 0.456, 0.406), std=(0.228, 0.224, 0.225)):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


def read_frame(frame_dir, scale_size=(480,), align=DEFAULT_ALIGN, align_mode="floor"):
    """Read a single frame, resize (aspect-preserving, aligned), and normalize."""
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if ori_h > ori_w:
            tw = _align(scale_size[0], align, align_mode)
            th = _align((tw * ori_h) / ori_w, align, align_mode)
        else:
            th = _align(scale_size[0], align, align_mode)
            tw = _align((th * ori_w) / ori_h, align, align_mode)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    return img, ori_h, ori_w


def to_one_hot(y_tensor, n_dims=None):
    """Convert integer mask to one-hot representation."""
    if n_dims is None:
        n_dims = int(y_tensor.max() + 1)
    _, h, w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h, w, n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)


def read_seg(seg_dir, target_hw, frame_hw=None, scale_size=(480,), align=DEFAULT_ALIGN, align_mode="floor"):
    """Read segmentation mask, resize to frame and feature sizes, convert to one-hot."""
    seg = Image.open(seg_dir)
    seg_ori = np.asarray(seg)
    _w, _h = seg.size  # PIL size is (w, h)
    if frame_hw is None:
        if len(scale_size) == 1:
            if _w > _h:
                _th = _align(scale_size[0], align, align_mode)
                _tw = _align((_th * _w) / _h, align, align_mode)
            else:
                _tw = _align(scale_size[0], align, align_mode)
                _th = _align((_tw * _h) / _w, align, align_mode)
        else:
            _th = scale_size[1]
            _tw = scale_size[0]
    else:
        _th, _tw = frame_hw
    seg = seg.resize((_tw, _th), 0)
    target_h, target_w = target_hw
    small_seg = np.array(seg.resize((target_w, target_h), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
    return to_one_hot(small_seg), seg_ori


def read_frame_list(video_dir):
    """Get sorted list of JPEG frames in a video directory."""
    frame_list = [img for img in glob.glob(os.path.join(video_dir, "*.jpg"))]
    frame_list = sorted(frame_list)
    return frame_list


def imwrite_indexed(filename, array, color_palette):
    """Save indexed PNG with color palette."""
    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")
    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


def load_color_palette(palette_path):
    """Load color palette from palette.txt."""
    color_palette = []
    with open(palette_path) as f:
        lines = f.read().splitlines()
    for line in lines:
        color_palette.append([int(i) for i in line.split(" ")])
    return np.asarray(color_palette, dtype=np.uint8).reshape(-1, 3)
