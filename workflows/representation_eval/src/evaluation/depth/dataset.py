"""Paired RGB/depth datasets and transforms for depth evaluation."""

from __future__ import annotations

import math
import os
import random
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import InterpolationMode


NYU_CROP_BOX = (43, 45, 608, 472)
KITTI_CROP_SIZE = (352, 1216)


def _parse_size(value: Any):
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [int(value[0]), int(value[1])]
    raise ValueError(f"Invalid size specification: {value}")


def _scaled_crop_box(width: int, height: int, crop_box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = crop_box
    orig_w, orig_h = 640, 480
    x1_new = int((x1 / orig_w) * width)
    x2_new = int((x2 / orig_w) * width)
    y1_new = int((y1 / orig_h) * height)
    y2_new = int((y2 / orig_h) * height)
    return x1_new, y1_new, x2_new, y2_new


def _kitti_crop_box(width: int, height: int) -> Tuple[int, int, int, int]:
    crop_h = min(KITTI_CROP_SIZE[0], height)
    crop_w = min(KITTI_CROP_SIZE[1], width)
    top = max(0, height - crop_h)
    left = max(0, (width - crop_w) // 2)
    return left, top, left + crop_w, top + crop_h


def _fixed_crop(
    image: torch.Tensor,
    depth: torch.Tensor,
    crop_type: str,
    *,
    crop_depth: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if crop_type == "FULL":
        return image, depth
    _, height, width = image.shape
    if crop_type == "NYU":
        x1, y1, x2, y2 = _scaled_crop_box(width, height, NYU_CROP_BOX)
    elif crop_type == "KITTI":
        x1, y1, x2, y2 = _kitti_crop_box(width, height)
    else:
        raise ValueError(f"Unsupported crop_type={crop_type}")
    image = image[:, y1:y2, x1:x2]
    if crop_depth:
        depth = depth[:, y1:y2, x1:x2]
    return image, depth


def _resize_pair(image: torch.Tensor, depth: torch.Tensor, size, resize_depth: bool) -> tuple[torch.Tensor, torch.Tensor]:
    parsed_size = _parse_size(size)
    if parsed_size is None:
        return image, depth
    image = TF.resize(image, parsed_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
    if resize_depth:
        depth = TF.resize(depth, parsed_size, interpolation=InterpolationMode.NEAREST)
    return image, depth


def _pad_to_min_size(image: torch.Tensor, depth: torch.Tensor, min_height: int, min_width: int):
    _, height, width = image.shape
    pad_h = max(0, min_height - height)
    pad_w = max(0, min_width - width)
    if pad_h == 0 and pad_w == 0:
        return image, depth
    padding = [0, 0, pad_w, pad_h]
    image = TF.pad(image, padding, fill=0)
    depth = TF.pad(depth, padding, fill=0.0)
    return image, depth


def _random_crop_pair(image: torch.Tensor, depth: torch.Tensor, crop_size: Sequence[int]):
    crop_h, crop_w = int(crop_size[0]), int(crop_size[1])
    image, depth = _pad_to_min_size(image, depth, crop_h, crop_w)
    top, left, height, width = torch.randint(0, image.shape[-2] - crop_h + 1, (1,)).item(), torch.randint(
        0, image.shape[-1] - crop_w + 1, (1,)
    ).item(), crop_h, crop_w
    image = TF.crop(image, top, left, height, width)
    depth = TF.crop(depth, top, left, height, width)
    return image, depth


def _color_aug(
    image: torch.Tensor,
    *,
    prob: float,
    gamma_range: tuple[float, float] = (0.9, 1.1),
    brightness_range: tuple[float, float] = (0.9, 1.1),
    color_range: tuple[float, float] = (0.9, 1.1),
):
    if random.random() >= prob:
        return image
    image = image.clamp(0.0, 1.0)
    gamma = random.uniform(*gamma_range)
    brightness = random.uniform(*brightness_range)
    colors = torch.tensor(
        [random.uniform(*color_range) for _ in range(3)],
        dtype=image.dtype,
        device=image.device,
    ).view(3, 1, 1)
    image = image.pow(gamma) * brightness * colors
    return image.clamp(0.0, 1.0)


class DepthTrainTransform:
    def __init__(
        self,
        *,
        img_size,
        normalization_constant: float,
        random_crop_size: Sequence[int],
        fixed_crop: str,
        gamma_range: Sequence[float],
        brightness_range: Sequence[float],
        color_range: Sequence[float],
        rotation_angle: float,
        hflip_prob: float,
        color_aug_prob: float,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        self.img_size = _parse_size(img_size)
        self.normalization_constant = float(normalization_constant)
        self.random_crop_size = [int(random_crop_size[0]), int(random_crop_size[1])]
        self.fixed_crop = fixed_crop
        self.gamma_range = (float(gamma_range[0]), float(gamma_range[1]))
        self.brightness_range = (float(brightness_range[0]), float(brightness_range[1]))
        self.color_range = (float(color_range[0]), float(color_range[1]))
        self.rotation_angle = float(rotation_angle)
        self.hflip_prob = float(hflip_prob)
        self.color_aug_prob = float(color_aug_prob)
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, image: Image.Image, depth: Image.Image):
        image_tensor = TF.to_tensor(image)
        depth_tensor = torch.from_numpy(np.array(depth, dtype=np.float32)).unsqueeze(0)
        image_tensor, depth_tensor = _fixed_crop(image_tensor, depth_tensor, self.fixed_crop)
        image_tensor, depth_tensor = _resize_pair(image_tensor, depth_tensor, self.img_size, resize_depth=True)
        depth_tensor = depth_tensor / self.normalization_constant

        if self.rotation_angle > 0 and random.random() < 0.5:
            angle = random.uniform(-self.rotation_angle, self.rotation_angle)
            image_tensor = TF.rotate(image_tensor, angle=angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            depth_tensor = TF.rotate(depth_tensor, angle=angle, interpolation=InterpolationMode.NEAREST, fill=0.0)

        if random.random() < self.hflip_prob:
            image_tensor = TF.hflip(image_tensor)
            depth_tensor = TF.hflip(depth_tensor)

        image_tensor, depth_tensor = _random_crop_pair(image_tensor, depth_tensor, self.random_crop_size)
        image_tensor = _color_aug(
            image_tensor,
            prob=self.color_aug_prob,
            gamma_range=self.gamma_range,
            brightness_range=self.brightness_range,
            color_range=self.color_range,
        )
        image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)
        return image_tensor, depth_tensor


class DepthEvalTransform:
    def __init__(
        self,
        *,
        img_size,
        normalization_constant: float,
        fixed_crop: str,
        fixed_crop_depth: bool,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        self.img_size = _parse_size(img_size)
        self.normalization_constant = float(normalization_constant)
        self.fixed_crop = fixed_crop
        self.fixed_crop_depth = bool(fixed_crop_depth)
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, image: Image.Image, depth: Image.Image):
        image_tensor = TF.to_tensor(image)
        depth_tensor = torch.from_numpy(np.array(depth, dtype=np.float32)).unsqueeze(0)
        image_tensor, depth_tensor = _fixed_crop(
            image_tensor,
            depth_tensor,
            self.fixed_crop,
            crop_depth=self.fixed_crop_depth,
        )
        image_tensor, depth_tensor = _resize_pair(image_tensor, depth_tensor, self.img_size, resize_depth=False)
        image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)
        depth_tensor = depth_tensor / self.normalization_constant
        return image_tensor, depth_tensor


def _resolve_sample_path(root: str, relpath: str, subdir: str | None) -> str:
    relpath = relpath.strip()
    if os.path.isabs(relpath) and os.path.exists(relpath):
        return relpath
    relpath = relpath.strip("/")
    if subdir:
        clean_subdir = subdir.strip("/")
        if clean_subdir and relpath != clean_subdir and not relpath.startswith(clean_subdir + os.sep):
            relpath = os.path.join(clean_subdir, relpath)
    return os.path.join(root, relpath)


class PairedDepthDataset(Dataset):
    def __init__(
        self,
        root: str,
        split_file: str,
        transform,
        *,
        dataset_name: str,
        image_dir: str | None = None,
        depth_dir: str | None = None,
    ):
        self.root = root
        self.transform = transform
        self.dataset_name = dataset_name
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        split_path = split_file if os.path.isabs(split_file) else os.path.join(root, split_file)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"{dataset_name} split file not found: {split_path}")

        self.samples: List[Tuple[str, str]] = []
        with open(split_path, "r", encoding="utf-8") as handle:
            for line in sorted(handle.readlines()):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    if not depth_dir:
                        raise ValueError(f"Expected image and depth paths in split file, got: {stripped}")
                    image_relpath = parts[0]
                    depth_relpath = parts[0]
                else:
                    image_relpath = parts[0]
                    depth_relpath = parts[1]
                self.samples.append((image_relpath, depth_relpath))

    def __getitem__(self, index: int):
        image_relpath, depth_relpath = self.samples[index]
        image = Image.open(_resolve_sample_path(self.root, image_relpath, self.image_dir)).convert("RGB")
        depth = Image.open(_resolve_sample_path(self.root, depth_relpath, self.depth_dir))
        image_tensor, depth_tensor = self.transform(image, depth)
        return image_tensor, depth_tensor, index

    def __len__(self):
        return len(self.samples)


class NYUv2DepthDataset(PairedDepthDataset):
    def __init__(self, root: str, split_file: str, transform):
        super().__init__(root=root, split_file=split_file, transform=transform, dataset_name="nyuv2")


class DistributedEvalSampler(Sampler[int]):
    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.total_size = int(math.ceil(len(dataset) / self.num_replicas)) * self.num_replicas
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # Pad with wraparound so all ranks get the same number of samples.
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        # Subsample for this rank.
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def build_depth_dataset(cfg, split: str, data_cfg=None):
    data_cfg = cfg["data"] if data_cfg is None else data_cfg
    if split == "train":
        transform = DepthTrainTransform(
            img_size=data_cfg["train"].get("img_size"),
            normalization_constant=float(data_cfg["normalization_constant"]),
            random_crop_size=data_cfg["train"]["random_crop_size"],
            fixed_crop=data_cfg["train"]["fixed_crop"],
            gamma_range=data_cfg["train"]["gamma_range"],
            brightness_range=data_cfg["train"]["brightness_range"],
            color_range=data_cfg["train"]["color_range"],
            rotation_angle=float(data_cfg["train"]["rotation_angle"]),
            hflip_prob=float(data_cfg["train"]["hflip_prob"]),
            color_aug_prob=float(data_cfg["train"]["color_aug_prob"]),
            mean=data_cfg["mean"],
            std=data_cfg["std"],
        )
        split_file = data_cfg["splits"]["train"]
    elif split in {"val", "test"}:
        transform = DepthEvalTransform(
            img_size=data_cfg["eval"].get("img_size"),
            normalization_constant=float(data_cfg["normalization_constant"]),
            fixed_crop=data_cfg["eval"]["fixed_crop"],
            fixed_crop_depth=bool(data_cfg["eval"].get("fixed_crop_depth", True)),
            mean=data_cfg["mean"],
            std=data_cfg["std"],
        )
        split_key = "val" if split == "val" else "test"
        split_file = data_cfg["splits"][split_key]
    else:
        raise ValueError(f"Unsupported split={split}")
    return PairedDepthDataset(
        root=data_cfg["root"],
        split_file=split_file,
        transform=transform,
        dataset_name=data_cfg.get("dataset", "depth"),
        image_dir=data_cfg.get("image_dir"),
        depth_dir=data_cfg.get("depth_dir"),
    )


def build_depth_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    split: str,
    num_workers: int,
    persistent_workers: bool,
    distributed: bool,
    rank: int,
    world_size: int,
    seed: int = 0,
):
    if split == "train":
        sampler = None
        shuffle = True
        drop_last = True
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
                seed=int(seed),
            )
            shuffle = False
    else:
        drop_last = False
        shuffle = False
        sampler = DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank) if distributed else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0 and persistent_workers),
    )
    return loader, sampler
