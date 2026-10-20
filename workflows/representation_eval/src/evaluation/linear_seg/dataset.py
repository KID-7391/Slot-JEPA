import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.cityscapes import Cityscapes
from torchvision.transforms import InterpolationMode


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def _to_image_tensor(image: Image.Image) -> torch.Tensor:
    return TF.to_tensor(image)


def _to_label_tensor(label: Image.Image) -> torch.Tensor:
    return torch.as_tensor(np.array(label), dtype=torch.int64)


def _parse_size(value: Any, default_size: int) -> Tuple[int, int]:
    if value is None:
        return default_size, default_size
    if isinstance(value, int):
        return value, value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Invalid size specification: {value}")


def _resize_label(label: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    resized = TF.resize(
        label.unsqueeze(0).float(),
        list(size),
        interpolation=InterpolationMode.NEAREST,
    ).squeeze(0)
    return resized.long()


def _apply_augmentations(
    image: torch.Tensor,
    label: Optional[torch.Tensor],
    augmentations: List[Dict[str, Any]],
    default_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    for aug in augmentations:
        name = str(aug.get("name", "")).lower()
        if name == "resize":
            size = _parse_size(aug.get("size"), default_size)
            image = TF.resize(
                image,
                list(size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            if label is not None:
                label = _resize_label(label, size)

        elif name == "random_resized_crop":
            size = _parse_size(aug.get("size"), default_size)
            scale = tuple(aug.get("scale", [0.5, 2.0]))
            ratio = tuple(aug.get("ratio", [3.0 / 4.0, 4.0 / 3.0]))
            i, j, h, w = T.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
            image = TF.resized_crop(
                image,
                i,
                j,
                h,
                w,
                list(size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            if label is not None:
                label = TF.resized_crop(
                    label.unsqueeze(0).float(),
                    i,
                    j,
                    h,
                    w,
                    list(size),
                    interpolation=InterpolationMode.NEAREST,
                ).squeeze(0).long()

        elif name == "random_horizontal_flip":
            p = float(aug.get("p", 0.5))
            if random.random() < p:
                image = TF.hflip(image)
                if label is not None:
                    label = TF.hflip(label.unsqueeze(0)).squeeze(0)

        elif name == "color_jitter":
            p = float(aug.get("p", 1.0))
            if random.random() < p:
                jitter = T.ColorJitter(
                    brightness=aug.get("brightness", 0.4),
                    contrast=aug.get("contrast", 0.4),
                    saturation=aug.get("saturation", 0.4),
                    hue=aug.get("hue", 0.1),
                )
                image = jitter(image)

        elif name == "random_grayscale":
            p = float(aug.get("p", 0.2))
            if random.random() < p:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)

        elif name == "gaussian_blur":
            p = float(aug.get("p", 1.0))
            if random.random() < p:
                kernel_size = int(aug.get("kernel_size", 23))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                sigma = aug.get("sigma", [0.1, 2.0])
                blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
                image = blur(image)

        elif name == "normalize":
            image = TF.normalize(
                image,
                mean=aug.get("mean", DEFAULT_MEAN),
                std=aug.get("std", DEFAULT_STD),
            )

        elif name in {"identity", "none"}:
            continue

        else:
            raise ValueError(f"Unsupported augmentation op: {name}")

    if label is not None:
        label = label.long()
    return image, label


def load_dataset(
    dataset_name,
    root,
    split="train",
    resolution=336,
    augmentations: Optional[Sequence[Dict[str, Any]]] = None,
    use_label=True,
):
    """
    Load a segmentation dataset with configurable paired augmentations.

    Args:
        dataset_name: One of 'voc', 'coco', 'ade20k', 'cityscapes'
        root: Root directory containing all datasets
        split: 'train' or 'val'
        resolution: Fallback image resolution
        augmentations: List of augmentation configs to apply in order
        use_label: Whether to load labels

    Returns:
        (dataset, num_classes)
    """
    if augmentations is None:
        augmentations = [
            {"name": "resize", "size": [resolution, resolution]},
            {"name": "normalize", "mean": DEFAULT_MEAN, "std": DEFAULT_STD},
        ]

    augmentations = list(augmentations)

    if dataset_name == "voc":
        return VOCDataset(root, split, augmentations, resolution, use_label), 21
    if dataset_name == "coco":
        return COCO(root, split, augmentations, resolution, use_label), 27
    if dataset_name == "cityscapes":
        return CityscapesSeg(root, split, augmentations, resolution, use_label), 19
    if dataset_name == "ade20k":
        return ADE20k(root, split, augmentations, resolution, use_label), 150
    raise ValueError(f"Unsupported dataset '{dataset_name}'. Choose from 'voc', 'coco', 'ade20k', 'cityscapes'.")


class ADE20k(Dataset):
    def __init__(self, root, split, augmentations, resolution, use_label=True):
        super(ADE20k, self).__init__()
        self.split = split
        self.root = os.path.join(root, "ADEChallengeData2016")
        self.augmentations = augmentations
        self.resolution = resolution
        self.use_label = use_label

        assert self.split in ["train", "val"]

        img_root = os.path.join(self.root, "images", split)
        lbl_root = os.path.join(self.root, "annotations", split)
        img_by_stem = {Path(f).stem: f for f in sorted(os.listdir(img_root))}
        lbl_by_stem = {Path(f).stem: f for f in sorted(os.listdir(lbl_root))}
        common_stems = sorted(img_by_stem.keys() & lbl_by_stem.keys())
        if len(common_stems) != len(img_by_stem) or len(common_stems) != len(lbl_by_stem):
            raise ValueError(
                f"ADE20K image/label count mismatch in {split}: "
                f"{len(img_by_stem)} images, {len(lbl_by_stem)} labels, {len(common_stems)} matched"
            )
        self.image_files = [os.path.join(img_root, img_by_stem[s]) for s in common_stems]
        self.label_files = [os.path.join(lbl_root, lbl_by_stem[s]) for s in common_stems]

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = _to_image_tensor(Image.open(image_path).convert("RGB"))

        label = None
        if self.use_label:
            label_path = self.label_files[index]
            label = _to_label_tensor(Image.open(label_path))

        image, label = _apply_augmentations(image, label, self.augmentations, self.resolution)

        if not self.use_label:
            return image, 0

        label -= 1
        label[label == -1] = 255
        return image, label

    def __len__(self):
        return len(self.image_files)


class VOCDataset(Dataset):
    def __init__(self, root, split, augmentations, resolution, use_label, subset=None):
        super(VOCDataset, self).__init__()
        self.split = split
        self.root = os.path.join(root, "VOCdevkit/VOC2012")
        self.augmentations = augmentations
        self.resolution = resolution
        self.use_label = use_label

        assert self.split in ["train", "val"]

        if split == "train":
            split = "trainaug"
            anno_set = "SegmentationClassAug"
        else:
            anno_set = "SegmentationClass"

        self.image_files = []
        self.label_files = []

        with open(os.path.join(self.root, "ImageSets/Segmentation/", split + ".txt"), "r", encoding="utf-8") as handle:
            img_ids = [fn.rstrip() for fn in handle.readlines()]
            for img_id in img_ids:
                self.image_files.append(os.path.join(self.root, "JPEGImages", img_id + ".jpg"))
                self.label_files.append(os.path.join(self.root, anno_set, img_id + ".png"))

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = _to_image_tensor(Image.open(image_path).convert("RGB"))

        if not self.use_label:
            image, _ = _apply_augmentations(image, None, self.augmentations, self.resolution)
            return image, 0

        label_path = self.label_files[index]
        label = _to_label_tensor(Image.open(label_path))

        image, label = _apply_augmentations(image, label, self.augmentations, self.resolution)
        label[label < 0] = 255
        saliency = torch.zeros_like(label, dtype=torch.long)
        return image, label, saliency

    def __len__(self):
        return len(self.image_files)


class CityscapesSeg(Dataset):
    def __init__(self, root, split, augmentations, resolution, use_label=True, **kwargs):
        super(CityscapesSeg, self).__init__()
        self.split = split
        self.root = os.path.join(root, "cityscapes")
        if split == "train":
            our_image_set = "train"
            mode = "fine"
        else:
            our_image_set = split
            mode = "fine"

        self.inner_loader = Cityscapes(
            self.root,
            our_image_set,
            mode=mode,
            target_type="semantic",
            transform=None,
            target_transform=None,
        )
        self.augmentations = augmentations
        self.resolution = resolution
        self.void_id = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30]
        self.id_map = [-1 for _ in range(34)]
        cnt_ok = 0
        for idx in range(34):
            if idx in self.void_id:
                self.id_map[idx] = 255
            else:
                self.id_map[idx] = cnt_ok
                cnt_ok += 1

    def __getitem__(self, index):
        image, target = self.inner_loader[index]
        image = _to_image_tensor(image)
        target = _to_label_tensor(target)
        image, target = _apply_augmentations(image, target, self.augmentations, self.resolution)

        mapped = torch.full_like(target, 255, dtype=torch.long)
        for idx, new_id in enumerate(self.id_map):
            mapped[target == idx] = new_id

        return image, mapped

    def __len__(self):
        return len(self.inner_loader)


class COCO(Dataset):
    def __init__(
        self,
        root,
        split,
        augmentations,
        resolution,
        use_label,
        subset=None,
        exclude_things=False,
        exclude_stuff=False,
    ):
        super(COCO, self).__init__()
        self.split = split
        self.root = os.path.join(root, "coco")
        self.augmentations = augmentations
        self.resolution = resolution
        self.use_label = use_label
        self.subset = subset
        self.exclude_things = exclude_things
        self.exclude_stuff = exclude_stuff

        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
            "train+val": ["train2017", "val2017"],
        }

        if self.subset is not None:
            self.image_files = []
            self.label_files = [] if self.use_label else None
            for split_dir in split_dirs[self.split]:
                curated_file = os.path.join(self.root, "curated", split_dir, f"{self.subset}.txt")
                with open(curated_file, "r", encoding="utf-8") as handle:
                    img_ids = [fn.rstrip() for fn in handle.readlines()]
                    for img_id in img_ids:
                        self.image_files.append(os.path.join(self.root, "images", split_dir, img_id + ".jpg"))
                        if self.use_label:
                            self.label_files.append(os.path.join(self.root, "annotations", split_dir, img_id + ".png"))
        else:
            self.image_files = []
            self.label_files = [] if self.use_label else None
            for split_dir in split_dirs[self.split]:
                img_ids = sorted(os.listdir(os.path.join(self.root, "images", split_dir)))
                for img_id in img_ids:
                    self.image_files.append(os.path.join(self.root, "images", split_dir, img_id))
                    if self.use_label:
                        self.label_files.append(
                            os.path.join(self.root, "annotations", split_dir, img_id.replace(".jpg", ".png"))
                        )

        self.fine_to_coarse = {
            0: 9,
            1: 11,
            2: 11,
            3: 11,
            4: 11,
            5: 11,
            6: 11,
            7: 11,
            8: 11,
            9: 8,
            10: 8,
            11: 8,
            12: 8,
            13: 8,
            14: 8,
            15: 7,
            16: 7,
            17: 7,
            18: 7,
            19: 7,
            20: 7,
            21: 7,
            22: 7,
            23: 7,
            24: 7,
            25: 6,
            26: 6,
            27: 6,
            28: 6,
            29: 6,
            30: 6,
            31: 6,
            32: 6,
            33: 10,
            34: 10,
            35: 10,
            36: 10,
            37: 10,
            38: 10,
            39: 10,
            40: 10,
            41: 10,
            42: 10,
            43: 5,
            44: 5,
            45: 5,
            46: 5,
            47: 5,
            48: 5,
            49: 5,
            50: 5,
            51: 2,
            52: 2,
            53: 2,
            54: 2,
            55: 2,
            56: 2,
            57: 2,
            58: 2,
            59: 2,
            60: 2,
            61: 3,
            62: 3,
            63: 3,
            64: 3,
            65: 3,
            66: 3,
            67: 3,
            68: 3,
            69: 3,
            70: 3,
            71: 0,
            72: 0,
            73: 0,
            74: 0,
            75: 0,
            76: 0,
            77: 1,
            78: 1,
            79: 1,
            80: 1,
            81: 1,
            82: 1,
            83: 4,
            84: 4,
            85: 4,
            86: 4,
            87: 4,
            88: 4,
            89: 4,
            90: 4,
            91: 17,
            92: 17,
            93: 22,
            94: 20,
            95: 20,
            96: 22,
            97: 15,
            98: 25,
            99: 16,
            100: 13,
            101: 12,
            102: 12,
            103: 17,
            104: 17,
            105: 23,
            106: 15,
            107: 15,
            108: 17,
            109: 15,
            110: 21,
            111: 15,
            112: 25,
            113: 13,
            114: 13,
            115: 13,
            116: 13,
            117: 13,
            118: 22,
            119: 26,
            120: 14,
            121: 14,
            122: 15,
            123: 22,
            124: 21,
            125: 21,
            126: 24,
            127: 20,
            128: 22,
            129: 15,
            130: 17,
            131: 16,
            132: 15,
            133: 22,
            134: 24,
            135: 21,
            136: 17,
            137: 25,
            138: 16,
            139: 21,
            140: 17,
            141: 22,
            142: 16,
            143: 21,
            144: 21,
            145: 25,
            146: 21,
            147: 26,
            148: 21,
            149: 24,
            150: 20,
            151: 17,
            152: 14,
            153: 21,
            154: 26,
            155: 15,
            156: 23,
            157: 20,
            158: 21,
            159: 24,
            160: 15,
            161: 24,
            162: 22,
            163: 25,
            164: 15,
            165: 20,
            166: 17,
            167: 17,
            168: 22,
            169: 14,
            170: 18,
            171: 18,
            172: 18,
            173: 18,
            174: 18,
            175: 18,
            176: 18,
            177: 26,
            178: 26,
            179: 19,
            180: 19,
            181: 24,
        }
        self.first_stuff_index = 12

        self._coarse_lut = torch.full((256,), 255, dtype=torch.int64)
        for fine, coarse in self.fine_to_coarse.items():
            self._coarse_lut[fine] = coarse

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = _to_image_tensor(Image.open(image_path).convert("RGB"))

        if not self.use_label:
            image, _ = _apply_augmentations(image, None, self.augmentations, self.resolution)
            return image, 0

        label_path = self.label_files[index]
        label = _to_label_tensor(Image.open(label_path))
        image, label = _apply_augmentations(image, label, self.augmentations, self.resolution)

        coarse_label = self._coarse_lut[label.clamp(0, 255).long()]

        if self.exclude_things:
            valid = (coarse_label >= self.first_stuff_index) & (coarse_label != 255)
            return image, coarse_label - self.first_stuff_index, valid
        if self.exclude_stuff:
            coarse_label[coarse_label >= self.first_stuff_index] = 255
            return image, coarse_label, (coarse_label != 255)
        return image, coarse_label, (coarse_label != 255)

    def __len__(self):
        return len(self.image_files)

