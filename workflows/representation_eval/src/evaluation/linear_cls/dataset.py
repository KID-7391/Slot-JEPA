"""Dataset and transform helpers for linear classification evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import InterpolationMode


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
FGVC_AIRCRAFT_LABEL_FILES = {
    "variant": "variants.txt",
    "family": "families.txt",
    "manufacturer": "manufacturers.txt",
}


@dataclass
class ClassificationDatasets:
    train_dataset: Dataset
    val_dataset: Optional[Dataset]
    test_dataset: Optional[Dataset]
    num_classes: int
    train_descriptor: str
    val_descriptor: Optional[str]
    test_descriptor: Optional[str] = None


def _parse_interpolation(value: Any) -> InterpolationMode:
    if isinstance(value, InterpolationMode):
        return value
    name = str(value).lower()
    mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "lanczos": InterpolationMode.LANCZOS,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported interpolation mode: {value}")
    return mapping[name]


def _parse_size(value: Any) -> Union[int, Tuple[int, int]]:
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Invalid size specification: {value}")


def build_classification_transform(augmentations: Sequence[Dict[str, Any]]) -> transforms.Compose:
    transform_list: List[Any] = []
    has_tensor_op = False

    for aug in augmentations:
        name = str(aug.get("name", "")).lower()
        if name == "random_resized_crop":
            transform_list.append(
                transforms.RandomResizedCrop(
                    _parse_size(aug.get("size", 224)),
                    scale=tuple(aug.get("scale", [0.08, 1.0])),
                    ratio=tuple(aug.get("ratio", [3.0 / 4.0, 4.0 / 3.0])),
                    interpolation=_parse_interpolation(aug.get("interpolation", "bicubic")),
                )
            )
        elif name == "random_horizontal_flip":
            transform_list.append(transforms.RandomHorizontalFlip(p=float(aug.get("p", 0.5))))
        elif name == "rand_augment":
            transform_list.append(
                transforms.RandAugment(
                    num_ops=int(aug.get("num_ops", 2)),
                    magnitude=int(aug.get("magnitude", 9)),
                    interpolation=_parse_interpolation(aug.get("interpolation", "bicubic")),
                )
            )
        elif name == "resize":
            transform_list.append(
                transforms.Resize(
                    _parse_size(aug.get("size", 256)),
                    interpolation=_parse_interpolation(aug.get("interpolation", "bicubic")),
                )
            )
        elif name == "center_crop":
            transform_list.append(transforms.CenterCrop(_parse_size(aug.get("size", 224))))
        elif name == "to_tensor":
            transform_list.append(transforms.ToTensor())
            has_tensor_op = True
        elif name == "normalize":
            if not has_tensor_op:
                transform_list.append(transforms.ToTensor())
                has_tensor_op = True
            transform_list.append(
                transforms.Normalize(
                    mean=aug.get("mean", DEFAULT_MEAN),
                    std=aug.get("std", DEFAULT_STD),
                )
            )
        elif name == "random_erasing":
            if not has_tensor_op:
                transform_list.append(transforms.ToTensor())
                has_tensor_op = True
            transform_list.append(
                transforms.RandomErasing(
                    p=float(aug.get("p", 0.25)),
                    scale=tuple(aug.get("scale", [0.02, 0.33])),
                    ratio=tuple(aug.get("ratio", [0.3, 3.3])),
                    value=aug.get("value", 0),
                    inplace=bool(aug.get("inplace", False)),
                )
            )
        elif name in {"identity", "none"}:
            continue
        else:
            raise ValueError(f"Unsupported classification augmentation op: {name}")

    if not has_tensor_op:
        transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def _resolve_fgvc_aircraft_roots(root: Path) -> Tuple[Path, Path]:
    root = root.expanduser()
    if (root / "data" / "images").exists():
        return root, root / "data"
    if (root / "images").exists() and any((root / filename).exists() for filename in FGVC_AIRCRAFT_LABEL_FILES.values()):
        return root.parent, root
    raise FileNotFoundError(
        "Unable to locate FGVC Aircraft data directory. "
        f"Expected either {root / 'data' / 'images'} or {root / 'images'} to exist."
    )


def _ensure_same_class_mapping(reference_dataset, candidate_dataset, split_name: str):
    if reference_dataset.class_to_idx != candidate_dataset.class_to_idx:
        raise ValueError(
            f"FGVC Aircraft {split_name} split class mapping does not match the training split."
        )


class FGVCAircraftClassificationDataset(Dataset):
    """FGVC Aircraft classification dataset backed by official split files."""

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        annotation_level: str,
        transform=None,
    ):
        dataset_root, data_root = _resolve_fgvc_aircraft_roots(Path(root))
        self.root = str(dataset_root)
        self.data_root = data_root
        self.split = str(split).lower()
        self.annotation_level = str(annotation_level).lower()
        self.transform = transform
        self.loader = default_loader

        if self.annotation_level not in FGVC_AIRCRAFT_LABEL_FILES:
            raise ValueError(
                f"Unsupported FGVC Aircraft annotation level: {annotation_level}. "
                f"Expected one of {sorted(FGVC_AIRCRAFT_LABEL_FILES)}."
            )

        class_names_path = self.data_root / FGVC_AIRCRAFT_LABEL_FILES[self.annotation_level]
        if not class_names_path.exists():
            raise FileNotFoundError(f"FGVC Aircraft class list does not exist: {class_names_path}")

        with class_names_path.open("r", encoding="utf-8") as handle:
            self.classes = [line.strip() for line in handle if line.strip()]
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        split_path = self.data_root / f"images_{self.annotation_level}_{self.split}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"FGVC Aircraft split file does not exist: {split_path}")

        images_dir = self.data_root / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"FGVC Aircraft images directory does not exist: {images_dir}")

        self.samples: List[Tuple[Path, int]] = []
        self.targets: List[int] = []

        with split_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                image_id, class_name = line.split(" ", 1)
                if class_name not in self.class_to_idx:
                    raise KeyError(
                        f"Unknown FGVC Aircraft class '{class_name}' encountered in {split_path}"
                    )

                image_path = images_dir / f"{image_id}.jpg"
                if not image_path.exists():
                    raise FileNotFoundError(f"FGVC Aircraft image does not exist: {image_path}")

                target = self.class_to_idx[class_name]
                self.samples.append((image_path, target))
                self.targets.append(target)

        self.imgs = self.samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, target = self.samples[index]
        image = self.loader(str(image_path))
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _load_imagefolder_datasets(data_cfg: Dict[str, Any]) -> ClassificationDatasets:
    if "train_dir" in data_cfg and "val_dir" in data_cfg:
        train_dir = Path(data_cfg["train_dir"])
        val_dir = Path(data_cfg["val_dir"])
    else:
        root = Path(data_cfg["root"])
        train_dir = root / data_cfg.get("train_subdir", "train")
        val_dir = root / data_cfg.get("val_subdir", "val")

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory does not exist: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory does not exist: {val_dir}")

    train_transform = build_classification_transform(data_cfg["augmentation"]["train"])
    val_transform = build_classification_transform(data_cfg["augmentation"]["val"])

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

    num_classes = data_cfg.get("num_classes")
    inferred_num_classes = len(train_dataset.classes)
    if num_classes is None:
        num_classes = inferred_num_classes
    else:
        num_classes = int(num_classes)
        if num_classes != inferred_num_classes:
            raise ValueError(
                f"Configured num_classes={num_classes} does not match "
                f"inferred class count={inferred_num_classes} from {train_dir}"
            )

    return ClassificationDatasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,
        num_classes=num_classes,
        train_descriptor=str(train_dir),
        val_descriptor=str(val_dir),
    )


def _load_fgvc_aircraft_datasets(data_cfg: Dict[str, Any]) -> ClassificationDatasets:
    root = Path(data_cfg["root"])
    annotation_level = str(data_cfg.get("annotation_level", "variant")).lower()
    train_split = str(data_cfg.get("train_split", "train")).lower()
    val_split = data_cfg.get("val_split")
    test_split = data_cfg.get("test_split", "test")

    train_transform = build_classification_transform(data_cfg["augmentation"]["train"])
    eval_transform = build_classification_transform(data_cfg["augmentation"]["val"])

    train_dataset = FGVCAircraftClassificationDataset(
        root=root,
        split=train_split,
        annotation_level=annotation_level,
        transform=train_transform,
    )
    val_dataset = None
    val_descriptor = None
    if val_split is not None:
        val_split = str(val_split).lower()
        val_dataset = FGVCAircraftClassificationDataset(
            root=root,
            split=val_split,
            annotation_level=annotation_level,
            transform=eval_transform,
        )
        _ensure_same_class_mapping(train_dataset, val_dataset, split_name=val_split)
        val_descriptor = f"{root} [{annotation_level}:{val_split}]"

    test_dataset = None
    test_descriptor = None
    if test_split is not None:
        test_dataset = FGVCAircraftClassificationDataset(
            root=root,
            split=str(test_split).lower(),
            annotation_level=annotation_level,
            transform=eval_transform,
        )
        _ensure_same_class_mapping(train_dataset, test_dataset, split_name=str(test_split).lower())
        test_descriptor = f"{root} [{annotation_level}:{str(test_split).lower()}]"

    num_classes = data_cfg.get("num_classes")
    inferred_num_classes = len(train_dataset.classes)
    if num_classes is None:
        num_classes = inferred_num_classes
    else:
        num_classes = int(num_classes)
        if num_classes != inferred_num_classes:
            raise ValueError(
                f"FGVC Aircraft num_classes={num_classes} does not match inferred classes={inferred_num_classes}"
            )

    return ClassificationDatasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_classes=num_classes,
        train_descriptor=f"{root} [{annotation_level}:{train_split}]",
        val_descriptor=val_descriptor,
        test_descriptor=test_descriptor,
    )


def load_classification_datasets(data_cfg: Dict[str, Any]) -> ClassificationDatasets:
    dataset_name = str(data_cfg.get("dataset", "imagenet")).lower()
    if dataset_name == "fgvc_aircraft":
        return _load_fgvc_aircraft_datasets(data_cfg)
    return _load_imagefolder_datasets(data_cfg)


def load_imagefolder_datasets(data_cfg: Dict[str, Any]):
    bundle = _load_imagefolder_datasets(data_cfg)
    return (
        bundle.train_dataset,
        bundle.val_dataset,
        bundle.num_classes,
        bundle.train_descriptor,
        bundle.val_descriptor,
    )
