"""YAML config loading/validation for linear segmentation evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.evaluation.common.bundle_paths import apply_bundle_paths


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
SUPPORTED_HEADS = {"linear", "bn_linear"}
SUPPORTED_OPTIMIZERS = {"adam", "adamw"}
SUPPORTED_LOAD_MODES = {"get_models", "get_models_slim", "dinov2_student_backbone", "direct_vit_checkpoint"}
SUPPORTED_FEATURE_MODES = {"single", "list"}
SUPPORTED_AMP_DTYPES = {"float16", "bfloat16", "float32"}


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _require_key(data: Dict[str, Any], key: str, parent: str):
    _require(key in data, f"Missing required key: {parent}.{key}")


def _ensure_list(value: Any, path: str) -> List[Any]:
    if isinstance(value, list):
        return value
    raise ValueError(f"{path} must be a list")


def _default_train_augmentation(size: int) -> List[Dict[str, Any]]:
    return [
        {"name": "random_resized_crop", "size": [size, size], "scale": [0.8, 1.0]},
        {"name": "random_horizontal_flip", "p": 0.5},
        {"name": "normalize", "mean": deepcopy(DEFAULT_MEAN), "std": deepcopy(DEFAULT_STD)},
    ]


def _default_val_augmentation(size: int) -> List[Dict[str, Any]]:
    return [
        {"name": "resize", "size": [size, size]},
        {"name": "normalize", "mean": deepcopy(DEFAULT_MEAN), "std": deepcopy(DEFAULT_STD)},
    ]


def load_linear_seg_config(config_path: str, paths_config: str | None = None) -> Dict[str, Any]:
    """Load and validate a linear-seg YAML config."""
    path = Path(config_path)
    _require(path.exists(), f"Config file does not exist: {config_path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    _require(isinstance(cfg, dict), "Config root must be a dictionary")
    cfg = apply_bundle_paths(cfg, config_path=path, paths_config=paths_config)
    return _validate_and_fill_defaults(cfg)


def _validate_and_fill_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(cfg)

    _require_key(cfg, "model", "root")
    _require_key(cfg, "data", "root")
    _require_key(cfg, "head_search", "root")
    _require_key(cfg, "train", "root")

    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault("name", "linear_seg")
    cfg.setdefault("distributed", {})
    cfg["distributed"].setdefault("num_workers", 4)
    cfg["distributed"].setdefault("persistent_workers", False)
    cfg.setdefault("eval", {})

    _validate_model(cfg["model"])
    _validate_data(cfg["data"])
    _validate_head_search(cfg["head_search"])
    _validate_train(cfg["train"])

    if "dataset" in cfg["data"] and "datasets" not in cfg["data"]:
        cfg["data"]["datasets"] = [cfg["data"]["dataset"]]
    cfg["data"]["datasets"] = _ensure_list(cfg["data"]["datasets"], "data.datasets")
    _require(len(cfg["data"]["datasets"]) > 0, "data.datasets cannot be empty")

    cfg["eval"].setdefault("metric", ["mIoU", "Accuracy"])
    cfg["eval"].setdefault("inference", {"mode": "single_scale"})

    return cfg


def _validate_model(model_cfg: Dict[str, Any]):
    _require_key(model_cfg, "model_type", "model")
    _require_key(model_cfg, "model_scale", "model")

    model_cfg.setdefault("load_mode", "get_models")
    _require(
        model_cfg["load_mode"] in SUPPORTED_LOAD_MODES,
        f"model.load_mode must be one of {sorted(SUPPORTED_LOAD_MODES)}",
    )

    load_mode = model_cfg["load_mode"]
    if load_mode in {"get_models", "get_models_slim"}:
        _require_key(model_cfg, "model_root", "model")
        _require_key(model_cfg, "epoch", "model")
    elif load_mode == "dinov2_student_backbone":
        _require_key(model_cfg, "ckpt_path", "model")
        model_cfg.setdefault("state_dict_key", "student")
        model_cfg.setdefault("prefix_filter", "backbone.")
        model_cfg.setdefault("strip_prefix", "backbone.")
        model_cfg.setdefault("patch_size", 14)
        model_cfg.setdefault("num_register_tokens", 4)
    elif load_mode == "direct_vit_checkpoint":
        _require_key(model_cfg, "ckpt_path", "model")
        model_cfg.setdefault("state_dict_key", None)
        model_cfg.setdefault("prefix_filter", None)
        model_cfg.setdefault("strip_prefix", "")
        if model_cfg["model_type"] == "dinov2":
            model_cfg.setdefault("patch_size", 14)
            model_cfg.setdefault("num_register_tokens", 4)
        else:
            model_cfg.setdefault("patch_size", 16)

    model_cfg.setdefault("embed_dim", None)
    if model_cfg["model_type"] == "vjepa2_slot":
        model_cfg.setdefault("patch_stride", model_cfg.get("patch_size", 16))
        model_cfg.setdefault("patch_padding", 0)
        model_cfg.setdefault("video_infer_frames", 2)
        model_cfg.setdefault("slot_alpha", 0.0)
        model_cfg.setdefault("slot_normalize_before_mix", False)
        model_cfg.setdefault("slot_window_size", [2, 4])
        model_cfg.setdefault("slot_slots_per_window", 4)
        model_cfg.setdefault("slot_global_k", 16)
        model_cfg.setdefault("slot_mask_temp", 1.0)
        model_cfg.setdefault("slot_l_max", 16)
        model_cfg.setdefault("slot_use_decoder_checkpoint", False)

        model_cfg["slot_alpha"] = float(model_cfg["slot_alpha"])
        _require(0.0 <= model_cfg["slot_alpha"] <= 1.0, "model.slot_alpha must be in [0, 1]")
        window_size = _ensure_list(model_cfg["slot_window_size"], "model.slot_window_size")
        _require(len(window_size) == 2, "model.slot_window_size must contain [temporal, spatial]")
        model_cfg["slot_window_size"] = [int(value) for value in window_size]
        _require(all(value > 0 for value in model_cfg["slot_window_size"]), "model.slot_window_size values must be > 0")
        for key in ("slot_slots_per_window", "slot_global_k", "slot_l_max"):
            model_cfg[key] = int(model_cfg[key])
            _require(model_cfg[key] > 0, f"model.{key} must be > 0")
        model_cfg["slot_mask_temp"] = float(model_cfg["slot_mask_temp"])
        _require(model_cfg["slot_mask_temp"] > 0.0, "model.slot_mask_temp must be > 0")

    model_cfg.setdefault("feature_source", {})
    _validate_feature_source(model_cfg)


def _validate_feature_source(model_cfg: Dict[str, Any]):
    feature_cfg = model_cfg["feature_source"]
    feature_cfg.setdefault("mode", "single")
    _require(
        feature_cfg["mode"] in SUPPORTED_FEATURE_MODES,
        f"model.feature_source.mode must be one of {sorted(SUPPORTED_FEATURE_MODES)}",
    )

    if feature_cfg["mode"] == "single":
        feature_cfg.setdefault("single", {})
        single_cfg = feature_cfg["single"]
        if model_cfg["model_type"] == "dinov2":
            single_cfg.setdefault("key", "x_norm_patchtokens")
            if single_cfg["key"] == "x_prenorm":
                single_cfg.setdefault("drop_prefix_tokens", None)
        else:
            single_cfg.setdefault("source", "get_embeddings")
            single_cfg.setdefault("index", -1)
    else:
        feature_cfg.setdefault("list", {})
        list_cfg = feature_cfg["list"]
        list_cfg.setdefault("source", "auto")
        list_cfg.setdefault("indices", [-1])
        list_cfg.setdefault("align", "max")
        _require(
            isinstance(list_cfg["indices"], list) and len(list_cfg["indices"]) > 0,
            "model.feature_source.list.indices must be a non-empty list",
        )
        _require(
            list_cfg["align"] in {"max", "min", "index"},
            "model.feature_source.list.align must be one of ['max', 'min', 'index']",
        )
        if list_cfg["align"] == "index":
            list_cfg.setdefault("align_index", 0)
        list_cfg.setdefault("norm", True)
        list_cfg.setdefault("projection", {})
        projection_cfg = list_cfg["projection"]
        projection_cfg.setdefault("enabled", True)
        if projection_cfg["enabled"]:
            projection_cfg.setdefault("out_dim", model_cfg.get("embed_dim"))
            _require(
                projection_cfg["out_dim"] is not None,
                "model.feature_source.list.projection.out_dim is required when projection.enabled=true "
                "and model.embed_dim is not set",
            )


def _validate_data(data_cfg: Dict[str, Any]):
    _require_key(data_cfg, "root", "data")
    _require("dataset" in data_cfg or "datasets" in data_cfg, "data.dataset or data.datasets is required")

    data_cfg.setdefault("resolution", {})
    data_cfg["resolution"].setdefault("train", 336)
    data_cfg["resolution"].setdefault("val", 336)

    train_res = int(data_cfg["resolution"]["train"])
    val_res = int(data_cfg["resolution"]["val"])
    _require(train_res > 0 and val_res > 0, "data.resolution.{train,val} must be positive")

    data_cfg.setdefault("augmentation", {})
    data_cfg["augmentation"].setdefault("train", _default_train_augmentation(train_res))
    data_cfg["augmentation"].setdefault("val", _default_val_augmentation(val_res))
    _ensure_list(data_cfg["augmentation"]["train"], "data.augmentation.train")
    _ensure_list(data_cfg["augmentation"]["val"], "data.augmentation.val")


def _validate_head_search(head_cfg: Dict[str, Any]):
    _require_key(head_cfg, "heads", "head_search")
    _require_key(head_cfg, "lrs", "head_search")
    heads = _ensure_list(head_cfg["heads"], "head_search.heads")
    lrs = _ensure_list(head_cfg["lrs"], "head_search.lrs")
    _require(len(heads) > 0, "head_search.heads cannot be empty")
    _require(len(lrs) > 0, "head_search.lrs cannot be empty")
    for raw_lr in lrs:
        lr = float(raw_lr)
        _require(lr > 0, f"head_search.lrs must be positive, got {raw_lr}")

    head_specs = _normalize_head_specs(heads)
    head_cfg["head_specs"] = head_specs
    head_cfg["heads"] = [spec["name"] for spec in head_specs]


def _normalize_head_specs(heads: List[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    used_names = set()

    for index, head_entry in enumerate(heads):
        if isinstance(head_entry, str):
            head_type = head_entry
            name = head_type
            projection_dim = None
            bias = True
            dropout = 0.0
            sync_bn = True
        elif isinstance(head_entry, dict):
            _require_key(head_entry, "type", f"head_search.heads[{index}]")
            head_type = str(head_entry["type"])
            name = str(head_entry.get("name", head_type))
            projection_dim = head_entry.get("projection_dim")
            bias = bool(head_entry.get("bias", True))
            dropout = float(head_entry.get("dropout", 0.0))
            sync_bn = bool(head_entry.get("sync_bn", True))
        else:
            raise ValueError(
                f"head_search.heads[{index}] must be either a string head type or a dictionary head spec"
            )

        _require(
            head_type in SUPPORTED_HEADS,
            f"Unsupported head type '{head_type}' in head_search.heads[{index}]. "
            f"Supported: {sorted(SUPPORTED_HEADS)}",
        )
        _require(dropout >= 0.0 and dropout < 1.0, f"Invalid dropout={dropout} for head_search.heads[{index}]")
        if projection_dim is not None:
            projection_dim = int(projection_dim)
            _require(projection_dim > 0, f"projection_dim must be positive in head_search.heads[{index}]")

        base_name = name
        suffix = 1
        while name in used_names:
            suffix += 1
            name = f"{base_name}_{suffix}"
        used_names.add(name)

        normalized.append(
            {
                "name": name,
                "type": head_type,
                "projection_dim": projection_dim,
                "bias": bias,
                "dropout": dropout,
                "sync_bn": sync_bn,
            }
        )

    return normalized


def _validate_train(train_cfg: Dict[str, Any]):
    _require_key(train_cfg, "max_steps", "train")
    _require(int(train_cfg["max_steps"]) > 0, "train.max_steps must be > 0")
    train_cfg.setdefault("global_batch_size", 256)
    _require(int(train_cfg["global_batch_size"]) > 0, "train.global_batch_size must be > 0")
    train_cfg.setdefault("grad_accum_steps", 1)
    _require(int(train_cfg["grad_accum_steps"]) > 0, "train.grad_accum_steps must be > 0")

    train_cfg.setdefault("optimizer", {})
    train_cfg["optimizer"].setdefault("type", "adam")
    train_cfg["optimizer"].setdefault("weight_decay", 0.0)
    train_cfg["optimizer"].setdefault("betas", [0.9, 0.999])
    _require(
        train_cfg["optimizer"]["type"] in SUPPORTED_OPTIMIZERS,
        f"train.optimizer.type must be one of {sorted(SUPPORTED_OPTIMIZERS)}",
    )

    train_cfg.setdefault("amp", {})
    train_cfg["amp"].setdefault("enabled", True)
    train_cfg["amp"].setdefault("dtype", "float16")
    _require(
        str(train_cfg["amp"]["dtype"]).lower() in SUPPORTED_AMP_DTYPES,
        f"train.amp.dtype must be one of {sorted(SUPPORTED_AMP_DTYPES)}",
    )
