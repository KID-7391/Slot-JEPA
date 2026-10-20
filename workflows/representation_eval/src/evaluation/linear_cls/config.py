"""YAML config loading/validation for linear classification evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.evaluation.common.bundle_paths import apply_bundle_paths


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
DEFAULT_LEARNING_RATES = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]
SUPPORTED_DATASETS = {"imagenet", "imagefolder", "fgvc_aircraft"}
FGVC_AIRCRAFT_ANNOTATION_LEVELS = {"variant", "family", "manufacturer"}
FGVC_AIRCRAFT_SPLITS = {"train", "val", "test", "trainval"}
SUPPORTED_LOAD_MODES = {"get_models", "get_models_slim", "dinov2_student_backbone", "direct_vit_checkpoint"}
SUPPORTED_FEATURE_SOURCES = {"auto", "get_embeddings", "get_intermediate_layers"}
SUPPORTED_OPTIMIZERS = {"sgd"}
SUPPORTED_AMP_DTYPES = {"float16", "bfloat16", "float32"}
SUPPORTED_POOLING_MODES = {"cls", "cls_avgpool", "avgpool"}


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _require_key(data: Dict[str, Any], key: str, parent: str):
    _require(key in data, f"Missing required key: {parent}.{key}")


def _ensure_list(value: Any, path: str) -> List[Any]:
    if isinstance(value, list):
        return value
    raise ValueError(f"{path} must be a list")


def _normalize_optional_split(value: Any, *, path: str):
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    _require(
        normalized in FGVC_AIRCRAFT_SPLITS,
        f"{path} must be one of {sorted(FGVC_AIRCRAFT_SPLITS)} or null",
    )
    return normalized


def _default_train_augmentation() -> List[Dict[str, Any]]:
    return [
        {"name": "random_resized_crop", "size": 224, "interpolation": "bicubic"},
        {"name": "random_horizontal_flip", "p": 0.5},
        {"name": "to_tensor"},
        {"name": "normalize", "mean": deepcopy(DEFAULT_MEAN), "std": deepcopy(DEFAULT_STD)},
    ]


def _default_val_augmentation() -> List[Dict[str, Any]]:
    return [
        {"name": "resize", "size": 256, "interpolation": "bicubic"},
        {"name": "center_crop", "size": 224},
        {"name": "to_tensor"},
        {"name": "normalize", "mean": deepcopy(DEFAULT_MEAN), "std": deepcopy(DEFAULT_STD)},
    ]


def load_linear_cls_config(config_path: str, paths_config: str | None = None) -> Dict[str, Any]:
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

    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault("name", "linear_cls")
    cfg.setdefault("distributed", {})
    cfg["distributed"].setdefault("num_workers", 8)
    cfg.setdefault("head_search", {})
    cfg.setdefault("train", {})
    cfg.setdefault("eval", {})

    _validate_model(cfg["model"])
    _validate_data(cfg["data"])
    _validate_head_search(cfg["head_search"])
    _validate_train(cfg["train"])
    _validate_eval(cfg["eval"], cfg["train"])

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

    model_cfg.setdefault("feature_source", {})
    feature_cfg = model_cfg["feature_source"]
    feature_cfg.setdefault("source", "auto")
    _require(
        feature_cfg["source"] in SUPPORTED_FEATURE_SOURCES,
        f"model.feature_source.source must be one of {sorted(SUPPORTED_FEATURE_SOURCES)}",
    )
    feature_cfg.setdefault("drop_prefix_tokens", None)
    feature_cfg.setdefault("norm", True)


def _validate_data(data_cfg: Dict[str, Any]):
    has_root = "root" in data_cfg
    has_dirs = "train_dir" in data_cfg and "val_dir" in data_cfg
    _require(has_root or has_dirs, "data.root or both data.train_dir/data.val_dir are required")

    data_cfg.setdefault("dataset", "imagenet")
    dataset_name = str(data_cfg["dataset"]).lower()
    _require(
        dataset_name in SUPPORTED_DATASETS,
        f"data.dataset must be one of {sorted(SUPPORTED_DATASETS)}",
    )
    data_cfg["dataset"] = dataset_name

    data_cfg.setdefault("num_classes", None)

    if dataset_name == "fgvc_aircraft":
        _require(has_root, "data.root is required for data.dataset=fgvc_aircraft")
        data_cfg.setdefault("annotation_level", "variant")
        data_cfg["annotation_level"] = str(data_cfg["annotation_level"]).lower()
        _require(
            data_cfg["annotation_level"] in FGVC_AIRCRAFT_ANNOTATION_LEVELS,
            f"data.annotation_level must be one of {sorted(FGVC_AIRCRAFT_ANNOTATION_LEVELS)}",
        )

        data_cfg.setdefault("train_split", "train")
        data_cfg.setdefault("val_split", "val")
        data_cfg.setdefault("test_split", "test")
        data_cfg["train_split"] = _normalize_optional_split(data_cfg["train_split"], path="data.train_split")
        data_cfg["val_split"] = _normalize_optional_split(data_cfg["val_split"], path="data.val_split")
        data_cfg["test_split"] = _normalize_optional_split(data_cfg["test_split"], path="data.test_split")
        _require(data_cfg["train_split"] is not None, "data.train_split cannot be null")
        _require(
            data_cfg["val_split"] is not None or data_cfg["test_split"] is not None,
            "At least one of data.val_split or data.test_split must be provided for data.dataset=fgvc_aircraft",
        )
    else:
        data_cfg.setdefault("train_subdir", "train")
        data_cfg.setdefault("val_subdir", "val")

    data_cfg.setdefault("augmentation", {})
    data_cfg["augmentation"].setdefault("train", _default_train_augmentation())
    data_cfg["augmentation"].setdefault("val", _default_val_augmentation())
    _ensure_list(data_cfg["augmentation"]["train"], "data.augmentation.train")
    _ensure_list(data_cfg["augmentation"]["val"], "data.augmentation.val")


def _validate_head_search(head_cfg: Dict[str, Any]):
    head_cfg.setdefault("learning_rates", deepcopy(DEFAULT_LEARNING_RATES))
    head_cfg.setdefault("n_last_blocks", [1, 4])
    head_cfg.setdefault("use_avgpool", [False, True])
    head_cfg.setdefault("pooling_mode", "cls")

    learning_rates = _ensure_list(head_cfg["learning_rates"], "head_search.learning_rates")
    n_last_blocks = _ensure_list(head_cfg["n_last_blocks"], "head_search.n_last_blocks")
    use_avgpool = _ensure_list(head_cfg["use_avgpool"], "head_search.use_avgpool")

    _require(len(learning_rates) > 0, "head_search.learning_rates cannot be empty")
    _require(len(n_last_blocks) > 0, "head_search.n_last_blocks cannot be empty")
    _require(len(use_avgpool) > 0, "head_search.use_avgpool cannot be empty")

    for raw_lr in learning_rates:
        lr = float(raw_lr)
        _require(lr > 0, f"head_search.learning_rates must be positive, got {raw_lr}")
    head_cfg["learning_rates"] = [float(raw_lr) for raw_lr in learning_rates]

    parsed_n_last_blocks = []
    for raw_n in n_last_blocks:
        value = int(raw_n)
        _require(value > 0, f"head_search.n_last_blocks must contain positive integers, got {raw_n}")
        parsed_n_last_blocks.append(value)
    head_cfg["n_last_blocks"] = parsed_n_last_blocks
    head_cfg["use_avgpool"] = [bool(value) for value in use_avgpool]
    head_cfg["pooling_mode"] = str(head_cfg["pooling_mode"]).lower()
    _require(
        head_cfg["pooling_mode"] in SUPPORTED_POOLING_MODES,
        f"head_search.pooling_mode must be one of {sorted(SUPPORTED_POOLING_MODES)}",
    )


def _validate_train(train_cfg: Dict[str, Any]):
    train_cfg.setdefault("epochs", 10)
    train_cfg.setdefault("epoch_length", 1250)
    train_cfg.setdefault("global_batch_size", None)
    train_cfg.setdefault("batch_size", None)
    train_cfg.setdefault("seed", 0)
    _require(int(train_cfg["epochs"]) > 0, "train.epochs must be > 0")
    _require(int(train_cfg["epoch_length"]) > 0, "train.epoch_length must be > 0")
    _require(
        train_cfg["global_batch_size"] is not None or train_cfg["batch_size"] is not None,
        "train.global_batch_size or train.batch_size must be provided",
    )
    if train_cfg["global_batch_size"] is not None:
        _require(int(train_cfg["global_batch_size"]) > 0, "train.global_batch_size must be > 0")
    if train_cfg["batch_size"] is not None:
        _require(int(train_cfg["batch_size"]) > 0, "train.batch_size must be > 0")

    train_cfg.setdefault("optimizer", {})
    train_cfg["optimizer"].setdefault("type", "sgd")
    train_cfg["optimizer"].setdefault("momentum", 0.9)
    train_cfg["optimizer"].setdefault("weight_decay", 0.0)
    _require(
        train_cfg["optimizer"]["type"] in SUPPORTED_OPTIMIZERS,
        f"train.optimizer.type must be one of {sorted(SUPPORTED_OPTIMIZERS)}",
    )

    train_cfg.setdefault("amp", {})
    train_cfg["amp"].setdefault("enabled", True)
    train_cfg["amp"].setdefault("dtype", "float16")
    _require(
        train_cfg["amp"]["dtype"] in SUPPORTED_AMP_DTYPES,
        f"train.amp.dtype must be one of {sorted(SUPPORTED_AMP_DTYPES)}",
    )


def _validate_eval(eval_cfg: Dict[str, Any], train_cfg: Dict[str, Any]):
    eval_cfg.setdefault("global_batch_size", train_cfg.get("global_batch_size"))
    eval_cfg.setdefault("batch_size", train_cfg.get("batch_size"))
    eval_cfg.setdefault("topk", [1, 5])
    eval_cfg.setdefault("eval_period_iterations", int(train_cfg["epoch_length"]))

    topk = _ensure_list(eval_cfg["topk"], "eval.topk")
    _require(len(topk) > 0, "eval.topk cannot be empty")
    eval_cfg["topk"] = [int(k) for k in topk]
    for k in eval_cfg["topk"]:
        _require(k > 0, f"eval.topk must contain positive integers, got {k}")

    _require(
        eval_cfg["global_batch_size"] is not None or eval_cfg["batch_size"] is not None,
        "eval.global_batch_size or eval.batch_size must be provided",
    )
    if eval_cfg["global_batch_size"] is not None:
        _require(int(eval_cfg["global_batch_size"]) > 0, "eval.global_batch_size must be > 0")
    if eval_cfg["batch_size"] is not None:
        _require(int(eval_cfg["batch_size"]) > 0, "eval.batch_size must be > 0")
    _require(int(eval_cfg["eval_period_iterations"]) >= 0, "eval.eval_period_iterations must be >= 0")
