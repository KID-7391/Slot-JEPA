"""YAML config loading/validation for linear depth evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.evaluation.common.bundle_paths import apply_bundle_paths
from src.evaluation.linear_cls.config import SUPPORTED_AMP_DTYPES
from src.evaluation.linear_seg.config import _validate_model


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]
SUPPORTED_DATASETS = {"nyuv2", "kitti", "sunrgbd"}
SUPPORTED_HEADS = {"linear"}
SUPPORTED_BINS_STRATEGIES = {"linear", "log"}
SUPPORTED_NORM_STRATEGIES = {"linear", "softmax", "sigmoid"}
SUPPORTED_LOSSES = {"sigloss", "gradient_loss", "gradient_log_loss", "l1"}
SUPPORTED_CROPS = {"FULL", "NYU", "KITTI"}
SUPPORTED_EVAL_MASKS = {"FULL", "NYU_EIGEN", "KITTI_GARG", "KITTI_EIGEN"}
SUPPORTED_SCHEDULERS = {"warmup_onecycle"}
SUPPORTED_RESULT_FORMATS = {"png", "jpg", "pth"}


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _require_key(data: Dict[str, Any], key: str, parent: str):
    _require(key in data, f"Missing required key: {parent}.{key}")


def _ensure_list(value: Any, path: str) -> List[Any]:
    if isinstance(value, list):
        return value
    raise ValueError(f"{path} must be a list")


def load_depth_config(config_path: str, paths_config: str | None = None) -> Dict[str, Any]:
    path = Path(config_path)
    _require(path.exists(), f"Config file does not exist: {config_path}")
    cfg = _load_yaml_with_extends(path)
    _require(isinstance(cfg, dict), "Config root must be a dictionary")
    cfg = apply_bundle_paths(cfg, config_path=path, paths_config=paths_config)
    return _validate_and_fill_defaults(cfg)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    if bool(override.get("_replace_", False)):
        return {key: deepcopy(value) for key, value in override.items() if key != "_replace_"}
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_with_extends(path: Path, stack: tuple[Path, ...] = ()) -> Dict[str, Any]:
    path = path.resolve()
    _require(path not in stack, f"Circular depth config inheritance: {' -> '.join(map(str, stack + (path,)))}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    _require(isinstance(cfg, dict), f"Config root must be a dictionary: {path}")
    base_ref = cfg.pop("extends", None)
    if base_ref is None:
        return cfg
    base_path = Path(str(base_ref)).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    _require(base_path.exists(), f"Base depth config does not exist: {base_path}")
    return _deep_merge(_load_yaml_with_extends(base_path, stack + (path,)), cfg)


def _validate_and_fill_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(cfg)

    _require_key(cfg, "model", "root")
    _require_key(cfg, "data", "root")
    _require_key(cfg, "head", "root")
    _require_key(cfg, "train", "root")

    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault("name", "depth")
    cfg["experiment"].setdefault("output_dir", None)
    cfg["experiment"].setdefault("seed", 0)
    cfg.setdefault("distributed", {})
    cfg["distributed"].setdefault("num_workers", 4)
    cfg["distributed"].setdefault("persistent_workers", False)

    _validate_model(cfg["model"])
    _validate_data(cfg["data"])
    if cfg.get("eval_data") is not None:
        _validate_data(cfg["eval_data"], parent="eval_data")
    _validate_head(cfg["head"])
    _validate_train(cfg["train"])
    cfg.setdefault("head_search", {})
    _validate_head_search(cfg["head_search"], cfg["train"])

    cfg.setdefault("loss", {"sigloss": 1.0})
    _validate_loss(cfg["loss"])

    cfg.setdefault("eval", {})
    _validate_eval(cfg["eval"], cfg["train"])

    cfg.setdefault("result", {})
    _validate_result(cfg["result"])
    if cfg["result"]["save_results"]:
        _require(bool(cfg["experiment"].get("output_dir")), "experiment.output_dir is required when result.save_results=true")
    return cfg


def _dataset_defaults(dataset: str) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "nyuv2": {
            "splits": {"train": "nyu_train.txt", "val": "nyu_test.txt", "test": "nyu_test.txt"},
            "normalization_constant": 1000.0,
            "train": {
                "fixed_crop": "NYU",
                "random_crop_size": [416, 544],
                "brightness_range": [0.75, 1.25],
            },
            "eval": {
                "img_size": None,
                "fixed_crop": "FULL",
                "eval_mask": "NYU_EIGEN",
            },
        },
        "kitti": {
            "splits": {
                "train": "kitti_eigen_train.txt",
                "val": "kitti_eigen_test.txt",
                "test": "kitti_eigen_test.txt",
            },
            "normalization_constant": 256.0,
            "image_dir": "input",
            "depth_dir": "gt_depth",
            "train": {
                "fixed_crop": "KITTI",
                "random_crop_size": [352, 704],
                "brightness_range": [0.9, 1.1],
            },
            "eval": {
                "img_size": [352, 1216],
                "fixed_crop": "KITTI",
                "fixed_crop_depth": True,
                "eval_mask": "KITTI_GARG",
            },
        },
        "sunrgbd": {
            "splits": {
                "train": "SUNRGBD_val_splits.txt",
                "val": "SUNRGBD_val_splits.txt",
                "test": "SUNRGBD_val_splits.txt",
            },
            "normalization_constant": 8000.0,
            "train": {
                "fixed_crop": "FULL",
                "random_crop_size": [416, 544],
                "brightness_range": [0.9, 1.1],
            },
            "eval": {
                "img_size": None,
                "fixed_crop": "FULL",
                "eval_mask": "FULL",
            },
        },
    }
    return deepcopy(defaults[dataset])


def _validate_data(data_cfg: Dict[str, Any], parent: str = "data"):
    _require_key(data_cfg, "root", parent)
    data_cfg.setdefault("dataset", "nyuv2")
    _require(data_cfg["dataset"] in SUPPORTED_DATASETS, f"{parent}.dataset must be one of {sorted(SUPPORTED_DATASETS)}")
    defaults = _dataset_defaults(data_cfg["dataset"])

    data_cfg.setdefault("splits", {})
    data_cfg["splits"].setdefault("train", defaults["splits"]["train"])
    data_cfg["splits"].setdefault("val", defaults["splits"]["val"])
    data_cfg["splits"].setdefault("test", data_cfg["splits"]["val"])

    data_cfg.setdefault("mean", deepcopy(DEFAULT_MEAN))
    data_cfg.setdefault("std", deepcopy(DEFAULT_STD))
    _require(len(data_cfg["mean"]) == 3, f"{parent}.mean must contain 3 values")
    _require(len(data_cfg["std"]) == 3, f"{parent}.std must contain 3 values")

    data_cfg.setdefault("normalization_constant", defaults["normalization_constant"])
    _require(float(data_cfg["normalization_constant"]) > 0, f"{parent}.normalization_constant must be > 0")
    data_cfg.setdefault("image_dir", defaults.get("image_dir"))
    data_cfg.setdefault("depth_dir", defaults.get("depth_dir"))

    data_cfg.setdefault("train", {})
    train_cfg = data_cfg["train"]
    train_cfg.setdefault("img_size", None)
    train_cfg.setdefault("fixed_crop", defaults["train"]["fixed_crop"])
    train_cfg.setdefault("random_crop_size", defaults["train"]["random_crop_size"])
    train_cfg.setdefault("gamma_range", [0.9, 1.1])
    train_cfg.setdefault("brightness_range", defaults["train"]["brightness_range"])
    train_cfg.setdefault("color_range", [0.9, 1.1])
    train_cfg.setdefault("rotation_angle", 2.5)
    train_cfg.setdefault("color_aug_prob", 0.5)
    train_cfg.setdefault("hflip_prob", 0.5)
    train_cfg.setdefault("eval_mask", "FULL")
    _require(
        train_cfg["fixed_crop"] in SUPPORTED_CROPS,
        f"{parent}.train.fixed_crop must be one of {sorted(SUPPORTED_CROPS)}",
    )
    _require(
        train_cfg["eval_mask"] in {"FULL"},
        f"{parent}.train.eval_mask currently supports 'FULL' only",
    )
    crop_size = _ensure_list(train_cfg["random_crop_size"], f"{parent}.train.random_crop_size")
    _require(len(crop_size) == 2, f"{parent}.train.random_crop_size must contain 2 integers")
    train_cfg["random_crop_size"] = [int(crop_size[0]), int(crop_size[1])]

    data_cfg.setdefault("eval", {})
    eval_cfg = data_cfg["eval"]
    eval_cfg.setdefault("img_size", defaults["eval"]["img_size"])
    eval_cfg.setdefault("fixed_crop", defaults["eval"]["fixed_crop"])
    eval_cfg.setdefault("fixed_crop_depth", defaults["eval"].get("fixed_crop_depth", True))
    eval_cfg.setdefault("eval_mask", defaults["eval"]["eval_mask"])
    _require(
        eval_cfg["fixed_crop"] in SUPPORTED_CROPS,
        f"{parent}.eval.fixed_crop must be one of {sorted(SUPPORTED_CROPS)}",
    )
    _require(
        eval_cfg["eval_mask"] in SUPPORTED_EVAL_MASKS,
        f"{parent}.eval.eval_mask must be one of {sorted(SUPPORTED_EVAL_MASKS)}",
    )


def _validate_head(head_cfg: Dict[str, Any]):
    head_cfg.setdefault("type", "linear")
    _require(head_cfg["type"] in SUPPORTED_HEADS, f"head.type must be one of {sorted(SUPPORTED_HEADS)}")
    head_cfg.setdefault("projection_dim", None)
    if head_cfg["projection_dim"] is not None:
        head_cfg["projection_dim"] = int(head_cfg["projection_dim"])
        _require(head_cfg["projection_dim"] > 0, "head.projection_dim must be > 0 when provided")
    head_cfg.setdefault("dropout", 0.0)
    _require(0.0 <= float(head_cfg["dropout"]) < 1.0, "head.dropout must be in [0, 1)")
    head_cfg.setdefault("bias", True)
    head_cfg.setdefault("use_batchnorm", True)
    head_cfg.setdefault("n_output_channels", 256)
    _require(int(head_cfg["n_output_channels"]) > 0, "head.n_output_channels must be > 0")
    head_cfg.setdefault("min_depth", 0.001)
    head_cfg.setdefault("max_depth", 10.0)
    _require(float(head_cfg["min_depth"]) > 0, "head.min_depth must be > 0")
    _require(float(head_cfg["max_depth"]) > float(head_cfg["min_depth"]), "head.max_depth must be > head.min_depth")
    head_cfg.setdefault("bins_strategy", "linear")
    head_cfg.setdefault("norm_strategy", "linear")
    _require(
        head_cfg["bins_strategy"] in SUPPORTED_BINS_STRATEGIES,
        f"head.bins_strategy must be one of {sorted(SUPPORTED_BINS_STRATEGIES)}",
    )
    _require(
        head_cfg["norm_strategy"] in SUPPORTED_NORM_STRATEGIES,
        f"head.norm_strategy must be one of {sorted(SUPPORTED_NORM_STRATEGIES)}",
    )


def _validate_train(train_cfg: Dict[str, Any]):
    train_cfg.setdefault("max_steps", 38400)
    _require(int(train_cfg["max_steps"]) > 0, "train.max_steps must be > 0")
    train_cfg.setdefault("global_batch_size", None)
    train_cfg.setdefault("batch_size", None)
    _require(
        train_cfg["global_batch_size"] is not None or train_cfg["batch_size"] is not None,
        "train.global_batch_size or train.batch_size must be provided",
    )
    if train_cfg["global_batch_size"] is not None:
        _require(int(train_cfg["global_batch_size"]) > 0, "train.global_batch_size must be > 0")
    if train_cfg["batch_size"] is not None:
        _require(int(train_cfg["batch_size"]) > 0, "train.batch_size must be > 0")
    train_cfg.setdefault("grad_clip", 35.0)
    _require(float(train_cfg["grad_clip"]) > 0, "train.grad_clip must be > 0")

    train_cfg.setdefault("optimizer", {})
    train_cfg["optimizer"].setdefault("type", "adamw")
    train_cfg["optimizer"].setdefault("lr", 3e-4)
    train_cfg["optimizer"].setdefault("weight_decay", 1e-4)
    train_cfg["optimizer"].setdefault("betas", [0.9, 0.99])
    _require(train_cfg["optimizer"]["type"] in {"adam", "adamw"}, "train.optimizer.type must be 'adam' or 'adamw'")
    _require(float(train_cfg["optimizer"]["lr"]) > 0, "train.optimizer.lr must be > 0")

    train_cfg.setdefault("scheduler", {})
    train_cfg["scheduler"].setdefault("type", "warmup_onecycle")
    train_cfg["scheduler"].setdefault("warmup_iters", 12800)
    train_cfg["scheduler"].setdefault("final_div_factor", 1000.0)
    train_cfg["scheduler"].setdefault("base_momentum", 0.85)
    train_cfg["scheduler"].setdefault("max_momentum", 0.95)
    _require(
        train_cfg["scheduler"]["type"] in SUPPORTED_SCHEDULERS,
        f"train.scheduler.type must be one of {sorted(SUPPORTED_SCHEDULERS)}",
    )

    train_cfg.setdefault("amp", {})
    train_cfg["amp"].setdefault("enabled", True)
    train_cfg["amp"].setdefault("dtype", "float16")
    _require(
        train_cfg["amp"]["dtype"] in SUPPORTED_AMP_DTYPES,
        f"train.amp.dtype must be one of {sorted(SUPPORTED_AMP_DTYPES)}",
    )


def _validate_head_search(head_search_cfg: Dict[str, Any], train_cfg: Dict[str, Any]):
    head_search_cfg.setdefault("lrs", [float(train_cfg["optimizer"]["lr"])])
    lrs = _ensure_list(head_search_cfg["lrs"], "head_search.lrs")
    _require(len(lrs) > 0, "head_search.lrs cannot be empty")
    parsed_lrs = []
    for lr in lrs:
        lr = float(lr)
        _require(lr > 0.0, "all head_search.lrs values must be > 0")
        parsed_lrs.append(lr)
    head_search_cfg["lrs"] = parsed_lrs


def _validate_loss(loss_cfg: Dict[str, Any]):
    _require(len(loss_cfg) > 0, "loss must define at least one loss term")
    for name, weight in list(loss_cfg.items()):
        lower_name = str(name).lower()
        _require(lower_name in SUPPORTED_LOSSES, f"Unsupported loss '{name}'. Supported: {sorted(SUPPORTED_LOSSES)}")
        _require(float(weight) >= 0, f"loss weight for '{name}' must be >= 0")
        if lower_name != name:
            loss_cfg.pop(name)
            loss_cfg[lower_name] = float(weight)
        else:
            loss_cfg[name] = float(weight)


def _validate_eval(eval_cfg: Dict[str, Any], train_cfg: Dict[str, Any]):
    eval_cfg.setdefault("batch_size", 1)
    eval_cfg.setdefault("global_batch_size", None)
    if eval_cfg["global_batch_size"] is not None:
        _require(int(eval_cfg["global_batch_size"]) > 0, "eval.global_batch_size must be > 0")
    if eval_cfg["batch_size"] is not None:
        _require(int(eval_cfg["batch_size"]) > 0, "eval.batch_size must be > 0")
    eval_cfg.setdefault("metrics", ["rmse", "abs_rel", "a1"])
    eval_cfg["metrics"] = [str(metric) for metric in _ensure_list(eval_cfg["metrics"], "eval.metrics")]
    _require(len(eval_cfg["metrics"]) > 0, "eval.metrics cannot be empty")
    eval_cfg.setdefault("min_depth", 0.001)
    eval_cfg.setdefault("max_depth", 10.0)
    _require(float(eval_cfg["min_depth"]) > 0, "eval.min_depth must be > 0")
    _require(float(eval_cfg["max_depth"]) > float(eval_cfg["min_depth"]), "eval.max_depth must be > eval.min_depth")
    eval_cfg.setdefault("ignored_value", 0.0)
    eval_cfg.setdefault("use_tta", True)
    eval_cfg.setdefault("align_least_squares", False)
    eval_cfg.setdefault("eval_period_iterations", 0)
    _require(int(eval_cfg["eval_period_iterations"]) >= 0, "eval.eval_period_iterations must be >= 0")
    eval_cfg.setdefault("val_split", "val")
    eval_cfg.setdefault("test_split", "test")


def _validate_result(result_cfg: Dict[str, Any]):
    result_cfg.setdefault("save_results", False)
    result_cfg.setdefault("save_format", "png")
    _require(
        result_cfg["save_format"] in SUPPORTED_RESULT_FORMATS,
        f"result.save_format must be one of {sorted(SUPPORTED_RESULT_FORMATS)}",
    )
    result_cfg.setdefault("save_resolution", None)
    if result_cfg["save_resolution"] is not None:
        _require(int(result_cfg["save_resolution"]) > 0, "result.save_resolution must be > 0")
