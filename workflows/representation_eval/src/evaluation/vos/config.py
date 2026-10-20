"""YAML config loading/validation for VOS evaluation."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from src.evaluation.common.bundle_paths import apply_bundle_paths
from .dataset_handlers import SUPPORTED_DATASETS

SUPPORTED_LOAD_MODES = {"get_models", "get_models_slim", "dinov2_student_backbone", "direct_vit_checkpoint"}
SUPPORTED_MASK_SHAPES = {"square", "circle", "none"}
SUPPORTED_ALIGN_MODES = {"floor", "nearest"}
SUPPORTED_DTYPES = {"float32", "float16"}
SUPPORTED_LABEL_AGG_DEVICES = {"auto", "cuda", "cpu"}
SUPPORTED_SLOT_FPS_INIT_MODES = {"random", "first_frame_mask_center"}


def _require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def _require_key(data: Dict[str, Any], key: str, parent: str):
    _require(key in data, f"Missing required key: {parent}.{key}")


def normalize_mask_shape(value: Any) -> str:
    """Normalize neighborhood mask shape aliases to canonical names."""
    aliases = {
        None: "square",
        "square": "square",
        "sq": "square",
        "circle": "circle",
        "disk": "circle",
        "none": "none",
        "-": "none",
        "off": "none",
        "disabled": "none",
    }
    key = value if value is None else str(value).strip().lower()
    if key not in aliases:
        raise ValueError(
            f"Unsupported mask_shape={value!r}; expected one of {sorted(SUPPORTED_MASK_SHAPES)}"
        )
    return aliases[key]


def parse_mask_size_neighborhood(value: Any) -> int | float:
    """Parse finite/infinite neighborhood size values from YAML or CLI."""
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"inf", "infty", "infinity"}:
            return float("inf")
        parsed = float(stripped)
    elif value is None:
        return 0
    else:
        parsed = float(value)

    if math.isfinite(parsed) and parsed.is_integer():
        return int(parsed)
    return parsed


def load_vos_config(config_path: str, paths_config: str | None = None) -> Dict[str, Any]:
    """Load and validate a VOS YAML config."""
    path = Path(config_path)
    _require(path.exists(), f"Config file does not exist: {config_path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    _require(isinstance(cfg, dict), "Config root must be a dictionary")
    cfg = apply_bundle_paths(cfg, config_path=path, paths_config=paths_config)
    return validate_vos_config(cfg)


def validate_vos_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate an in-memory VOS config dict and fill defaults."""
    _require(isinstance(cfg, dict), "Config root must be a dictionary")
    return _validate_and_fill_defaults(cfg)


def _validate_and_fill_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(cfg)

    _require_key(cfg, "experiment", "root")
    _require_key(cfg, "model", "root")
    _require_key(cfg, "data", "root")

    cfg.setdefault("vos", {})
    cfg.setdefault("distributed", {})

    _validate_experiment(cfg["experiment"])
    _validate_model(cfg["model"])
    _validate_data(cfg["data"])
    _validate_vos(cfg["vos"])
    _validate_distributed(cfg["distributed"])

    if cfg["model"]["model_type"] in {"vjepa2", "vjepa2_slot"}:
        cfg["vos"]["use_video_clip_features"] = bool(cfg["vos"].get("use_video_clip_features", True))
        cfg["vos"]["video_infer_frames"] = int(cfg["model"]["video_infer_frames"])
        if cfg["vos"].get("feature_chunk_size") is None:
            cfg["vos"]["feature_chunk_size"] = int(cfg["model"]["video_infer_frames"])
    else:
        cfg["vos"]["use_video_clip_features"] = bool(cfg["vos"].get("use_video_clip_features", False))
        cfg["vos"]["video_infer_frames"] = 1
        if cfg["vos"].get("feature_chunk_size") is None:
            cfg["vos"]["feature_chunk_size"] = 1

    cfg["vos"]["feature_chunk_size"] = int(cfg["vos"]["feature_chunk_size"])
    _require(cfg["vos"]["feature_chunk_size"] >= 1, "vos.feature_chunk_size must be >= 1")
    if cfg["vos"].get("num_workers") is None:
        cfg["vos"]["num_workers"] = 1 if cfg["model"]["model_type"] == "vjepa2_slot" else 4
    cfg["vos"]["num_workers"] = int(cfg["vos"]["num_workers"])
    _require(cfg["vos"]["num_workers"] >= 1, "vos.num_workers must be >= 1")
    _apply_protocol_frame_short_side(cfg)

    return cfg


def resolve_protocol_frame_short_side(
    data_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    vos_cfg: Dict[str, Any],
) -> int:
    frame_short_side = int(vos_cfg.get("frame_short_side", 480))
    if not bool(vos_cfg.get("apply_davis_patch14_short420", True)):
        return frame_short_side
    dataset = str(data_cfg.get("dataset", "")).strip().lower()
    patch_size = int(model_cfg.get("patch_size", 0) or 0)
    if dataset == "davis" and patch_size == 14 and frame_short_side == 480:
        return 420
    return frame_short_side


def _apply_protocol_frame_short_side(cfg: Dict[str, Any]) -> None:
    cfg["vos"]["frame_short_side"] = resolve_protocol_frame_short_side(
        cfg["data"],
        cfg["model"],
        cfg["vos"],
    )


def _validate_experiment(exp_cfg: Dict[str, Any]):
    _require_key(exp_cfg, "name", "experiment")
    _require_key(exp_cfg, "output_dir", "experiment")
    exp_cfg.setdefault("seed", 0)
    exp_cfg["seed"] = int(exp_cfg["seed"])


def _validate_model(model_cfg: Dict[str, Any]):
    _require_key(model_cfg, "model_type", "model")
    _require_key(model_cfg, "model_scale", "model")
    _require_key(model_cfg, "ckpt_path", "model")

    model_cfg.setdefault("load_mode", "get_models")
    _require(
        model_cfg["load_mode"] in SUPPORTED_LOAD_MODES,
        f"model.load_mode must be one of {sorted(SUPPORTED_LOAD_MODES)}",
    )

    if model_cfg["load_mode"] == "dinov2_student_backbone":
        model_cfg.setdefault("state_dict_key", "student")
        model_cfg.setdefault("prefix_filter", "backbone.")
        model_cfg.setdefault("strip_prefix", "backbone.")
        if model_cfg["model_type"] == "dinov2":
            model_cfg.setdefault("patch_size", 14)
        model_cfg.setdefault("num_register_tokens", 4)
    elif model_cfg["load_mode"] == "direct_vit_checkpoint":
        model_cfg.setdefault("state_dict_key", None)
        model_cfg.setdefault("prefix_filter", None)
        model_cfg.setdefault("strip_prefix", "")
        if model_cfg["model_type"] == "dinov2":
            model_cfg.setdefault("patch_size", 14)
            model_cfg.setdefault("num_register_tokens", 4)
        else:
            model_cfg.setdefault("patch_size", 16)

    model_cfg.setdefault("patch_size", 0)
    model_cfg.setdefault("patch_stride", None)
    model_cfg.setdefault("patch_padding", None)
    if model_cfg["model_type"] in {"vjepa2", "vjepa2_slot"}:
        patch_size = int(model_cfg.get("patch_size", 16) or 16)
        patch_stride = model_cfg.get("patch_stride")
        patch_padding = model_cfg.get("patch_padding")
        if patch_stride is None:
            patch_stride = patch_size
        if patch_padding is None:
            patch_padding = 0 if int(patch_stride) == patch_size else int(patch_stride) // 2
        model_cfg["patch_stride"] = int(patch_stride)
        model_cfg["patch_padding"] = int(patch_padding)
        model_cfg.setdefault("video_infer_frames", 16)
        model_cfg["video_infer_frames"] = int(model_cfg["video_infer_frames"])
        _require(model_cfg["video_infer_frames"] >= 2, "model.video_infer_frames must be >= 2 for vjepa2")
        _require(
            model_cfg["video_infer_frames"] % 2 == 0,
            "model.video_infer_frames must be divisible by 2 for vjepa2 tubelets",
        )
        if model_cfg["model_type"] == "vjepa2_slot":
            model_cfg.setdefault("slot_alpha", 0.0)
            model_cfg["slot_alpha"] = float(model_cfg["slot_alpha"])
            _require(0.0 <= model_cfg["slot_alpha"] <= 1.0, "model.slot_alpha must be in [0, 1]")
            model_cfg.setdefault("slot_normalize_before_mix", False)
            _require(
                isinstance(model_cfg["slot_normalize_before_mix"], bool),
                "model.slot_normalize_before_mix must be a boolean",
            )
            model_cfg.setdefault("slot_window_size", [2, 4])
            _require(
                isinstance(model_cfg["slot_window_size"], (list, tuple)) and len(model_cfg["slot_window_size"]) == 2,
                "model.slot_window_size must be a list/tuple of length 2",
            )
            model_cfg["slot_window_size"] = [int(v) for v in model_cfg["slot_window_size"]]
            model_cfg.setdefault("slot_slots_per_window", 4)
            model_cfg.setdefault("slot_global_k", 16)
            model_cfg.setdefault("slot_mask_temp", 1.0)
            model_cfg.setdefault("slot_l_max", 16)
            model_cfg.setdefault("slot_use_decoder_checkpoint", False)
            model_cfg["slot_slots_per_window"] = int(model_cfg["slot_slots_per_window"])
            model_cfg["slot_global_k"] = int(model_cfg["slot_global_k"])
            model_cfg["slot_mask_temp"] = float(model_cfg["slot_mask_temp"])
            model_cfg["slot_l_max"] = int(model_cfg["slot_l_max"])
            model_cfg["slot_use_decoder_checkpoint"] = bool(model_cfg["slot_use_decoder_checkpoint"])


def _validate_data(data_cfg: Dict[str, Any]):
    _require_key(data_cfg, "dataset", "data")
    _require_key(data_cfg, "data_path", "data")
    _require(
        data_cfg["dataset"] in SUPPORTED_DATASETS,
        f"data.dataset must be one of {sorted(SUPPORTED_DATASETS)}",
    )
    data_cfg.setdefault("subset", "val")
    data_cfg.setdefault("video_list_file", None)
    if data_cfg["video_list_file"] is not None:
        _require(
            str(data_cfg["video_list_file"]).strip() != "",
            "data.video_list_file must be a non-empty string when provided",
        )


def _validate_vos(vos_cfg: Dict[str, Any]):
    vos_cfg.setdefault("n_last_frames", 7)
    vos_cfg.setdefault("size_mask_neighborhood", 12)
    vos_cfg.setdefault("mask_shape", "square")
    vos_cfg.setdefault("topk", 5)
    vos_cfg.setdefault("temperature", 0.1)
    vos_cfg.setdefault("matmul_chunk_size", 2048)
    vos_cfg.setdefault("frame_short_side", 480)
    vos_cfg.setdefault("apply_davis_patch14_short420", True)
    vos_cfg.setdefault("frame_align", None)
    vos_cfg.setdefault("frame_align_mode", "floor")
    vos_cfg.setdefault("affinity_dtype", "float32")
    vos_cfg.setdefault("feature_dtype", "float32")
    vos_cfg.setdefault("label_agg_device", "auto")
    vos_cfg.setdefault("cpu_fallback_videos", [])
    vos_cfg.setdefault("skip_completed_videos", True)
    vos_cfg.setdefault("shift_and_stitch_enabled", False)
    vos_cfg.setdefault("shift_and_stitch_factor", 1)
    vos_cfg.setdefault("visualize_outputs", False)
    vos_cfg.setdefault("align_corners", True)
    vos_cfg.setdefault("feature_chunk_size", None)
    vos_cfg.setdefault("use_video_clip_features", None)
    vos_cfg.setdefault("slot_reference_num_frames", 8)
    vos_cfg.setdefault("slot_fps_init_mode", "first_frame_mask_center")
    vos_cfg.setdefault("num_workers", None)

    vos_cfg["size_mask_neighborhood"] = parse_mask_size_neighborhood(vos_cfg["size_mask_neighborhood"])
    vos_cfg["mask_shape"] = normalize_mask_shape(vos_cfg["mask_shape"])
    vos_cfg["frame_align_mode"] = str(vos_cfg["frame_align_mode"]).strip().lower()

    _require(int(vos_cfg["n_last_frames"]) >= 1, "vos.n_last_frames must be >= 1")
    size_mask = float(vos_cfg["size_mask_neighborhood"])
    _require(
        math.isinf(size_mask) or size_mask >= 0,
        "vos.size_mask_neighborhood must be >= 0 or 'inf'",
    )
    _require(int(vos_cfg["topk"]) >= 1, "vos.topk must be >= 1")
    _require(float(vos_cfg["temperature"]) > 0, "vos.temperature must be > 0")
    _require(int(vos_cfg["matmul_chunk_size"]) >= 1, "vos.matmul_chunk_size must be >= 1")
    _require(int(vos_cfg["frame_short_side"]) >= 1, "vos.frame_short_side must be >= 1")
    vos_cfg["apply_davis_patch14_short420"] = bool(vos_cfg["apply_davis_patch14_short420"])
    if vos_cfg["frame_align"] is not None:
        _require(int(vos_cfg["frame_align"]) >= 1, "vos.frame_align must be >= 1 when provided")
    _require(
        vos_cfg["frame_align_mode"] in SUPPORTED_ALIGN_MODES,
        f"vos.frame_align_mode must be one of {sorted(SUPPORTED_ALIGN_MODES)}",
    )
    _require(
        str(vos_cfg["affinity_dtype"]).lower() in {"float32", "float16"},
        "vos.affinity_dtype must be one of {'float32', 'float16'}",
    )
    _require(
        str(vos_cfg["feature_dtype"]).lower() in SUPPORTED_DTYPES,
        "vos.feature_dtype must be one of {'float32', 'float16'}",
    )
    _require(
        str(vos_cfg["label_agg_device"]).lower() in SUPPORTED_LABEL_AGG_DEVICES,
        "vos.label_agg_device must be one of {'auto', 'cuda', 'cpu'}",
    )
    if vos_cfg["cpu_fallback_videos"] is None:
        vos_cfg["cpu_fallback_videos"] = []
    _require(
        isinstance(vos_cfg["cpu_fallback_videos"], list),
        "vos.cpu_fallback_videos must be a list of video names",
    )
    vos_cfg["cpu_fallback_videos"] = [str(v).strip() for v in vos_cfg["cpu_fallback_videos"] if str(v).strip()]
    vos_cfg["skip_completed_videos"] = bool(vos_cfg["skip_completed_videos"])
    vos_cfg["shift_and_stitch_enabled"] = bool(vos_cfg["shift_and_stitch_enabled"])
    vos_cfg["shift_and_stitch_factor"] = int(vos_cfg["shift_and_stitch_factor"])
    vos_cfg["visualize_outputs"] = bool(vos_cfg["visualize_outputs"])
    vos_cfg["align_corners"] = bool(vos_cfg["align_corners"])
    vos_cfg["slot_reference_num_frames"] = int(vos_cfg["slot_reference_num_frames"])
    vos_cfg["slot_fps_init_mode"] = str(vos_cfg["slot_fps_init_mode"]).strip().lower()
    _require(
        vos_cfg["shift_and_stitch_factor"] >= 1,
        "vos.shift_and_stitch_factor must be >= 1",
    )
    _require(
        vos_cfg["slot_reference_num_frames"] >= 1,
        "vos.slot_reference_num_frames must be >= 1",
    )
    _require(
        vos_cfg["slot_fps_init_mode"] in SUPPORTED_SLOT_FPS_INIT_MODES,
        "vos.slot_fps_init_mode must be one of "
        f"{sorted(SUPPORTED_SLOT_FPS_INIT_MODES)}",
    )


def _validate_distributed(dist_cfg: Dict[str, Any]):
    dist_cfg.setdefault("num_gpus", None)
    dist_cfg.setdefault("skip_existing", True)
    if dist_cfg["num_gpus"] is not None:
        _require(int(dist_cfg["num_gpus"]) >= 1, "distributed.num_gpus must be >= 1 when provided")
