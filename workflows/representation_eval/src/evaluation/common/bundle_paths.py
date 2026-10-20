"""Path resolution helpers for the standalone downstream eval bundle."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def find_bundle_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _as_path(value: Any, bundle_root: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = bundle_root / path
    return str(path.resolve())


def _load_paths_config(paths_config: str | Path | None = None) -> tuple[Dict[str, Any], Path]:
    bundle_root = find_bundle_root()
    path = Path(paths_config).expanduser() if paths_config is not None else bundle_root / "configs" / "paths.yaml"
    if not path.is_absolute():
        path = bundle_root / path
    if not path.exists():
        return {"data": {}, "checkpoints": {}, "outputs": {}}, bundle_root
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Path config root must be a dictionary: {path}")
    return cfg, bundle_root


def _resolve_section_ref(paths_cfg: Dict[str, Any], section: str, ref: str, bundle_root: Path) -> str:
    section_cfg = paths_cfg.get(section, {})
    if not isinstance(section_cfg, dict):
        raise ValueError(f"paths.{section} must be a dictionary")
    key, _, rest = str(ref).partition("/")
    if key in section_cfg:
        base = _as_path(section_cfg[key], bundle_root)
        return str((Path(base) / rest).resolve()) if rest else base
    if section == "checkpoints" and "baselines" in section_cfg:
        return str((Path(_as_path(section_cfg["baselines"], bundle_root)) / str(ref)).resolve())
    raise KeyError(f"Unknown {section} reference: {ref}")


def _resolve_output(exp_cfg: Dict[str, Any], paths_cfg: Dict[str, Any], bundle_root: Path) -> None:
    output_subdir = exp_cfg.pop("output_subdir", None)
    if output_subdir is not None:
        outputs_cfg = paths_cfg.get("outputs", {})
        output_root = outputs_cfg.get("root", "outputs") if isinstance(outputs_cfg, dict) else "outputs"
        exp_cfg["output_dir"] = str((Path(_as_path(output_root, bundle_root)) / str(output_subdir)).resolve())
    elif exp_cfg.get("output_dir") is not None:
        exp_cfg["output_dir"] = _as_path(exp_cfg["output_dir"], bundle_root)


def _resolve_model(model_cfg: Dict[str, Any], paths_cfg: Dict[str, Any], bundle_root: Path) -> None:
    ckpt_ref = model_cfg.pop("ckpt_ref", None)
    if ckpt_ref is not None:
        model_cfg["ckpt_path"] = _resolve_section_ref(paths_cfg, "checkpoints", str(ckpt_ref), bundle_root)
    elif model_cfg.get("ckpt_path") is not None:
        model_cfg["ckpt_path"] = _as_path(model_cfg["ckpt_path"], bundle_root)

    model_root_ref = model_cfg.pop("model_root_ref", None)
    if model_root_ref is not None:
        model_cfg["model_root"] = _resolve_section_ref(paths_cfg, "checkpoints", str(model_root_ref), bundle_root)
    elif model_cfg.get("model_root") is not None:
        model_cfg["model_root"] = _as_path(model_cfg["model_root"], bundle_root)


def _resolve_data(data_cfg: Dict[str, Any], paths_cfg: Dict[str, Any], bundle_root: Path) -> None:
    root_ref = data_cfg.pop("root_ref", None)
    if root_ref is not None:
        data_cfg["root"] = _resolve_section_ref(paths_cfg, "data", str(root_ref), bundle_root)
    elif data_cfg.get("root") is not None:
        data_cfg["root"] = _as_path(data_cfg["root"], bundle_root)

    data_path_ref = data_cfg.pop("data_path_ref", None)
    if data_path_ref is not None:
        data_cfg["data_path"] = _resolve_section_ref(paths_cfg, "data", str(data_path_ref), bundle_root)
    elif data_cfg.get("data_path") is not None:
        data_cfg["data_path"] = _as_path(data_cfg["data_path"], bundle_root)

    for key in ("train_dir", "val_dir", "test_dir", "video_list_file"):
        if data_cfg.get(key) is not None:
            data_cfg[key] = _as_path(data_cfg[key], bundle_root)


def apply_bundle_paths(
    cfg: Dict[str, Any],
    *,
    config_path: str | Path | None = None,
    paths_config: str | Path | None = None,
) -> Dict[str, Any]:
    """Resolve bundle-local path references before task-specific validation.

    Existing absolute paths are preserved. Relative explicit paths and reference
    keys are interpreted from the standalone bundle root, not the current shell
    directory, so configs stay portable after copying the folder.
    """
    del config_path
    cfg = deepcopy(cfg)
    paths_cfg, bundle_root = _load_paths_config(paths_config)

    if isinstance(cfg.get("experiment"), dict):
        _resolve_output(cfg["experiment"], paths_cfg, bundle_root)
    if isinstance(cfg.get("model"), dict):
        _resolve_model(cfg["model"], paths_cfg, bundle_root)
    if isinstance(cfg.get("data"), dict):
        _resolve_data(cfg["data"], paths_cfg, bundle_root)
    if isinstance(cfg.get("eval_data"), dict):
        _resolve_data(cfg["eval_data"], paths_cfg, bundle_root)
    if isinstance(cfg.get("checkpoint"), dict) and cfg["checkpoint"].get("resume") is not None:
        cfg["checkpoint"]["resume"] = _as_path(cfg["checkpoint"]["resume"], bundle_root)

    return cfg
