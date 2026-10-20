#!/usr/bin/env python3
"""Validate the standalone downstream eval bundle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Callable, Dict

import yaml


BUNDLE_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "slot_jepa_matplotlib"))
sys.path.insert(0, str(BUNDLE_ROOT))

from src.evaluation.depth import load_depth_config
from src.evaluation.linear_seg.config import load_linear_seg_config
from src.evaluation.vos import load_vos_config


LOADERS: Dict[str, Callable[..., dict]] = {
    "linear_seg": load_linear_seg_config,
    "depth": load_depth_config,
    "vos": load_vos_config,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate downstream_eval bundle structure and configs.")
    parser.add_argument("--paths_config", type=str, default="configs/paths.yaml")
    parser.add_argument("--validate-configs", action="store_true")
    parser.add_argument("--check-checkpoints", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path):
    path = Path(path)
    if not path.is_absolute():
        path = BUNDLE_ROOT / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_configs(paths_config: str):
    errors = []
    for task_dir in sorted((BUNDLE_ROOT / "configs" / "tasks").iterdir()):
        if not task_dir.is_dir():
            continue
        loader = LOADERS.get(task_dir.name)
        if loader is None:
            continue
        for config_path in sorted(task_dir.glob("*.yaml")):
            try:
                loader(str(config_path), paths_config=paths_config)
                print(f"[ok][config] {config_path.relative_to(BUNDLE_ROOT)}")
            except Exception as exc:  # noqa: BLE001
                errors.append((config_path, exc))
                print(f"[fail][config] {config_path.relative_to(BUNDLE_ROOT)}: {exc}")
    if errors:
        raise SystemExit(1)


def check_checkpoints(paths_config: str):
    errors = []
    refs = set()
    paths_cfg = load_yaml(paths_config)
    checkpoint_roots = paths_cfg.get("checkpoints") or {}
    # Only released Slot-JEPA configs are required to resolve from the bundled
    # artifact package. Baseline checkpoints remain optional downloads.
    for config_path in (BUNDLE_ROOT / "configs" / "tasks").glob("*/*_slot_jepa.yaml"):
        cfg = load_yaml(config_path)
        ref = (cfg.get("model") or {}).get("ckpt_ref")
        if ref:
            refs.add(str(ref))
    for ref in sorted(refs):
        key, _, rest = ref.partition("/")
        root = checkpoint_roots.get(key, checkpoint_roots.get("baselines"))
        if root is None:
            errors.append(f"unknown checkpoint reference {ref}")
            print(f"[fail][checkpoint] unknown reference {ref}")
            continue
        path = Path(str(root)).expanduser()
        if not path.is_absolute():
            path = BUNDLE_ROOT / path
        path = path / rest if rest else path
        path = path.resolve()
        if not path.exists():
            errors.append(f"missing {path}")
            print(f"[fail][checkpoint] missing {path}")
            continue
        print(f"[ok][checkpoint] {path} ({path.stat().st_size} bytes)")
    if errors:
        raise SystemExit(1)


def validate_data(paths_config: str):
    cfg = load_yaml(paths_config)
    errors = []
    for key, raw_path in sorted((cfg.get("data") or {}).items()):
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = BUNDLE_ROOT / path
        if path.exists():
            print(f"[ok][data] {key}: {path}")
        else:
            errors.append((key, path))
            print(f"[missing][data] {key}: {path}")
    if errors:
        raise SystemExit(1)


def main():
    args = parse_args()
    if not any([args.validate_configs, args.check_checkpoints, args.validate_data]):
        args.validate_configs = True
        args.check_checkpoints = True
    if args.validate_configs:
        validate_configs(args.paths_config)
    if args.check_checkpoints:
        check_checkpoints(args.paths_config)
    if args.validate_data and not args.skip_data:
        validate_data(args.paths_config)


if __name__ == "__main__":
    main()
