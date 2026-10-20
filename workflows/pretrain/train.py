#!/usr/bin/env python3
"""Launch Slot-JEPA pretraining from a YAML configuration."""

import argparse
import logging
import os
import pprint
import sys
from pathlib import Path

import yaml


WORKFLOW_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = WORKFLOW_DIR.parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_DIR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", required=True, help="YAML configuration to load")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    logging.getLogger().setLevel(logging.INFO if rank == 0 else logging.ERROR)
    with Path(args.fname).open("r", encoding="utf-8") as stream:
        params = yaml.safe_load(stream)
    if rank == 0:
        pprint.pp(params)
        output_dir = Path(params["folder"])
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "params-pretrain.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(params, stream, sort_keys=False)
    if params.get("app") != "slot_jepa":
        raise ValueError(f"Unsupported pretraining app: {params.get('app')!r}")
    from trainer import main as train
    train(args=args, resume_preempt=False)


if __name__ == "__main__":
    main()
