"""YAML-driven linear depth evaluation entrypoint."""

import argparse
import logging
import os
import random
import sys
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.depth import evaluate_depth, load_depth_config
from src.models.direct_checkpoint import (
    apply_prefix_rules,
    extract_state_dict_from_checkpoint,
    load_direct_checkpoint_model,
)
from src.models import get_models, get_models_slim


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ssl_eval.depth")


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        else:
            dist.init_process_group(backend="gloo", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_vjepa2_model(model_cfg: Dict[str, Any], device: torch.device):
    from src.models.vjepa_2_1.vit_encoder import init_module as init_vjepa2_module

    patch_size = int(model_cfg.get("patch_size", 16) or 16)
    patch_stride = int(model_cfg.get("patch_stride", patch_size) or patch_size)
    patch_padding = int(model_cfg.get("patch_padding", patch_stride // 2) or 0)
    video_infer_frames = int(model_cfg.get("video_infer_frames", 2) or 2)
    return init_vjepa2_module(
        resolution=384,
        model_kwargs={
            "encoder": {
                "model_name": model_cfg["model_scale"],
                "patch_size": patch_size,
                "patch_stride": patch_stride,
                "patch_padding": patch_padding,
                "tubelet_size": 2,
                "uniform_power": True,
                "use_rope": True,
            }
        },
        wrapper_kwargs={"img_as_video_nframes": video_infer_frames},
    ).to(device)


def _load_vjepa2_checkpoint(model: nn.Module, model_cfg: Dict[str, Any]):
    checkpoint = torch.load(model_cfg["ckpt_path"], map_location="cpu")
    state_dict_key = model_cfg.get("state_dict_key")
    if state_dict_key is not None:
        if state_dict_key not in checkpoint:
            raise KeyError(f"VJEPA2 checkpoint does not contain model.state_dict_key={state_dict_key!r}")
        state_dict = checkpoint[state_dict_key]
    else:
        state_dict = checkpoint
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned_state_dict[key] = value
    state_dict = cleaned_state_dict

    for key, value in model.state_dict().items():
        if key in state_dict and state_dict[key].shape != value.shape:
            logger.warning(
                "Ignoring VJEPA2 checkpoint tensor with incompatible shape: %s checkpoint=%s model=%s",
                key,
                tuple(state_dict[key].shape),
                tuple(value.shape),
            )
            state_dict[key] = value
    return model.load_state_dict(state_dict, strict=False)


def _create_vjepa2_slot_model(model_cfg: Dict[str, Any], device: torch.device):
    from src.models.vjepa_2_1.slot_wrapper import VJEPA2SlotWrapper

    patch_size = int(model_cfg.get("patch_size", 16) or 16)
    patch_stride = int(model_cfg.get("patch_stride", patch_size) or patch_size)
    patch_padding = int(model_cfg.get("patch_padding", patch_stride // 2) or 0)
    video_infer_frames = int(model_cfg.get("video_infer_frames", 2) or 2)
    return VJEPA2SlotWrapper(
        resolution=384,
        model_scale=model_cfg["model_scale"],
        patch_size=patch_size,
        patch_stride=patch_stride,
        patch_padding=patch_padding,
        video_infer_frames=video_infer_frames,
        alpha=float(model_cfg.get("slot_alpha", 0.0)),
        normalize_before_mix=bool(model_cfg.get("slot_normalize_before_mix", False)),
        window_size=tuple(model_cfg.get("slot_window_size", [2, 4])),
        slots_per_window=int(model_cfg.get("slot_slots_per_window", 4)),
        global_k=int(model_cfg.get("slot_global_k", 16)),
        mask_temp=float(model_cfg.get("slot_mask_temp", 1.0)),
        l_max=int(model_cfg.get("slot_l_max", 16)),
        use_decoder_checkpoint=bool(model_cfg.get("slot_use_decoder_checkpoint", False)),
    ).to(device)


def load_model_from_config(model_cfg: Dict[str, Any], device: torch.device, rank: int):
    model_type = model_cfg["model_type"]
    model_scale = model_cfg["model_scale"]
    load_mode = model_cfg["load_mode"]

    if model_type == "vjepa2":
        model = _create_vjepa2_model(model_cfg, device)
        msg = _load_vjepa2_checkpoint(model, model_cfg)
        if rank == 0:
            logger.info("Model loaded: %s", msg)
    elif model_type == "vjepa2_slot":
        model = _create_vjepa2_slot_model(model_cfg, device)
        checkpoint = torch.load(model_cfg["ckpt_path"], map_location="cpu")
        encoder_msg, extractor_msg = model.load_combined_checkpoint(checkpoint)
        if rank == 0:
            logger.info("Model loaded: encoder=%s extractor=%s", encoder_msg, extractor_msg)
    elif load_mode == "get_models":
        model = get_models(
            model_type=model_type,
            model_scale=model_scale,
            model_root=model_cfg["model_root"],
            epoch=int(model_cfg["epoch"]),
            device=device,
        )
    elif load_mode == "get_models_slim":
        model = get_models_slim(
            model_type=model_type,
            model_scale=model_scale,
            model_root=model_cfg["model_root"],
            epoch=int(model_cfg["epoch"]),
            device=device,
        )
    elif load_mode == "dinov2_student_backbone":
        if model_scale != "vit_small":
            raise NotImplementedError("dinov2_student_backbone currently supports model_scale=vit_small only")

        from src.models.dinov2.vision_transformer import vit_small as dinov2_vit_small

        model = dinov2_vit_small(
            patch_size=int(model_cfg.get("patch_size", 14)),
            num_register_tokens=int(model_cfg.get("num_register_tokens", 4)),
        )

        checkpoint = torch.load(model_cfg["ckpt_path"], map_location="cpu")
        state_dict = extract_state_dict_from_checkpoint(checkpoint, model_cfg)
        state_dict = apply_prefix_rules(state_dict, model_cfg)

        if "pos_embed" in state_dict:
            model.pos_embed = nn.Parameter(state_dict["pos_embed"])

        msg = model.load_state_dict(state_dict, strict=False)
        if rank == 0:
            logger.info(f"Model loaded: {msg}")
        model = model.to(device)
    elif load_mode == "direct_vit_checkpoint":
        model, msg = load_direct_checkpoint_model(model_cfg, device=device)
        if rank == 0:
            logger.info(f"Model loaded: {msg}")
    else:
        raise ValueError(f"Unsupported model.load_mode={load_mode}")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="YAML-driven linear depth evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--paths_config", type=str, default=None, help="Optional bundle path-map YAML")
    args = parser.parse_args()

    distributed, rank, world_size, local_rank = init_distributed()
    cfg = load_depth_config(args.config, paths_config=args.paths_config)
    seed = int(cfg.get("experiment", {}).get("seed", 0))
    set_seed(seed + rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if rank == 0:
        logger.info(f"Loaded config: {args.config}")
        logger.info(f"Distributed: {distributed}, world_size={world_size}")
        logger.info(f"Experiment name: {cfg.get('experiment', {}).get('name', 'depth')}")

    model = load_model_from_config(cfg["model"], device=device, rank=rank)
    results = evaluate_depth(model=model, cfg=cfg)

    if rank == 0:
        print("\n" + "=" * 60)
        print("FINAL DEPTH RESULTS")
        print("=" * 60)
        for split_name in ("val", "test"):
            split_results = results.get(split_name, {})
            if split_results:
                print(f"{split_name}: " + ", ".join(f"{name}={value:.4f}" for name, value in split_results.items()))
        print("=" * 60)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
