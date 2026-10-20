"""YAML-driven VOS (Video Object Segmentation) evaluation entrypoint."""

import argparse
import csv
import json
import logging
import os
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.vos import evaluate_vos, load_vos_config, validate_vos_config
from src.models.direct_checkpoint import (
    apply_prefix_rules,
    extract_state_dict_from_checkpoint,
    load_direct_checkpoint_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ssl_eval.vos")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _create_model(model_type, model_scale, model_cfg, device):
    """Create model architecture based on model_type and model_scale."""
    if model_type == "mae":
        from src.models.mae.models_mae import mae_vit_small_patch16 as vit_small
        return vit_small().to(device)

    elif model_type == "dino":
        from src.models.dino.vision_transformer import vit_small
        if model_scale == "vit_small":
            return vit_small().to(device)
        raise NotImplementedError(f"dino does not support model_scale={model_scale}")

    elif model_type in ("ibot", "mugs"):
        from src.models.ibot.vision_transformer import vit_small
        if model_scale == "vit_small":
            return vit_small().to(device)
        raise NotImplementedError(f"{model_type} does not support model_scale={model_scale}")

    elif model_type == "ijepa":
        from src.models.ijepa.deit import deit_base
        if model_scale == "vit_base":
            return deit_base().to(device)
        raise NotImplementedError(f"ijepa does not support model_scale={model_scale}")

    elif model_type in ("mocov3", "RINCE"):
        from src.models.mocov3.vit_moco import vit_small
        if model_scale == "vit_small":
            return vit_small().to(device)
        raise NotImplementedError(f"{model_type} does not support model_scale={model_scale}")

    elif model_type in ("mec", "resa", "simsiam", "densecl", "swav", "vicreg", "vicregl", "barlowtwins"):
        if model_scale == "resnet50":
            from src.models.resnet import resnet50
            return resnet50().to(device)
        raise NotImplementedError(f"{model_type} does not support model_scale={model_scale}")

    elif model_type == "byol":
        from src.models.byol.byol import EncoderwithProjection
        return EncoderwithProjection().to(device)

    elif model_type == "esvit":
        from src.models.esvit.swin_transformer import SwinTransformer
        if model_scale == "swint_tiny":
            return SwinTransformer(
                img_size=[224, 224], in_chans=3, num_classes=1000, patch_size=4,
                embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                window_size=7, mlp_ratio=4, qkv_bias=True, drop_rate=0,
                attn_drop_rate=0, drop_path_rate=0.1,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                ape=False, patch_norm=True, use_dense_prediction=False,
            ).to(device)
        raise NotImplementedError(f"esvit does not support model_scale={model_scale}")

    elif model_type == "dinov2":
        from src.models.dinov2.vision_transformer import vit_small as dinov2_vit_small
        if model_scale == "vit_small":
            patch_size = int(model_cfg.get("patch_size", 14)) or 14
            num_register_tokens = int(model_cfg.get("num_register_tokens", 4))
            return dinov2_vit_small(patch_size=patch_size, num_register_tokens=num_register_tokens).to(device)
        raise NotImplementedError(f"dinov2 does not support model_scale={model_scale}")

    elif model_type == "vjepa2":
        from src.models.vjepa_2_1.vit_encoder import init_module as vjepa2_model
        patch_size = int(model_cfg.get("patch_size", 16) or 16)
        patch_stride = int(model_cfg.get("patch_stride", patch_size) or patch_size)
        patch_padding = int(model_cfg.get("patch_padding", patch_stride // 2) or 0)
        video_infer_frames = int(model_cfg.get("video_infer_frames", 16) or 16)
        return vjepa2_model(
            resolution=384,
            model_kwargs={
                "encoder": {
                    "model_name": model_scale,
                    "patch_size": patch_size,
                    "patch_stride": patch_stride,
                    "patch_padding": patch_padding,
                    "tubelet_size": 2,
                    "uniform_power": True,
                    "use_rope": True
                }
            },
            wrapper_kwargs={
                "img_as_video_nframes": video_infer_frames
            },
        ).to(device)
    elif model_type == "vjepa2_slot":
        from src.models.vjepa_2_1.slot_wrapper import VJEPA2SlotWrapper

        patch_size = int(model_cfg.get("patch_size", 16) or 16)
        patch_stride = int(model_cfg.get("patch_stride", patch_size) or patch_size)
        patch_padding = int(model_cfg.get("patch_padding", patch_stride // 2) or 0)
        video_infer_frames = int(model_cfg.get("video_infer_frames", 16) or 16)
        return VJEPA2SlotWrapper(
            resolution=384,
            model_scale=model_scale,
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
    else:
        raise NotImplementedError(f"Model type '{model_type}' not supported")


def _is_slim_checkpoint(checkpoint):
    """Check if checkpoint is a slim (pre-extracted) checkpoint."""
    return (isinstance(checkpoint, dict)
            and "state_dict" in checkpoint
            and "model_type" in checkpoint)


def _extract_state_dict_full(checkpoint, model_type, state_dict_key=None):
    """Extract encoder state_dict from a full (non-slim) checkpoint."""
    if model_type == "mae":
        return checkpoint["model"]

    elif model_type == "dino":
        sd = checkpoint["teacher"]
        return {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}

    elif model_type in ("ibot", "mugs"):
        sd = checkpoint["teacher"]
        return {k.replace("backbone.", ""): v for k, v in sd.items() if k.startswith("backbone.")}

    elif model_type == "ijepa":
        sd = checkpoint["encoder"]
        return {k.replace("module.", ""): v for k, v in sd.items()}

    elif model_type in ("mocov3", "RINCE"):
        sd = checkpoint["state_dict"]
        sd = {k: v for k, v in sd.items() if k.startswith("module.momentum_encoder.")}
        return {k.replace("module.momentum_encoder.", ""): v for k, v in sd.items()}

    elif model_type in ("mec", "resa", "simsiam"):
        sd = checkpoint["state_dict"]
        sd = {k: v for k, v in sd.items() if k.startswith("module.encoder.")}
        return {k.replace("module.encoder.", ""): v for k, v in sd.items()}

    elif model_type == "densecl":
        sd = checkpoint["state_dict"]
        sd = {k: v for k, v in sd.items() if k.startswith("backbone.")}
        return {k.replace("backbone.", ""): v for k, v in sd.items()}

    elif model_type == "swav":
        sd = checkpoint.get("state_dict", checkpoint)
        return {k.replace("module.", ""): v for k, v in sd.items()}

    elif model_type in ("vicreg", "vicregl"):
        sd = checkpoint["model"]
        if "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k: v for k, v in sd.items() if "encoder" not in k}
        return {k.replace("module.backbone.", ""): v for k, v in sd.items()}

    elif model_type == "byol":
        sd = checkpoint["model"]
        sd = {k: v for k, v in sd.items() if k.startswith("module.target_network.")}
        return {k.replace("module.target_network.", ""): v for k, v in sd.items()}

    elif model_type in ("barlowtwins",):
        sd = checkpoint["model"]
        if "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k: v for k, v in sd.items() if "encoder" not in k}
        return {k.replace("module.backbone.", ""): v for k, v in sd.items()}

    elif model_type == "esvit":
        return checkpoint["teacher"]

    elif model_type == "dinov2":
        return checkpoint

    elif model_type == "vjepa2":
        pretrained_dict = checkpoint[state_dict_key]
        # --
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        return pretrained_dict
    elif model_type == "vjepa2_slot":
        return checkpoint

    else:
        raise NotImplementedError(f"Full checkpoint extraction not supported for '{model_type}'")


def load_model_from_config(model_cfg: Dict[str, Any], device: torch.device):
    """Load a model from a checkpoint path specified in config."""
    model_type = model_cfg["model_type"]
    model_scale = model_cfg["model_scale"]
    load_mode = model_cfg["load_mode"]
    ckpt_path = model_cfg["ckpt_path"]

    if load_mode == "dinov2_student_backbone":
        # Special handling: load student backbone with prefix filtering
        from src.models.dinov2.vision_transformer import vit_small as dinov2_vit_small

        if model_scale != "vit_small":
            raise NotImplementedError("dinov2_student_backbone currently supports model_scale=vit_small only")

        patch_size = int(model_cfg.get("patch_size", 14)) or 14
        num_register_tokens = int(model_cfg.get("num_register_tokens", 4))
        model = dinov2_vit_small(patch_size=patch_size, num_register_tokens=num_register_tokens).to(device)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = extract_state_dict_from_checkpoint(checkpoint, model_cfg)
        state_dict = apply_prefix_rules(state_dict, model_cfg)

        if "pos_embed" in state_dict:
            model.pos_embed = nn.Parameter(state_dict["pos_embed"].to(device))

        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Model loaded: {msg}")
    elif load_mode == "direct_vit_checkpoint" and model_type != "vjepa2_slot":
        model, msg = load_direct_checkpoint_model(model_cfg, device=device)
        logger.info(f"Model loaded: {msg}")
    elif model_type == "vjepa2_slot":
        model = _create_model(model_type, model_scale, model_cfg, device)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        encoder_msg, extractor_msg = model.load_combined_checkpoint(checkpoint)
        logger.info("Model loaded: encoder=%s extractor=%s", encoder_msg, extractor_msg)
    else:
        # get_models / get_models_slim: create architecture then load checkpoint
        model = _create_model(model_type, model_scale, model_cfg, device)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if _is_slim_checkpoint(checkpoint):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = _extract_state_dict_full(checkpoint, model_type, state_dict_key=model_cfg.get("state_dict_key"))

        # Handle dinov2 pos_embed size mismatch
        if model_type == "dinov2" and "pos_embed" in state_dict:
            model.pos_embed = nn.Parameter(state_dict["pos_embed"].to(device))

        for k, v in model.state_dict().items():
            if k not in state_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif state_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and loaded state dict')
                state_dict[k] = v
        
        msg = model.load_state_dict(state_dict, strict=False)
        logger.info(f"Model loaded: {msg}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _parse_metric_value(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def write_effective_config(cfg: Dict[str, Any]) -> Path:
    """Persist the fully resolved configuration before evaluation starts."""
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "effective_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    logger.info("Wrote effective VOS config: %s", config_path)
    return config_path


def write_summary_if_available(cfg: Dict[str, Any]) -> None:
    """Persist a compact JSON summary next to the official VOS CSV outputs."""
    output_dir = Path(cfg["experiment"]["output_dir"])
    subset = cfg["data"]["subset"]
    global_csv = output_dir / f"global_results-{subset}.csv"
    if not global_csv.exists():
        logger.warning("VOS global metrics CSV not found, summary.json not written: %s", global_csv)
        return

    with global_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)

    summary = {
        "experiment": {
            "name": cfg["experiment"]["name"],
            "output_dir": str(output_dir),
            "seed": cfg["experiment"].get("seed"),
        },
        "dataset": {
            "name": cfg["data"]["dataset"],
            "subset": subset,
            "data_path": cfg["data"]["data_path"],
            "video_list_file": cfg["data"].get("video_list_file"),
        },
        "model": {
            "model_type": cfg["model"]["model_type"],
            "model_scale": cfg["model"]["model_scale"],
            "ckpt_path": cfg["model"]["ckpt_path"],
            "patch_size": cfg["model"].get("patch_size"),
            "patch_stride": cfg["model"].get("patch_stride"),
            "patch_padding": cfg["model"].get("patch_padding"),
            "video_infer_frames": cfg["model"].get("video_infer_frames"),
            "slot_alpha": cfg["model"].get("slot_alpha"),
            "slot_normalize_before_mix": cfg["model"].get("slot_normalize_before_mix"),
        },
        "vos": {
            "n_last_frames": cfg["vos"]["n_last_frames"],
            "size_mask_neighborhood": cfg["vos"]["size_mask_neighborhood"],
            "mask_shape": cfg["vos"].get("mask_shape"),
            "topk": cfg["vos"]["topk"],
            "temperature": cfg["vos"]["temperature"],
            "frame_short_side": cfg["vos"].get("frame_short_side"),
            "apply_davis_patch14_short420": cfg["vos"].get("apply_davis_patch14_short420"),
            "frame_align": cfg["vos"].get("frame_align"),
            "frame_align_mode": cfg["vos"].get("frame_align_mode"),
            "affinity_dtype": cfg["vos"].get("affinity_dtype"),
            "feature_dtype": cfg["vos"].get("feature_dtype"),
            "shift_and_stitch_enabled": cfg["vos"].get("shift_and_stitch_enabled"),
            "shift_and_stitch_factor": cfg["vos"].get("shift_and_stitch_factor"),
            "visualize_outputs": cfg["vos"].get("visualize_outputs"),
            "align_corners": cfg["vos"].get("align_corners"),
            "feature_chunk_size": cfg["vos"].get("feature_chunk_size"),
            "use_video_clip_features": cfg["vos"].get("use_video_clip_features"),
            "slot_reference_num_frames": cfg["vos"].get("slot_reference_num_frames"),
            "slot_fps_init_mode": cfg["vos"].get("slot_fps_init_mode"),
            "num_workers": cfg["vos"].get("num_workers"),
        },
        "metrics": {key: _parse_metric_value(value) for key, value in row.items()},
        "files": {
            "global_results": str(global_csv),
            "per_sequence_results": str(output_dir / f"per-sequence_results-{subset}.csv"),
        },
        "distributed": {
            "num_gpus": cfg.get("distributed", {}).get("num_gpus"),
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Wrote VOS summary: %s", summary_path)

def main():
    parser = argparse.ArgumentParser(description="YAML-driven VOS evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--paths_config", type=str, default=None, help="Optional bundle path-map YAML")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Optional override for model.ckpt_path")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional override for experiment.output_dir")
    parser.add_argument("--dataset", type=str, default=None, help="Optional override for data.dataset")
    parser.add_argument("--data_path", type=str, default=None, help="Optional override for data.data_path")
    parser.add_argument("--subset", type=str, default=None, help="Optional override for data.subset")
    parser.add_argument("--video_list_file", type=str, default=None, help="Optional override for data.video_list_file")
    parser.add_argument("--num_gpus", type=int, default=None, help="Optional override for distributed.num_gpus")
    parser.add_argument("--seed", type=int, default=None, help="Optional override for experiment.seed")
    parser.add_argument("--slot_alpha", type=float, default=None, help="Optional override for model.slot_alpha")
    parser.add_argument(
        "--slot_normalize_before_mix",
        type=int,
        choices=(0, 1),
        default=None,
        help="Optional override for model.slot_normalize_before_mix (1 or 0; default: 0)",
    )
    parser.add_argument("--n_last_frames", type=int, default=None, help="Optional override for vos.n_last_frames")
    parser.add_argument("--num_workers", type=int, default=None, help="Optional override for vos.num_workers")
    parser.add_argument(
        "--slot_reference_num_frames",
        type=int,
        default=None,
        help="Optional override for vos.slot_reference_num_frames",
    )
    parser.add_argument(
        "--slot_fps_init_mode",
        choices=("random", "first_frame_mask_center"),
        default=None,
        help="Optional override for vos.slot_fps_init_mode",
    )
    parser.add_argument(
        "--size_mask_neighborhood",
        type=str,
        default=None,
        help="Optional override for vos.size_mask_neighborhood (supports 'inf')",
    )
    parser.add_argument(
        "--mask_shape",
        type=str,
        default=None,
        help="Optional override for vos.mask_shape: square | circle | none",
    )
    parser.add_argument("--topk", type=int, default=None, help="Optional override for vos.topk")
    parser.add_argument("--temperature", type=float, default=None, help="Optional override for vos.temperature")
    parser.add_argument(
        "--affinity_dtype",
        type=str,
        default=None,
        help="Optional override for vos.affinity_dtype: float32 | float16",
    )
    parser.add_argument(
        "--feature_dtype",
        type=str,
        default=None,
        help="Optional override for vos.feature_dtype: float32 | float16",
    )
    parser.add_argument(
        "--label_agg_device",
        type=str,
        default=None,
        help="Optional override for vos.label_agg_device: auto | cuda | cpu",
    )
    parser.add_argument(
        "--cpu_fallback_videos",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of video names that should run fully on CPU",
    )
    parser.add_argument(
        "--skip_completed_videos",
        type=int,
        default=None,
        help="Optional override for vos.skip_completed_videos (1 or 0)",
    )
    parser.add_argument(
        "--matmul_chunk_size",
        type=int,
        default=None,
        help="Optional override for vos.matmul_chunk_size",
    )
    parser.add_argument(
        "--frame_short_side",
        type=int,
        default=None,
        help="Optional override for vos.frame_short_side",
    )
    parser.add_argument(
        "--frame_align_mode",
        type=str,
        default=None,
        help="Optional override for vos.frame_align_mode: floor | nearest",
    )
    parser.add_argument(
        "--shift_and_stitch",
        type=int,
        default=None,
        help="Optional override for vos.shift_and_stitch_enabled (1 or 0)",
    )
    parser.add_argument(
        "--shift_and_stitch_factor",
        type=int,
        default=None,
        help="Optional override for vos.shift_and_stitch_factor",
    )
    parser.add_argument(
        "--visualize_outputs",
        type=int,
        default=None,
        help="Optional override for vos.visualize_outputs (1 or 0)",
    )
    parser.add_argument(
        "--align_corners",
        type=int,
        default=None,
        help="Optional override for vos.align_corners (1 or 0)",
    )
    args = parser.parse_args()

    cfg = load_vos_config(args.config, paths_config=args.paths_config)
    if args.ckpt_path is not None:
        cfg["model"]["ckpt_path"] = args.ckpt_path
    if args.output_dir is not None:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.dataset is not None:
        cfg["data"]["dataset"] = args.dataset
    if args.data_path is not None:
        cfg["data"]["data_path"] = args.data_path
    if args.subset is not None:
        cfg["data"]["subset"] = args.subset
    if args.video_list_file is not None:
        cfg["data"]["video_list_file"] = args.video_list_file
    if args.num_gpus is not None:
        cfg.setdefault("distributed", {})
        cfg["distributed"]["num_gpus"] = args.num_gpus
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.slot_alpha is not None:
        cfg["model"]["slot_alpha"] = args.slot_alpha
    if args.slot_normalize_before_mix is not None:
        cfg["model"]["slot_normalize_before_mix"] = bool(args.slot_normalize_before_mix)
    if args.n_last_frames is not None:
        cfg["vos"]["n_last_frames"] = args.n_last_frames
    if args.num_workers is not None:
        cfg["vos"]["num_workers"] = args.num_workers
    elif os.environ.get("VOS_TEST_WORKERS") is not None:
        cfg["vos"]["num_workers"] = int(os.environ["VOS_TEST_WORKERS"])
    if args.slot_reference_num_frames is not None:
        cfg["vos"]["slot_reference_num_frames"] = args.slot_reference_num_frames
    if args.slot_fps_init_mode is not None:
        cfg["vos"]["slot_fps_init_mode"] = args.slot_fps_init_mode
    if args.size_mask_neighborhood is not None:
        cfg["vos"]["size_mask_neighborhood"] = args.size_mask_neighborhood
    if args.mask_shape is not None:
        cfg["vos"]["mask_shape"] = args.mask_shape
    if args.topk is not None:
        cfg["vos"]["topk"] = args.topk
    if args.temperature is not None:
        cfg["vos"]["temperature"] = args.temperature
    if args.affinity_dtype is not None:
        cfg["vos"]["affinity_dtype"] = args.affinity_dtype
    if args.feature_dtype is not None:
        cfg["vos"]["feature_dtype"] = args.feature_dtype
    if args.label_agg_device is not None:
        cfg["vos"]["label_agg_device"] = args.label_agg_device
    if args.cpu_fallback_videos is not None:
        cfg["vos"]["cpu_fallback_videos"] = args.cpu_fallback_videos
    if args.skip_completed_videos is not None:
        cfg["vos"]["skip_completed_videos"] = bool(args.skip_completed_videos)
    if args.matmul_chunk_size is not None:
        cfg["vos"]["matmul_chunk_size"] = args.matmul_chunk_size
    if args.frame_short_side is not None:
        cfg["vos"]["frame_short_side"] = args.frame_short_side
    if args.frame_align_mode is not None:
        cfg["vos"]["frame_align_mode"] = args.frame_align_mode
    if args.shift_and_stitch is not None:
        cfg["vos"]["shift_and_stitch_enabled"] = bool(args.shift_and_stitch)
    if args.shift_and_stitch_factor is not None:
        cfg["vos"]["shift_and_stitch_factor"] = args.shift_and_stitch_factor
    if args.visualize_outputs is not None:
        cfg["vos"]["visualize_outputs"] = bool(args.visualize_outputs)
    if args.align_corners is not None:
        cfg["vos"]["align_corners"] = bool(args.align_corners)
    if cfg["model"].get("model_type") == "vjepa2_slot" and cfg["vos"].get("use_video_clip_features"):
        logger.info("Forcing vos.use_video_clip_features=False for vjepa2_slot; slot mixing now runs on full-video raw features")
        cfg["vos"]["use_video_clip_features"] = False
    cfg = validate_vos_config(cfg)

    seed = cfg.get("experiment", {}).get("seed")
    if seed is not None:
        set_seed(int(seed))

    requested_num_gpus = cfg["distributed"].get("num_gpus")
    available_cuda_gpus = torch.cuda.device_count()
    if requested_num_gpus is None:
        num_gpus = available_cuda_gpus
    else:
        num_gpus = int(requested_num_gpus)
    if num_gpus > 0 and available_cuda_gpus <= 0:
        logger.warning("distributed.num_gpus=%s requested, but CUDA is unavailable; falling back to CPU", num_gpus)
        num_gpus = 0
    if num_gpus > available_cuda_gpus:
        logger.warning(
            "Requested num_gpus=%s but only %s CUDA devices are visible; clamping",
            num_gpus,
            available_cuda_gpus,
        )
        num_gpus = available_cuda_gpus
    device = "cuda:0" if num_gpus > 0 else "cpu"

    logger.info(f"Loaded config: {args.config}")
    logger.info("Experiment seed: %s", cfg["experiment"].get("seed"))
    logger.info(f"Dataset: {cfg['data']['dataset']} ({cfg['data']['subset']})")
    if cfg["data"].get("video_list_file"):
        logger.info(f"Video list file: {cfg['data']['video_list_file']}")
    logger.info(f"Model: {cfg['model']['model_type']} ({cfg['model']['model_scale']})")
    logger.info(f"Checkpoint: {cfg['model']['ckpt_path']}")
    logger.info(
        "Model patch config: patch_size=%s patch_stride=%s patch_padding=%s video_infer_frames=%s "
        "slot_alpha=%s slot_normalize_before_mix=%s",
        cfg["model"].get("patch_size"),
        cfg["model"].get("patch_stride"),
        cfg["model"].get("patch_padding"),
        cfg["model"].get("video_infer_frames"),
        cfg["model"].get("slot_alpha"),
        cfg["model"].get("slot_normalize_before_mix"),
    )
    logger.info(f"Output: {cfg['experiment']['output_dir']}")
    logger.info("Distributed: num_gpus=%s", cfg.get("distributed", {}).get("num_gpus"))
    logger.info(
        (
            "VOS params: n_last_frames=%s size_mask_neighborhood=%s mask_shape=%s "
            "topk=%s temperature=%s affinity_dtype=%s feature_dtype=%s label_agg_device=%s "
            "matmul_chunk_size=%s cpu_fallback_videos=%s skip_completed_videos=%s "
            "frame_short_side=%s frame_align_mode=%s shift_and_stitch_enabled=%s "
            "shift_and_stitch_factor=%s visualize_outputs=%s "
            "align_corners=%s feature_chunk_size=%s use_video_clip_features=%s "
            "slot_reference_num_frames=%s slot_fps_init_mode=%s num_workers=%s"
        ),
        cfg["vos"]["n_last_frames"],
        cfg["vos"]["size_mask_neighborhood"],
        cfg["vos"].get("mask_shape", "square"),
        cfg["vos"]["topk"],
        cfg["vos"]["temperature"],
        cfg["vos"].get("affinity_dtype"),
        cfg["vos"].get("feature_dtype"),
        cfg["vos"].get("label_agg_device"),
        cfg["vos"].get("matmul_chunk_size"),
        cfg["vos"].get("cpu_fallback_videos"),
        cfg["vos"].get("skip_completed_videos"),
        cfg["vos"].get("frame_short_side"),
        cfg["vos"].get("frame_align_mode"),
        cfg["vos"].get("shift_and_stitch_enabled"),
        cfg["vos"].get("shift_and_stitch_factor"),
        cfg["vos"].get("visualize_outputs"),
        cfg["vos"].get("align_corners"),
        cfg["vos"].get("feature_chunk_size"),
        cfg["vos"].get("use_video_clip_features"),
        cfg["vos"].get("slot_reference_num_frames"),
        cfg["vos"].get("slot_fps_init_mode"),
        cfg["vos"].get("num_workers"),
    )
    logger.info("Device: %s (num_gpus=%s)", device, num_gpus)
    if num_gpus > 1:
        logger.info("VOS evaluation currently uses single-process shared-GPU parallelism; ignoring extra GPUs")

    write_effective_config(cfg)
    model = load_model_from_config(cfg["model"], device=device)
    cpu_fallback_model = None
    if cfg["vos"].get("cpu_fallback_videos") and str(device).startswith("cuda"):
        logger.info("Loading CPU fallback model for selected videos")
        cpu_fallback_model = load_model_from_config(cfg["model"], device="cpu")
    evaluate_vos(
        model,
        cfg,
        device,
        cpu_fallback_model=cpu_fallback_model,
        run_official_eval=True,
    )
    write_summary_if_available(cfg)

    logger.info("Done.")


if __name__ == "__main__":
    main()
