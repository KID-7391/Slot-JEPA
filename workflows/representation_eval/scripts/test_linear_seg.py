"""YAML-driven linear segmentation evaluation entrypoint."""

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
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.linear_seg import evaluate_linear_seg
from src.evaluation.linear_seg.config import load_linear_seg_config
from src.models.direct_checkpoint import (
    apply_prefix_rules,
    extract_state_dict_from_checkpoint,
    load_direct_checkpoint_model,
)
from src.models import get_models, get_models_slim


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ssl_eval")


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Distributed environment not found. Please launch with torchrun.")
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    dist.barrier()
    return True, rank, world_size, local_rank


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
                img_size=[224, 224],
                in_chans=3,
                num_classes=1000,
                patch_size=4,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0,
                attn_drop_rate=0,
                drop_path_rate=0.1,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                ape=False,
                patch_norm=True,
                use_dense_prediction=False,
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
        video_infer_frames = int(model_cfg.get("video_infer_frames", 2) or 2)
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
                    "use_rope": True,
                }
            },
            wrapper_kwargs={"img_as_video_nframes": video_infer_frames},
        ).to(device)

    elif model_type == "vjepa2_slot":
        from src.models.vjepa_2_1.slot_wrapper import VJEPA2SlotWrapper

        patch_size = int(model_cfg.get("patch_size", 16) or 16)
        patch_stride = int(model_cfg.get("patch_stride", patch_size) or patch_size)
        patch_padding = int(model_cfg.get("patch_padding", patch_stride // 2) or 0)
        video_infer_frames = int(model_cfg.get("video_infer_frames", 2) or 2)
        return VJEPA2SlotWrapper(
            resolution=384,
            model_scale=model_scale,
            patch_size=patch_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            video_infer_frames=video_infer_frames,
            alpha=float(model_cfg.get("slot_alpha", 0.0)),
            window_size=tuple(model_cfg.get("slot_window_size", [2, 4])),
            slots_per_window=int(model_cfg.get("slot_slots_per_window", 4)),
            global_k=int(model_cfg.get("slot_global_k", 16)),
            mask_temp=float(model_cfg.get("slot_mask_temp", 1.0)),
            l_max=int(model_cfg.get("slot_l_max", 16)),
            use_decoder_checkpoint=bool(model_cfg.get("slot_use_decoder_checkpoint", False)),
        ).to(device)

    raise NotImplementedError(f"Model type '{model_type}' not supported")


def _is_slim_checkpoint(checkpoint):
    """Check if checkpoint is a slim (pre-extracted) checkpoint."""
    return isinstance(checkpoint, dict) and "state_dict" in checkpoint and "model_type" in checkpoint


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
        pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        return pretrained_dict
    elif model_type == "vjepa2_slot":
        return checkpoint

    raise NotImplementedError(f"Full checkpoint extraction not supported for '{model_type}'")


def load_model_from_config(model_cfg: Dict[str, Any], device: torch.device, rank: int):
    model_type = model_cfg["model_type"]
    model_scale = model_cfg["model_scale"]
    load_mode = model_cfg["load_mode"]

    if model_type == "vjepa2":
        model = _create_model(model_type, model_scale, model_cfg, device)

        checkpoint = torch.load(model_cfg["ckpt_path"], map_location="cpu")
        if _is_slim_checkpoint(checkpoint):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = _extract_state_dict_full(
                checkpoint,
                model_type,
                state_dict_key=model_cfg.get("state_dict_key"),
            )

        for key, value in model.state_dict().items():
            if key not in state_dict:
                logger.info(f'key "{key}" could not be found in loaded state dict')
            elif state_dict[key].shape != value.shape:
                logger.info(f'key "{key}" is of different shape in model and loaded state dict')
                state_dict[key] = value

        msg = model.load_state_dict(state_dict, strict=False)
        if rank == 0:
            logger.info(f"Model loaded: {msg}")
    elif model_type == "vjepa2_slot":
        model = _create_model(model_type, model_scale, model_cfg, device)
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
            inference_block_sequence=model_cfg.get("inference_block_sequence"),
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
    parser = argparse.ArgumentParser(description="YAML-driven linear segmentation evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--paths_config", type=str, default=None, help="Optional bundle path-map YAML")
    args = parser.parse_args()

    distributed, rank, world_size, local_rank = init_distributed()
    if not distributed:
        return

    cfg = load_linear_seg_config(args.config, paths_config=args.paths_config)
    seed = cfg.get("experiment", {}).get("seed")
    if seed is not None:
        set_seed(int(seed))

    device = torch.device(f"cuda:{local_rank}")
    if rank == 0:
        logger.info(f"Loaded config: {args.config}")
        logger.info(f"Running on {world_size} GPUs")
        logger.info(f"Experiment name: {cfg.get('experiment', {}).get('name', 'linear_seg')}")

    model = load_model_from_config(cfg["model"], device=device, rank=rank)
    results = evaluate_linear_seg(model=model, cfg=cfg)

    if rank == 0:
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        for dataset_name, head_results in results.items():
            print(f"\nDataset: {dataset_name}")
            for head_type, values in head_results.items():
                print(f"  Head: {head_type}")
                print(f"    Best LR:   {values['best_lr']}")
                print(f"    Best mIoU: {values['best_miou']:.2f}%")
                print(f"    Best Acc:  {values['best_acc']:.2f}%")
                if values.get("per_lr_results"):
                    print("    Per-LR breakdown:")
                    for lr, lr_values in sorted(values["per_lr_results"].items()):
                        print(f"      LR={lr}: mIoU={lr_values['mIoU']:.2f}, Acc={lr_values['Accuracy']:.2f}")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
