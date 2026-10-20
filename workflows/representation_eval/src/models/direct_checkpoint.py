"""Helpers for inferring and loading direct checkpoints used by evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


def _normalized_checkpoint_name(ckpt_path: str) -> str:
    name = Path(ckpt_path).name.lower()
    for suffix in (".pth.tar", ".pth", ".ckpt", ".tar"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _try_import_timm_backbone(model_type: str, model_scale: str, patch_size: int):
    if model_type in {"mae", "mocov3", "mugs"}:
        from .dino.vision_transformer import vit_base, vit_large, vit_small

        if model_scale == "vit_small":
            return vit_small(patch_size=patch_size)
        if model_scale == "vit_base":
            return vit_base(patch_size=patch_size)
        if model_scale == "vit_large":
            return vit_large(patch_size=patch_size)
        raise NotImplementedError(f"{model_type} direct checkpoint does not support model_scale={model_scale}")

    if model_type == "cae":
        from .ibot.vision_transformer import vit_base, vit_large

        if model_scale == "vit_base":
            return vit_base(patch_size=patch_size, init_values=float(1.0))
        if model_scale == "vit_large":
            return vit_large(patch_size=patch_size, init_values=float(1.0))
        raise NotImplementedError(f"cae direct checkpoint does not support model_scale={model_scale}")

    raise ValueError(f"Unexpected timm-backed model_type={model_type}")


def infer_peisong_model_cfg(ckpt_path: str) -> Dict[str, Any]:
    """Infer model-loading config for checkpoints stored under checkpoints/peisong/*."""
    folder_name = Path(ckpt_path).resolve().parent.name.lower()

    if "ibot" in folder_name:
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "ibot",
            "model_scale": "vit_small",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "student",
            "prefix_filter": "backbone.",
            "strip_prefix": "backbone.",
            "init_values": 1.0,
            "masked_im_modeling": True,
        }
    if "ijpea" in folder_name or "ijepa" in folder_name:
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "ijepa",
            "model_scale": "vit_base",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "student",
            "prefix_filter": "module.",
            "strip_prefix": "module.",
        }
    if "mae" in folder_name:
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "mae",
            "model_scale": "vit_small",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "model",
            "prefix_filter": None,
            "strip_prefix": "",
        }
    if "moco-v3" in folder_name or "mocov3" in folder_name:
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "mocov3",
            "model_scale": "vit_small",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "student",
            "prefix_filter": None,
            "strip_prefix": "",
            "num_classes": 0,
        }

    raise ValueError(f"Unable to infer peisong checkpoint type from folder name: {folder_name}")


def infer_baseline_model_cfg(ckpt_path: str) -> Dict[str, Any]:
    """Infer model-loading config for checkpoints stored under checkpoints/baselines/*."""
    checkpoint_name = _normalized_checkpoint_name(ckpt_path)

    if checkpoint_name in {"cae_base_800ep", "cae_base_1600ep"}:
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "cae",
            "model_scale": "vit_base",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "model",
            "prefix_filter": "encoder.",
            "strip_prefix": "encoder.",
        }
    if checkpoint_name == "cae_large_1600ep":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "cae",
            "model_scale": "vit_large",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "model",
            "prefix_filter": "encoder.",
            "strip_prefix": "encoder.",
        }
    if checkpoint_name == "dino_resnet50_pretrain":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "dino",
            "model_scale": "resnet50",
            "ckpt_path": ckpt_path,
            "patch_size": 32,
            "state_dict_key": "state_dict",
            "prefix_filter": "model.",
            "strip_prefix": "model.",
        }
    if checkpoint_name == "dino_vits16":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "dino",
            "model_scale": "vit_small",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": None,
            "prefix_filter": None,
            "strip_prefix": "",
        }
    if checkpoint_name == "dino_vitb16":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "dino",
            "model_scale": "vit_base",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": None,
            "prefix_filter": None,
            "strip_prefix": "",
        }
    if checkpoint_name == "dino_vitb8":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "dino",
            "model_scale": "vit_base",
            "ckpt_path": ckpt_path,
            "patch_size": 8,
            "state_dict_key": None,
            "prefix_filter": None,
            "strip_prefix": "",
        }
    if checkpoint_name == "leopart_vits16":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "dino",
            "model_scale": "vit_small",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": None,
            "prefix_filter": "model.",
            "strip_prefix": "model.",
        }
    if checkpoint_name == "dinov2_vits14_reg4":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "dinov2",
            "model_scale": "vit_small",
            "ckpt_path": ckpt_path,
            "patch_size": 14,
            "num_register_tokens": 4,
            "state_dict_key": None,
            "prefix_filter": None,
            "strip_prefix": "",
        }
    if checkpoint_name == "dinov2_vitb14_reg4":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "dinov2",
            "model_scale": "vit_base",
            "ckpt_path": ckpt_path,
            "patch_size": 14,
            "num_register_tokens": 4,
            "state_dict_key": None,
            "prefix_filter": None,
            "strip_prefix": "",
        }
    if checkpoint_name == "dinov2_vitl14_reg4":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "dinov2",
            "model_scale": "vit_large",
            "ckpt_path": ckpt_path,
            "patch_size": 14,
            "num_register_tokens": 4,
            "state_dict_key": None,
            "prefix_filter": None,
            "strip_prefix": "",
        }
    if checkpoint_name == "dinov2_vitg14_reg4_pretrain":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "dinov2",
            "model_scale": "vit_giant",
            "ckpt_path": ckpt_path,
            "patch_size": 14,
            "num_register_tokens": 4,
            "state_dict_key": None,
            "prefix_filter": None,
            "strip_prefix": "",
            "embed_dim": 1536,
        }
    if checkpoint_name == "dinov2_reproduced":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "dinov2",
            "model_scale": "vit_small",
            "ckpt_path": ckpt_path,
            "patch_size": 14,
            "num_register_tokens": 4,
            "state_dict_key": "student",
            "prefix_filter": "backbone.",
            "strip_prefix": "backbone.",
        }
    if checkpoint_name == "dinov2_reproduced_vitb14":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "dinov2",
            "model_scale": "vit_base",
            "ckpt_path": ckpt_path,
            "patch_size": 14,
            "num_register_tokens": 4,
            "state_dict_key": "teacher",
            "prefix_filter": "backbone.",
            "strip_prefix": "backbone.",
            "embed_dim": 768,
        }
    if checkpoint_name == "dinov2_reproduced_vitl14":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "dinov2",
            "model_scale": "vit_large",
            "ckpt_path": ckpt_path,
            "patch_size": 14,
            "num_register_tokens": 4,
            "state_dict_key": "teacher",
            "prefix_filter": "backbone.",
            "strip_prefix": "backbone.",
            "embed_dim": 1024,
            "block_chunks": 4,
        }
    if checkpoint_name == "ibot_vits16":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "ibot",
            "model_scale": "vit_small",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "state_dict",
            "prefix_filter": None,
            "strip_prefix": "",
            "init_values": 0.0,
            "masked_im_modeling": False,
        }
    if checkpoint_name == "ibot_vitb16":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "ibot",
            "model_scale": "vit_base",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "state_dict",
            "prefix_filter": None,
            "strip_prefix": "",
            "init_values": 0.0,
            "masked_im_modeling": False,
        }
    if checkpoint_name == "ibot_vitl16":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "ibot",
            "model_scale": "vit_large",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "state_dict",
            "prefix_filter": None,
            "strip_prefix": "",
            "init_values": 0.0,
            "masked_im_modeling": False,
        }
    if checkpoint_name == "mae_pretrain_vit_base":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "mae",
            "model_scale": "vit_base",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "model",
            "prefix_filter": None,
            "strip_prefix": "",
        }
    if checkpoint_name == "moco_v2_800ep_pretrain":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "mocov2",
            "model_scale": "resnet50",
            "ckpt_path": ckpt_path,
            "patch_size": 32,
            "state_dict_key": "state_dict",
            "prefix_filter": "model.",
            "strip_prefix": "model.",
        }
    if checkpoint_name == "mocov3-vits16":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "mocov3",
            "model_scale": "vit_small",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "state_dict",
            "prefix_filter": "module.base_encoder.",
            "strip_prefix": "module.base_encoder.",
            "num_classes": 0,
        }
    if checkpoint_name == "mocov3-vitb16":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "mocov3",
            "model_scale": "vit_base",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "state_dict",
            "prefix_filter": "module.base_encoder.",
            "strip_prefix": "module.base_encoder.",
            "num_classes": 0,
        }
    if checkpoint_name == "mugs_vit_small_backbone_800ep":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "mugs",
            "model_scale": "vit_small",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "state_dict",
            "prefix_filter": None,
            "strip_prefix": "",
        }
    if checkpoint_name == "mugs_vit_base_backbone_400ep":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "mugs",
            "model_scale": "vit_base",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "state_dict",
            "prefix_filter": None,
            "strip_prefix": "",
        }
    if checkpoint_name == "mugs_vit_large_backbone_250ep":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "mugs",
            "model_scale": "vit_large",
            "ckpt_path": ckpt_path,
            "patch_size": 16,
            "state_dict_key": "state_dict",
            "prefix_filter": None,
            "strip_prefix": "",
        }
    if checkpoint_name == "ijepa-in1k-vit.h.14-300e":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "ijepa",
            "model_scale": "vit_huge",
            "ckpt_path": ckpt_path,
            "patch_size": 14,
            "state_dict_key": "encoder",
            "prefix_filter": "module.",
            "strip_prefix": "module.",
            "embed_dim": 1280,
        }
    if checkpoint_name == "swav_800ep_pretrain":
        return {
            "load_mode": "direct_vit_checkpoint",
            "model_type": "swav",
            "model_scale": "resnet50",
            "ckpt_path": ckpt_path,
            "patch_size": 32,
            "state_dict_key": "state_dict",
            "prefix_filter": "model.",
            "strip_prefix": "model.",
        }

    raise ValueError(f"Unable to infer baseline checkpoint type from filename: {Path(ckpt_path).name}")


def extract_state_dict_from_checkpoint(checkpoint: Dict[str, Any], model_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Resolve a state_dict from a checkpoint dict according to config overrides."""
    state_dict_key = model_cfg.get("state_dict_key")
    if state_dict_key is not None:
        if state_dict_key not in checkpoint:
            raise KeyError(f"state_dict_key='{state_dict_key}' not found in checkpoint keys={list(checkpoint.keys())}")
        state_dict = checkpoint[state_dict_key]
    else:
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Resolved state_dict must be a dictionary")
    return state_dict


def apply_prefix_rules(state_dict: Dict[str, torch.Tensor], model_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Apply prefix filtering / stripping configured for a direct checkpoint."""
    prefix_filter = model_cfg.get("prefix_filter")
    strip_prefix = model_cfg.get("strip_prefix", "")

    filtered_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if prefix_filter is not None and not key.startswith(prefix_filter):
            continue
        if strip_prefix and key.startswith(strip_prefix):
            key = key[len(strip_prefix) :]
        filtered_state_dict[key] = value

    return filtered_state_dict


def _adapt_state_dict_for_model(state_dict: Dict[str, torch.Tensor], model_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    model_type = model_cfg["model_type"]
    adapted: Dict[str, torch.Tensor] = {}
    qkv_biases: Dict[str, Dict[str, torch.Tensor]] = {}

    for key, value in state_dict.items():
        if model_type == "ibot":
            key = key.replace(".ls1.gamma", ".gamma_1").replace(".ls2.gamma", ".gamma_2")
            if key == "mask_token":
                if model_cfg.get("masked_im_modeling", False):
                    key = "masked_embed"
                    if value.ndim == 1:
                        value = value.unsqueeze(0)
                else:
                    continue
            if key.startswith("head."):
                continue
        elif model_type == "mocov3":
            if key.startswith("head."):
                continue
        elif model_type == "mae":
            if key == "mask_token":
                continue
            if key.startswith("decoder_") or key.startswith("decoder_blocks."):
                continue
        elif model_type == "cae":
            if key == "mask_token":
                continue
            if key.endswith(".attn.q_bias") or key.endswith(".attn.v_bias"):
                prefix, bias_name = key.rsplit(".", 1)
                qkv_biases.setdefault(prefix, {})[bias_name] = value
                continue
        elif model_type in {"dino", "mocov2", "swav", "mugs"}:
            if model_type == "mugs" and key.startswith("relation_blocks."):
                continue
            if key.startswith("fc."):
                continue
            if key.startswith("projection_head."):
                continue
            if key.startswith("prototypes."):
                continue

        adapted[key] = value

    if model_type == "cae":
        for prefix, biases in qkv_biases.items():
            q_bias = biases.get("q_bias")
            v_bias = biases.get("v_bias")
            if q_bias is None or v_bias is None:
                continue
            k_bias = torch.zeros_like(q_bias)
            adapted[f"{prefix}.qkv.bias"] = torch.cat((q_bias, k_bias, v_bias), dim=0)

    return adapted


def build_direct_vit_model(model_cfg: Dict[str, Any], device: torch.device | str):
    """Instantiate a model for direct checkpoint loading."""
    model_type = model_cfg["model_type"]
    model_scale = model_cfg["model_scale"]
    patch_size = int(model_cfg.get("patch_size", 16))

    if model_type == "ibot":
        from .ibot.vision_transformer import vit_base as ibot_vit_base
        from .ibot.vision_transformer import vit_large as ibot_vit_large
        from .ibot.vision_transformer import vit_small as ibot_vit_small

        if model_scale == "vit_small":
            model_fn = ibot_vit_small
        elif model_scale == "vit_base":
            model_fn = ibot_vit_base
        elif model_scale == "vit_large":
            model_fn = ibot_vit_large
        else:
            raise NotImplementedError(f"ibot direct checkpoint does not support model_scale={model_scale}")
        model = model_fn(
            patch_size=int(model_cfg.get("patch_size", 16)),
            init_values=float(model_cfg.get("init_values", 0.0)),
            masked_im_modeling=bool(model_cfg.get("masked_im_modeling", False)),
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
            inference_block_sequence=model_cfg.get("inference_block_sequence"),
        )
    elif model_type == "dino":
        from .dino.vision_transformer import vit_base as dino_vit_base
        from .dino.vision_transformer import vit_large as dino_vit_large
        from .dino.vision_transformer import vit_small as dino_vit_small

        if model_scale == "vit_small":
            model = dino_vit_small(patch_size=patch_size)
        elif model_scale == "vit_base":
            model = dino_vit_base(patch_size=patch_size)
        elif model_scale == "vit_large":
            model = dino_vit_large(patch_size=patch_size)
        elif model_scale == "resnet50":
            from .resnet import resnet50

            model = resnet50()
        else:
            raise NotImplementedError(f"dino direct checkpoint does not support model_scale={model_scale}")
    elif model_type == "dinov2":
        from .dinov2.vision_transformer import vit_base as dinov2_vit_base
        from .dinov2.vision_transformer import vit_giant as dinov2_vit_giant
        from .dinov2.vision_transformer import vit_large as dinov2_vit_large
        from .dinov2.vision_transformer import vit_small as dinov2_vit_small

        if model_scale == "vit_small":
            model_fn = dinov2_vit_small
        elif model_scale == "vit_base":
            model_fn = dinov2_vit_base
        elif model_scale == "vit_large":
            model_fn = dinov2_vit_large
        elif model_scale in {"vit_giant", "vit_giant2"}:
            model_fn = dinov2_vit_giant
        else:
            raise NotImplementedError(f"dinov2 direct checkpoint does not support model_scale={model_scale}")
        model = model_fn(
            patch_size=int(model_cfg.get("patch_size", 14)),
            num_register_tokens=int(model_cfg.get("num_register_tokens", 4)),
            block_chunks=int(model_cfg.get("block_chunks", 0)),
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
            inference_block_sequence=model_cfg.get("inference_block_sequence"),
        )
    elif model_type == "mae":
        model = _try_import_timm_backbone(model_type, model_scale, patch_size)
    elif model_type == "mocov3":
        model = _try_import_timm_backbone(model_type, model_scale, patch_size)
    elif model_type in {"cae", "mugs"}:
        model = _try_import_timm_backbone(model_type, model_scale, patch_size)
    elif model_type == "ijepa":
        from .ijepa.deit import deit_base, deit_giant, deit_huge, deit_large

        if model_scale == "vit_base":
            model = deit_base(patch_size=patch_size)
        elif model_scale == "vit_large":
            model = deit_large(patch_size=patch_size)
        elif model_scale == "vit_huge":
            model = deit_huge(patch_size=patch_size)
        elif model_scale == "vit_giant":
            model = deit_giant(patch_size=patch_size)
        else:
            raise NotImplementedError(f"ijepa direct checkpoint does not support model_scale={model_scale}")
    elif model_type in {"mocov2", "swav"}:
        if model_scale != "resnet50":
            raise NotImplementedError(f"{model_type} direct checkpoint does not support model_scale={model_scale}")
        from .resnet import resnet50

        model = resnet50()
    else:
        raise ValueError(f"Unsupported model_type for direct_vit_checkpoint: {model_type}")

    return model.to(device)


def load_direct_checkpoint_model(
    model_cfg: Dict[str, Any],
    device: torch.device | str,
) -> Tuple[nn.Module, Any]:
    """Create a model, load a direct checkpoint, and return both model and load message."""
    model = build_direct_vit_model(model_cfg, device=device)

    checkpoint = torch.load(model_cfg["ckpt_path"], map_location="cpu", weights_only=False)
    state_dict = extract_state_dict_from_checkpoint(checkpoint, model_cfg)
    state_dict = apply_prefix_rules(state_dict, model_cfg)
    state_dict = _adapt_state_dict_for_model(state_dict, model_cfg)

    if model_cfg["model_type"] == "dinov2" and "pos_embed" in state_dict:
        model.pos_embed = nn.Parameter(state_dict["pos_embed"])

    msg = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, msg


def _ignored_missing_key_prefixes(model_cfg: Dict[str, Any]) -> Tuple[str, ...]:
    prefixes: List[str] = list(model_cfg.get("ignore_missing_key_prefixes", []))
    if model_cfg["model_type"] == "mae":
        prefixes.extend(
            [
                "mask_token",
                "decoder_",
                "decoder_blocks.",
                "decoder_norm.",
                "decoder_pred.",
            ]
        )
    return tuple(prefixes)


def _ignored_unexpected_key_prefixes(model_cfg: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(model_cfg.get("ignore_unexpected_key_prefixes", []))


def _filter_keys(keys: Iterable[str], ignored_prefixes: Tuple[str, ...]) -> List[str]:
    if not ignored_prefixes:
        return list(keys)
    return [key for key in keys if not any(key.startswith(prefix) for prefix in ignored_prefixes)]


def normalize_load_result(model_cfg: Dict[str, Any], load_msg: Any) -> Tuple[List[str], List[str]]:
    """Filter expected missing/unexpected keys for checkpoint families with encoder-only weights."""
    missing_keys = _filter_keys(list(load_msg.missing_keys), _ignored_missing_key_prefixes(model_cfg))
    unexpected_keys = _filter_keys(list(load_msg.unexpected_keys), _ignored_unexpected_key_prefixes(model_cfg))
    return missing_keys, unexpected_keys
