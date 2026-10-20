import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _ensure_writable_torch_home():
    if os.environ.get("TORCH_HOME"):
        return
    repo_root = Path(__file__).resolve().parents[4]
    torch_home = repo_root / ".cache" / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)


class DinoV2PatchEncoder(nn.Module):
    def __init__(
        self,
        arch="dinov2_vitb14",
        img_size=336,
        patch_size=14,
        num_register_tokens=0,
        checkpoint=None,
        checkpoint_key=None,
    ):
        super().__init__()
        if arch.startswith("dinov2_"):
            from slot_jepa.encoders.dinov2.hub import backbones as dinov2_backbones

            if not hasattr(dinov2_backbones, arch):
                raise ValueError(f"Unsupported DINOv2 hub model: {arch}")
            _ensure_writable_torch_home()
            # DINOv2 checkpoints store positional embeddings at the pretraining
            # grid size (518/14=37 for ViT-B/14). Keep that parameter shape and
            # let DINOv2 interpolate it at forward time for smaller crops.
            checkpoint_img_size = 518
            self.model = getattr(dinov2_backbones, arch)(
                img_size=checkpoint_img_size,
                pretrained=checkpoint is None,
            )
        else:
            from slot_jepa.encoders.dinov2.models import vision_transformer as dinov2_vit

            if arch not in dinov2_vit.__dict__:
                raise ValueError(f"Unsupported DINOv2 arch: {arch}")
            self.model = dinov2_vit.__dict__[arch](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
            )
        self.embed_dim = self.model.embed_dim
        self.patch_size = getattr(self.model, "patch_size", patch_size)
        if checkpoint:
            self._load_checkpoint(checkpoint, checkpoint_key=checkpoint_key)

    def _select_state_dict(self, checkpoint, checkpoint_key=None):
        if checkpoint_key:
            if checkpoint_key not in checkpoint:
                raise KeyError(f"Checkpoint key `{checkpoint_key}` not found")
            return checkpoint[checkpoint_key]
        if isinstance(checkpoint, dict):
            for key in ("teacher", "student", "model", "state_dict", "backbone", "encoder"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
        return checkpoint

    def _load_checkpoint(self, checkpoint_path, checkpoint_key=None):
        logger.info("Loading DINOv2 checkpoint from %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = self._select_state_dict(checkpoint, checkpoint_key=checkpoint_key)
        cleaned = {}
        prefixes = (
            "module.",
            "backbone.",
            "teacher.",
            "student.",
            "model.",
            "encoder.",
        )
        for key, value in state_dict.items():
            clean_key = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix) :]
                        changed = True
            if clean_key in self.model.state_dict() and self.model.state_dict()[clean_key].shape != value.shape:
                logger.info(
                    "Skipping DINOv2 checkpoint key %s due to shape mismatch: %s vs %s",
                    clean_key,
                    tuple(value.shape),
                    tuple(self.model.state_dict()[clean_key].shape),
                )
                continue
            cleaned[clean_key] = value
        msg = self.model.load_state_dict(cleaned, strict=False)
        logger.info("Loaded DINOv2 checkpoint with msg: %s", msg)

    def forward(self, x):
        return self.model.forward_features(x)["x_norm_patchtokens"]


def init_encoder(
    encoder_impl,
    resolution,
    checkpoint,
    model_name="vit_base",
    checkpoint_key="ema_encoder",
    patch_size=16,
    tubelet_size=2,
    uniform_power=True,
    use_rope=True,
    img_as_video_nframes=2,
    num_register_tokens=0,
):
    if encoder_impl in ("slot_jepa", "vjepa", "vjepa2"):
        from .vit_encoder import init_module as init_vjepa_module

        return init_vjepa_module(
            resolution=resolution,
            checkpoint=checkpoint,
            model_kwargs={
                "encoder": {
                    "checkpoint_key": checkpoint_key,
                    "model_name": model_name,
                    "patch_size": patch_size,
                    "tubelet_size": tubelet_size,
                    "uniform_power": uniform_power,
                    "use_rope": use_rope,
                }
            },
            wrapper_kwargs={"img_as_video_nframes": img_as_video_nframes},
        )
    if encoder_impl in ("dinov2", "dinov2_vitb14", "dinov2_vitb14_reg"):
        if checkpoint is None and model_name == "vit_base" and patch_size == 14:
            model_name = "dinov2_vitb14"
        return DinoV2PatchEncoder(
            arch=model_name,
            img_size=resolution,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            checkpoint=checkpoint,
            checkpoint_key=checkpoint_key,
        )
    raise ValueError(f"Unsupported encoder_impl: {encoder_impl}")
