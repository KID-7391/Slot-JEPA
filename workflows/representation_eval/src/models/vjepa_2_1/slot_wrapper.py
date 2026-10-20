"""VJEPA2 encoder + slot-based latent reconstruction wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_RELEASE_SRC = str(Path(__file__).resolve().parents[5] / "src")
if _RELEASE_SRC not in sys.path:
    sys.path.insert(0, _RELEASE_SRC)

from .vit_encoder import init_module as init_vjepa2_module
from slot_jepa.slots import TwoStageExtractor


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    if not any(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


class VJEPA2SlotWrapper(nn.Module):
    """Blend raw VJEPA tokens with slot-decoder reconstruction."""

    def __init__(
        self,
        resolution: int,
        model_scale: str,
        patch_size: int,
        patch_stride: int,
        patch_padding: int,
        video_infer_frames: int,
        alpha: float = 0.0,
        normalize_before_mix: bool = False,
        window_size: Tuple[int, int] = (2, 4),
        slots_per_window: int = 4,
        global_k: int = 16,
        mask_temp: float = 1.0,
        l_max: int = 16,
        use_decoder_checkpoint: bool = False,
    ):
        super().__init__()
        self.encoder = init_vjepa2_module(
            resolution=resolution,
            model_kwargs={
                "encoder": {
                    "model_name": model_scale,
                    "patch_size": int(patch_size),
                    "patch_stride": int(patch_stride),
                    "patch_padding": int(patch_padding),
                    "tubelet_size": 2,
                    "uniform_power": True,
                    "use_rope": True,
                }
            },
            wrapper_kwargs={"img_as_video_nframes": int(video_infer_frames)},
        )
        self.extractor = TwoStageExtractor(
            window_size=list(window_size),
            slots_per_window=int(slots_per_window),
            global_k=int(global_k),
            D_in=int(getattr(self.encoder, "embed_dim", 768)),
            mask_temp=float(mask_temp),
            L_max=int(l_max),
            use_decoder_checkpoint=bool(use_decoder_checkpoint),
        )
        self.alpha = float(alpha)
        self.normalize_before_mix = bool(normalize_before_mix)
        self.embed_dim = int(getattr(self.encoder, "embed_dim"))
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.patch_padding = int(patch_padding)
        self.tubelet_size = int(getattr(self.encoder, "tubelet_size", 1) or 1)
        self.img_temporal_dim_size = int(video_infer_frames)

    def _infer_temporal_spatial_shape(self, x: torch.Tensor, tokens: torch.Tensor) -> Tuple[int, int, int]:
        if x.ndim == 4:
            _, _, frame_h, frame_w = x.shape
            temporal_steps = 1
        elif x.ndim == 5:
            _, _, clip_frames, frame_h, frame_w = x.shape
            temporal_steps = max(int(clip_frames) // self.tubelet_size, 1)
        else:
            raise ValueError(f"Expected 4D or 5D input, got shape={tuple(x.shape)}")

        feat_h = int(self.encoder._compute_spatial_tokens(frame_h))
        feat_w = int(self.encoder._compute_spatial_tokens(frame_w))
        expected_tokens = temporal_steps * feat_h * feat_w
        if expected_tokens != int(tokens.shape[1]):
            raise ValueError(
                "Token shape mismatch for VJEPA2 slot wrapper: "
                f"expected_tokens={expected_tokens} actual_tokens={tokens.shape[1]} "
                f"temporal_steps={temporal_steps} feat_hw=({feat_h},{feat_w})"
            )
        return temporal_steps, feat_h, feat_w

    def get_embeddings(self, x: torch.Tensor):
        return self.encoder.get_embeddings(x)

    def forward(self, x: torch.Tensor):
        _, patch_embeddings = self.get_embeddings(x)
        return patch_embeddings[-1]

    def reshape_patch_embeddings(self, x: torch.Tensor, tokens: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        temporal_steps, feat_h, feat_w = self._infer_temporal_spatial_shape(x, tokens)
        batch_size, _, feat_dim = tokens.shape
        return tokens.reshape(batch_size, temporal_steps, feat_h, feat_w, feat_dim), feat_h, feat_w

    def extract_video_patch_embeddings(self, video: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Extract raw patch tokens for a video or image sequence.

        Args:
            video: [T, C, H, W] or [B, T, C, H, W]

        Returns:
            patch_tokens: [B, T_eff, H_feat * W_feat, D]
        """
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video with shape [T,C,H,W] or [B,T,C,H,W], got {tuple(video.shape)}")
        video_input = video.permute(0, 2, 1, 3, 4).contiguous()
        _, patch_embeddings = self.encoder.get_embeddings(video_input)
        token_maps, feat_h, feat_w = self.reshape_patch_embeddings(video_input, patch_embeddings[-1])
        batch_size, temporal_steps, _, _, feat_dim = token_maps.shape
        return token_maps.reshape(batch_size, temporal_steps, feat_h * feat_w, feat_dim), feat_h, feat_w

    def _sample_video_indices(
        self,
        num_frames: int,
        device: torch.device,
        max_reference_frames: int = 8,
    ) -> torch.Tensor:
        max_reference_frames = int(max_reference_frames)
        if max_reference_frames <= 0:
            raise ValueError("max_reference_frames must be positive")
        num_samples = min(int(num_frames), max_reference_frames)
        if num_samples <= 0:
            raise ValueError("num_frames must be positive")
        return torch.linspace(0, num_frames - 1, num_samples, device=device).long()

    def extract_obj_latents(
        self,
        token_maps: torch.Tensor,
        max_reference_frames: int = 8,
        fps_seed: int | None = None,
        fps_initial_token_indices: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract object latents from uniformly sampled video frames.

        Args:
            token_maps: [B, T, H, W, D]

        Notes:
            Frames are sampled uniformly from the provided sequence. By default,
            up to 8 sampled frames are used to build object latents.
        """
        if token_maps.ndim != 5:
            raise ValueError(f"Expected token maps with shape [B,T,H,W,D], got {tuple(token_maps.shape)}")
        sampled_indices = self._sample_video_indices(
            token_maps.shape[1],
            token_maps.device,
            max_reference_frames=max_reference_frames,
        )
        sampled_maps = token_maps[:, sampled_indices]
        fps_generator = None
        if fps_seed is not None:
            fps_generator = torch.Generator(device=token_maps.device)
            fps_generator.manual_seed(int(fps_seed))
        obj_latents, _ = self.extractor.merger(
            sampled_maps,
            fps_generator=fps_generator,
            fps_initial_token_indices=fps_initial_token_indices,
        )
        return obj_latents, sampled_indices

    def reconstruct_video_token_maps(
        self,
        token_maps: torch.Tensor,
        obj_latents: torch.Tensor,
        decode_chunk_size: int = 4,
    ) -> torch.Tensor:
        """Reconstruct per-frame token maps from object latents."""
        if token_maps.ndim != 5:
            raise ValueError(f"Expected token maps with shape [B,T,H,W,D], got {tuple(token_maps.shape)}")
        if obj_latents.ndim != 3:
            raise ValueError(f"Expected obj_latents with shape [B,K,D], got {tuple(obj_latents.shape)}")
        batch_size, temporal_steps, feat_h, feat_w, feat_dim = token_maps.shape
        flat_tokens = token_maps.reshape(batch_size, temporal_steps, feat_h * feat_w, feat_dim)
        recon_chunks = []
        for t_start in range(0, temporal_steps, int(decode_chunk_size)):
            t_end = min(t_start + int(decode_chunk_size), temporal_steps)
            decoded_state = self.extractor.decoder(obj_latents, flat_tokens[:, t_start:t_end], feat_h, feat_w)
            mask_chunk = decoded_state["mask"]
            recon_chunk = torch.einsum("btkn,bkd->btnd", mask_chunk.to(obj_latents.dtype), obj_latents)
            recon_chunks.append(recon_chunk)
        recon_tokens = torch.cat(recon_chunks, dim=1)
        return recon_tokens.reshape(batch_size, temporal_steps, feat_h, feat_w, feat_dim)

    def mix_with_obj_latents(
        self,
        target_token_maps: torch.Tensor,
        obj_latents: torch.Tensor,
        alpha: float | None = None,
        decode_chunk_size: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct and mix target token maps with externally provided object latents.

        Notes:
            The target token maps can cover all video frames, even if the object
            latents were extracted from only a sampled subset of frames.
        """
        if target_token_maps.ndim != 5:
            raise ValueError(
                f"Expected target_token_maps with shape [B,T,H,W,D], got {tuple(target_token_maps.shape)}"
            )
        mix_alpha = self.alpha if alpha is None else float(alpha)
        if mix_alpha <= 0.0:
            return target_token_maps, target_token_maps
        reconstructed = self.reconstruct_video_token_maps(
            target_token_maps,
            obj_latents,
            decode_chunk_size=decode_chunk_size,
        )
        if self.normalize_before_mix:
            target_token_maps = F.normalize(target_token_maps, p=2, dim=-1)
            reconstructed = F.normalize(reconstructed, p=2, dim=-1)
        mixed = ((1.0 - mix_alpha) * target_token_maps) + (mix_alpha * reconstructed.to(target_token_maps.dtype))
        return mixed, reconstructed

    def reshape_video_patch_embeddings(
        self,
        video_tokens: torch.Tensor,
        feat_h: int,
        feat_w: int,
    ) -> torch.Tensor:
        """Reshape flattened video patch embeddings to 5D token maps."""
        if video_tokens.ndim == 3:
            video_tokens = video_tokens.unsqueeze(0)
        if video_tokens.ndim != 4:
            raise ValueError(f"Expected video_tokens with shape [T,N,D] or [B,T,N,D], got {tuple(video_tokens.shape)}")
        batch_size, temporal_steps, num_tokens, feat_dim = video_tokens.shape
        expected_tokens = int(feat_h) * int(feat_w)
        if num_tokens != expected_tokens:
            raise ValueError(
                f"Token count mismatch: num_tokens={num_tokens} expected={expected_tokens} from feat_h={feat_h} feat_w={feat_w}"
            )
        return video_tokens.reshape(batch_size, temporal_steps, int(feat_h), int(feat_w), feat_dim)

    def flatten_video_token_maps(self, token_maps: torch.Tensor) -> torch.Tensor:
        if token_maps.ndim != 5:
            raise ValueError(f"Expected token_maps with shape [B,T,H,W,D], got {tuple(token_maps.shape)}")
        batch_size, temporal_steps, feat_h, feat_w, feat_dim = token_maps.shape
        return token_maps.reshape(batch_size, temporal_steps, feat_h * feat_w, feat_dim)

    def mix_video_patch_embeddings(
        self,
        target_video_tokens: torch.Tensor,
        feat_h: int,
        feat_w: int,
        obj_latents: torch.Tensor,
        alpha: float | None = None,
        decode_chunk_size: int = 4,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Mix target video patch embeddings with externally extracted object latents."""
        target_token_maps = self.reshape_video_patch_embeddings(target_video_tokens, feat_h=feat_h, feat_w=feat_w)
        mixed_maps, reconstructed = self.mix_with_obj_latents(
            target_token_maps,
            obj_latents=obj_latents,
            alpha=alpha,
            decode_chunk_size=decode_chunk_size,
        )
        mixed_tokens = self.flatten_video_token_maps(mixed_maps)
        return mixed_tokens, {
            "reconstructed": reconstructed,
            "obj_latents": obj_latents,
        }

    def load_combined_checkpoint(self, checkpoint: Dict[str, object]):
        encoder_state = checkpoint.get("student_state", checkpoint.get("ema_encoder"))
        extractor_state = checkpoint.get("extractor_state")
        if encoder_state is None:
            raise KeyError("Checkpoint must contain 'student_state' or 'ema_encoder' for vjepa2_slot")

        encoder_state = _strip_prefix_if_present(encoder_state, "module.")
        encoder_state = _strip_prefix_if_present(encoder_state, "backbone.")
        extractor_state = _strip_prefix_if_present(extractor_state, "module.") if extractor_state is not None else None

        encoder_msg = self.encoder.load_state_dict(encoder_state, strict=False)
        extractor_msg = None
        if extractor_state is not None:
            extractor_msg = self.extractor.load_state_dict(extractor_state, strict=False)
        return encoder_msg, extractor_msg
