"""Image patch masking and shuffling helpers for classification robustness."""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.nn.functional as F


ATTENTION_HIGH_MODES = {"attention", "attention_high", "high_attention", "representative", "top"}
ATTENTION_LOW_MODES = {"attention_low", "low_attention", "bottom"}
RANDOM_MODES = {"random", "rand"}
SUPPORTED_MASK_MODES = RANDOM_MODES | ATTENTION_HIGH_MODES | ATTENTION_LOW_MODES


def normalize_mask_mode(mode: str | None) -> str:
    normalized = str(mode or "random").lower()
    if normalized not in SUPPORTED_MASK_MODES:
        raise ValueError(
            f"Unsupported patch mask mode '{mode}'. "
            f"Expected one of {sorted(SUPPORTED_MASK_MODES)}."
        )
    if normalized in RANDOM_MODES:
        return "random"
    if normalized in ATTENTION_LOW_MODES:
        return "attention_low"
    return "attention_high"


def patch_mask_enabled(mask_cfg: Dict) -> bool:
    return bool(mask_cfg.get("enabled", False)) and float(mask_cfg.get("ratio", 0.0)) > 0.0


def patch_shuffle_enabled(shuffle_cfg: Dict) -> bool:
    return bool(shuffle_cfg.get("enabled", False)) and int(shuffle_cfg.get("grid_size", 1)) > 1


def patch_mask_count(num_patches: int, ratio: float) -> int:
    return max(0, min(int(round(float(num_patches) * float(ratio))), int(num_patches)))


def _valid_flags(valid: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
    if valid is None:
        return torch.ones(batch_size, dtype=torch.bool)
    return valid.detach().cpu().bool().view(-1)


def _sample_indices(sample_indices: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
    if sample_indices is None:
        return torch.arange(batch_size, dtype=torch.long)
    return sample_indices.detach().cpu().long().view(-1)


def _patch_grid(images: torch.Tensor, patch_size: int) -> tuple[int, int]:
    if images.ndim != 4:
        raise ValueError(f"Expected image tensor [B, C, H, W], got {tuple(images.shape)}")
    height, width = int(images.shape[-2]), int(images.shape[-1])
    if height % int(patch_size) != 0 or width % int(patch_size) != 0:
        raise ValueError(f"patch_size={patch_size} must divide image size {height}x{width}")
    return height // int(patch_size), width // int(patch_size)


def random_patch_mask(
    *,
    batch_size: int,
    num_patches: int,
    ratio: float,
    seed: int,
    sample_indices: Optional[torch.Tensor],
    valid: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    mask_count = patch_mask_count(num_patches, ratio)
    selected = torch.zeros((batch_size, num_patches), device=device, dtype=torch.bool)
    if mask_count == 0:
        return selected

    sample_indices_cpu = _sample_indices(sample_indices, batch_size)
    valid_cpu = _valid_flags(valid, batch_size)
    for batch_index in range(batch_size):
        if not bool(valid_cpu[batch_index]):
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(sample_indices_cpu[batch_index].item()))
        indices = torch.randperm(num_patches, generator=generator)[:mask_count].to(device=device)
        selected[batch_index, indices] = True
    return selected


def attention_patch_mask(
    attention: torch.Tensor,
    *,
    ratio: float,
    mode: str,
    valid: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if attention.ndim != 2:
        raise ValueError(f"Expected attention [B, P], got {tuple(attention.shape)}")
    batch_size, num_patches = int(attention.shape[0]), int(attention.shape[1])
    mask_count = patch_mask_count(num_patches, ratio)
    selected = torch.zeros((batch_size, num_patches), device=device, dtype=torch.bool)
    if mask_count == 0:
        return selected

    valid_cpu = _valid_flags(valid, batch_size)
    largest = normalize_mask_mode(mode) == "attention_high"
    scores = attention.detach().float().to(device=device)
    for batch_index in range(batch_size):
        if not bool(valid_cpu[batch_index]):
            continue
        indices = torch.topk(scores[batch_index], k=mask_count, largest=largest, sorted=False).indices
        selected[batch_index, indices] = True
    return selected


def apply_patch_mask(images: torch.Tensor, mask: torch.Tensor, *, patch_size: int, value: float) -> torch.Tensor:
    if not bool(mask.any()):
        return images

    batch_size = int(images.shape[0])
    grid_h, grid_w = _patch_grid(images, int(patch_size))
    if mask.ndim == 2:
        if int(mask.shape[1]) != grid_h * grid_w:
            raise ValueError(f"Mask has {mask.shape[1]} patches but image grid is {grid_h}x{grid_w}.")
        patch_mask = mask.to(device=images.device, dtype=torch.bool).view(batch_size, 1, grid_h, grid_w)
    elif mask.ndim == 3:
        if tuple(mask.shape[-2:]) != (grid_h, grid_w):
            raise ValueError(f"Mask grid {tuple(mask.shape[-2:])} does not match image grid {grid_h}x{grid_w}.")
        patch_mask = mask.to(device=images.device, dtype=torch.bool).view(batch_size, 1, grid_h, grid_w)
    else:
        raise ValueError(f"Expected mask [B, P] or [B, H, W], got {tuple(mask.shape)}")

    pixel_mask = patch_mask.repeat_interleave(int(patch_size), dim=2).repeat_interleave(int(patch_size), dim=3)
    return images.clone().masked_fill_(pixel_mask, float(value))


def flatten_blocks(blocks: Iterable[torch.nn.Module]) -> list[torch.nn.Module]:
    flat: list[torch.nn.Module] = []
    for block in blocks:
        if isinstance(block, torch.nn.Identity):
            continue
        if isinstance(block, (torch.nn.ModuleList, torch.nn.Sequential)):
            flat.extend(flatten_blocks(block))
        else:
            flat.append(block)
    return flat


def get_last_attention_module(model: torch.nn.Module):
    model = getattr(model, "module", model)
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return None
    for block in reversed(flatten_blocks(blocks)):
        attn = getattr(block, "attn", None)
        if attn is not None and getattr(attn, "qkv", None) is not None:
            return attn
    return None


def _prefix_token_count(model: torch.nn.Module) -> int:
    return 1 + int(getattr(model, "num_register_tokens", 0) or 0)


def _attention_from_qkv(model: torch.nn.Module, attn_module, qkv: torch.Tensor) -> torch.Tensor:
    batch_size, token_count, qkv_dim = qkv.shape
    num_heads = int(getattr(attn_module, "num_heads", getattr(model, "num_heads", 0)))
    if num_heads <= 0:
        raise RuntimeError("Unable to infer attention head count.")
    head_dim = qkv_dim // (3 * num_heads)
    if 3 * num_heads * head_dim != qkv_dim:
        raise ValueError(f"Unexpected qkv shape: {tuple(qkv.shape)}")

    qkv = qkv.reshape(batch_size, token_count, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    scale = float(getattr(attn_module, "scale", head_dim ** -0.5))
    query = qkv[0] * scale
    key = qkv[1]
    attn = (query @ key.transpose(-2, -1)).softmax(dim=-1)
    return _patch_attention_from_full_attention(model, attn)


def _patch_attention_from_full_attention(model: torch.nn.Module, attention: torch.Tensor) -> torch.Tensor:
    if attention.ndim != 4:
        raise ValueError(f"Expected full attention [B, H, T, T], got {tuple(attention.shape)}")
    prefix_tokens = _prefix_token_count(getattr(model, "module", model))
    if prefix_tokens >= int(attention.shape[-1]):
        raise ValueError(f"Cannot remove {prefix_tokens} prefix tokens from attention with {attention.shape[-1]} tokens.")
    patch_attention = attention[:, :, 0, prefix_tokens:].mean(dim=1).float()
    patch_attention = patch_attention / patch_attention.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return patch_attention


@torch.no_grad()
def extract_cls_patch_attention(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    forward_fn: Optional[Callable[[torch.Tensor], object]] = None,
) -> torch.Tensor:
    """Extract final-layer CLS-to-image-patch attention as [B, P]."""
    base_model = getattr(model, "module", model)
    attn_module = get_last_attention_module(base_model)
    if attn_module is not None:
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            captured["qkv"] = output

        handle = attn_module.qkv.register_forward_hook(hook)
        try:
            if forward_fn is None:
                _ = base_model(images)
            else:
                _ = forward_fn(images)
        finally:
            handle.remove()
        if "qkv" not in captured:
            raise RuntimeError("Failed to capture final-block qkv activations for attention masking.")
        return _attention_from_qkv(base_model, attn_module, captured["qkv"])

    if hasattr(base_model, "get_last_selfattention"):
        attention = base_model.get_last_selfattention(images)
        return _patch_attention_from_full_attention(base_model, attention)

    raise RuntimeError("Attention-based patch masking requires a ViT model with final self-attention access.")


@torch.no_grad()
def apply_patch_mask_from_config(
    images: torch.Tensor,
    sample_indices: Optional[torch.Tensor],
    valid: Optional[torch.Tensor],
    mask_cfg: Dict,
    *,
    model: Optional[torch.nn.Module] = None,
    forward_fn: Optional[Callable[[torch.Tensor], object]] = None,
) -> torch.Tensor:
    if not patch_mask_enabled(mask_cfg):
        return images

    patch_size = int(mask_cfg.get("patch_size", getattr(getattr(model, "module", model), "patch_size", 16)))
    grid_h, grid_w = _patch_grid(images, patch_size)
    num_patches = grid_h * grid_w
    mode = normalize_mask_mode(mask_cfg.get("mode", "random"))
    ratio = float(mask_cfg.get("ratio", 0.0))
    seed = int(mask_cfg.get("seed", 0))
    value = float(mask_cfg.get("value", 0.0))

    if mode == "random":
        patch_mask = random_patch_mask(
            batch_size=int(images.shape[0]),
            num_patches=num_patches,
            ratio=ratio,
            seed=seed,
            sample_indices=sample_indices,
            valid=valid,
            device=images.device,
        )
    else:
        if model is None:
            raise ValueError("Attention patch masking requires model=... to be passed.")
        attention = extract_cls_patch_attention(model, images, forward_fn=forward_fn)
        if int(attention.shape[1]) != num_patches:
            raise ValueError(
                f"Attention has {attention.shape[1]} patches but mask grid from patch_size={patch_size} "
                f"is {grid_h}x{grid_w}."
            )
        patch_mask = attention_patch_mask(
            attention,
            ratio=ratio,
            mode=mode,
            valid=valid,
            device=images.device,
        )

    return apply_patch_mask(images, patch_mask, patch_size=patch_size, value=value)


def _factor_shuffle_grid(grid_size: int) -> tuple[int, int]:
    if int(grid_size) <= 0:
        raise ValueError(f"patch shuffle grid_size must be positive, got {grid_size}")
    grid_h = int(math.isqrt(int(grid_size)))
    while grid_h > 1 and int(grid_size) % grid_h != 0:
        grid_h -= 1
    return grid_h, int(grid_size) // grid_h


def _resolve_shuffle_grid(shuffle_cfg: Dict, grid_size: int) -> tuple[int, int]:
    if "grid_h" in shuffle_cfg or "grid_w" in shuffle_cfg:
        grid_h = int(shuffle_cfg.get("grid_h", 0))
        grid_w = int(shuffle_cfg.get("grid_w", 0))
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(f"patch shuffle grid_h/grid_w must be positive, got {grid_h}x{grid_w}")
        if grid_h * grid_w != int(grid_size):
            raise ValueError(f"patch shuffle grid_h*grid_w={grid_h * grid_w} does not match grid_size={grid_size}")
        return grid_h, grid_w
    return _factor_shuffle_grid(grid_size)


def _boundaries(length: int, parts: int) -> list[int]:
    if int(parts) <= 0:
        raise ValueError(f"parts must be positive, got {parts}")
    if int(parts) > int(length):
        raise ValueError(f"Cannot split length={length} into {parts} non-empty cells")
    values = [int(round(index * int(length) / int(parts))) for index in range(int(parts) + 1)]
    values[0] = 0
    values[-1] = int(length)
    return values


@torch.no_grad()
def apply_patch_shuffle_from_config(
    images: torch.Tensor,
    sample_indices: Optional[torch.Tensor],
    valid: Optional[torch.Tensor],
    shuffle_cfg: Dict,
) -> torch.Tensor:
    if not patch_shuffle_enabled(shuffle_cfg):
        return images

    if images.ndim != 4:
        raise ValueError(f"Expected image tensor [B, C, H, W], got {tuple(images.shape)}")

    grid_size = int(shuffle_cfg.get("grid_size", 1))
    grid_h, grid_w = _resolve_shuffle_grid(shuffle_cfg, grid_size)
    batch_size, channels, height, width = (int(value) for value in images.shape)
    seed = int(shuffle_cfg.get("seed", 0))
    sample_indices_cpu = _sample_indices(sample_indices, batch_size)
    valid_cpu = _valid_flags(valid, batch_size)

    if height % grid_h == 0 and width % grid_w == 0:
        tile_h = height // grid_h
        tile_w = width // grid_w
        tiles = images.reshape(batch_size, channels, grid_h, tile_h, grid_w, tile_w)
        tiles = tiles.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, grid_size, channels, tile_h, tile_w)
        shuffled = tiles.clone()
        for batch_index in range(batch_size):
            if not bool(valid_cpu[batch_index]):
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + int(sample_indices_cpu[batch_index].item()))
            order = torch.randperm(grid_size, generator=generator).to(device=images.device)
            shuffled[batch_index] = tiles[batch_index, order]

        output = shuffled.reshape(batch_size, grid_h, grid_w, channels, tile_h, tile_w)
        output = output.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)
        return output.contiguous()

    y_bounds = _boundaries(height, grid_h)
    x_bounds = _boundaries(width, grid_w)
    output = images.clone()
    for batch_index in range(batch_size):
        if not bool(valid_cpu[batch_index]):
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + int(sample_indices_cpu[batch_index].item()))
        order = torch.randperm(grid_size, generator=generator).tolist()
        for dst_index, src_index in enumerate(order):
            dst_y, dst_x = divmod(int(dst_index), grid_w)
            src_y, src_x = divmod(int(src_index), grid_w)
            dy0, dy1 = y_bounds[dst_y], y_bounds[dst_y + 1]
            dx0, dx1 = x_bounds[dst_x], x_bounds[dst_x + 1]
            sy0, sy1 = y_bounds[src_y], y_bounds[src_y + 1]
            sx0, sx1 = x_bounds[src_x], x_bounds[src_x + 1]
            source = images[batch_index : batch_index + 1, :, sy0:sy1, sx0:sx1]
            dst_size = (dy1 - dy0, dx1 - dx0)
            if tuple(source.shape[-2:]) != dst_size:
                source = F.interpolate(source, size=dst_size, mode="bilinear", align_corners=False)
            output[batch_index : batch_index + 1, :, dy0:dy1, dx0:dx1] = source
    return output.contiguous()
