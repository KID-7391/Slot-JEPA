"""Shared spatial feature extraction helpers for dense evaluation protocols."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F


def _is_tensor_or_none(value: Any) -> bool:
    return value is None or isinstance(value, torch.Tensor)


def _normalize_feature_outputs(outputs: Any) -> List[Any]:
    if isinstance(outputs, (list, tuple)):
        # Some backbones return a single feature as (tensor, aux) instead of a list of features.
        if len(outputs) > 0 and isinstance(outputs[0], torch.Tensor) and all(_is_tensor_or_none(v) for v in outputs[1:]):
            return [outputs]
        return list(outputs)
    return [outputs]


def _is_square(value: int) -> bool:
    if value <= 0:
        return False
    root = int(math.sqrt(value))
    return root * root == value


def _resolve_indices(length: int, indices: Sequence[int]) -> List[int]:
    resolved = []
    for raw_index in indices:
        index = int(raw_index)
        if index < 0:
            index = length + index
        if index < 0 or index >= length:
            raise IndexError(f"Feature index {raw_index} is out of range for length {length}")
        resolved.append(index)
    return resolved


def _grid_from_image_hw(model, image_hw: tuple[int, int] | None, token_count: int | None = None):
    if image_hw is None:
        return None
    input_pad_size = getattr(model, "input_pad_size", None)
    if input_pad_size is None:
        input_pad_size = getattr(model, "patch_size", None)
    if input_pad_size is None and hasattr(model, "patch_embed"):
        input_pad_size = getattr(model.patch_embed, "patch_size", None)
    if input_pad_size is not None:
        if isinstance(input_pad_size, (tuple, list)):
            input_pad_size = int(input_pad_size[0])
        height, width = image_hw
        return int(height // input_pad_size), int(width // input_pad_size)
    # Model does not expose patch size; try common downsampling strides.
    if token_count is not None and token_count > 0:
        height, width = image_hw
        for stride in (32, 16, 14, 8, 7, 4):
            gh = int(height // stride)
            gw = int(width // stride)
            if gh > 0 and gw > 0 and gh * gw == token_count:
                return gh, gw
            gh = int(math.ceil(height / stride))
            gw = int(math.ceil(width / stride))
            if gh > 0 and gw > 0 and gh * gw == token_count:
                return gh, gw
    return None


def _strip_prefix_tokens(
    tokens: torch.Tensor,
    model,
    drop_prefix_tokens: int | None = None,
    image_hw: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, tuple[int, int] | None]:
    _, token_count, _ = tokens.shape
    if drop_prefix_tokens is not None:
        drop_count = int(drop_prefix_tokens)
        if drop_count <= 0:
            remain = token_count
            return tokens, _grid_from_image_hw(model, image_hw, token_count=remain)
        if drop_count >= token_count:
            raise ValueError(f"drop_prefix_tokens={drop_count} removes all tokens (token_count={token_count}).")
        remain = token_count - drop_count
        return tokens[:, drop_count:], _grid_from_image_hw(model, image_hw, token_count=remain)

    grid_shape = _grid_from_image_hw(model, image_hw, token_count=token_count)
    if grid_shape is not None:
        grid_h, grid_w = grid_shape
        candidate_drops = []
        num_register_tokens = int(getattr(model, "num_register_tokens", 0))
        candidate_drops.append(1 + num_register_tokens)
        candidate_drops.append(1)
        candidate_drops.append(0)
        for drop_count in candidate_drops:
            remain = token_count - drop_count
            if remain == grid_h * grid_w:
                return tokens[:, drop_count:], (grid_h, grid_w)

    # Try candidate prefix drops with known image_hw to infer non-square grids.
    num_register_tokens = int(getattr(model, "num_register_tokens", 0))
    candidate_drops = [1 + num_register_tokens, 1, 0] if num_register_tokens > 0 else [1, 0]
    for drop_count in candidate_drops:
        remain = token_count - drop_count
        if remain <= 0:
            continue
        if _is_square(remain):
            side = int(math.sqrt(remain))
            return tokens[:, drop_count:], (side, side)
        # Try inferring a non-square grid from image_hw with common strides.
        inferred = _grid_from_image_hw(model, image_hw, token_count=remain)
        if inferred is not None:
            return tokens[:, drop_count:], inferred

    raise ValueError(
        f"Patch token count {token_count} is not square and no valid cls/register strip rule was found."
    )


def _tokens_to_spatial(
    tokens: torch.Tensor,
    model,
    drop_prefix_tokens: int | None = None,
    image_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    if tokens.ndim == 4:
        return tokens.contiguous()
    if tokens.ndim != 3:
        raise ValueError(f"Expected feature tensor with ndim=3 or 4, got shape={tuple(tokens.shape)}")

    tokens, grid_shape = _strip_prefix_tokens(
        tokens,
        model=model,
        drop_prefix_tokens=drop_prefix_tokens,
        image_hw=image_hw,
    )
    batch_size, token_count, channels = tokens.shape
    if grid_shape is None:
        height = int(math.sqrt(token_count))
        if height * height != token_count:
            raise ValueError(f"Patch token count {token_count} cannot be reshaped to square spatial map.")
        width = height
    else:
        height, width = grid_shape
        if height * width != token_count:
            raise ValueError(
                f"Patch token count {token_count} does not match inferred spatial map {height}x{width}."
            )
    return tokens.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()


def _align_and_concat(features: List[torch.Tensor], list_cfg: Dict[str, Any]) -> torch.Tensor:
    if len(features) == 0:
        raise ValueError("No features selected for list fusion.")
    if len(features) == 1:
        return features[0]

    align_mode = list_cfg.get("align", "max")
    if align_mode == "max":
        target_h = max(feature.shape[-2] for feature in features)
        target_w = max(feature.shape[-1] for feature in features)
    elif align_mode == "min":
        target_h = min(feature.shape[-2] for feature in features)
        target_w = min(feature.shape[-1] for feature in features)
    elif align_mode == "index":
        align_index = int(list_cfg.get("align_index", 0))
        resolved_index = _resolve_indices(len(features), [align_index])[0]
        target_h = features[resolved_index].shape[-2]
        target_w = features[resolved_index].shape[-1]
    else:
        raise ValueError(f"Unsupported list align mode: {align_mode}")

    aligned = []
    for feature in features:
        if feature.shape[-2:] == (target_h, target_w):
            aligned.append(feature)
        else:
            aligned.append(F.interpolate(feature, size=(target_h, target_w), mode="bilinear", align_corners=False))
    return torch.cat(aligned, dim=1)


def _mix_vjepa2_slot_tokens(
    model,
    tokens: torch.Tensor,
    image_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    if not (hasattr(model, "reshape_patch_embeddings") and hasattr(model, "extract_obj_latents") and hasattr(model, "mix_with_obj_latents")):
        return tokens
    if tokens.ndim != 3:
        raise ValueError(f"Expected VJEPA2 slot tokens with shape [B,N,D], got {tuple(tokens.shape)}")
    if image_hw is None:
        raise ValueError("image_hw is required for VJEPA2 slot feature mixing")
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "_compute_spatial_tokens"):
        feat_h = int(encoder._compute_spatial_tokens(int(image_hw[0])))
        feat_w = int(encoder._compute_spatial_tokens(int(image_hw[1])))
    else:
        feat_h, feat_w = _grid_from_image_hw(model, image_hw, token_count=tokens.shape[1]) or (None, None)
    if feat_h is None or feat_w is None:
        raise ValueError(
            f"Could not infer VJEPA2 slot feature grid from image_hw={image_hw} and token_count={tokens.shape[1]}"
        )
    token_maps = tokens.reshape(tokens.shape[0], 1, feat_h, feat_w, tokens.shape[-1])
    obj_latents, _ = model.extract_obj_latents(token_maps)
    mixed_maps, _ = model.mix_with_obj_latents(token_maps, obj_latents=obj_latents)
    return model.flatten_video_token_maps(mixed_maps)[:, 0]


def _extract_single_feature(model, images, model_type: str, feature_cfg: Dict[str, Any]) -> torch.Tensor:
    single_cfg = feature_cfg.get("single", {})
    image_hw = tuple(images.shape[-2:])
    if model_type == "dinov2":
        feats = model.forward_features(images)
        key = single_cfg.get("key", "x_norm_patchtokens")
        if key not in feats:
            raise KeyError(f"Feature key '{key}' not found in model.forward_features output.")
        tokens = feats[key]
        drop_prefix_tokens = single_cfg.get("drop_prefix_tokens") if key == "x_prenorm" else None
        return _tokens_to_spatial(tokens, model=model, drop_prefix_tokens=drop_prefix_tokens, image_hw=image_hw)

    source = single_cfg.get("source", "auto")
    if source == "auto":
        source = "get_intermediate_layers" if hasattr(model, "get_intermediate_layers") else "get_embeddings"

    index = int(single_cfg.get("index", -1))
    if source == "get_embeddings":
        _, patch_embeddings = model.get_embeddings(images)
        resolved_index = _resolve_indices(len(patch_embeddings), [index])[0]
        tokens = patch_embeddings[resolved_index]
        if model_type == "vjepa2_slot":
            tokens = _mix_vjepa2_slot_tokens(model, tokens, image_hw=image_hw)
        return _tokens_to_spatial(
            tokens,
            model=model,
            drop_prefix_tokens=single_cfg.get("drop_prefix_tokens"),
            image_hw=image_hw,
        )

    if source == "get_intermediate_layers" and hasattr(model, "get_intermediate_layers"):
        outputs = model.get_intermediate_layers(images)
        outputs = _normalize_feature_outputs(outputs)
        resolved_index = _resolve_indices(len(outputs), [index])[0]
        output = outputs[resolved_index]
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                raise ValueError("Encountered empty tuple/list output while extracting features.")
            output = output[0]
        return _tokens_to_spatial(
            output,
            model=model,
            drop_prefix_tokens=single_cfg.get("drop_prefix_tokens"),
            image_hw=image_hw,
        )

    raise ValueError(f"Unsupported single feature source for non-dinov2 model: {source}")


def _collect_feature_list(model, images, model_type: str, list_cfg: Dict[str, Any]) -> List[torch.Tensor]:
    image_hw = tuple(images.shape[-2:])
    source = list_cfg.get("source", "auto")
    if source == "auto":
        if hasattr(model, "get_intermediate_layers"):
            source = "get_intermediate_layers"
        elif hasattr(model, "get_embeddings"):
            source = "get_embeddings"
        else:
            source = "model_forward"

    outputs: List[Any]
    if source == "get_intermediate_layers" and hasattr(model, "get_intermediate_layers"):
        if hasattr(model, "get_num_layers"):
            total_layers = int(model.get_num_layers())
            layer_ids = list(range(total_layers))
            if model_type == "dinov2":
                outputs = list(
                    model.get_intermediate_layers(images, n=layer_ids, reshape=False, norm=bool(list_cfg.get("norm", True)))
                )
            else:
                outputs = _normalize_feature_outputs(model.get_intermediate_layers(images, n=layer_ids))
        else:
            outputs = model.get_intermediate_layers(images)
            outputs = _normalize_feature_outputs(outputs)
    elif source == "get_embeddings" and hasattr(model, "get_embeddings"):
        _, patch_embeddings = model.get_embeddings(images)
        outputs = list(patch_embeddings)
    elif source == "model_forward":
        outputs = model(images)
        outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
    else:
        raise ValueError(f"Unsupported list feature source: {source}")

    indices = _resolve_indices(len(outputs), list_cfg.get("indices", [-1]))
    selected_outputs = [outputs[index] for index in indices]

    spatial_features: List[torch.Tensor] = []
    for output in selected_outputs:
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                raise ValueError("Encountered empty tuple/list output while extracting features.")
            output = output[0]
        if model_type == "vjepa2_slot" and source == "get_embeddings":
            output = _mix_vjepa2_slot_tokens(model, output, image_hw=image_hw)
        spatial_features.append(
            _tokens_to_spatial(
                output,
                model=model,
                drop_prefix_tokens=list_cfg.get("drop_prefix_tokens"),
                image_hw=image_hw,
            )
        )
    return spatial_features


def extract_spatial_features(model, images, model_type: str, feature_cfg: Dict[str, Any]) -> torch.Tensor:
    mode = feature_cfg.get("mode", "single")
    if mode == "single":
        return _extract_single_feature(model, images, model_type=model_type, feature_cfg=feature_cfg)
    if mode == "list":
        list_cfg = feature_cfg.get("list", {})
        feature_list = _collect_feature_list(model, images, model_type=model_type, list_cfg=list_cfg)
        return _align_and_concat(feature_list, list_cfg=list_cfg)
    raise ValueError(f"Unsupported feature mode: {mode}")
