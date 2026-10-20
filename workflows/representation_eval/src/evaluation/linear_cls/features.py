"""Feature extraction helpers for DINOv2-style linear probing."""

from __future__ import annotations

import inspect
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def _is_square(value: int) -> bool:
    if value <= 0:
        return False
    root = int(math.sqrt(value))
    return root * root == value


def _flatten_patch_tokens(patch_tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if patch_tokens is None:
        return None
    if patch_tokens.ndim == 3:
        return patch_tokens
    if patch_tokens.ndim == 4:
        return patch_tokens.flatten(2).transpose(1, 2).contiguous()
    if patch_tokens.ndim == 2:
        return patch_tokens.unsqueeze(1)
    raise ValueError(f"Unsupported patch tensor shape: {tuple(patch_tokens.shape)}")


def _resolve_cls_token(cls_token: Optional[torch.Tensor], patch_tokens: Optional[torch.Tensor]) -> torch.Tensor:
    if cls_token is None:
        if patch_tokens is None:
            raise ValueError("Both cls_token and patch_tokens are None.")
        return patch_tokens.mean(dim=1)
    if cls_token.ndim == 2:
        return cls_token
    if cls_token.ndim == 3:
        if cls_token.shape[1] == 1:
            return cls_token[:, 0]
        return cls_token.mean(dim=1)
    if cls_token.ndim == 4:
        return cls_token.flatten(2).mean(dim=2)
    raise ValueError(f"Unsupported cls tensor shape: {tuple(cls_token.shape)}")


def _split_sequence_tokens(
    tokens: torch.Tensor,
    model,
    drop_prefix_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, token_count, _ = tokens.shape

    if drop_prefix_tokens is not None:
        drop_count = int(drop_prefix_tokens)
        if drop_count < 0 or drop_count >= token_count:
            raise ValueError(
                f"Invalid drop_prefix_tokens={drop_count} for token_count={token_count}."
            )
        if drop_count == 0:
            return tokens, tokens.mean(dim=1)
        return tokens[:, drop_count:], tokens[:, 0]

    candidate_drops: List[int] = []
    num_register_tokens = int(getattr(model, "num_register_tokens", 0))
    candidate_drops.extend([1 + num_register_tokens, 2, 1, 0])

    seen = set()
    for drop_count in candidate_drops:
        if drop_count in seen:
            continue
        seen.add(drop_count)
        remain = token_count - drop_count
        if remain <= 0:
            continue
        if _is_square(remain):
            if drop_count == 0:
                return tokens, tokens.mean(dim=1)
            return tokens[:, drop_count:], tokens[:, 0]

    if token_count > 1:
        return tokens[:, 1:], tokens[:, 0]
    return tokens, tokens[:, 0]


def _parse_intermediate_output(
    output: Any,
    model,
    drop_prefix_tokens: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, (list, tuple)) and len(output) == 2:
        first, second = output
        if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
            if first.ndim in (3, 4) and second.ndim == 2:
                patch_tokens = _flatten_patch_tokens(first)
                return patch_tokens, second

    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("Encountered empty output while parsing intermediate features.")
        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Unsupported intermediate output type: {type(output)}")

    if output.ndim == 4:
        patch_tokens = _flatten_patch_tokens(output)
        return patch_tokens, patch_tokens.mean(dim=1)
    if output.ndim == 3:
        return _split_sequence_tokens(output, model=model, drop_prefix_tokens=drop_prefix_tokens)
    if output.ndim == 2:
        return output.unsqueeze(1), output
    raise ValueError(f"Unsupported intermediate output shape: {tuple(output.shape)}")


def _normalize_embedding_lists(
    cls_embeddings: Sequence[Any],
    patch_embeddings: Sequence[Any],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if len(cls_embeddings) != len(patch_embeddings):
        raise ValueError(
            f"cls_embeddings and patch_embeddings length mismatch: {len(cls_embeddings)} vs {len(patch_embeddings)}"
        )

    normalized_cls: List[torch.Tensor] = []
    normalized_patch: List[torch.Tensor] = []

    for cls_token, patch_tokens in zip(cls_embeddings, patch_embeddings):
        patch_tokens = _flatten_patch_tokens(patch_tokens if isinstance(patch_tokens, torch.Tensor) else None)
        cls_token = cls_token if isinstance(cls_token, torch.Tensor) else None
        if cls_token is None and patch_tokens is None:
            continue
        normalized_patch.append(patch_tokens if patch_tokens is not None else cls_token.unsqueeze(1))
        normalized_cls.append(_resolve_cls_token(cls_token, patch_tokens))

    if len(normalized_cls) == 0:
        raise ValueError("No valid embeddings were returned by the model.")

    return normalized_cls, normalized_patch


def _call_get_intermediate_layers(model, images: torch.Tensor, max_n_blocks: int, feature_cfg: Dict[str, Any]):
    kwargs = {}
    signature = inspect.signature(model.get_intermediate_layers)
    if "reshape" in signature.parameters:
        kwargs["reshape"] = False
    if "norm" in signature.parameters:
        kwargs["norm"] = bool(feature_cfg.get("norm", True))
    if "return_class_token" in signature.parameters:
        kwargs["return_class_token"] = True
    return model.get_intermediate_layers(images, n=max_n_blocks, **kwargs)


def extract_embedding_lists(
    model,
    images: torch.Tensor,
    model_type: str,
    feature_cfg: Dict[str, Any],
    max_n_blocks: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    source = feature_cfg.get("source", "auto")
    drop_prefix_tokens = feature_cfg.get("drop_prefix_tokens")

    if source in {"auto", "get_embeddings"} and hasattr(model, "get_embeddings"):
        cls_embeddings, patch_embeddings = model.get_embeddings(images)
        return _normalize_embedding_lists(cls_embeddings, patch_embeddings)

    if source in {"auto", "get_intermediate_layers"} and hasattr(model, "get_intermediate_layers"):
        outputs = _call_get_intermediate_layers(model, images, max_n_blocks=max_n_blocks, feature_cfg=feature_cfg)
        outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]

        cls_embeddings: List[torch.Tensor] = []
        patch_embeddings: List[torch.Tensor] = []
        for output in outputs:
            patch_tokens, cls_token = _parse_intermediate_output(
                output,
                model=model,
                drop_prefix_tokens=drop_prefix_tokens,
            )
            patch_embeddings.append(patch_tokens)
            cls_embeddings.append(cls_token)
        return cls_embeddings, patch_embeddings

    if model_type == "dinov2" and hasattr(model, "forward_features"):
        outputs = model.forward_features(images)
        patch_tokens = outputs["x_norm_patchtokens"]
        cls_token = outputs["x_norm_clstoken"]
        return [cls_token], [patch_tokens]

    raise ValueError(
        f"Unable to extract embeddings for model_type={model_type}. "
        "Expected get_embeddings(), get_intermediate_layers(), or a DINOv2-style forward_features()."
    )


def resolve_n_last_blocks(available_blocks: int, requested_n_last_blocks: Sequence[int]) -> List[int]:
    resolved = sorted({int(value) for value in requested_n_last_blocks if int(value) <= available_blocks})
    if len(resolved) == 0:
        raise ValueError(
            f"No valid n_last_blocks remain after filtering requested={list(requested_n_last_blocks)} "
            f"against available_blocks={available_blocks}."
        )
    return resolved


def create_linear_input(
    embedding_output: Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]],
    use_n_blocks: int,
    use_avgpool: bool,
    pooling_mode: str = "cls",
) -> torch.Tensor:
    cls_embeddings, patch_embeddings = embedding_output
    pooling_mode = str(pooling_mode).lower()
    if use_n_blocks <= 0:
        raise ValueError(f"use_n_blocks must be positive, got {use_n_blocks}")
    if use_n_blocks > len(cls_embeddings):
        raise ValueError(
            f"Requested use_n_blocks={use_n_blocks}, but only {len(cls_embeddings)} blocks are available."
        )

    if pooling_mode not in {"cls", "cls_avgpool", "avgpool"}:
        raise ValueError(f"Unsupported pooling_mode={pooling_mode}")

    if pooling_mode == "avgpool":
        last_patch_tokens = patch_embeddings[-1]
        if last_patch_tokens is None:
            raise ValueError("pooling_mode=avgpool requires patch tokens for the last selected block.")
        return last_patch_tokens.mean(dim=1).float()

    selected_cls = list(cls_embeddings[-use_n_blocks:])
    output = torch.cat(selected_cls, dim=-1)

    if use_avgpool or pooling_mode == "cls_avgpool":
        last_patch_tokens = patch_embeddings[-1]
        if last_patch_tokens is None:
            raise ValueError("use_avgpool=True requires patch tokens for the last selected block.")
        output = torch.cat((output, last_patch_tokens.mean(dim=1)), dim=-1)

    return output.float()
