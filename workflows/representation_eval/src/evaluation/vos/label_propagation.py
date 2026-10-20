"""Core VOS label propagation algorithm."""

from contextlib import nullcontext
import math
import warnings

import torch
from torch.nn import functional as F


def _infer_feature_hw(frame_h, frame_w, num_tokens, patch_size=None):
    """Infer spatial feature map dimensions from token count.

    Uses multiple strategies with fallbacks for robustness across model architectures.
    """
    if patch_size and patch_size > 0:
        for rounding in (math.floor, math.ceil):
            h = int(rounding(frame_h / float(patch_size)))
            w = int(rounding(frame_w / float(patch_size)))
            if h * w == num_tokens:
                return h, w
    stride = int(round(math.sqrt((frame_h * frame_w) / float(num_tokens))))
    stride = max(stride, 1)
    for stride_guess in (stride, max(stride - 1, 1), stride + 1):
        h = frame_h // stride_guess
        w = frame_w // stride_guess
        if h * w == num_tokens:
            return h, w
        h = int(round(frame_h / float(stride_guess)))
        w = int(round(frame_w / float(stride_guess)))
        if h * w == num_tokens:
            return h, w
    aspect = frame_h / float(frame_w)
    h = int(round(math.sqrt(num_tokens * aspect)))
    w = int(round(num_tokens / float(max(h, 1))))
    if h * w == num_tokens:
        return h, w
    w = int(round(math.sqrt(num_tokens / aspect)))
    h = int(round(num_tokens / float(max(w, 1))))
    if h * w == num_tokens:
        return h, w
    # Fallback: factorize num_tokens and pick closest aspect ratio.
    best = None
    for h in range(1, int(math.sqrt(num_tokens)) + 1):
        if num_tokens % h != 0:
            continue
        w = num_tokens // h
        for cand_h, cand_w in ((h, w), (w, h)):
            aspect_diff = abs((cand_h / float(cand_w)) - aspect)
            stride_diff = abs((frame_h / float(cand_h)) - (frame_w / float(cand_w)))
            score = (aspect_diff, stride_diff)
            if best is None or score < best[0]:
                best = (score, cand_h, cand_w)
    if best is not None:
        return best[1], best[2]
    raise ValueError(
        f"Cannot infer feature map size: frame=({frame_h},{frame_w}) tokens={num_tokens}"
    )


def _resolve_feature_dtype(vos_params):
    mode = vos_params.get("feature_dtype", "float32")
    if isinstance(mode, torch.dtype):
        return mode
    mode = str(mode).lower()
    if mode == "float32":
        return torch.float32
    if mode == "float16":
        return torch.float16
    raise ValueError(f"Unsupported feature_dtype={mode}")


def _resolve_label_agg_device(vos_params):
    mode = str(vos_params.get("label_agg_device", "auto")).lower()
    if mode in {"auto", "cuda", "cpu"}:
        return mode
    raise ValueError(f"Unsupported label_agg_device={mode}")


def _run_embedding_forward(model, inputs, device, feature_dtype):
    input_device = inputs.to(device, non_blocking=True)
    autocast_context = nullcontext()
    if input_device.device.type == "cuda" and feature_dtype != torch.float32:
        autocast_context = torch.autocast(device_type="cuda", dtype=feature_dtype)
    with autocast_context:
        cls_embeddings, patch_embeddings = model.get_embeddings(input_device)
    return patch_embeddings[-1].float()


def _forward_frame_batch(model, frames, device, patch_size=None, feature_dtype=torch.float32):
    tokens = _run_embedding_forward(model, frames, device, feature_dtype)
    num_tokens = tokens.shape[1]
    h0, w0 = _infer_feature_hw(frames.shape[2], frames.shape[3], num_tokens, patch_size)
    dim0 = tokens.shape[-1]
    tokens = tokens.reshape(frames.shape[0], h0, w0, dim0)
    return tokens, h0, w0


def extract_video_clip_target_features(
    model,
    clips,
    device,
    return_h_w=False,
    patch_size=None,
    feature_dtype=torch.float32,
    shift_and_stitch_factor=1,
    temporal_pool="last",
):
    """Extract one target feature map per clip using the final VJEPA temporal slot."""
    feature_dtype = _resolve_feature_dtype({"feature_dtype": feature_dtype})
    factor = int(shift_and_stitch_factor)
    if factor < 1:
        raise ValueError(f"shift_and_stitch_factor must be >= 1, got {shift_and_stitch_factor}")
    if clips.ndim != 5:
        raise ValueError(f"Expected clips with shape [B, T, C, H, W], got {tuple(clips.shape)}")

    if factor != 1:
        feats = []
        h = w = None
        for clip in clips:
            feat, h_i, w_i = extract_feature(
                model,
                clip[-1],
                device,
                return_h_w=True,
                patch_size=patch_size,
                feature_dtype=feature_dtype,
                shift_and_stitch_factor=factor,
            )
            feats.append(feat)
            if h is None:
                h, w = h_i, w_i
            elif h != h_i or w != w_i:
                raise ValueError(
                    "Shift-and-stitch clip fallback produced inconsistent feature shapes: "
                    f"expected=({h},{w}) got=({h_i},{w_i})"
                )
        out = torch.stack(feats, dim=0)
        if return_h_w:
            return out, h, w
        return out

    clip_batch = clips.permute(0, 2, 1, 3, 4).contiguous()
    tokens = _run_embedding_forward(model, clip_batch, device, feature_dtype)

    tubelet_size = int(getattr(model, "tubelet_size", 1) or 1)
    temporal_steps = max(clips.shape[1] // tubelet_size, 1)
    if tokens.shape[1] % temporal_steps != 0:
        raise ValueError(
            "Cannot split VJEPA clip tokens into temporal steps: "
            f"num_tokens={tokens.shape[1]} temporal_steps={temporal_steps}"
        )

    tokens_per_step = tokens.shape[1] // temporal_steps
    h0, w0 = _infer_feature_hw(clips.shape[3], clips.shape[4], tokens_per_step, patch_size)
    dim0 = tokens.shape[-1]
    maps = tokens.reshape(clips.shape[0], temporal_steps, h0, w0, dim0)

    if temporal_pool == "last":
        maps = maps[:, -1]
    elif temporal_pool == "mean":
        maps = maps.mean(dim=1)
    else:
        raise ValueError(f"Unsupported temporal_pool={temporal_pool}")

    out = maps.reshape(clips.shape[0], -1, dim0)
    if return_h_w:
        return out, h0, w0
    return out


def extract_feature(
    model,
    frame,
    device,
    return_h_w=False,
    patch_size=None,
    feature_dtype=torch.float32,
    shift_and_stitch_factor=1,
):
    """Extract patch embeddings from a single frame."""
    feature_dtype = _resolve_feature_dtype({"feature_dtype": feature_dtype})
    factor = int(shift_and_stitch_factor)
    if factor < 1:
        raise ValueError(f"shift_and_stitch_factor must be >= 1, got {shift_and_stitch_factor}")

    out_maps, h, w = _forward_frame_batch(
        model,
        frame.unsqueeze(0),
        device,
        patch_size=patch_size,
        feature_dtype=feature_dtype,
    )
    out_map = out_maps[0]
    if factor == 1:
        out = out_map.reshape(-1, out_map.shape[-1])
        if return_h_w:
            return out, h, w
        return out

    if not patch_size or patch_size <= 0:
        raise ValueError("shift-and-stitch requires a positive patch_size")
    if patch_size % factor != 0:
        raise ValueError(f"patch_size={patch_size} must be divisible by shift_and_stitch_factor={factor}")

    phase_step = patch_size // factor
    dense = out_map.new_empty((h * factor, w * factor, out_map.shape[-1]))
    shifted_frames = []
    phase_indices = []

    for ay in range(factor):
        for ax in range(factor):
            offset_y = ay * phase_step
            offset_x = ax * phase_step
            if ay == 0 and ax == 0:
                dense[ay::factor, ax::factor] = out_map
                continue
            shifted = frame[:, offset_y:, offset_x:]
            shifted = F.pad(
                shifted.unsqueeze(0),
                (0, offset_x, 0, offset_y),
                mode="replicate",
            )[0]
            shifted_frames.append(shifted)
            phase_indices.append((ay, ax))

    if shifted_frames:
        shifted_batch = torch.stack(shifted_frames, dim=0)
        shifted_maps, shift_h, shift_w = _forward_frame_batch(
            model,
            shifted_batch,
            device,
            patch_size=patch_size,
            feature_dtype=feature_dtype,
        )
        if shift_h != h or shift_w != w:
            raise ValueError(
                "Shift-and-stitch produced inconsistent feature shapes: "
                f"base=({h},{w}) shifted=({shift_h},{shift_w})"
            )
        for idx, (ay, ax) in enumerate(phase_indices):
            dense[ay::factor, ax::factor] = shifted_maps[idx]

    out = dense.reshape(-1, dense.shape[-1])
    h *= factor
    w *= factor
    if return_h_w:
        return out, h, w
    return out


def _normalize_mask_shape(mask_shape):
    aliases = {
        None: "square",
        "square": "square",
        "sq": "square",
        "circle": "circle",
        "disk": "circle",
        "none": "none",
        "-": "none",
        "off": "none",
        "disabled": "none",
    }
    key = mask_shape if mask_shape is None else str(mask_shape).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported mask_shape={mask_shape!r}")
    return aliases[key]


def _use_neighborhood_mask(size_mask_neighborhood, mask_shape):
    mask_shape = _normalize_mask_shape(mask_shape)
    size_value = float(size_mask_neighborhood)
    return mask_shape != "none" and math.isfinite(size_value) and size_value > 0


def _canonicalize_spatial_context(spatial_context, fallback_hw=None):
    if spatial_context is None:
        if fallback_hw is None:
            return None
        full_h, full_w = int(fallback_hw[0]), int(fallback_hw[1])
        return {
            "full_hw": (full_h, full_w),
            "bbox": (0, 0, full_w, full_h),
        }
    full_h, full_w = spatial_context.get("full_hw", fallback_hw or (None, None))
    if full_h is None or full_w is None:
        raise ValueError("spatial_context must provide full_hw")
    full_h, full_w = int(full_h), int(full_w)
    bbox = spatial_context.get("bbox")
    if bbox is None:
        bbox = (0, 0, full_w, full_h)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return {
        "full_hw": (full_h, full_w),
        "bbox": (x1, y1, x2, y2),
    }


def _spatial_context_to_key(spatial_context):
    if spatial_context is None:
        return None
    canonical = _canonicalize_spatial_context(spatial_context)
    return canonical["full_hw"] + canonical["bbox"]


def _feature_grid_centers(h, w, spatial_context):
    spatial_context = _canonicalize_spatial_context(spatial_context, fallback_hw=(h, w))
    x1, y1, x2, y2 = spatial_context["bbox"]
    box_h = max(float(y2 - y1), 1.0)
    box_w = max(float(x2 - x1), 1.0)

    rows = y1 + (torch.arange(h, device="cpu", dtype=torch.float32) + 0.5) * (box_h / float(h))
    cols = x1 + (torch.arange(w, device="cpu", dtype=torch.float32) + 0.5) * (box_w / float(w))
    yy = rows.repeat_interleave(w).unsqueeze(1)
    xx = cols.repeat(h).unsqueeze(1)
    step_y = box_h / float(h)
    step_x = box_w / float(w)
    return yy, xx, step_y, step_x


def restrict_neighborhood(
    source_h,
    source_w,
    target_h,
    target_w,
    size_mask_neighborhood,
    device,
    mask_shape="square",
    source_spatial=None,
    target_spatial=None,
):
    """Create spatial locality mask restricting attention using original-image coordinates."""
    mask_shape = _normalize_mask_shape(mask_shape)
    radius = float(size_mask_neighborhood)
    if mask_shape == "none" or not math.isfinite(radius):
        return None

    src_r, src_c, src_step_y, src_step_x = _feature_grid_centers(source_h, source_w, source_spatial)
    tgt_r, tgt_c, tgt_step_y, tgt_step_x = _feature_grid_centers(target_h, target_w, target_spatial)
    ref_step_y = max((src_step_y + tgt_step_y) * 0.5, 1e-6)
    ref_step_x = max((src_step_x + tgt_step_x) * 0.5, 1e-6)

    dr = (tgt_r - src_r.squeeze(1).unsqueeze(0)) / ref_step_y
    dc = (tgt_c - src_c.squeeze(1).unsqueeze(0)) / ref_step_x
    if mask_shape == "square":
        mask = ((dr.abs() <= radius) & (dc.abs() <= radius)).float()
    elif mask_shape == "circle":
        mask = (((dr * dr) + (dc * dc)) <= (radius * radius)).float()
    else:
        raise ValueError(f"Unsupported mask_shape={mask_shape}")
    return mask.to(device, non_blocking=True)


def norm_mask(mask):
    """Per-channel min-max normalization of mask probabilities."""
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt, :, :]
        if mask_cnt.max() > 0:
            mask_cnt = mask_cnt - mask_cnt.min()
            mask_cnt = mask_cnt / mask_cnt.max()
            mask[cnt, :, :] = mask_cnt
    return mask


def upsample_and_norm_mask(mask, size, align_corners=True):
    """Resize segmentation logits to image resolution and normalize per channel."""
    mask = F.interpolate(
        mask,
        size=size,
        mode="bilinear",
        align_corners=bool(align_corners),
    )[0]
    return norm_mask(mask)


def _resolve_affinity_dtype(vos_params, device):
    mode = str(vos_params.get("affinity_dtype", "float32")).lower()
    if mode == "float32":
        return torch.float32
    if mode == "float16":
        return torch.float16
    raise ValueError(f"Unsupported affinity_dtype={mode}")


def _chunked_label_matmul(segs, aff, chunk_size):
    """Multiply segs @ aff in target-column chunks to reduce peak GEMM pressure."""
    if chunk_size is None or int(chunk_size) <= 0 or aff.shape[1] <= int(chunk_size):
        return torch.mm(segs, aff)

    chunk_size = int(chunk_size)
    out = torch.empty(segs.shape[0], aff.shape[1], device=segs.device, dtype=segs.dtype)
    for start in range(0, aff.shape[1], chunk_size):
        end = min(start + chunk_size, aff.shape[1])
        out[:, start:end] = torch.mm(segs, aff[:, start:end])
    return out


def _chunked_label_matmul_cpu(segs, aff, chunk_size):
    """CPU fallback for the final label aggregation when CUDA GEMM is unstable."""
    chunk_size = int(chunk_size) if chunk_size is not None else aff.shape[1]
    if chunk_size <= 0:
        chunk_size = aff.shape[1]

    segs_cpu = segs.float().cpu()
    out_cpu = torch.empty(segs_cpu.shape[0], aff.shape[1], device="cpu", dtype=torch.float32)
    for start in range(0, aff.shape[1], chunk_size):
        end = min(start + chunk_size, aff.shape[1])
        aff_chunk_cpu = aff[:, start:end].float().cpu()
        out_cpu[:, start:end] = torch.mm(segs_cpu, aff_chunk_cpu)
    return out_cpu


def _safe_label_matmul(segs, aff, chunk_size):
    """Run label aggregation on CUDA when possible, otherwise fall back to CPU."""
    try:
        return _chunked_label_matmul(segs, aff, chunk_size)
    except RuntimeError as error:
        message = str(error)
        if segs.is_cuda and (
            "CUBLAS_STATUS_EXECUTION_FAILED" in message
            or "cublasGemmEx" in message
            or "out of memory" in message.lower()
        ):
            warnings.warn(
                "CUDA label aggregation failed; retrying segs @ aff on CPU. "
                f"chunk_size={chunk_size}, segs={tuple(segs.shape)}, aff={tuple(aff.shape)}"
            )
            try:
                return _chunked_label_matmul_cpu(segs.detach(), aff.detach(), chunk_size)
            except Exception as fallback_error:
                raise RuntimeError(
                    "CUDA label aggregation failed and CPU fallback could not complete"
                ) from fallback_error
        raise


def _get_neighborhood_mask(
    mask_cache,
    source_h,
    source_w,
    target_h,
    target_w,
    size_mask_neighborhood,
    device,
    mask_shape,
    source_spatial,
    target_spatial,
):
    cache_key = (
        int(source_h),
        int(source_w),
        int(target_h),
        int(target_w),
        float(size_mask_neighborhood),
        _normalize_mask_shape(mask_shape),
        _spatial_context_to_key(source_spatial),
        _spatial_context_to_key(target_spatial),
    )
    if isinstance(mask_cache, dict) and cache_key in mask_cache:
        return mask_cache[cache_key]
    mask = restrict_neighborhood(
        source_h,
        source_w,
        target_h,
        target_w,
        size_mask_neighborhood,
        device,
        mask_shape=mask_shape,
        source_spatial=source_spatial,
        target_spatial=target_spatial,
    )
    if isinstance(mask_cache, dict):
        mask_cache[cache_key] = mask
    return mask


def _label_propagation_streaming_topk(
    vos_params,
    feat_tar,
    feat_sources,
    list_segs,
    device,
    h,
    w,
    target_spatial=None,
    source_spatials=None,
    mask_neighborhood=None,
):
    """Memory-efficient equivalent of dense top-k label propagation.

    The historical implementation builds a dense matrix with shape
    (n_context * h * w, h * w). For DAVIS at short-side 960 and n_last=15,
    this crosses 1B affinity elements. We instead keep only the exact global
    top-k source patches per target patch while streaming over context frames.
    """
    ncontext = len(feat_sources)
    num_target_patches = h * w
    source_patch_counts = [int(source.shape[1]) for source in feat_sources]
    total_source_patches = sum(source_patch_counts)
    topk = min(int(vos_params["topk"]), total_source_patches)

    affinity_dtype = _resolve_affinity_dtype(vos_params, device)
    feat_tar_mm = F.normalize(feat_tar, dim=1, p=2).to(dtype=affinity_dtype)
    if source_spatials is None:
        source_spatials = [None] * ncontext
    elif len(source_spatials) != ncontext:
        raise ValueError(f"source_spatials length mismatch: expected {ncontext}, got {len(source_spatials)}")

    mask_shape = vos_params.get("mask_shape", "square")
    if _use_neighborhood_mask(vos_params["size_mask_neighborhood"], mask_shape):
        if mask_neighborhood is None:
            mask_neighborhood = {}
    else:
        mask_neighborhood = None

    best_vals = None
    best_indices = None
    temperature = float(vos_params["temperature"])
    source_offset = 0
    for ctx_idx in range(ncontext):
        source = feat_sources[ctx_idx].to(device, non_blocking=True)
        source_h = int(list_segs[ctx_idx].shape[-2])
        source_w = int(list_segs[ctx_idx].shape[-1])
        if source_h * source_w != source.shape[1]:
            raise ValueError(
                "Feature/mask shape mismatch in context frame: "
                f"feat={tuple(source.shape)} seg={tuple(list_segs[ctx_idx].shape)}"
            )
        source = F.normalize(source, dim=0, p=2).to(dtype=affinity_dtype)
        aff_ctx = torch.exp(torch.mm(feat_tar_mm, source) / temperature)  # target x source
        if mask_neighborhood is not None:
            mask_for_aff = _get_neighborhood_mask(
                mask_neighborhood,
                source_h,
                source_w,
                h,
                w,
                vos_params["size_mask_neighborhood"],
                device,
                mask_shape,
                source_spatials[ctx_idx],
                target_spatial,
            ).to(device=device, dtype=affinity_dtype, non_blocking=True)
            aff_ctx = aff_ctx * mask_for_aff

        per_context_topk = min(topk, source.shape[1])
        vals, indices = torch.topk(aff_ctx, dim=1, k=per_context_topk)
        vals = vals.T.contiguous()
        indices = (indices.T.contiguous() + source_offset).long()

        if best_vals is None:
            best_vals = vals
            best_indices = indices
        else:
            merged_vals = torch.cat([best_vals, vals], dim=0)
            merged_indices = torch.cat([best_indices, indices], dim=0)
            best_vals, selected = torch.topk(merged_vals, dim=0, k=topk)
            best_indices = torch.gather(merged_indices, 0, selected)

        del aff_ctx, vals, indices
        source_offset += source.shape[1]

    if best_vals is None or best_indices is None:
        raise RuntimeError("No context frames available for VOS label propagation")

    weights = best_vals.float()
    weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-12)

    labels = []
    num_classes = None
    for ctx_idx, seg in enumerate(list_segs):
        seg = seg.to(device, non_blocking=True).float()
        if seg.shape[0] != 1:
            raise ValueError(f"Expected seg batch=1, got {tuple(seg.shape)}")
        _, current_classes, seg_h, seg_w = seg.shape
        if seg_h * seg_w != source_patch_counts[ctx_idx]:
            raise ValueError(
                "Segmentation/context shape mismatch: "
                f"seg={tuple(seg.shape)} feat={tuple(feat_sources[ctx_idx].shape)}"
            )
        if num_classes is None:
            num_classes = current_classes
        elif current_classes != num_classes:
            raise ValueError(f"Class-count mismatch across contexts: {current_classes} vs {num_classes}")
        labels.append(seg[0].reshape(current_classes, -1).T)
    labels = torch.cat(labels, dim=0)
    selected_labels = labels.index_select(0, best_indices.reshape(-1))
    selected_labels = selected_labels.reshape(topk, num_target_patches, num_classes)
    seg_tar = (selected_labels * weights.reshape(topk, num_target_patches, 1)).sum(dim=0)
    seg_tar = seg_tar.T.reshape(1, num_classes, h, w)
    return seg_tar, mask_neighborhood


def label_propagation(vos_params, model, frame_tar, list_frame_feats, list_segs,
                      device, mask_neighborhood=None, patch_size=None,
                      target_spatial=None, source_spatials=None, precomputed_feat_tar=None):
    """Propagate segmentation masks from context frames to target frame.

    Args:
        vos_params: dict with keys: size_mask_neighborhood, mask_shape, topk, temperature
        model: model with get_embeddings() method
        frame_tar: target frame tensor [3, H, W]
        list_frame_feats: list of context frame features [dim, h*w] each
        list_segs: list of context segmentation one-hot tensors
        device: torch device
        mask_neighborhood: precomputed neighborhood mask or None
        patch_size: patch size override (0 or None = auto)

    Returns:
        seg_tar: predicted segmentation [1, C, h, w]
        return_feat_tar: target frame features [dim, h*w]
        mask_neighborhood: neighborhood mask (for reuse)
    """
    feature_dtype = _resolve_feature_dtype(vos_params)
    shift_and_stitch_factor = int(vos_params.get("shift_and_stitch_factor", 1))
    if not bool(vos_params.get("shift_and_stitch_enabled", False)):
        shift_and_stitch_factor = 1
    if precomputed_feat_tar is None:
        feat_tar, h, w = extract_feature(
            model,
            frame_tar,
            device,
            return_h_w=True,
            patch_size=patch_size,
            feature_dtype=feature_dtype,
            shift_and_stitch_factor=shift_and_stitch_factor,
        )
        return_feat_tar = feat_tar.T  # dim x h*w
    else:
        return_feat_tar = precomputed_feat_tar.to(device, non_blocking=True).float()
        num_tokens = int(return_feat_tar.shape[1])
        h, w = _infer_feature_hw(frame_tar.shape[1], frame_tar.shape[2], num_tokens, patch_size)
        feat_tar = return_feat_tar.T.contiguous()

    ncontext = len(list_frame_feats)
    feat_sources = list(list_frame_feats)

    propagation_impl = str(vos_params.get("propagation_impl", "streaming_topk")).lower()
    if propagation_impl in {"streaming_topk", "streaming", "stream"}:
        seg_tar, mask_neighborhood = _label_propagation_streaming_topk(
            vos_params=vos_params,
            feat_tar=feat_tar,
            feat_sources=feat_sources,
            list_segs=list_segs,
            device=device,
            h=h,
            w=w,
            target_spatial=target_spatial,
            source_spatials=source_spatials,
            mask_neighborhood=mask_neighborhood,
        )
        return seg_tar, return_feat_tar, mask_neighborhood
    if propagation_impl not in {"dense", "legacy"}:
        raise ValueError(f"Unsupported propagation_impl={propagation_impl!r}")

    affinity_dtype = _resolve_affinity_dtype(vos_params, device)
    mask_shape = vos_params.get("mask_shape", "square")
    if source_spatials is None:
        source_spatials = [None] * ncontext
    elif len(source_spatials) != ncontext:
        raise ValueError(f"source_spatials length mismatch: expected {ncontext}, got {len(source_spatials)}")
    if _use_neighborhood_mask(vos_params["size_mask_neighborhood"], mask_shape):
        if mask_neighborhood is None:
            mask_neighborhood = {}
    else:
        mask_neighborhood = None

    feat_tar_mm = F.normalize(feat_tar, dim=1, p=2).to(dtype=affinity_dtype)
    aff_parts = []
    label_parts = []
    num_classes = None
    for ctx_idx, source in enumerate(feat_sources):
        source = source.to(device, non_blocking=True)
        seg = list_segs[ctx_idx].to(device, non_blocking=True).float()
        if seg.shape[0] != 1:
            raise ValueError(f"Expected seg batch=1, got {tuple(seg.shape)}")
        _, current_classes, source_h, source_w = seg.shape
        if source_h * source_w != source.shape[1]:
            raise ValueError(
                "Feature/mask shape mismatch in context frame: "
                f"feat={tuple(source.shape)} seg={tuple(seg.shape)}"
            )
        if num_classes is None:
            num_classes = current_classes
        elif current_classes != num_classes:
            raise ValueError(f"Class-count mismatch across contexts: {current_classes} vs {num_classes}")

        source_mm = F.normalize(source, dim=0, p=2).to(dtype=affinity_dtype)
        aff_ctx = torch.exp(torch.mm(feat_tar_mm, source_mm) / vos_params["temperature"])
        if mask_neighborhood is not None:
            aff_ctx = aff_ctx * _get_neighborhood_mask(
                mask_neighborhood,
                source_h,
                source_w,
                h,
                w,
                vos_params["size_mask_neighborhood"],
                device,
                mask_shape,
                source_spatials[ctx_idx],
                target_spatial,
            ).to(dtype=aff_ctx.dtype)
        aff_parts.append(aff_ctx.T)
        label_parts.append(seg[0].reshape(current_classes, -1).T)
    aff = torch.cat(aff_parts, dim=0)
    dense_topk = min(int(vos_params["topk"]), aff.shape[0])
    tk_val, _ = torch.topk(aff, dim=0, k=dense_topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, dim=0, keepdim=True).clamp_min(1e-12)

    segs = torch.cat(label_parts, dim=0).T
    matmul_chunk_size = int(vos_params.get("matmul_chunk_size", 2048))
    label_agg_device = _resolve_label_agg_device(vos_params)
    segs_mm = segs.to(dtype=affinity_dtype)
    if label_agg_device == "cpu":
        seg_tar = _chunked_label_matmul_cpu(segs_mm, aff, matmul_chunk_size)
    elif label_agg_device == "cuda":
        seg_tar = _chunked_label_matmul(segs_mm, aff, matmul_chunk_size)
    else:
        seg_tar = _safe_label_matmul(segs_mm, aff, chunk_size=matmul_chunk_size)
    seg_tar = seg_tar.float()
    seg_tar = seg_tar.reshape(1, num_classes, h, w)
    return seg_tar, return_feat_tar, mask_neighborhood
