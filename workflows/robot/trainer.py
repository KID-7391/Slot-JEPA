import gc
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

_RELEASE_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _RELEASE_SRC not in sys.path:
    sys.path.insert(0, _RELEASE_SRC)

from slot_jepa.predictors import ActionConditionalMaskPredictor
from slot_jepa.data.droid import init_data as init_raw_data
from slot_jepa.data.droid_frame import init_data as init_frame_data
from slot_jepa.data.transforms import make_gpu_transforms, make_transforms
from training import init_opt
from slot_jepa.losses import feature_recon_loss
from slot_jepa.slots import TwoStageExtractor
from slot_jepa.encoders import init_encoder
from slot_jepa.utils.distributed import init_distributed
from slot_jepa.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer


log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50
_GLOBAL_SEED = 0

random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)
_OT_COST_CACHE = {}


def _scheduler_state_dict(scheduler):
    if hasattr(scheduler, "state_dict"):
        return scheduler.state_dict()
    state = {}
    for key, value in scheduler.__dict__.items():
        if key == "optimizer":
            continue
        state[key] = value
    return state


def _scheduler_load_state_dict(scheduler, state_dict):
    if state_dict is None:
        return
    if hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(state_dict)
        return
    for key, value in state_dict.items():
        setattr(scheduler, key, value)


def _unnormalize_frame(frame):
    mean = torch.tensor([0.485, 0.456, 0.406], device=frame.device, dtype=frame.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=frame.device, dtype=frame.dtype).view(3, 1, 1)
    return (frame * std + mean).clamp(0.0, 1.0)


def _feature_to_pca_image(feat, pca_mean, pca_basis, pca_min, pca_max, patch_size, pred_border=False):
    feat_proj = (feat - pca_mean) @ pca_basis
    feat_proj = ((feat_proj - pca_min) / (pca_max - pca_min + 1.0e-8)).clamp(0.0, 1.0)
    spatial_size = int(feat_proj.size(0) ** 0.5)
    feat_img = feat_proj.view(spatial_size, spatial_size, 3).permute(2, 0, 1).unsqueeze(0)
    feat_img = F.interpolate(feat_img, scale_factor=patch_size, mode="nearest").squeeze(0)
    if pred_border:
        border = min(4, max(1, patch_size // 4))
        feat_img = feat_img.clone()
        feat_img[0, :border, :] = 1.0
        feat_img[1:, :border, :] = 0.0
        feat_img[0, -border:, :] = 1.0
        feat_img[1:, -border:, :] = 0.0
        feat_img[0, :, :border] = 1.0
        feat_img[1:, :, :border] = 0.0
        feat_img[0, :, -border:] = 1.0
        feat_img[1:, :, -border:] = 0.0
    return feat_img


def _get_spatial_transport_cost(spatial_size, device):
    cache_key = (spatial_size, device.type, device.index)
    if cache_key in _OT_COST_CACHE:
        return _OT_COST_CACHE[cache_key]

    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(0.0, 1.0, spatial_size, device=device),
            torch.linspace(0.0, 1.0, spatial_size, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    cost = torch.cdist(coords, coords, p=2).pow(2)
    _OT_COST_CACHE[cache_key] = cost
    return cost


def _sinkhorn_mask_transport_loss(
    pred_mask_probs,
    target_mask_probs,
    spatial_size,
    sinkhorn_iters=5,
    entropy_reg=0.05,
    chunk_size=32,
):
    return _sinkhorn_mask_transport_costs(
        pred_mask_probs,
        target_mask_probs,
        spatial_size=spatial_size,
        sinkhorn_iters=sinkhorn_iters,
        entropy_reg=entropy_reg,
        chunk_size=chunk_size,
    ).mean()


def _recon_l1_loss(pred_recon, target_recon):
    return (pred_recon.float() - target_recon.float()).abs().mean()


def _mask_l1_loss(pred_mask_probs, target_mask_probs):
    return (pred_mask_probs.float() - target_mask_probs.float()).abs().mean()


def _sinkhorn_mask_transport_costs(
    pred_mask_probs,
    target_mask_probs,
    spatial_size,
    sinkhorn_iters=5,
    entropy_reg=0.05,
    chunk_size=32,
):
    if pred_mask_probs.shape != target_mask_probs.shape:
        raise ValueError(
            f"Expected pred and target masks to share shape, got {tuple(pred_mask_probs.shape)} vs {tuple(target_mask_probs.shape)}"
        )

    leading_shape = pred_mask_probs.shape[:-1]
    pred_mass = pred_mask_probs.float().reshape(-1, pred_mask_probs.size(-1)).clamp_min(1.0e-8)
    target_mass = target_mask_probs.float().reshape(-1, target_mask_probs.size(-1)).clamp_min(1.0e-8)
    pred_mass = pred_mass / pred_mass.sum(dim=-1, keepdim=True)
    target_mass = target_mass / target_mass.sum(dim=-1, keepdim=True)

    num_distributions, num_patches = pred_mass.shape
    if num_distributions == 0:
        return pred_mask_probs.new_zeros(leading_shape)

    cost = _get_spatial_transport_cost(spatial_size, pred_mask_probs.device).float()
    kernel = torch.exp(-cost / entropy_reg).clamp_min(1.0e-8)

    per_distribution_cost = pred_mass.new_zeros(num_distributions)
    for start in range(0, num_distributions, chunk_size):
        end = min(start + chunk_size, num_distributions)
        a = pred_mass[start:end]
        b = target_mass[start:end]
        u = torch.ones_like(a) / num_patches
        v = torch.ones_like(b) / num_patches

        for _ in range(sinkhorn_iters):
            kv = torch.matmul(v, kernel.transpose(0, 1)).clamp_min(1.0e-8)
            u = a / kv
            ktu = torch.matmul(u, kernel).clamp_min(1.0e-8)
            v = b / ktu

        transport_plan = u.unsqueeze(-1) * kernel.unsqueeze(0) * v.unsqueeze(-2)
        per_distribution_cost[start:end] = (transport_plan * cost.unsqueeze(0)).sum(dim=(1, 2))

    return per_distribution_cost.view(*leading_shape)

def _sample_context_target_indices(total_frames, device, strategy="extrapolation", target_length=4):
    if total_frames < 2:
        raise ValueError(f"Need at least 2 frames to sample context/target, got {total_frames}")
    if target_length <= 0 or target_length >= total_frames:
        raise ValueError(f"target_length must be in [1, total_frames - 1], got {target_length} for {total_frames} frames")

    if strategy == "interpolation":
        if total_frames < target_length + 2:
            raise ValueError(
                f"Interpolation sampling requires at least {target_length + 2} frames, got {total_frames}"
            )
        min_start = 1
        max_start = total_frames - target_length - 1
        start = torch.randint(min_start, max_start + 1, (1,), device=device).item()
        target_indices = torch.arange(start, start + target_length, device=device, dtype=torch.long)
    elif strategy == "extrapolation":
        if torch.randint(0, 2, (1,), device=device).item() == 0:
            target_indices = torch.arange(0, target_length, device=device, dtype=torch.long)
        else:
            target_indices = torch.arange(total_frames - target_length, total_frames, device=device, dtype=torch.long)
    elif strategy == "front_extrapolation":
        target_indices = torch.arange(0, target_length, device=device, dtype=torch.long)
    elif strategy == "back_extrapolation":
        target_indices = torch.arange(total_frames - target_length, total_frames, device=device, dtype=torch.long)
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    target_mask = torch.zeros(total_frames, device=device, dtype=torch.bool)
    target_mask[target_indices] = True
    context_indices = torch.arange(total_frames, device=device, dtype=torch.long)[~target_mask]
    return context_indices, target_indices


def _compute_group_losses(
    sample_mask,
    pred_recon,
    target_recon,
    pred_mask_probs,
    target_mask_probs,
    spatial_size,
    sinkhorn_iters,
    sinkhorn_entropy_reg,
    recon_loss_weight,
    transport_loss_weight,
    mask_l1_loss_weight,
):
    if not sample_mask.any():
        return None

    group_recon = _recon_l1_loss(
        pred_recon[sample_mask],
        target_recon[sample_mask],
    )
    group_transport = _sinkhorn_mask_transport_loss(
        pred_mask_probs[sample_mask],
        target_mask_probs[sample_mask],
        spatial_size=spatial_size,
        sinkhorn_iters=sinkhorn_iters,
        entropy_reg=sinkhorn_entropy_reg,
    )
    group_mask_l1 = _mask_l1_loss(
        pred_mask_probs[sample_mask],
        target_mask_probs[sample_mask],
    )

    group_total = (
        recon_loss_weight * group_recon
        + transport_loss_weight * group_transport
        + mask_l1_loss_weight * group_mask_l1
    )
    return {
        "loss": float(group_total),
        "recon": float(group_recon),
        "transport": float(group_transport),
        "mask_l1": float(group_mask_l1),
    }


def _make_rollout_pca_grid(
    clips,
    target_reps,
    pred_reps,
    patch_size,
    max_vis_frames=4,
    aux_reps=None,
    pca_source_reps=None,
    pred_start_idx=1,
    pred_frame_indices=None,
):
    num_gt_frames = min(max_vis_frames, clips.size(2), target_reps.size(1))
    if pred_frame_indices is None:
        pred_start_idx = min(max(pred_start_idx, 0), num_gt_frames)
        num_pred_frames = min(max(0, num_gt_frames - pred_start_idx), pred_reps.size(1))
    else:
        pred_frame_indices = [idx for idx in pred_frame_indices if 0 <= idx < num_gt_frames]
        num_pred_frames = min(len(pred_frame_indices), pred_reps.size(1))
    if num_gt_frames <= 0:
        return None

    orig_imgs = [_unnormalize_frame(clips[0, :, t].detach().float()) for t in range(num_gt_frames)]
    gt_feats = [target_reps[0, t].detach().float() for t in range(num_gt_frames)]
    pred_feats = [pred_reps[0, t].detach().float() for t in range(num_pred_frames)]
    aux_feats = None
    if aux_reps is not None:
        aux_frames = min(num_gt_frames, aux_reps.size(1))
        aux_feats = [aux_reps[0, t].detach().float() for t in range(aux_frames)]

    pca_feats = list(gt_feats)
    if aux_feats is not None:
        pca_feats.extend(aux_feats)
    pca_feats.extend(pred_feats)
    if pca_source_reps is not None:
        pca_source = pca_source_reps[0].detach().float()
        pca_feats.append(pca_source.reshape(-1, pca_source.size(-1)))
    gt_all = torch.cat(pca_feats, dim=0)
    pca_mean = gt_all.mean(dim=0, keepdim=True)
    gt_centered = gt_all - pca_mean
    _, _, pca_basis = torch.pca_lowrank(gt_centered, q=3)
    gt_proj = gt_centered @ pca_basis
    pca_min = gt_proj.amin(dim=0, keepdim=True)
    pca_max = gt_proj.amax(dim=0, keepdim=True)

    gt_imgs = [
        _feature_to_pca_image(feat, pca_mean, pca_basis, pca_min, pca_max, patch_size, pred_border=False)
        for feat in gt_feats
    ]
    pred_imgs = [
        _feature_to_pca_image(feat, pca_mean, pca_basis, pca_min, pca_max, patch_size, pred_border=True)
        for feat in pred_feats
    ]

    blank = torch.ones_like(gt_imgs[0])
    pred_row = [blank.clone() for _ in range(num_gt_frames)]
    if pred_frame_indices is None:
        for idx, pred_img in enumerate(pred_imgs, start=pred_start_idx):
            if idx < num_gt_frames:
                pred_row[idx] = pred_img
    else:
        for idx, pred_img in zip(pred_frame_indices[:num_pred_frames], pred_imgs[:num_pred_frames]):
            pred_row[idx] = pred_img

    rows = [
        make_grid(orig_imgs, nrow=num_gt_frames, padding=4, pad_value=1.0),
        make_grid(gt_imgs, nrow=num_gt_frames, padding=4, pad_value=1.0),
    ]
    if aux_feats is not None:
        aux_imgs = [
            _feature_to_pca_image(feat, pca_mean, pca_basis, pca_min, pca_max, patch_size, pred_border=False)
            for feat in aux_feats
        ]
        while len(aux_imgs) < num_gt_frames:
            aux_imgs.append(blank.clone())
        rows.append(make_grid(aux_imgs[:num_gt_frames], nrow=num_gt_frames, padding=4, pad_value=1.0))
    rows.append(make_grid(pred_row[:num_gt_frames], nrow=num_gt_frames, padding=4, pad_value=1.0))
    return torch.cat(rows, dim=1)


def _make_multi_pred_pca_grid(
    clips,
    target_reps,
    pred_payloads,
    patch_size,
    max_vis_frames=4,
    aux_reps=None,
    pca_source_reps=None,
):
    num_gt_frames = min(max_vis_frames, clips.size(2), target_reps.size(1))
    if num_gt_frames <= 0 or not pred_payloads:
        return None

    orig_imgs = [_unnormalize_frame(clips[0, :, t].detach().float()) for t in range(num_gt_frames)]
    gt_feats = [target_reps[0, t].detach().float() for t in range(num_gt_frames)]
    aux_feats = None
    if aux_reps is not None:
        aux_frames = min(num_gt_frames, aux_reps.size(1))
        aux_feats = [aux_reps[0, t].detach().float() for t in range(aux_frames)]

    pca_feats = list(gt_feats)
    if aux_feats is not None:
        pca_feats.extend(aux_feats)
    for pred_reps, pred_frame_indices in pred_payloads:
        pred_frame_indices = [idx for idx in pred_frame_indices if 0 <= idx < num_gt_frames]
        num_pred_frames = min(len(pred_frame_indices), pred_reps.size(1))
        pca_feats.extend([pred_reps[0, t].detach().float() for t in range(num_pred_frames)])
    if pca_source_reps is not None:
        pca_source = pca_source_reps[0].detach().float()
        pca_feats.append(pca_source.reshape(-1, pca_source.size(-1)))
    gt_all = torch.cat(pca_feats, dim=0)
    pca_mean = gt_all.mean(dim=0, keepdim=True)
    gt_centered = gt_all - pca_mean
    _, _, pca_basis = torch.pca_lowrank(gt_centered, q=3)
    gt_proj = gt_centered @ pca_basis
    pca_min = gt_proj.amin(dim=0, keepdim=True)
    pca_max = gt_proj.amax(dim=0, keepdim=True)

    gt_imgs = [
        _feature_to_pca_image(feat, pca_mean, pca_basis, pca_min, pca_max, patch_size, pred_border=False)
        for feat in gt_feats
    ]
    blank = torch.ones_like(gt_imgs[0])

    rows = [
        make_grid(orig_imgs, nrow=num_gt_frames, padding=4, pad_value=1.0),
        make_grid(gt_imgs, nrow=num_gt_frames, padding=4, pad_value=1.0),
    ]
    if aux_feats is not None:
        aux_imgs = [
            _feature_to_pca_image(feat, pca_mean, pca_basis, pca_min, pca_max, patch_size, pred_border=False)
            for feat in aux_feats
        ]
        while len(aux_imgs) < num_gt_frames:
            aux_imgs.append(blank.clone())
        rows.append(make_grid(aux_imgs[:num_gt_frames], nrow=num_gt_frames, padding=4, pad_value=1.0))

    for pred_reps, pred_frame_indices in pred_payloads:
        pred_frame_indices = [idx for idx in pred_frame_indices if 0 <= idx < num_gt_frames]
        num_pred_frames = min(len(pred_frame_indices), pred_reps.size(1))
        pred_feats = [pred_reps[0, t].detach().float() for t in range(num_pred_frames)]
        pred_imgs = [
            _feature_to_pca_image(feat, pca_mean, pca_basis, pca_min, pca_max, patch_size, pred_border=True)
            for feat in pred_feats
        ]
        pred_row = [blank.clone() for _ in range(num_gt_frames)]
        for idx, pred_img in zip(pred_frame_indices[:num_pred_frames], pred_imgs):
            pred_row[idx] = pred_img
        rows.append(make_grid(pred_row, nrow=num_gt_frames, padding=4, pad_value=1.0))

    return torch.cat(rows, dim=1)


def _flatten_multiview_batch(clips, actions, states, extrinsics):
    if clips.dim() == 5:
        return clips, actions, states, extrinsics, 1
    if clips.dim() != 6:
        raise ValueError(f"Expected clips to have 5 or 6 dims, got shape {tuple(clips.shape)}")    

    batch_size, num_views, channels, num_frames, height, width = clips.shape
    
    clips = clips.reshape(batch_size * num_views, channels, num_frames, height, width)
    actions = actions.reshape(batch_size * num_views, actions.size(-2), actions.size(-1))
    states = states.reshape(batch_size * num_views, states.size(-2), states.size(-1))
    extrinsics = extrinsics.reshape(batch_size * num_views, extrinsics.size(-2), extrinsics.size(-1))
    return clips, actions, states, extrinsics, num_views


def _reshape_latent_sequence_for_extractor(student_feats, batch_size, num_views, total_frames, spatial_size):
    feat_dim = student_feats.size(-1)
    student_6d = student_feats.view(batch_size, num_views, total_frames, spatial_size, spatial_size, feat_dim)
    joint_sequence = student_6d.permute(0, 1, 2, 3, 4, 5).reshape(
        batch_size,
        num_views * total_frames,
        spatial_size,
        spatial_size,
        feat_dim,
    )
    return joint_sequence, feat_dim


def _reshape_single_view_sequence_for_extractor(student_feats, batch_size, total_frames, spatial_size):
    feat_dim = student_feats.size(-1)
    return student_feats.view(batch_size, total_frames, spatial_size, spatial_size, feat_dim), feat_dim


def _split_joint_view_time_tensor(tensor, batch_size, num_views, total_frames):
    return tensor.view(batch_size, num_views, total_frames, *tensor.shape[2:])


def _get_predictor_modules(predictor):
    if isinstance(predictor, DistributedDataParallel):
        predictor = predictor.module
    if isinstance(predictor, torch.nn.ModuleDict):
        return predictor
    return torch.nn.ModuleDict({"default": predictor})


def _find_wrist_view_indices(view_names):
    return [idx for idx, name in enumerate(view_names) if "wrist" in name.lower()]


def _build_view_index_tensor(batch_size, num_views, device):
    return torch.arange(num_views, device=device).unsqueeze(0).expand(batch_size, -1).reshape(-1)


def _build_copied_warped_masks(context_masks, num_queries):
    context_mask_probs = F.softmax(context_masks, dim=2)
    return context_mask_probs.unsqueeze(1).expand(-1, num_queries, -1, -1, -1).contiguous()


def _run_view_routed_predictor(
    predictor,
    view_indices,
    wrist_view_indices,
    obj_latents,
    context_masks,
    states,
    next_states,
    extrinsics=None,
    next_extrinsics=None,
):
    predictor_modules = _get_predictor_modules(predictor)
    wrist_view_set = set(wrist_view_indices)
    if not wrist_view_set or "wrist" not in predictor_modules:
        predictor_module = predictor_modules["default"]
        if not getattr(predictor_module, "use_warp_concat", False) or not wrist_view_set:
            return predictor_module(
                obj_latents,
                context_masks,
                states,
                next_states,
                extrinsics,
                next_extrinsics,
            )

        wrist_mask = torch.zeros_like(view_indices, dtype=torch.bool)
        for wrist_idx in wrist_view_set:
            wrist_mask |= view_indices == wrist_idx
        default_mask = ~wrist_mask
        outputs = None
        if default_mask.any():
            default_output = predictor_module(
                obj_latents[default_mask],
                context_masks[default_mask],
                states[default_mask],
                next_states[default_mask],
                extrinsics[default_mask] if extrinsics is not None else None,
                next_extrinsics[default_mask] if next_extrinsics is not None else None,
                external_warped_mask_probs=_build_copied_warped_masks(
                    context_masks[default_mask],
                    next_states.size(1),
                ),
            )
            outputs = default_output.new_empty(
                obj_latents.size(0),
                default_output.size(1),
                default_output.size(2),
                default_output.size(3),
            )
            outputs[default_mask] = default_output
        if wrist_mask.any():
            wrist_output = predictor_module(
                obj_latents[wrist_mask],
                context_masks[wrist_mask],
                states[wrist_mask],
                next_states[wrist_mask],
                extrinsics[wrist_mask] if extrinsics is not None else None,
                next_extrinsics[wrist_mask] if next_extrinsics is not None else None,
            )
            if outputs is None:
                outputs = wrist_output.new_empty(
                    obj_latents.size(0),
                    wrist_output.size(1),
                    wrist_output.size(2),
                    wrist_output.size(3),
                )
            outputs[wrist_mask] = wrist_output
        return outputs

    wrist_mask = torch.zeros_like(view_indices, dtype=torch.bool)
    for wrist_idx in wrist_view_set:
        wrist_mask |= view_indices == wrist_idx

    outputs = None
    default_mask = ~wrist_mask
    if default_mask.any():
        default_warped = None
        if getattr(predictor_modules["default"], "use_warp_concat", False):
            default_warped = _build_copied_warped_masks(context_masks[default_mask], next_states.size(1))
        default_output = predictor_modules["default"](
            obj_latents[default_mask],
            context_masks[default_mask],
            states[default_mask],
            next_states[default_mask],
            extrinsics[default_mask] if extrinsics is not None else None,
            next_extrinsics[default_mask] if next_extrinsics is not None else None,
            external_warped_mask_probs=default_warped,
        )
        outputs = default_output.new_empty(
            obj_latents.size(0),
            default_output.size(1),
            default_output.size(2),
            default_output.size(3),
        )
        outputs[default_mask] = default_output

    if wrist_mask.any():
        wrist_output = predictor_modules["wrist"](
            obj_latents[wrist_mask],
            context_masks[wrist_mask],
            states[wrist_mask],
            next_states[wrist_mask],
            extrinsics[wrist_mask] if extrinsics is not None else None,
            next_extrinsics[wrist_mask] if next_extrinsics is not None else None,
        )
        if outputs is None:
            outputs = wrist_output.new_empty(
                obj_latents.size(0),
                wrist_output.size(1),
                wrist_output.size(2),
                wrist_output.size(3),
            )
        outputs[wrist_mask] = wrist_output

    return outputs


def _apply_gpu_video_transform_batch(clips, gpu_transform):
    if gpu_transform is None:
        return clips
    if clips.dim() == 5:
        return torch.stack([gpu_transform(clip) for clip in clips], dim=0)
    if clips.dim() == 6:
        bsz, nviews = clips.shape[:2]
        flat = clips.reshape(bsz * nviews, *clips.shape[2:])
        flat = torch.stack([gpu_transform(clip) for clip in flat], dim=0)
        return flat.reshape(bsz, nviews, *flat.shape[1:])
    raise ValueError(f"Expected clips to have 5 or 6 dims, got shape {tuple(clips.shape)}")


def _resize_flat_frames(frames, size):
    if frames.shape[-2:] == (size, size):
        return frames
    return F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)


def _transformer_block_index(param_name):
    parts = param_name.split(".")
    if "blocks" not in parts:
        return None
    block_pos = parts.index("blocks")
    for part in parts[block_pos + 1 :]:
        if part.isdigit():
            return int(part)
    return None


def _set_trainable_last_n_transformer_blocks(module, last_n_layers):
    block_indices = [
        block_idx
        for name, _ in module.named_parameters()
        for block_idx in [_transformer_block_index(name)]
        if block_idx is not None
    ]
    if not block_indices:
        raise RuntimeError("No transformer block parameters found in the student encoder.")
    min_trainable_block = max(block_indices) + 1 - int(last_n_layers)
    trainable_params = 0
    total_params = 0
    for name, param in module.named_parameters():
        block_idx = _transformer_block_index(name)
        param.requires_grad = block_idx is not None and block_idx >= min_trainable_block
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if trainable_params == 0:
        raise RuntimeError(
            "No trainable student encoder parameters were selected. "
            "Check the encoder parameter names and block-freezing policy."
        )
    return trainable_params, total_params, min_trainable_block


def _load_slot_checkpoint(slot_ckpt_path, extractor, student, predictor, load_predictor_shared=True, log_enabled=True):
    if log_enabled:
        logger.info(f"Loading SLOT pretrain checkpoint from {slot_ckpt_path}")
    ckpt = torch.load(slot_ckpt_path, map_location="cpu")

    if "student_state" in ckpt:
        msg = student.load_state_dict(ckpt["student_state"], strict=False)
        if log_enabled:
            logger.info(f"loaded slot student with msg: {msg}")

    if "extractor_state" in ckpt:
        msg = extractor.load_state_dict(ckpt["extractor_state"], strict=False)
        if log_enabled:
            logger.info(f"loaded slot extractor with msg: {msg}")

    if load_predictor_shared and "mask_predictor_state" in ckpt:
        predictor_modules = _get_predictor_modules(predictor)
        legacy_recon_key_map = {
            "obj_latent_reduce.": "recon_branch.obj_latent_reduce.",
            "predictor_embed.": "recon_branch.predictor_embed.",
            "state_encoder.": "recon_branch.state_encoder.",
            "extrinsics_encoder.": "recon_branch.extrinsics_encoder.",
            "obj_encoder.": "recon_branch.obj_encoder.",
            "predictor_blocks.": "recon_branch.blocks.",
            "predictor_norm.": "recon_branch.norm.",
        }
        for predictor_name, predictor_module in predictor_modules.items():
            predictor_state = predictor_module.state_dict()
            shared_state = {}
            for key, value in ckpt["mask_predictor_state"].items():
                candidate_keys = [key]
                if key == "query_tokens":
                    candidate_keys.append("recon_branch.query_tokens")
                for old_prefix, new_prefix in legacy_recon_key_map.items():
                    if key.startswith(old_prefix):
                        candidate_keys.append(new_prefix + key[len(old_prefix) :])
                for candidate_key in candidate_keys:
                    if candidate_key in predictor_state and predictor_state[candidate_key].shape == value.shape:
                        shared_state[candidate_key] = value
                        break
            msg = predictor_module.load_state_dict(shared_state, strict=False)
            if log_enabled:
                logger.info(f"loaded shared mask predictor weights into {predictor_name} with msg: {msg}")

    return ckpt


def main(args, resume_preempt=False):
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    cfgs_model = args.get("model")
    cfgs_data = args.get("data")
    cfgs_data_aug = args.get("data_aug")
    cfgs_loss = args.get("loss")
    cfgs_opt = args.get("optimization")

    r_file = cfgs_meta.get("resume_checkpoint", None)
    slot_ckpt_path = cfgs_meta.get("slot_pretrain_checkpoint")
    student_checkpoint_path = cfgs_meta.get(
        "student_pretrain_checkpoint",
        cfgs_meta.get("vjepa_checkpoint_path", cfgs_meta.get("student_checkpoint_path")),
    )
    teacher_checkpoint_path = cfgs_meta.get("teacher_pretrain_checkpoint", cfgs_meta.get("pretrain_checkpoint"))
    vjepa_checkpoint_path = student_checkpoint_path
    load_predictor_shared = cfgs_meta.get("load_predictor_shared", True)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype", "bfloat16")
    tb_log_freq = cfgs_meta.get("tb_log_freq", log_freq)
    tb_vis_freq = cfgs_meta.get("tb_vis_freq", 100)
    tb_vis_frames = cfgs_meta.get("tb_vis_frames", None)
    overfit_one_batch = cfgs_meta.get("overfit_one_batch", False)

    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    slot_feature_dim = cfgs_model.get("slot_feature_dim", cfgs_model.get("feature_dim", 196))
    predictor_embed_dim = cfgs_model.get("predictor_embed_dim", cfgs_model.get("pred_embed_dim", 192))
    predictor_depth = cfgs_model.get("predictor_depth", cfgs_model.get("pred_depth", 12))
    predictor_num_heads = cfgs_model.get("predictor_num_heads", cfgs_model.get("pred_num_heads", 3))
    object_branch_scales = cfgs_model.get("object_branch_scales", None)
    mlp_ratio = cfgs_model.get("mlp_ratio", 4.0)
    drop_rate = cfgs_model.get("drop_rate", 0.0)
    embed_dim = cfgs_model.get("embed_dim", 768)
    num_objects = cfgs_model.get("num_objects", cfgs_model.get("global_k", 24))
    concat_mask_input = cfgs_model.get("concat_mask_input", True)
    state_dim = cfgs_model.get("state_dim", 7)
    use_extrinsics = cfgs_model.get("use_extrinsics", True)
    use_wrist_warp_concat = cfgs_model.get("use_wrist_warp_concat", True)
    share_obj_latents_across_views = bool(cfgs_model.get("share_obj_latents_across_views", False))
    warp_hfov_deg = float(cfgs_model.get("warp_hfov_deg", 60.0))
    warp_extrinsics_format = cfgs_model.get("warp_extrinsics_format", "c2w")
    warp_rotation_mode = cfgs_model.get("warp_rotation_mode", "euler_xyz")
    warp_depth_bins = cfgs_model.get("warp_depth_bins", [0.15, 0.25, 0.4, 0.7, 1.2, 2.0])
    extrinsics_dim = cfgs_model.get("extrinsics_dim", 6)
    freeze_extractor = cfgs_model.get("freeze_extractor", True)
    freeze_student = cfgs_model.get("freeze_student", True)
    student_encoder_impl = cfgs_model.get("student_encoder_impl", "slot_jepa")
    student_model_name = cfgs_model.get(
        "student_model_name",
        "vit_base",
    )
    student_checkpoint_key = cfgs_model.get(
        "student_encoder_checkpoint_key",
        "ema_encoder",
    )
    student_patch_size = int(cfgs_model.get("student_encoder_patch_size", cfgs_data.get("patch_size", 16)))
    student_num_register_tokens = int(
        cfgs_model.get("student_encoder_num_register_tokens", 0)
    )
    student_train_last_n_layers = int(cfgs_model.get("student_train_last_n_layers", 6))
    teacher_encoder_impl = cfgs_model.get("teacher_encoder_impl", cfgs_model.get("encoder_impl", "dinov2_vitb14_reg"))
    teacher_model_name = cfgs_model.get("teacher_model_name", cfgs_model.get("model_name", "dinov2_vitb14_reg"))
    if teacher_encoder_impl in ("slot_jepa", "vjepa", "vjepa2") and str(teacher_model_name).startswith("dinov2"):
        teacher_model_name = "vit_base"
    if student_encoder_impl in ("slot_jepa", "vjepa", "vjepa2") and str(student_model_name).startswith("dinov2"):
        student_model_name = "vit_base"
    teacher_checkpoint_key = cfgs_model.get("teacher_encoder_checkpoint_key", cfgs_model.get("encoder_checkpoint_key", None))
    if teacher_encoder_impl in ("slot_jepa", "vjepa", "vjepa2") and teacher_checkpoint_key is None:
        teacher_checkpoint_key = "ema_encoder"
    if student_encoder_impl in ("slot_jepa", "vjepa", "vjepa2") and student_checkpoint_key is None:
        student_checkpoint_key = "ema_encoder"
    teacher_patch_size = int(cfgs_model.get("teacher_encoder_patch_size", cfgs_model.get("encoder_patch_size", 14)))
    teacher_num_register_tokens = int(cfgs_model.get("teacher_encoder_num_register_tokens", cfgs_model.get("encoder_num_register_tokens", 4)))
    teacher_crop_size = int(cfgs_model.get("teacher_crop_size", 336 if teacher_patch_size == 14 else cfgs_data.get("crop_size", 384)))
    encoder_impl = student_encoder_impl
    encoder_patch_size = student_patch_size
    vjepa_model_name = student_model_name
    vjepa_checkpoint_key = student_checkpoint_key
    encoder_num_register_tokens = student_num_register_tokens
    vjepa_img_as_video_nframes = cfgs_model.get("vjepa_img_as_video_nframes", 2)
    recon_loss_type = cfgs_loss.get("recon_loss_type", "l1")
    recon_loss_weight = float(cfgs_loss.get("recon_loss_weight", 0.1))
    transport_loss_weight = float(cfgs_loss.get("transport_loss_weight", 1.0))
    mask_l1_loss_weight = float(cfgs_loss.get("mask_l1_loss_weight", 1.0))
    sinkhorn_iters = int(cfgs_loss.get("sinkhorn_iters", 5))
    sinkhorn_entropy_reg = float(cfgs_loss.get("sinkhorn_entropy_reg", 0.05))
    auto_steps = cfgs_loss.get("auto_steps", None)
    sampling_strategies = cfgs_loss.get("sampling_strategies")
    if sampling_strategies is None:
        legacy_sampling_strategy = cfgs_loss.get("sampling_strategy")
        if legacy_sampling_strategy is not None:
            sampling_strategies = [legacy_sampling_strategy]
        else:
            sampling_strategies = ["interpolation", "extrapolation"]
    if isinstance(sampling_strategies, str):
        sampling_strategies = [sampling_strategies]
    expanded_sampling_strategies = []
    for strategy in sampling_strategies:
        strategy = str(strategy)
        if strategy == "extrapolation":
            expanded_sampling_strategies.extend(["front_extrapolation", "back_extrapolation"])
        else:
            expanded_sampling_strategies.append(strategy)
    sampling_strategies = list(dict.fromkeys(expanded_sampling_strategies))
    target_num_frames = int(cfgs_loss.get("target_num_frames", 4))

    dataset_path = cfgs_data.get("datasets", [None])[0]
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    if tb_vis_frames is None:
        tb_vis_frames = max_num_frames
    if auto_steps is None:
        auto_steps = max_num_frames - 1
    auto_steps = min(auto_steps, max_num_frames - 1)
    dataset_format = cfgs_data.get("dataset_format", "raw_droid")
    camera_frame = cfgs_data.get("camera_frame", False)
    camera_views = cfgs_data.get("camera_views", ["left_mp4_path"])
    multiview_training = cfgs_data.get("multiview_training", False)
    stereo_view = cfgs_data.get("stereo_view", False)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 384)
    patch_size = cfgs_data.get("patch_size", encoder_patch_size)
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    horizontal_flip = cfgs_data_aug.get("horizontal_flip", False)
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay", 1.0e-4))
    final_wd = float(cfgs_opt.get("final_weight_decay", wd))
    num_epochs = cfgs_opt.get("epochs")
    anneal = cfgs_opt.get("anneal", num_epochs)
    warmup = cfgs_opt.get("warmup", 0)
    start_lr = float(cfgs_opt.get("start_lr", cfgs_opt.get("lr")))
    lr = float(cfgs_opt.get("lr"))
    final_lr = float(cfgs_opt.get("final_lr", lr * 0.1))
    enc_lr_scale = float(cfgs_opt.get("enc_lr_scale", 1.0))
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = float(cfgs_opt.get("eps", 1.0e-8))

    if slot_ckpt_path is None:
        raise ValueError("Expected meta.slot_pretrain_checkpoint")
    if vjepa_checkpoint_path is None and encoder_impl == "slot_jepa":
        raise ValueError("Expected meta.vjepa_checkpoint_path for the V-JEPA encoder")

    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    world_size, rank = init_distributed()
    is_main_process = rank == 0
    if is_main_process:
        logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    resume_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        resume_path = None

    writer = SummaryWriter(log_dir=os.path.join(folder, "tensorboard")) if is_main_process else None
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
        mode="+a",
    )

    teacher_spatial_size = teacher_crop_size // teacher_patch_size
    if crop_size // patch_size != teacher_spatial_size:
        raise ValueError(
            "Student and teacher patch grids must match: "
            f"{crop_size}//{patch_size} != {teacher_crop_size}//{teacher_patch_size}"
        )

    teacher_auto_download = teacher_encoder_impl not in ("slot_jepa", "vjepa", "vjepa2") and teacher_checkpoint_path is None
    student_auto_download = student_encoder_impl not in ("slot_jepa", "vjepa", "vjepa2") and vjepa_checkpoint_path is None
    encoder_auto_download = teacher_auto_download or student_auto_download
    if encoder_auto_download and dist.is_available() and dist.is_initialized() and rank != 0:
        dist.barrier()

    teacher = init_encoder(
        encoder_impl=teacher_encoder_impl,
        resolution=teacher_crop_size,
        checkpoint=teacher_checkpoint_path,
        model_name=teacher_model_name,
        checkpoint_key=teacher_checkpoint_key,
        patch_size=teacher_patch_size,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
        img_as_video_nframes=vjepa_img_as_video_nframes,
        num_register_tokens=teacher_num_register_tokens,
    ).to(device)
    if encoder_auto_download and dist.is_available() and dist.is_initialized() and rank == 0:
        dist.barrier()
    student = init_encoder(
        encoder_impl=student_encoder_impl,
        resolution=crop_size,
        checkpoint=vjepa_checkpoint_path,
        model_name=student_model_name,
        checkpoint_key=student_checkpoint_key,
        patch_size=student_patch_size,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
        img_as_video_nframes=vjepa_img_as_video_nframes,
        num_register_tokens=student_num_register_tokens,
    ).to(device)

    spatial_size = crop_size // patch_size
    extractor = TwoStageExtractor(
        window_size=cfgs_model.get("window_size", [2, 4]),
        slots_per_window=cfgs_model.get("slots_per_window", 4),
        global_k=num_objects,
        D_in=embed_dim,
        mask_temp=cfgs_model.get("mask_temp", 0.05),
        L_max=cfgs_model.get("L_max", num_objects),
        use_decoder_checkpoint=cfgs_model.get("use_decoder_checkpoint", False),
    ).to(device)

    predictor = ActionConditionalMaskPredictor(
        embed_dim=embed_dim,
        feature_dim=slot_feature_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=predictor_depth,
        num_heads=predictor_num_heads,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        num_context_frames=max_num_frames - 1,
        spatial_size=spatial_size,
        num_objects=num_objects,
        state_dim=state_dim,
        use_extrinsics=use_extrinsics,
        extrinsics_dim=extrinsics_dim,
        num_obj_tokens=cfgs_model.get("num_obj_tokens", 1),
        concat_mask_input=concat_mask_input,
        use_warp_concat=True,
        warp_hfov_deg=warp_hfov_deg,
        warp_extrinsics_format=warp_extrinsics_format,
        warp_rotation_mode=warp_rotation_mode,
        warp_depth_bins=warp_depth_bins,
        patch_size=patch_size,
        use_checkpoint=use_activation_checkpointing,
        object_branch_scales=object_branch_scales,
    ).to(device)

    _load_slot_checkpoint(
        slot_ckpt_path,
        extractor,
        student,
        predictor,
        load_predictor_shared=load_predictor_shared,
        log_enabled=is_main_process,
    )

    teacher.eval()
    modules_to_freeze = [teacher]
    if freeze_student:
        student.eval()
        modules_to_freeze.append(student)
    else:
        trainable_student_params, total_student_params, min_trainable_block = _set_trainable_last_n_transformer_blocks(
            student,
            student_train_last_n_layers,
        )
        if is_main_process:
            logger.info(
                "Student trainable params: %.2fM / %.2fM; train blocks >= %d",
                trainable_student_params / 1e6,
                total_student_params / 1e6,
                min_trainable_block,
            )
    if freeze_extractor:
        extractor.eval()
        modules_to_freeze.append(extractor)
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

    if compile_model:
        torch._dynamo.config.optimize_ddp = False
        predictor.compile()

    if world_size > 1:
        predictor = DistributedDataParallel(predictor, device_ids=[device.index], output_device=device.index, static_graph=False, find_unused_parameters=False)

    video_collator = torch.utils.data.default_collate
    cpu_transform = make_transforms(
        random_horizontal_flip=horizontal_flip,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )
    gpu_transform = None
    transform = cpu_transform
    if dataset_format == "exported_frames":
        transform = None
        gpu_transform = make_gpu_transforms(
            random_horizontal_flip=horizontal_flip,
            random_resize_aspect_ratio=ar_range,
            random_resize_scale=rr_scale,
            reprob=reprob,
            auto_augment=use_aa,
            motion_shift=motion_shift,
            crop_size=crop_size,
        )
    data_init_fn = init_frame_data if dataset_format == "exported_frames" else init_raw_data
    unsupervised_loader, unsupervised_sampler = data_init_fn(
        data_path=dataset_path,
        batch_size=batch_size,
        frames_per_clip=max_num_frames,
        tubelet_size=1,
        fps=fps,
        camera_views=camera_views,
        camera_frame=camera_frame,
        stereo_view=stereo_view,
        transform=transform,
        collator=video_collator,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        rank=rank,
    )
    dataset_view_names = getattr(unsupervised_loader.dataset, "selected_views", list(camera_views))
    wrist_view_indices = _find_wrist_view_indices(dataset_view_names)
    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    overfit_iterations = ipe if overfit_one_batch else None
    if overfit_one_batch:
        ipe = overfit_iterations
        num_epochs = 1
        anneal = 1
        warmup = min(warmup, 1)
        tb_vis_freq = 100
    if is_main_process:
        logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    class _EmptyEncoder(torch.nn.Module):
        def named_parameters(self, prefix="", recurse=True):
            return iter(())

    target_predictor_for_opt = predictor.module if isinstance(predictor, DistributedDataParallel) else predictor
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=_EmptyEncoder(),
        predictor=target_predictor_for_opt,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        enc_lr_scale=enc_lr_scale,
        iterations_per_epoch=ipe,
        anneal=anneal,
        warmup=warmup,
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )
    start_epoch = 0
    if resume_path is not None and os.path.exists(resume_path):
        if is_main_process:
            logger.info(f"Loading SLOT-AC checkpoint from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")
        pred_state = ckpt.get("slot_ac_predictor_state", ckpt.get("predictor_state"))
        if pred_state is not None:
            target_predictor = predictor.module if isinstance(predictor, DistributedDataParallel) else predictor
            if any(key.startswith("default.") or key.startswith("wrist.") for key in pred_state.keys()):
                remapped_state = {}
                for key, value in pred_state.items():
                    if key.startswith("default."):
                        remapped_state[key[len("default."):]] = value
                    elif key.startswith("wrist."):
                        continue
                    else:
                        remapped_state[key] = value
                msg = target_predictor.load_state_dict(remapped_state, strict=False)
                if is_main_process:
                    logger.info(f"loaded merged slot-ac predictor with msg: {msg}")
            else:
                msg = target_predictor.load_state_dict(pred_state, strict=False)
                if is_main_process:
                    logger.info(f"loaded slot-ac predictor with msg: {msg}")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt:
            _scheduler_load_state_dict(scheduler, ckpt["scheduler"])
        if "wd_scheduler" in ckpt:
            _scheduler_load_state_dict(wd_scheduler, ckpt["wd_scheduler"])
        start_epoch = int(ckpt.get("epoch", 0))
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        start_epoch_tensor = torch.tensor([start_epoch], device=device, dtype=torch.long)
        dist.broadcast(start_epoch_tensor, src=0)
        start_epoch = int(start_epoch_tensor.item())
    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        target_predictor = predictor.module if isinstance(predictor, DistributedDataParallel) else predictor
        save_dict = {
            "epoch": epoch,
            "slot_ac_predictor_state": target_predictor.state_dict(),
            "extractor_state": extractor.state_dict(),
            "student_state": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": _scheduler_state_dict(scheduler),
            "wd_scheduler": _scheduler_state_dict(wd_scheduler),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        torch.save(save_dict, path)

    loader = iter(unsupervised_loader)
    if skip_batches > 0:
        for _ in range(skip_batches):
            try:
                next(loader)
            except StopIteration:
                unsupervised_sampler.set_epoch(start_epoch)
                loader = iter(unsupervised_loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    fixed_overfit_sample = None

    for epoch in range(start_epoch, num_epochs):
        if is_main_process:
            logger.info("Epoch %d" % (epoch + 1))
        if hasattr(unsupervised_sampler, "set_epoch"):
            unsupervised_sampler.set_epoch(epoch)

        predictor.train()

        loss_meter = AverageMeter()
        recon_meter = AverageMeter()
        transport_meter = AverageMeter()
        mask_l1_meter = AverageMeter()
        entropy_meter = AverageMeter()
        strategy_loss_meters = {strategy: AverageMeter() for strategy in sampling_strategies}
        strategy_recon_meters = {strategy: AverageMeter() for strategy in sampling_strategies}
        strategy_transport_meters = {strategy: AverageMeter() for strategy in sampling_strategies}
        strategy_mask_l1_meters = {strategy: AverageMeter() for strategy in sampling_strategies}
        strategy_entropy_meters = {strategy: AverageMeter() for strategy in sampling_strategies}
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()
            if overfit_one_batch and fixed_overfit_sample is not None:
                sample = fixed_overfit_sample
            else:
                try:
                    sample = next(loader)
                except StopIteration:
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                    sample = next(loader)
                if overfit_one_batch and fixed_overfit_sample is None:
                    fixed_overfit_sample = sample

            def load_clips():
                clips = sample[0].to(device, non_blocking=True)
                actions = sample[1].to(device, dtype=torch.float32, non_blocking=True)
                states = sample[2].to(device, dtype=torch.float32, non_blocking=True)
                extrinsics = sample[3].to(device, dtype=torch.float32, non_blocking=True)
                clips = _apply_gpu_video_transform_batch(clips, gpu_transform)
                return clips, actions, states, extrinsics

            clips, actions, states, extrinsics = load_clips()
            clips_flat, actions_flat, states_flat, extrinsics_flat, batch_num_views = _flatten_multiview_batch(
                clips, actions, states, extrinsics
            )
            total_frames = clips_flat.size(2)
            sampled_indices = {
                strategy: _sample_context_target_indices(
                    total_frames,
                    device,
                    strategy=strategy,
                    target_length=target_num_frames,
                )
                for strategy in sampling_strategies
            }
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                gc.collect()

            vis_payload = None

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                extractor_context = torch.no_grad() if (freeze_student and freeze_extractor) else torch.enable_grad()
                with extractor_context:
                    batch_size_eff = clips.size(0) if clips.dim() == 6 else clips_flat.size(0)
                    flat_view_indices = _build_view_index_tensor(batch_size_eff, batch_num_views, device)
                    clips_frames = clips_flat.permute(0, 2, 1, 3, 4).flatten(0, 1)
                    student_feats = student(clips_frames)
                    if batch_num_views > 1 and share_obj_latents_across_views:
                        student_joint_5d, D = _reshape_latent_sequence_for_extractor(
                            student_feats,
                            batch_size=batch_size_eff,
                            num_views=batch_num_views,
                            total_frames=total_frames,
                            spatial_size=spatial_size,
                        )

                        class_state_all, obj_latents = extractor(
                            student_joint_5d,
                            spatial_dims=(spatial_size, spatial_size),
                            T_dim=batch_num_views * total_frames,
                        )
                        decoded_mask_logits_joint = class_state_all.get_layer(-1)["mask_logits"]
                        decoded_mask_logits = _split_joint_view_time_tensor(
                            decoded_mask_logits_joint,
                            batch_size=batch_size_eff,
                            num_views=batch_num_views,
                            total_frames=total_frames,
                        ).reshape(clips_flat.size(0), total_frames, decoded_mask_logits_joint.size(2), decoded_mask_logits_joint.size(3))
                        shared_obj_latents = obj_latents.unsqueeze(1).expand(-1, batch_num_views, -1, -1).reshape(
                            clips_flat.size(0),
                            obj_latents.size(1),
                            obj_latents.size(2),
                        )
                    else:
                        student_single_5d, D = _reshape_single_view_sequence_for_extractor(
                            student_feats,
                            batch_size=clips_flat.size(0),
                            total_frames=total_frames,
                            spatial_size=spatial_size,
                        )
                        class_state_all, shared_obj_latents = extractor(
                            student_single_5d,
                            spatial_dims=(spatial_size, spatial_size),
                            T_dim=total_frames,
                        )
                        decoded_mask_logits = class_state_all.get_layer(-1)["mask_logits"]
                    decoded_mask_probs = F.softmax(decoded_mask_logits, dim=2)
                    decoder_recon_all = torch.einsum(
                        "btkn,bkd->btnd", decoded_mask_probs.to(shared_obj_latents.dtype), shared_obj_latents
                    )

                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    wrist_sample_mask = torch.zeros_like(flat_view_indices, dtype=torch.bool)
                    for wrist_idx in wrist_view_indices:
                        wrist_sample_mask |= flat_view_indices == wrist_idx
                    non_wrist_sample_mask = ~wrist_sample_mask
                    strategy_metrics = {}
                    total_loss = None
                    total_recon_loss = 0.0
                    total_transport_loss = 0.0
                    total_mask_l1_loss = 0.0
                    total_pred_mask_entropy = 0.0
                    for strategy, (context_indices, target_indices) in sampled_indices.items():
                        context_mask_logits = decoded_mask_logits.index_select(1, context_indices).detach()
                        context_states = states_flat.index_select(1, context_indices)
                        target_states = states_flat.index_select(1, target_indices)
                        context_extrinsics = extrinsics_flat.index_select(1, context_indices) if use_extrinsics else None
                        target_extrinsics = extrinsics_flat.index_select(1, target_indices) if use_extrinsics else None
                        target_mask_probs = decoded_mask_probs.index_select(1, target_indices).detach()
                        target_recon = decoder_recon_all.index_select(1, target_indices).detach()

                        pred_logits = _run_view_routed_predictor(
                            predictor,
                            flat_view_indices,
                            wrist_view_indices,
                            shared_obj_latents.detach(),
                            context_mask_logits,
                            context_states,
                            target_states,
                            context_extrinsics,
                            target_extrinsics,
                        )
                        pred_mask_probs = F.softmax(pred_logits, dim=2)
                        pred_mask_entropy = (
                            -(pred_mask_probs.clamp_min(1.0e-8) * pred_mask_probs.clamp_min(1.0e-8).log()).sum(dim=2).mean()
                        )
                        pred_recon = torch.einsum(
                            "btkn,bkd->btnd", pred_mask_probs.to(shared_obj_latents.dtype), shared_obj_latents.detach()
                        )
                        recon_loss = _recon_l1_loss(pred_recon, target_recon)
                        transport_loss = _sinkhorn_mask_transport_loss(
                            pred_mask_probs,
                            target_mask_probs,
                            spatial_size=spatial_size,
                            sinkhorn_iters=sinkhorn_iters,
                            entropy_reg=sinkhorn_entropy_reg,
                        )
                        mask_l1_loss = _mask_l1_loss(pred_mask_probs, target_mask_probs)
                        branch_loss = (
                            recon_loss_weight * recon_loss
                            + transport_loss_weight * transport_loss
                            + mask_l1_loss_weight * mask_l1_loss
                        )
                        total_loss = branch_loss if total_loss is None else total_loss + branch_loss
                        total_recon_loss = total_recon_loss + recon_loss
                        total_transport_loss = total_transport_loss + transport_loss
                        total_mask_l1_loss = total_mask_l1_loss + mask_l1_loss
                        total_pred_mask_entropy = total_pred_mask_entropy + pred_mask_entropy
                        strategy_metrics[strategy] = {
                            "loss": float(branch_loss),
                            "recon_loss": float(recon_loss),
                            "transport_loss": float(transport_loss),
                            "mask_l1_loss": float(mask_l1_loss),
                            "pred_mask_entropy": float(pred_mask_entropy),
                            "pred_recon_cpu": pred_recon.detach().cpu(),
                            "target_indices_cpu": target_indices.detach().cpu(),
                            "non_wrist_group_losses": _compute_group_losses(
                                non_wrist_sample_mask,
                                pred_recon,
                                target_recon,
                                pred_mask_probs,
                                target_mask_probs,
                                spatial_size,
                                sinkhorn_iters,
                                sinkhorn_entropy_reg,
                                recon_loss_weight,
                                transport_loss_weight,
                                mask_l1_loss_weight,
                            ),
                            "wrist_group_losses": _compute_group_losses(
                                wrist_sample_mask,
                                pred_recon,
                                target_recon,
                                pred_mask_probs,
                                target_mask_probs,
                                spatial_size,
                                sinkhorn_iters,
                                sinkhorn_entropy_reg,
                                recon_loss_weight,
                                transport_loss_weight,
                                mask_l1_loss_weight,
                            ),
                        }

                    num_sampling_strategies = len(sampling_strategies)
                    loss = total_loss / num_sampling_strategies
                    recon_loss = total_recon_loss / num_sampling_strategies
                    transport_loss = total_transport_loss / num_sampling_strategies
                    mask_l1_loss = total_mask_l1_loss / num_sampling_strategies
                    pred_mask_entropy = total_pred_mask_entropy / num_sampling_strategies

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                return (
                    float(loss),
                    float(recon_loss),
                    float(transport_loss),
                    float(mask_l1_loss),
                    float(pred_mask_entropy),
                    float(_new_lr),
                    float(_new_wd),
                    decoder_recon_all.detach().cpu(),
                    strategy_metrics,
                )

            (
                loss,
                recon_loss,
                transport_loss,
                mask_l1_loss,
                pred_mask_entropy,
                _new_lr,
                _new_wd,
                decoder_vis_cpu,
                strategy_metrics,
            ), gpu_etime_ms = gpu_timer(train_step)

            if is_main_process and (((itr % tb_vis_freq) == 0) or (itr == ipe - 1)):
                clips_vis = clips_flat.detach().cpu()
                with torch.no_grad():
                    teacher_frames = clips_flat.permute(0, 2, 1, 3, 4).flatten(0, 1)
                    teacher_frames = _resize_flat_frames(teacher_frames, teacher_crop_size)
                    teacher_feats = teacher(teacher_frames)
                    D = teacher_feats.size(-1)
                    if teacher_feats.size(1) != spatial_size * spatial_size:
                        raise RuntimeError(
                            "Teacher feature token count must match the student spatial grid: "
                            f"teacher={teacher_feats.size(1)}, grid={spatial_size * spatial_size}"
                        )
                    h_vis_cpu = teacher_feats.view(clips_flat.size(0), total_frames, spatial_size * spatial_size, D).cpu()
                clips_vis = clips_vis.view(-1, batch_num_views, clips_vis.size(1), clips_vis.size(2), clips_vis.size(3), clips_vis.size(4))
                h_vis = h_vis_cpu.view(-1, batch_num_views, total_frames, h_vis_cpu.size(2), D)
                decoder_vis = decoder_vis_cpu.view(-1, batch_num_views, total_frames, decoder_vis_cpu.size(2), D)
                strategy_vis_payload = {}
                for strategy, metrics in strategy_metrics.items():
                    z_pred_vis_cpu = metrics["pred_recon_cpu"]
                    target_indices_cpu = metrics["target_indices_cpu"]
                    num_target_frames = target_indices_cpu.numel()
                    strategy_vis_payload[strategy] = (
                        z_pred_vis_cpu.view(-1, batch_num_views, num_target_frames, z_pred_vis_cpu.size(2), D),
                        target_indices_cpu,
                    )
                vis_payload = (clips_vis, h_vis, decoder_vis, strategy_vis_payload)

            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            recon_meter.update(recon_loss)
            transport_meter.update(transport_loss)
            mask_l1_meter.update(mask_l1_loss)
            entropy_meter.update(pred_mask_entropy)
            for strategy, metrics in strategy_metrics.items():
                strategy_loss_meters[strategy].update(metrics["loss"])
                strategy_recon_meters[strategy].update(metrics["recon_loss"])
                strategy_transport_meters[strategy].update(metrics["transport_loss"])
                strategy_mask_l1_meters[strategy].update(metrics["mask_l1_loss"])
                strategy_entropy_meters[strategy].update(metrics["pred_mask_entropy"])
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            global_step = epoch * ipe + itr
            
            if is_main_process:
                csv_logger.log(epoch + 1, itr, loss, iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms)
            if is_main_process and writer is not None and (((itr % tb_log_freq) == 0) or (itr == ipe - 1)):
                writer.add_scalar("Train/Loss", loss, global_step)
                writer.add_scalar("Train/Loss_Recon", recon_loss, global_step)
                writer.add_scalar("Train/Loss_Transport", transport_loss, global_step)
                writer.add_scalar("Train/Loss_MaskL1", mask_l1_loss, global_step)
                writer.add_scalar("Train/PredMaskEntropy", pred_mask_entropy, global_step)
                for strategy, metrics in strategy_metrics.items():
                    writer.add_scalar(f"Train_{strategy}/Loss", metrics["loss"], global_step)
                    writer.add_scalar(f"Train_{strategy}/Loss_Recon", metrics["recon_loss"], global_step)
                    writer.add_scalar(f"Train_{strategy}/Loss_Transport", metrics["transport_loss"], global_step)
                    writer.add_scalar(f"Train_{strategy}/Loss_MaskL1", metrics["mask_l1_loss"], global_step)
                    writer.add_scalar(f"Train_{strategy}/PredMaskEntropy", metrics["pred_mask_entropy"], global_step)
                    non_wrist_group_losses = metrics["non_wrist_group_losses"]
                    wrist_group_losses = metrics["wrist_group_losses"]
                    if non_wrist_group_losses is not None:
                        writer.add_scalar(f"Train_{strategy}_NonWrist/Loss", non_wrist_group_losses["loss"], global_step)
                        writer.add_scalar(f"Train_{strategy}_NonWrist/Loss_Recon", non_wrist_group_losses["recon"], global_step)
                        writer.add_scalar(
                            f"Train_{strategy}_NonWrist/Loss_Transport", non_wrist_group_losses["transport"], global_step
                        )
                        writer.add_scalar(
                            f"Train_{strategy}_NonWrist/Loss_MaskL1", non_wrist_group_losses["mask_l1"], global_step
                        )
                    if wrist_group_losses is not None:
                        writer.add_scalar(f"Train_{strategy}_Wrist/Loss", wrist_group_losses["loss"], global_step)
                        writer.add_scalar(f"Train_{strategy}_Wrist/Loss_Recon", wrist_group_losses["recon"], global_step)
                        writer.add_scalar(
                            f"Train_{strategy}_Wrist/Loss_Transport", wrist_group_losses["transport"], global_step
                        )
                        writer.add_scalar(
                            f"Train_{strategy}_Wrist/Loss_MaskL1", wrist_group_losses["mask_l1"], global_step
                        )
                writer.add_scalar("Train/LR", _new_lr, global_step)
                writer.add_scalar("Train/WeightDecay", _new_wd, global_step)
                writer.add_scalar("Train/IterTimeMs", iter_elapsed_time_ms, global_step)
                writer.add_scalar("Train/GPUTimeMs", gpu_etime_ms, global_step)
                writer.add_scalar("Train/DataTimeMs", data_elapsed_time_ms, global_step)
                if torch.cuda.is_available():
                    writer.add_scalar("Train/MaxMemoryMB", torch.cuda.max_memory_allocated() / 1024.0**2, global_step)

            if is_main_process and writer is not None and vis_payload is not None:
                clips_vis, h_vis, decoder_vis, strategy_vis_payload = vis_payload
                num_vis_views = min(clips_vis.size(1), len(dataset_view_names))
                has_bidir_extrapolation = (
                    "front_extrapolation" in strategy_vis_payload and "back_extrapolation" in strategy_vis_payload
                )
                for view_idx in range(num_vis_views):
                    view_name = dataset_view_names[view_idx]
                    if has_bidir_extrapolation:
                        front_pred, front_indices = strategy_vis_payload["front_extrapolation"]
                        back_pred, back_indices = strategy_vis_payload["back_extrapolation"]
                        pca_grid_extrap = _make_multi_pred_pca_grid(
                            clips=clips_vis[:, view_idx],
                            target_reps=h_vis[:, view_idx],
                            pred_payloads=[
                                (front_pred[:, view_idx], front_indices.tolist()),
                                (back_pred[:, view_idx], back_indices.tolist()),
                            ],
                            patch_size=patch_size,
                            max_vis_frames=tb_vis_frames,
                            aux_reps=decoder_vis[:, view_idx],
                            pca_source_reps=h_vis,
                        )
                        if pca_grid_extrap is not None:
                            writer.add_image(
                                f"Train_Vis/extrapolation/{view_name}/PCA_QueryPred",
                                pca_grid_extrap,
                                global_step,
                            )
                for strategy, (z_pred_vis, target_indices_vis) in strategy_vis_payload.items():
                    if has_bidir_extrapolation and strategy in {"front_extrapolation", "back_extrapolation"}:
                        continue
                    for view_idx in range(num_vis_views):
                        view_name = dataset_view_names[view_idx]
                        pca_grid_pred = _make_rollout_pca_grid(
                            clips=clips_vis[:, view_idx],
                            target_reps=h_vis[:, view_idx],
                            pred_reps=z_pred_vis[:, view_idx],
                            patch_size=patch_size,
                            max_vis_frames=tb_vis_frames,
                            aux_reps=decoder_vis[:, view_idx],
                            pca_source_reps=h_vis,
                            pred_frame_indices=target_indices_vis.tolist(),
                        )
                        if pca_grid_pred is not None:
                            writer.add_image(f"Train_Vis/{strategy}/{view_name}/PCA_QueryPred", pca_grid_pred, global_step)

                strategy_log = " ".join(
                    [
                        f"[{strategy}: loss {strategy_loss_meters[strategy].avg:.3f}, rec {strategy_recon_meters[strategy].avg:.2f}, ot {strategy_transport_meters[strategy].avg:.2f}, ml1 {strategy_mask_l1_meters[strategy].avg:.2f}, ent {strategy_entropy_meters[strategy].avg:.3f}]"
                        for strategy in sampling_strategies
                    ]
                )
                logger.info(
                    "[%d, %5d/%5d] loss: %.3f [rec: %.2f, ot: %.2f, ml1: %.2f] [ent: %.3f] %s [lr: %.2e] [mem: %.2e] [iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms]"
                    % (
                        epoch + 1,
                        itr,
                        ipe,
                        loss_meter.avg,
                        recon_meter.avg,
                        transport_meter.avg,
                        mask_l1_meter.avg,
                        entropy_meter.avg,
                        strategy_log,
                        _new_lr,
                        torch.cuda.max_memory_allocated() / 1024.0**2 if torch.cuda.is_available() else 0.0,
                        iter_time_meter.avg,
                        gpu_time_meter.avg,
                        data_elapsed_time_meter.avg,
                    )
                )
            assert not np.isnan(loss), "loss is nan"

        if is_main_process:
            logger.info("avg. loss %.3f" % loss_meter.avg)
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_checkpoint(epoch + 1, os.path.join(folder, f"e{epoch}.pt"))

    if writer is not None:
        writer.close()
