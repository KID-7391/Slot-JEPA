import os
# for env in [
#     "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", 
#     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_DEVICE_MAX_CONNECTIONS"
# ]:
#     os.environ[env] = "1"
# # 禁用 decord 的内部多线程处理
# os.environ["DECORD_NUM_THREADS"] = "1"
import sys
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt

from pathlib import Path

# Ensure the installable source tree is available when launched directly.
release_src = str(Path(__file__).resolve().parents[2] / "src")
if release_src not in sys.path:
    sys.path.insert(0, release_src)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm
from torchvision import transforms as T
import random
from functools import lru_cache

from slot_jepa.encoders import init_encoder
from slot_jepa.losses import feature_recon_loss, slot_orthogonality_loss
from slot_jepa.slots import TwoStageExtractor
from slot_jepa.predictors import ActionConditionalMaskPredictor, InverseConditionModule
from visualization import make_pca_grid, make_pca_grid_with_points
from slot_jepa.data.transforms import make_gpu_transforms

# use the reusable Kinetics video dataset implementation
from slot_jepa.data.kinetics_video import KineticsVideoDataset, KineticsLMDBDataset


_OT_COST_CACHE = {}


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


def _apply_gpu_video_transform_batch(clips, gpu_transform):
    if gpu_transform is None:
        if isinstance(clips, list):
            return torch.stack(clips, dim=0)
        return clips
    if isinstance(clips, list):
        return torch.stack([gpu_transform(clip) for clip in clips], dim=0)
    if clips.dim() != 5:
        raise ValueError(f"Expected clips to have shape [B, C, T, H, W], got {tuple(clips.shape)}")
    return torch.stack([gpu_transform(clip) for clip in clips], dim=0)


def _resize_flat_frames(frames, size):
    if frames.shape[-2:] == (size, size):
        return frames
    return F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False)


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


def _sinkhorn_mask_transport_loss(pred_mask_probs, target_mask_probs, spatial_size, sinkhorn_iters=5, entropy_reg=0.05):
    leading_shape = pred_mask_probs.shape[:-1]
    pred_mass = pred_mask_probs.float().reshape(-1, pred_mask_probs.size(-1)).clamp_min(1.0e-8)
    target_mass = target_mask_probs.float().reshape(-1, target_mask_probs.size(-1)).clamp_min(1.0e-8)
    pred_mass = pred_mass / pred_mass.sum(dim=-1, keepdim=True)
    target_mass = target_mass / target_mass.sum(dim=-1, keepdim=True)
    cost = _get_spatial_transport_cost(spatial_size, pred_mask_probs.device).float()
    kernel = torch.exp(-cost / entropy_reg).clamp_min(1.0e-8)
    u = torch.ones_like(pred_mass) / pred_mass.size(-1)
    v = torch.ones_like(target_mass) / target_mass.size(-1)
    for _ in range(sinkhorn_iters):
        kv = torch.matmul(v, kernel.transpose(0, 1)).clamp_min(1.0e-8)
        u = pred_mass / kv
        ktu = torch.matmul(u, kernel).clamp_min(1.0e-8)
        v = target_mass / ktu
    transport_plan = u.unsqueeze(-1) * kernel.unsqueeze(0) * v.unsqueeze(-2)
    transport_cost = (transport_plan * cost.unsqueeze(0)).sum(dim=(1, 2))
    return transport_cost.view(*leading_shape).mean()


def _mask_l1_loss(pred_mask_probs, target_mask_probs):
    return (pred_mask_probs.float() - target_mask_probs.float()).abs().mean()


def _recon_l1_loss(pred_recon, target_recon):
    return (pred_recon.float() - target_recon.float()).abs().mean()


def _warp_masks_by_local_correspondence(
    context_feats,
    target_feats,
    context_mask_probs,
    spatial_size,
    jitter_std=0.0,
    radius=2,
):
    with torch.no_grad():
        B, T_ctx, N, D = context_feats.shape
        T_query = target_feats.size(1)
        H = W = spatial_size
        window = 2 * radius + 1
        K = context_mask_probs.size(2)
        context_feat_2d = context_feats.float().reshape(B * T_ctx, H, W, D).permute(0, 3, 1, 2)
        target_feat_2d = target_feats.float().reshape(B * T_query, H, W, D).permute(0, 3, 1, 2)
        context_feat_2d = F.normalize(context_feat_2d, dim=1, eps=1.0e-6)
        target_feat_2d = F.normalize(target_feat_2d, dim=1, eps=1.0e-6)
        unfolded_feats = F.unfold(context_feat_2d, kernel_size=window, padding=radius)
        unfolded_feats = unfolded_feats.view(B, T_ctx, D, window * window, N).permute(0, 1, 4, 3, 2)
        unfolded_masks = F.unfold(
            context_mask_probs.float().reshape(B * T_ctx, K, H, W),
            kernel_size=window,
            padding=radius,
        )
        unfolded_masks = unfolded_masks.view(B, T_ctx, K, window * window, N).permute(0, 1, 4, 2, 3)
        target_flat = target_feat_2d.permute(0, 2, 3, 1).reshape(B, T_query, N, D)

        warped = []
        for q in range(T_query):
            target_q = target_flat[:, q]
            per_context = []
            for t in range(T_ctx):
                cand_feats = unfolded_feats[:, t]
                sim = (cand_feats * target_q.unsqueeze(2)).sum(dim=-1)
                if jitter_std > 0:
                    sim = sim + torch.randn_like(sim) * jitter_std
                best_idx = sim.argmax(dim=-1)
                gather_idx = best_idx[:, :, None, None].expand(-1, -1, K, 1)
                gathered = torch.gather(unfolded_masks[:, t], dim=-1, index=gather_idx).squeeze(-1)
                per_context.append(gathered.permute(0, 2, 1))
            warped.append(torch.stack(per_context, dim=1))
        return torch.stack(warped, dim=1)


def _sample_context_target_indices(total_frames, device, strategy="front_extrapolation", target_length=4):
    if total_frames < 2:
        raise ValueError(f"Need at least 2 frames to sample context/target, got {total_frames}")
    if target_length <= 0 or target_length >= total_frames:
        raise ValueError(f"target_length must be in [1, total_frames - 1], got {target_length} for {total_frames} frames")

    if strategy == "front_extrapolation":
        target_indices = torch.arange(0, target_length, device=device, dtype=torch.long)
    elif strategy == "back_extrapolation":
        target_indices = torch.arange(total_frames - target_length, total_frames, device=device, dtype=torch.long)
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    target_mask = torch.zeros(total_frames, device=device, dtype=torch.bool)
    target_mask[target_indices] = True
    context_indices = torch.arange(total_frames, device=device, dtype=torch.long)[~target_mask]
    return context_indices, target_indices


def setup_distributed(args):
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    else:
        local_rank = args.local_rank
    return rank, world_size, local_rank


def custom_collate_fn(batch):
    # batch is a list of tuples: (imgs, v_name, f_indices)
    imgs = [item[0] for item in batch]
    v_names = [item[1] for item in batch]
    f_indices = [item[2] for item in batch]
    return imgs, v_names, f_indices


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


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8, help='Number of videos per iteration.')
    parser.add_argument('--clip-size', type=int, default=7, help='Number of frames to extract spatiotemporally')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--vjepa-checkpoint-path', type=str, default='checkpoints/vjepa2_1_vitb_dist_vitG_384.pt')
    parser.add_argument('--encoder-impl', type=str, default='slot_jepa')
    parser.add_argument('--encoder-model-name', type=str, default='vit_base')
    parser.add_argument('--encoder-checkpoint-key', type=str, default='ema_encoder')
    parser.add_argument('--encoder-patch-size', type=int, default=16)
    parser.add_argument('--encoder-num-register-tokens', type=int, default=0)
    parser.add_argument('--student-encoder-impl', type=str, default='slot_jepa')
    parser.add_argument('--student-model-name', type=str, default='vit_base')
    parser.add_argument('--student-encoder-checkpoint-key', type=str, default='ema_encoder')
    parser.add_argument('--student-encoder-patch-size', type=int, default=16)
    parser.add_argument('--student-encoder-num-register-tokens', type=int, default=0)
    parser.add_argument('--student-checkpoint-path', type=str, default='checkpoints/vjepa2_1_vitb_dist_vitG_384.pt')
    parser.add_argument('--student-train-last-n-layers', type=int, default=6)
    parser.add_argument('--teacher-encoder-impl', type=str, default='dinov2_vitb14_reg')
    parser.add_argument('--teacher-model-name', type=str, default='dinov2_vitb14_reg')
    parser.add_argument('--teacher-encoder-checkpoint-key', type=str, default=None)
    parser.add_argument('--teacher-encoder-patch-size', type=int, default=14)
    parser.add_argument('--teacher-encoder-num-register-tokens', type=int, default=4)
    parser.add_argument('--teacher-checkpoint-path', type=str, default=None)
    parser.add_argument('--teacher-input-size', type=int, default=336)
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume training from')
    parser.add_argument('--save-dir', type=str, default='save/state_init_slot')
    parser.add_argument('--recon-loss-type', type=str, default='l1', choices=['kl', 'l1'])
    parser.add_argument('--cache-to-ram', action='store_true', help='Copy dataset to RAM (/dev/shm) to prevent IO bottlenecks')
    parser.add_argument('--use-lmdb', action='store_true', help='Use LMDB dataset instead of MP4/Frames')
    parser.add_argument('--lmdb-path', type=str, default='/dev/shm/Kinetics-400/kinetics_train.lmdb', help='Path to LMDB dataset')
    parser.add_argument('--dataset-root', type=str, default='./data/Kinetics-400', help='Path to the pretraining video dataset root.')
    parser.add_argument('--valid-list', type=str, default='valid_video_names.txt', help='Validation list file used by the pretraining dataset.')
    parser.add_argument('--input-size', type=int, default=384, help='Input image dynamic spatial size')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='Prefetch factor when num_workers > 0')
    parser.add_argument('--checkpoint-decoder', action='store_true', help='Use activation checkpointing in the decoder to reduce memory')
    parser.add_argument('--checkpoint-predictor', action='store_true', help='Use activation checkpointing in the mask predictor to reduce memory')
    parser.add_argument('--tau', type=float, default=2.25, help='Threshold for truncated gaussians')
    parser.add_argument('--num-slots', type=int, default=256, help='Number of initial slots (N)')
    parser.add_argument('--mask-temp', type=float, default=1.0, help='Temperature for slot masks')
    
    # New Two Stage Slot Attention Args
    parser.add_argument('--window-size', type=int, nargs=2, default=[2, 4], help='Temporal and Spatial Window Size')
    parser.add_argument('--slots-per-window', type=int, default=4, help='Number of slots per window in stage 1')
    parser.add_argument('--global-k', type=int, default=16, help='Number of slots in stage 2 (FPS)')
    parser.add_argument('--L-max', type=int, default=16, help='Maximum number of instance slots for mask prediction')
    parser.add_argument('--slot-feature-dim', type=int, default=196, help='Low-dimensional slot reconstruction feature dim.')
    parser.add_argument('--pred-embed-dim', type=int, default=192, help='Predictor hidden dim. Must match downstream slot_ac.')
    parser.add_argument('--pred-depth', type=int, default=12, help='Predictor depth. Must match downstream slot_ac.')
    parser.add_argument('--pred-num-heads', type=int, default=3, help='Predictor attention heads.')
    parser.add_argument('--num-obj-tokens', type=int, default=1, help='Number of reduced obj-latent condition tokens.')
    parser.add_argument('--no-concat-mask-input', action='store_true', help='Disable concatenating K-way mask probabilities to the reconstructed patch token input.')
    parser.add_argument('--use-extrinsics', action='store_true', help='Emit pseudo extrinsics and feed them into predictor.')
    parser.add_argument('--inverse-depth', type=int, default=8, help='Depth of pseudo-condition inverse module.')
    parser.add_argument('--inverse-heads', type=int, default=4, help='Heads of pseudo-condition inverse module.')
    parser.add_argument('--inverse-token-dim', type=int, default=32, help='Compressed inverse condition token dim.')
    parser.add_argument('--target-num-frames', type=int, default=3, help='Number of target frames for each predictor sampling strategy.')
    parser.add_argument('--predictor-transport-loss-weight', type=float, default=3.0, help='Weight for predictor OT loss.')
    parser.add_argument('--predictor-mask-l1-loss-weight', type=float, default=3.0, help='Weight for predictor mask L1 loss.')
    parser.add_argument('--predictor-recon-loss-weight', type=float, default=0.1, help='Weight for predictor recon L1 loss.')
    parser.add_argument('--predictor-sinkhorn-iters', type=int, default=5, help='Sinkhorn iterations for predictor OT loss.')
    parser.add_argument('--predictor-sinkhorn-entropy-reg', type=float, default=0.05, help='Entropy regularization for predictor OT loss.')
    parser.add_argument('--correspondence-jitter-std', type=float, default=0.05, help='Score jitter used in stop-grad correspondence warp.')
    parser.add_argument('--correspondence-radius', type=int, default=2, help='Local search radius for stop-grad correspondence warp.')
    return parser


def _namespace_from_config(config):
    parser = build_arg_parser()
    args = parser.parse_args([])
    cfg_meta = config.get("meta", {})
    cfg_model = config.get("model", {})
    cfg_data = config.get("data", {})
    cfg_data_aug = config.get("data_aug", {})
    cfg_loss = config.get("loss", {})
    cfg_opt = config.get("optimization", {})

    args.epochs = int(cfg_opt.get("epochs", args.epochs))
    args.batch_size = int(cfg_data.get("batch_size", args.batch_size))
    args.clip_size = int(max(cfg_data.get("dataset_fpcs", [args.clip_size])))
    args.lr = float(cfg_opt.get("lr", args.lr))
    args.local_rank = 0
    student_checkpoint = cfg_meta.get(
        "student_pretrain_checkpoint",
        cfg_meta.get("vjepa_checkpoint_path", args.student_checkpoint_path),
    )
    teacher_checkpoint = cfg_meta.get("teacher_pretrain_checkpoint", cfg_meta.get("pretrain_checkpoint", args.teacher_checkpoint_path))
    args.student_checkpoint_path = None if student_checkpoint is None else str(student_checkpoint)
    args.teacher_checkpoint_path = None if teacher_checkpoint is None else str(teacher_checkpoint)
    args.vjepa_checkpoint_path = args.student_checkpoint_path
    args.resume = str(cfg_meta.get("resume_checkpoint") or args.resume)
    args.save_dir = str(config.get("folder", cfg_meta.get("save_dir", args.save_dir)))
    args.recon_loss_type = str(cfg_loss.get("recon_loss_type", args.recon_loss_type))
    args.cache_to_ram = bool(cfg_data.get("cache_to_ram", args.cache_to_ram))
    args.use_lmdb = bool(cfg_data.get("use_lmdb", args.use_lmdb))
    args.lmdb_path = str(cfg_data.get("lmdb_path", args.lmdb_path))
    args.dataset_root = str(cfg_data.get("dataset_root", args.dataset_root))
    args.valid_list = str(cfg_data.get("valid_list", args.valid_list))
    args.input_size = int(cfg_data.get("crop_size", args.input_size))
    args.num_workers = int(cfg_data.get("num_workers", args.num_workers))
    args.prefetch_factor = int(cfg_data.get("prefetch_factor", args.prefetch_factor))
    args.pin_mem = bool(cfg_data.get("pin_mem", True))
    args.persistent_workers = bool(cfg_data.get("persistent_workers", True))
    args.horizontal_flip = bool(cfg_data_aug.get("horizontal_flip", False))
    args.random_resize_aspect_ratio = tuple(cfg_data_aug.get("random_resize_aspect_ratio", [0.75, 1.35]))
    args.random_resize_scale = tuple(cfg_data_aug.get("random_resize_scale", [1.777, 1.777]))
    args.motion_shift = bool(cfg_data_aug.get("motion_shift", False))
    args.reprob = float(cfg_data_aug.get("reprob", 0.0))
    args.auto_augment = bool(cfg_data_aug.get("auto_augment", False))
    args.student_encoder_impl = str(cfg_model.get("student_encoder_impl", args.student_encoder_impl))
    args.student_model_name = str(cfg_model.get("student_model_name", args.student_model_name))
    args.student_encoder_checkpoint_key = cfg_model.get(
        "student_encoder_checkpoint_key", args.student_encoder_checkpoint_key
    )
    args.student_encoder_patch_size = int(cfg_model.get("student_encoder_patch_size", args.student_encoder_patch_size))
    args.student_encoder_num_register_tokens = int(
        cfg_model.get("student_encoder_num_register_tokens", args.student_encoder_num_register_tokens)
    )
    args.student_train_last_n_layers = int(cfg_model.get("student_train_last_n_layers", args.student_train_last_n_layers))
    legacy_teacher_impl = cfg_model.get("encoder_impl")
    args.teacher_encoder_impl = str(cfg_model.get("teacher_encoder_impl", legacy_teacher_impl or args.teacher_encoder_impl))
    default_teacher_model_name = "vit_base" if args.teacher_encoder_impl in ("slot_jepa", "vjepa", "vjepa2") else args.teacher_model_name
    args.teacher_model_name = str(cfg_model.get("teacher_model_name", cfg_model.get("model_name", default_teacher_model_name)))
    args.teacher_encoder_checkpoint_key = cfg_model.get(
        "teacher_encoder_checkpoint_key",
        cfg_model.get("encoder_checkpoint_key", args.teacher_encoder_checkpoint_key),
    )
    args.teacher_encoder_patch_size = int(cfg_model.get("teacher_encoder_patch_size", cfg_model.get("encoder_patch_size", args.teacher_encoder_patch_size)))
    args.teacher_encoder_num_register_tokens = int(
        cfg_model.get("teacher_encoder_num_register_tokens", cfg_model.get("encoder_num_register_tokens", args.teacher_encoder_num_register_tokens))
    )
    default_teacher_input_size = 336 if args.teacher_encoder_patch_size == 14 else args.input_size
    args.teacher_input_size = int(cfg_model.get("teacher_crop_size", cfg_model.get("teacher_input_size", default_teacher_input_size)))
    args.encoder_impl = args.student_encoder_impl
    args.encoder_model_name = args.student_model_name
    args.encoder_checkpoint_key = args.student_encoder_checkpoint_key
    args.encoder_patch_size = args.student_encoder_patch_size
    args.encoder_num_register_tokens = args.student_encoder_num_register_tokens
    args.checkpoint_decoder = bool(cfg_model.get("use_decoder_checkpoint", args.checkpoint_decoder))
    args.checkpoint_predictor = bool(cfg_model.get("use_activation_checkpointing", args.checkpoint_predictor))
    args.tau = float(cfg_model.get("tau", args.tau))
    args.num_slots = int(cfg_model.get("num_slots", args.num_slots))
    args.mask_temp = float(cfg_model.get("mask_temp", args.mask_temp))
    args.window_size = list(cfg_model.get("window_size", args.window_size))
    args.slots_per_window = int(cfg_model.get("slots_per_window", args.slots_per_window))
    args.global_k = int(cfg_model.get("global_k", cfg_model.get("num_objects", args.global_k)))
    args.L_max = int(cfg_model.get("L_max", args.L_max))
    args.slot_feature_dim = int(cfg_model.get("slot_feature_dim", cfg_model.get("feature_dim", args.slot_feature_dim)))
    args.pred_embed_dim = int(cfg_model.get("predictor_embed_dim", cfg_model.get("pred_embed_dim", args.pred_embed_dim)))
    args.pred_depth = int(cfg_model.get("predictor_depth", cfg_model.get("pred_depth", args.pred_depth)))
    args.pred_num_heads = int(cfg_model.get("predictor_num_heads", cfg_model.get("pred_num_heads", args.pred_num_heads)))
    args.num_obj_tokens = int(cfg_model.get("num_obj_tokens", args.num_obj_tokens))
    args.no_concat_mask_input = not bool(cfg_model.get("concat_mask_input", not args.no_concat_mask_input))
    args.use_extrinsics = bool(cfg_model.get("use_extrinsics", args.use_extrinsics))
    args.inverse_depth = int(cfg_model.get("inverse_depth", args.inverse_depth))
    args.inverse_heads = int(cfg_model.get("inverse_heads", args.inverse_heads))
    args.inverse_token_dim = int(cfg_model.get("inverse_token_dim", args.inverse_token_dim))
    args.target_num_frames = int(cfg_loss.get("target_num_frames", args.target_num_frames))
    args.predictor_transport_loss_weight = float(cfg_loss.get("transport_loss_weight", args.predictor_transport_loss_weight))
    args.predictor_mask_l1_loss_weight = float(cfg_loss.get("mask_l1_loss_weight", args.predictor_mask_l1_loss_weight))
    args.predictor_recon_loss_weight = float(cfg_loss.get("recon_loss_weight", args.predictor_recon_loss_weight))
    args.predictor_sinkhorn_iters = int(cfg_loss.get("sinkhorn_iters", args.predictor_sinkhorn_iters))
    args.predictor_sinkhorn_entropy_reg = float(cfg_loss.get("sinkhorn_entropy_reg", args.predictor_sinkhorn_entropy_reg))
    args.correspondence_jitter_std = float(cfg_model.get("correspondence_jitter_std", args.correspondence_jitter_std))
    args.correspondence_radius = int(cfg_model.get("correspondence_radius", args.correspondence_radius))
    return args


def main(args=None, resume_preempt=False):
    del resume_preempt
    torch.backends.cudnn.benchmark = True
    parser = build_arg_parser()
    if args is None:
        args = parser.parse_args()
    elif isinstance(args, dict):
        args = _namespace_from_config(args)
    elif isinstance(args, argparse.Namespace):
        pass
    else:
        args = parser.parse_args(args=args)
    if args.teacher_encoder_impl not in ("slot_jepa", "vjepa", "vjepa2") and args.teacher_encoder_checkpoint_key == "ema_encoder":
        args.teacher_encoder_checkpoint_key = None
    if args.student_encoder_impl not in ("slot_jepa", "vjepa", "vjepa2") and args.student_encoder_checkpoint_key == "ema_encoder":
        args.student_encoder_checkpoint_key = None
    if args.teacher_encoder_impl in ("slot_jepa", "vjepa", "vjepa2") and args.teacher_encoder_checkpoint_key is None:
        args.teacher_encoder_checkpoint_key = "ema_encoder"
    if args.student_encoder_impl in ("slot_jepa", "vjepa", "vjepa2") and args.student_encoder_checkpoint_key is None:
        args.student_encoder_checkpoint_key = "ema_encoder"
    if args.teacher_encoder_impl in ("slot_jepa", "vjepa", "vjepa2") and str(args.teacher_model_name).startswith("dinov2"):
        args.teacher_model_name = "vit_base"
    if args.student_encoder_impl in ("slot_jepa", "vjepa", "vjepa2") and str(args.student_model_name).startswith("dinov2"):
        args.student_model_name = "vit_base"
    if args.batch_size < 1:
        raise ValueError('--batch-size must be >= 1')
    if not hasattr(args, "pin_mem"):
        args.pin_mem = True
    if not hasattr(args, "persistent_workers"):
        args.persistent_workers = True
    if not hasattr(args, "horizontal_flip"):
        args.horizontal_flip = False
    if not hasattr(args, "random_resize_aspect_ratio"):
        args.random_resize_aspect_ratio = (0.75, 1.35)
    if not hasattr(args, "random_resize_scale"):
        args.random_resize_scale = (1.777, 1.777)
    if not hasattr(args, "motion_shift"):
        args.motion_shift = False
    if not hasattr(args, "reprob"):
        args.reprob = 0.0
    if not hasattr(args, "auto_augment"):
        args.auto_augment = False
    if args.pred_embed_dim % args.pred_num_heads != 0:
        raise ValueError('--pred-embed-dim must be divisible by --pred-num-heads')
    if args.pred_embed_dim % args.inverse_heads != 0:
        raise ValueError('--pred-embed-dim must be divisible by --inverse-heads')
    if args.target_num_frames <= 0 or args.target_num_frames >= args.clip_size:
        raise ValueError('--target-num-frames must be in [1, clip-size - 1]')
    student_spatial_size = args.input_size // args.student_encoder_patch_size
    teacher_spatial_size = args.teacher_input_size // args.teacher_encoder_patch_size
    if student_spatial_size != teacher_spatial_size:
        raise ValueError(
            "Student and teacher patch grids must match: "
            f"{args.input_size}//{args.student_encoder_patch_size} != "
            f"{args.teacher_input_size}//{args.teacher_encoder_patch_size}"
        )
    rank, world_size, local_rank = setup_distributed(args)

    is_main_process = (rank == 0)

    if is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        import json
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    writer = None
    if is_main_process:
        try:
            writer = SummaryWriter(log_dir=args.save_dir)
        except Exception as e:
            print('Warning: Failed to create SummaryWriter:', e)

    device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    teacher_auto_download = (
        args.teacher_encoder_impl not in ("slot_jepa", "vjepa", "vjepa2")
        and args.teacher_checkpoint_path is None
    )
    student_auto_download = (
        args.student_encoder_impl not in ("slot_jepa", "vjepa", "vjepa2")
        and args.student_checkpoint_path is None
    )
    encoder_auto_download = teacher_auto_download or student_auto_download
    if encoder_auto_download and dist.is_available() and dist.is_initialized() and rank != 0:
        dist.barrier()

    vjepa_teacher = init_encoder(
        encoder_impl=args.teacher_encoder_impl,
        resolution=args.teacher_input_size,
        checkpoint=args.teacher_checkpoint_path,
        model_name=args.teacher_model_name,
        checkpoint_key=args.teacher_encoder_checkpoint_key,
        patch_size=args.teacher_encoder_patch_size,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
        img_as_video_nframes=2,
        num_register_tokens=args.teacher_encoder_num_register_tokens,
    ).to(device)
    if encoder_auto_download and dist.is_available() and dist.is_initialized() and rank == 0:
        dist.barrier()
    vjepa_teacher.eval() # 冻结 Teacher 模型
    for param in vjepa_teacher.parameters():
        param.requires_grad = False
    print("VJEPA teacher created.")
    
    vjepa_student = init_encoder(
        encoder_impl=args.student_encoder_impl,
        resolution=args.input_size,
        checkpoint=args.student_checkpoint_path,
        model_name=args.student_model_name,
        checkpoint_key=args.student_encoder_checkpoint_key,
        patch_size=args.student_encoder_patch_size,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
        img_as_video_nframes=2,
        num_register_tokens=args.student_encoder_num_register_tokens,
    ).to(device)
    vjepa_student.train() # 训练 Student 模型
    # for param in vjepa_student.parameters():
    #     param.requires_grad = False

    # 1. 冻结 patch embed 和 pos embed, CLS token 等，只开放最后几层的注意力权重更新。
    trainable_student_params, total_student_params, min_trainable_block = _set_trainable_last_n_transformer_blocks(
        vjepa_student,
        args.student_train_last_n_layers,
    )
    if is_main_process:
        print(
            f"Student trainable params: {trainable_student_params / 1e6:.2f}M / "
            f"{total_student_params / 1e6:.2f}M; train blocks >= {min_trainable_block}",
            flush=True,
        )
    print("VJEPA student created.")

    spatial_size = student_spatial_size

    # The Feature extractor handles slot iterations
    extractor = TwoStageExtractor(
        window_size=args.window_size,
        slots_per_window=args.slots_per_window,
        global_k=args.global_k,
        D_in=768,
        mask_temp=args.mask_temp,
        L_max=args.L_max,
        use_decoder_checkpoint=args.checkpoint_decoder,
    ).to(device)
    
    inverse_condition = InverseConditionModule(
        embed_dim=768,
        feature_dim=args.slot_feature_dim,
        predictor_embed_dim=args.pred_embed_dim,
        num_objects=args.global_k,
        state_dim=args.inverse_token_dim,
        extrinsics_dim=args.inverse_token_dim,
        depth=args.inverse_depth,
        num_heads=args.inverse_heads,
        spatial_size=spatial_size,
        concat_mask_input=not args.no_concat_mask_input,
        use_checkpoint=args.checkpoint_predictor,
    ).to(device)

    mask_predictor = ActionConditionalMaskPredictor(
        embed_dim=768,
        feature_dim=args.slot_feature_dim,
        predictor_embed_dim=args.pred_embed_dim,
        depth=args.pred_depth,
        num_heads=args.pred_num_heads,
        num_context_frames=args.clip_size - 1,
        spatial_size=spatial_size,
        num_objects=args.global_k,
        state_dim=args.inverse_token_dim,
        use_extrinsics=args.use_extrinsics,
        extrinsics_dim=args.inverse_token_dim,
        num_obj_tokens=args.num_obj_tokens,
        concat_mask_input=not args.no_concat_mask_input,
        use_warp_concat=True,
        use_checkpoint=args.checkpoint_predictor,
    ).to(device)
    
    # if args.resume and os.path.isfile(args.resume):
    #     if is_main_process:
    #         print(f">>> [Rank {rank}] Resuming from checkpoint: {args.resume}")
    #     ckpt = torch.load(args.resume, map_location='cpu')
    #     extractor.load_state_dict(ckpt['extractor_state'])
    #     if 'student_state' in ckpt:
    #         vjepa_student.load_state_dict(ckpt['student_state'])
    #     if 'mask_predictor_state' in ckpt:
    #         mask_predictor.load_state_dict(ckpt['mask_predictor_state'], strict=False)

    if world_size > 1:
        extractor = DDP(extractor, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        extractor._set_static_graph()
        inverse_condition = DDP(inverse_condition, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        inverse_condition._set_static_graph()
        mask_predictor = DDP(mask_predictor, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        mask_predictor._set_static_graph()
        vjepa_student = DDP(vjepa_student, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    start_epoch = 0
    global_step = 0

    extractor_params = list(extractor.parameters())
    inverse_params = list(inverse_condition.parameters())
    predictor_params = list(mask_predictor.parameters())
    student_params = [p for p in vjepa_student.parameters() if p.requires_grad]
    active_params = extractor_params + inverse_params + predictor_params + student_params
    optimizer = optim.AdamW(active_params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    
    if args.resume and os.path.isfile(args.resume):
        if is_main_process:
            print(f">>> [Rank {rank}] Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')

        if isinstance(extractor, DDP):
            vjepa_student.module.load_state_dict(ckpt['student_state'])
        else:
            vjepa_student.load_state_dict(ckpt['student_state'])

        if isinstance(extractor, DDP):
            extractor.module.load_state_dict(ckpt['extractor_state'])
        else:
            extractor.load_state_dict(ckpt['extractor_state'])

        if 'inverse_condition_state' in ckpt:
            if isinstance(inverse_condition, DDP):
                inverse_condition.module.load_state_dict(ckpt['inverse_condition_state'])
            else:
                inverse_condition.load_state_dict(ckpt['inverse_condition_state'])
            
        if isinstance(mask_predictor, DDP):
            mask_predictor.module.load_state_dict(ckpt['mask_predictor_state'])
        else:
            mask_predictor.load_state_dict(ckpt['mask_predictor_state'])
            
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        # if 'scheduler' in ckpt:
        #     scheduler.load_state_dict(ckpt['scheduler'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        if 'global_step' in ckpt:
            global_step = ckpt['global_step']

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    
    gpu_transform = make_gpu_transforms(
        random_horizontal_flip=args.horizontal_flip,
        random_resize_aspect_ratio=args.random_resize_aspect_ratio,
        random_resize_scale=args.random_resize_scale,
        reprob=args.reprob,
        auto_augment=args.auto_augment,
        motion_shift=args.motion_shift,
        crop_size=args.input_size,
    )

    valid_list = args.valid_list
    
    clip_size = args.clip_size
    
    if args.use_lmdb:
        if rank == 0:
            print(f">>> [Rank {rank}] Initializing LMDB Dataset from {args.lmdb_path}")
        train_ds = KineticsLMDBDataset(lmdb_path=args.lmdb_path, split='train', clip_size=clip_size, transforms=None, return_info=True)
    else:
        train_ds = KineticsVideoDataset(root=str(dataset_root), valid_list=valid_list, split='train', clip_size=clip_size, transforms=None, return_info=True)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader_kwargs = dict(
        dataset=train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        persistent_workers=args.num_workers > 0 and args.persistent_workers,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_mem,
        collate_fn=custom_collate_fn,
    )
    if args.num_workers > 0:
        train_loader_kwargs['prefetch_factor'] = args.prefetch_factor
    train_loader = torch.utils.data.DataLoader(**train_loader_kwargs)
    if is_main_process:
        effective_videos_per_update = args.batch_size
        effective_frames_per_update = effective_videos_per_update * clip_size
        print(
            f'Batch config: {effective_videos_per_update} videos/update '
            f'({effective_frames_per_update} frames)'
        )
    
    # val_ds = KineticsVideoDataset(root=str(kinetics_root), valid_list=valid_list, split='val', clip_size=1, transforms=val_transforms, return_info=True)
    # val_loader = torch.utils.data.DataLoader(
    #     val_ds, 
    #     batch_size=args.batch_size, 
    #     shuffle=False, 
    #     persistent_workers=False,
    #     num_workers=6,
    #     pin_memory=True,
    #     prefetch_factor=2
    # )

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        extractor.train()
        optimizer.zero_grad(set_to_none=True)

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch} train", disable=not is_main_process)
        
        for i, batch in enumerate(train_iter):
            imgs_list, v_n, f_i = batch
            current_video_batch = len(imgs_list)
            clip_tensors = []
            for img in imgs_list:
                clip = img.to(device, non_blocking=True)
                if clip.dim() != 4:
                    raise ValueError(f"Expected clip tensor with shape [T, C, H, W], got {tuple(clip.shape)}")
                clip_tensors.append(clip.permute(1, 0, 2, 3).contiguous())
            clips = _apply_gpu_video_transform_batch(clip_tensors, gpu_transform)
            norm_imgs_student = clips.permute(0, 2, 1, 3, 4).contiguous()
            norm_imgs_student_flat = norm_imgs_student.reshape(-1, 3, args.input_size, args.input_size)
            norm_imgs_teacher = _resize_flat_frames(norm_imgs_student_flat, args.teacher_input_size)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.inference_mode():
                    teacher_features = vjepa_teacher(norm_imgs_teacher)
                    
                student_features = vjepa_student(norm_imgs_student_flat)
                    
            B, _, D = teacher_features.shape
            if teacher_features.size(1) != spatial_size * spatial_size or student_features.size(1) != spatial_size * spatial_size:
                raise RuntimeError(
                    "Student and teacher feature token counts must match the configured spatial grid: "
                    f"student={student_features.size(1)}, teacher={teacher_features.size(1)}, grid={spatial_size * spatial_size}"
                )
            target_h = spatial_size
            target_w = spatial_size
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                Z0_teacher = teacher_features
                T_dim = args.clip_size
                B_videos = B // T_dim

                Z0_5d_teacher = Z0_teacher.view(B_videos, T_dim, target_h, target_w, D)
                Z0_5d_student = student_features.view(B_videos, T_dim, target_h, target_w, D)

                class_state_all, obj_latents = extractor(
                    Z0_5d_student,
                    spatial_dims=(target_h, target_w),
                    T_dim=T_dim,
                )

                decoded_mask_logits = class_state_all.get_layer(-1)['mask_logits']
                decoded_mask_probs = F.softmax(decoded_mask_logits, dim=2)

                K = obj_latents.shape[1]
                decoder_recon_all = torch.einsum('btkn,bkd->btnd', decoded_mask_probs.to(obj_latents.dtype), obj_latents)
                target_all = Z0_5d_teacher.reshape(B_videos, T_dim, -1, D).float()
                decoder_recon_loss = feature_recon_loss(
                    decoder_recon_all.reshape(B_videos * T_dim, -1, D).float(),
                    target_all.reshape(B_videos * T_dim, -1, D),
                    loss_type=args.recon_loss_type,
                )

                (
                    pseudo_states,
                    pseudo_extrinsics,
                ) = inverse_condition(
                    obj_latents,
                    decoded_mask_logits[:, :-1],
                    decoded_mask_logits[:, 1:],
                    all_masks=decoded_mask_logits,
                )
                # Predictor supervision should only update the encoder through obj_latents.
                # Stop gradients through decoder-derived masks/reconstructions and
                # inverse-condition outputs.
                predictor_states = pseudo_states.detach()
                predictor_extrinsics = pseudo_extrinsics.detach() if args.use_extrinsics else None
                teacher_feats_all = Z0_5d_teacher.reshape(B_videos, T_dim, -1, D).float()
                strategy_results = {}
                predictor_recon_loss = 0.0
                predictor_transport_loss = 0.0
                predictor_mask_l1_loss = 0.0
                for strategy in ("front_extrapolation", "back_extrapolation"):
                    context_indices, target_indices = _sample_context_target_indices(
                        T_dim,
                        decoded_mask_logits.device,
                        strategy=strategy,
                        target_length=args.target_num_frames,
                    )
                    context_feats = teacher_feats_all.index_select(1, context_indices)
                    target_feats = teacher_feats_all.index_select(1, target_indices)
                    context_mask_logits = decoded_mask_logits.index_select(1, context_indices)
                    context_mask_probs = decoded_mask_probs.index_select(1, context_indices)
                    target_mask_probs = decoded_mask_probs.index_select(1, target_indices).detach()
                    target_recon = decoder_recon_all.index_select(1, target_indices).detach()
                    context_states = predictor_states.index_select(1, context_indices)
                    target_states = predictor_states.index_select(1, target_indices)
                    context_extrinsics = (
                        predictor_extrinsics.index_select(1, context_indices) if predictor_extrinsics is not None else None
                    )
                    target_extrinsics = (
                        predictor_extrinsics.index_select(1, target_indices) if predictor_extrinsics is not None else None
                    )
                    warped_mask_probs = _warp_masks_by_local_correspondence(
                        context_feats.detach(),
                        target_feats,
                        context_mask_probs.detach(),
                        spatial_size=spatial_size,
                        jitter_std=args.correspondence_jitter_std,
                        radius=args.correspondence_radius,
                    )
                    pred_logits = mask_predictor(
                        obj_latents,
                        context_mask_logits.detach(),
                        context_states,
                        target_states,
                        context_extrinsics,
                        target_extrinsics,
                        external_warped_mask_probs=warped_mask_probs,
                    )
                    pred_mask_probs = F.softmax(pred_logits, dim=2)
                    pred_recon = torch.einsum(
                        'btkn,bkd->btnd',
                        pred_mask_probs.to(obj_latents.dtype),
                        obj_latents,
                    )
                    recon_loss = _recon_l1_loss(pred_recon, target_recon)
                    transport_loss = _sinkhorn_mask_transport_loss(
                        pred_mask_probs,
                        target_mask_probs,
                        spatial_size=spatial_size,
                        sinkhorn_iters=args.predictor_sinkhorn_iters,
                        entropy_reg=args.predictor_sinkhorn_entropy_reg,
                    )
                    mask_l1_loss = _mask_l1_loss(pred_mask_probs, target_mask_probs)
                    strategy_results[strategy] = {
                        "context_indices": context_indices,
                        "target_indices": target_indices,
                        "pred_mask_probs": pred_mask_probs,
                        "pred_recon": pred_recon,
                        "recon_loss": recon_loss,
                        "transport_loss": transport_loss,
                        "mask_l1_loss": mask_l1_loss,
                    }
                    predictor_recon_loss = predictor_recon_loss + recon_loss
                    predictor_transport_loss = predictor_transport_loss + transport_loss
                    predictor_mask_l1_loss = predictor_mask_l1_loss + mask_l1_loss
                num_predictor_strategies = len(strategy_results)
                predictor_recon_loss = predictor_recon_loss / num_predictor_strategies
                predictor_transport_loss = predictor_transport_loss / num_predictor_strategies
                predictor_mask_l1_loss = predictor_mask_l1_loss / num_predictor_strategies
                predictor_loss = (
                    args.predictor_recon_loss_weight * predictor_recon_loss
                    + args.predictor_transport_loss_weight * predictor_transport_loss
                    + args.predictor_mask_l1_loss_weight * predictor_mask_l1_loss
                )

                loss_ortho = slot_orthogonality_loss(obj_latents.float())
                loss = (
                    0.001 * loss_ortho
                    + decoder_recon_loss
                    + predictor_loss
                )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Rank {rank}] loss is NaN/Inf! Exiting...", flush=True)
                dist.destroy_process_group() if dist.is_initialized() else None
                sys.exit(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(extractor_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(inverse_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(student_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(predictor_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # # EMA Update for Teacher
            # with torch.no_grad():
            #     m = 0.9999
            #     student_m = vjepa_student.module if isinstance(vjepa_student, DDP) else vjepa_student
            #     for param_q, param_k in zip(student_m.parameters(), vjepa_teacher.parameters()):
            #         param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            if i % 10 == 0 and is_main_process:
                train_iter.set_postfix({'loss': f'{loss.item():.4f}'})

            if (global_step % 100) == 0 and is_main_process and writer is not None:
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Train/LR', lr, global_step)
                writer.add_scalar('Train/Loss_Total', loss.item(), global_step)
                writer.add_scalar('Train/Loss_Ortho', loss_ortho.item(), global_step)
                writer.add_scalar('Train/Loss_Recon_Decoder', decoder_recon_loss.item(), global_step)
                for strategy, strategy_data in strategy_results.items():
                    strategy_tag = "FrontExtrapolation" if strategy == "front_extrapolation" else "BackExtrapolation"
                    writer.add_scalar(
                        f'Train/Loss_Recon_Predictor_{strategy_tag}',
                        strategy_data["recon_loss"].item(),
                        global_step,
                    )
                    writer.add_scalar(
                        f'Train/Loss_Transport_Predictor_{strategy_tag}',
                        strategy_data["transport_loss"].item(),
                        global_step,
                    )
                    writer.add_scalar(
                        f'Train/Loss_MaskL1_Predictor_{strategy_tag}',
                        strategy_data["mask_l1_loss"].item(),
                        global_step,
                    )
                writer.add_scalar('Train/Loss_Predictor_Total', predictor_loss.item(), global_step)

            if ((global_step % 1000) == 0 or (global_step < 1000 and global_step % 200 == 0)) \
                                                            and is_main_process and writer is not None:
                # Visualization
                from visualization import generate_gaussian_mask_grid
                vis_frames_full = list(range(args.clip_size))
                
                gt_feats_vis = []
                student_feats_vis = []
                decoder_recon_feats_vis = []
                orig_imgs_vis = {}
                
                # Show the student input image so feature rows align with the
                # representation that receives gradients.
                norm_imgs_cpu = norm_imgs_student_flat.detach().cpu().float()
                
                for t in vis_frames_full:
                    gt_feats_vis.append(Z0_5d_teacher[0:1, t].detach().cpu().float().view(1, -1, D))
                    student_feats_vis.append(Z0_5d_student[0:1, t].detach().cpu().float().view(1, -1, D))
                    recon_Z_t = decoder_recon_all[:, t]
                    decoder_recon_feats_vis.append(recon_Z_t.detach().cpu().float()[0:1])
                        
                    orig_img = norm_imgs_cpu[t]
                    orig_img = orig_img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    orig_img = orig_img.clamp(0, 1)
                    orig_img = F.interpolate(
                        orig_img.unsqueeze(0),
                        size=(target_h * args.encoder_patch_size, target_w * args.encoder_patch_size),
                        mode='bilinear',
                        align_corners=False,
                    ).squeeze(0)
                    orig_imgs_vis[t] = orig_img

                orig_imgs_list = [orig_imgs_vis[t] for t in vis_frames_full]

                recon_pca_feats = [feat[0] for feat in gt_feats_vis + decoder_recon_feats_vis]
                for strategy in ("front_extrapolation", "back_extrapolation"):
                    strategy_data = strategy_results[strategy]
                    pred_recon_cpu = strategy_data["pred_recon"].detach().cpu().float()[0]
                    recon_pca_feats.extend([pred_recon_cpu[pred_idx] for pred_idx in range(pred_recon_cpu.size(0))])
                pca_source = torch.cat(recon_pca_feats, dim=0)
                pca_mean = pca_source.mean(dim=0, keepdim=True)
                pca_centered = pca_source - pca_mean
                _, _, pca_basis = torch.pca_lowrank(pca_centered, q=3)
                pca_proj = pca_centered @ pca_basis
                pca_min = pca_proj.amin(dim=0, keepdim=True)
                pca_max = pca_proj.amax(dim=0, keepdim=True)

                student_all = torch.cat(student_feats_vis, dim=1).squeeze(0)
                student_pca_mean = student_all.mean(dim=0, keepdim=True)
                student_centered = student_all - student_pca_mean
                _, _, student_pca_basis = torch.pca_lowrank(student_centered, q=3)
                student_proj = student_centered @ student_pca_basis
                student_pca_min = student_proj.amin(dim=0, keepdim=True)
                student_pca_max = student_proj.amax(dim=0, keepdim=True)
                gt_imgs = [
                    _feature_to_pca_image(feat[0], pca_mean, pca_basis, pca_min, pca_max, patch_size=args.encoder_patch_size, pred_border=False)
                    for feat in gt_feats_vis
                ]
                student_imgs = [
                    _feature_to_pca_image(
                        feat[0],
                        student_pca_mean,
                        student_pca_basis,
                        student_pca_min,
                        student_pca_max,
                        patch_size=args.encoder_patch_size,
                        pred_border=False,
                    )
                    for feat in student_feats_vis
                ]
                decoder_imgs = [
                    _feature_to_pca_image(feat[0], pca_mean, pca_basis, pca_min, pca_max, patch_size=args.encoder_patch_size, pred_border=False)
                    for feat in decoder_recon_feats_vis
                ]
                blank_img = torch.ones_like(orig_imgs_list[0])
                pred_rows = []
                for strategy in ("front_extrapolation", "back_extrapolation"):
                    strategy_data = strategy_results[strategy]
                    pred_row = [blank_img.clone() for _ in vis_frames_full]
                    target_indices_cpu = strategy_data["target_indices"].detach().cpu().tolist()
                    pred_recon_cpu = strategy_data["pred_recon"].detach().cpu().float()[0]
                    for pred_idx, frame_idx in enumerate(target_indices_cpu):
                        pred_row[frame_idx] = _feature_to_pca_image(
                            pred_recon_cpu[pred_idx],
                            pca_mean,
                            pca_basis,
                            pca_min,
                            pca_max,
                            patch_size=args.encoder_patch_size,
                            pred_border=True,
                        )
                    pred_rows.append(pred_row)
                feature_rows = [
                    orig_imgs_list,
                    gt_imgs,
                    student_imgs,
                    decoder_imgs,
                    pred_rows[0],
                    pred_rows[1],
                ]
                extrapolation_grid = vutils.make_grid(
                    [img for row in feature_rows for img in row],
                    nrow=args.clip_size,
                    padding=2,
                    pad_value=1.0,
                )
                writer.add_image('Train_Vis/PCA_Features_Extrapolation', extrapolation_grid, global_step)

                t_center = 0
                image_shape_ref = orig_imgs_vis[t_center]
                W_img, H_img = image_shape_ref.shape[-1], image_shape_ref.shape[-2]

                K = obj_latents.shape[1]
                sort_idx = torch.arange(K)

                hs = torch.linspace(0.0, 1.0, K, device=device)
                colors = torch.stack([
                    (0.5 * (torch.sin(2 * math.pi * (hs + 0.0)) + 1.0)),
                    (0.5 * (torch.sin(2 * math.pi * (hs + 1.0/3.0)) + 1.0)),
                    (0.5 * (torch.sin(2 * math.pi * (hs + 2.0/3.0)) + 1.0)),
                ], dim=1)  # [K, 3]

                # Temporal Slot Mask Visualization
                for t in vis_frames_full:
                    slot_vis_list_t = []
                    img_rgb_t = orig_imgs_vis[t]
                    raw_mask_t = decoded_mask_probs[0, t].detach().cpu().float() # [K, N]
                        
                    raw_mask_spatial_t = raw_mask_t.view(K, target_h, target_w)
                    attn_map_up_t = F.interpolate(raw_mask_spatial_t.unsqueeze(0), size=(H_img, W_img), mode='bilinear', align_corners=False).squeeze(0) # [K, H, W]
                    
                    for k_idx in sort_idx.tolist():
                        mask_k_t = attn_map_up_t[k_idx].unsqueeze(0) # [1, H, W]
                        color_k = colors[k_idx].view(3, 1, 1).to(img_rgb_t.device)
                        vis_k_t = img_rgb_t * (1.0 - mask_k_t * 0.65) + color_k * (mask_k_t * 0.65)
                        slot_vis_list_t.append(vis_k_t)

                    slot_grid = vutils.make_grid(slot_vis_list_t, nrow=int(math.ceil(math.sqrt(K))))
                    writer.add_image(f'Train_Vis/Decoder_Slot_Masks_t{t}', slot_grid, global_step)

                pred_last_mask_probs = strategy_results["back_extrapolation"]["pred_mask_probs"]
                last_target_index = int(strategy_results["back_extrapolation"]["target_indices"][-1].item())
                pred_mask_last = pred_last_mask_probs[0, -1].detach().cpu().float().view(K, target_h, target_w)
                pred_mask_last_up = F.interpolate(pred_mask_last.unsqueeze(0), size=(H_img, W_img), mode='bilinear', align_corners=False).squeeze(0)
                pred_slot_vis_list = []
                last_img_rgb = orig_imgs_vis[last_target_index]
                for k_idx in sort_idx.tolist():
                    mask_k_t = pred_mask_last_up[k_idx].unsqueeze(0)
                    color_k = colors[k_idx].view(3, 1, 1).to(last_img_rgb.device)
                    vis_k_t = last_img_rgb * (1.0 - mask_k_t * 0.65) + color_k * (mask_k_t * 0.65)
                    pred_slot_vis_list.append(vis_k_t)
                pred_slot_grid = vutils.make_grid(pred_slot_vis_list, nrow=int(math.ceil(math.sqrt(K))))
                writer.add_image('Train_Vis/Predictor_Last_Frame_Masks', pred_slot_grid, global_step)

            global_step += 1

        scheduler.step()

        if epoch % 1 == 0 and is_main_process:
            state_dict_ext = extractor.module.state_dict() if isinstance(extractor, DDP) else extractor.state_dict()
            state_dict_student = vjepa_student.module.state_dict() if isinstance(vjepa_student, DDP) else vjepa_student.state_dict()
            state_dict_inverse = inverse_condition.module.state_dict() if isinstance(inverse_condition, DDP) else inverse_condition.state_dict()
            state_dict_mask_predictor = mask_predictor.module.state_dict() if isinstance(mask_predictor, DDP) else mask_predictor.state_dict()
            ckpt = {
                'epoch': epoch,
                'global_step': global_step,
                'extractor_state': state_dict_ext,
                'student_state': state_dict_student,
                'inverse_condition_state': state_dict_inverse,
                'mask_predictor_state': state_dict_mask_predictor,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(ckpt, os.path.join(args.save_dir, f'init_epoch_{epoch}.pt'))
            torch.save(ckpt, os.path.join(args.save_dir, 'latest.pt'))

    if is_main_process and writer is not None:
        writer.close()
    if is_main_process:
        state_dict_ext = extractor.module.state_dict() if isinstance(extractor, DDP) else extractor.state_dict()
        state_dict_student = vjepa_student.module.state_dict() if isinstance(vjepa_student, DDP) else vjepa_student.state_dict()
        state_dict_inverse = inverse_condition.module.state_dict() if isinstance(inverse_condition, DDP) else inverse_condition.state_dict()
        state_dict_mask_predictor = mask_predictor.module.state_dict() if isinstance(mask_predictor, DDP) else mask_predictor.state_dict()
        ckpt = {
            'epoch': args.epochs - 1,
            'global_step': global_step,
            'extractor_state': state_dict_ext,
            'student_state': state_dict_student,
            'inverse_condition_state': state_dict_inverse,
            'mask_predictor_state': state_dict_mask_predictor,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'latest.pt'))

if __name__ == '__main__':
    main()
