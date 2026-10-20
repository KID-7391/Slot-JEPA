import math
import os
import sys
import time
import logging
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision import transforms
from torchvision.transforms import functional as TF
from einops import rearrange


_RELEASE_ROOT = Path(__file__).resolve().parents[3]
_SHARED_SRC = _RELEASE_ROOT / "src"
for _path in (_SHARED_SRC,):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from slot_jepa.losses import feature_recon_loss
from slot_jepa.predictors import ActionConditionalMaskPredictor
from models.sequential_slot_predictor import (
    SequentialActionConditionalMaskPredictor,
)
from slot_jepa.slots import TwoStageExtractor
from slot_jepa.encoders import vision_transformer as realm_vit
from planning.objectives import (
    _mask_kl_loss,
    _mask_l1_loss,
    _mask_sobel_loss,
    _slot_area_normalizer,
    _slot_change_weights,
    _uot_mask_loss,
)
from planning.pushobj_context import PUSHOBJ_STRUCTURED_CONTEXT_NAMES


log = logging.getLogger(__name__)


def _build_student_from_config(model_cfg, crop_size, patch_size):
    img_as_video_nframes = int(model_cfg.get("vjepa_img_as_video_nframes", 2))
    model_name = model_cfg.get("vjepa_model_name", model_cfg.get("model_name", "vit_base"))
    model = realm_vit.__dict__[model_name](
        img_size=crop_size,
        num_frames=img_as_video_nframes,
        patch_size=patch_size,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
        interpolate_rope=True,
    )
    # RoPE interpolation must use this model's training grid as its coordinate
    # reference. The upstream attention module otherwise defaults to the
    # original 256px pretraining grid (16 for patch_size=16), which is wrong
    # for our 384px training inputs. At training resolution this is an identity
    # mapping; larger inference grids are compressed back into this range.
    rope_reference_grid_size = int(crop_size) // int(patch_size)
    for block in model.blocks:
        attention = getattr(block, "attn", None)
        if attention is not None and hasattr(attention, "interpolate_rope"):
            attention.interpolate_rope = True
            attention.pretrained_grid_size = rope_reference_grid_size

    def forward_prehook(_module, inputs):
        image = inputs[0]
        return (image.unsqueeze(2).repeat(1, 1, img_as_video_nframes, 1, 1),)

    model.register_forward_pre_hook(forward_prehook)
    return model


def _build_scaled_object_branch_scales(
    input_size,
    patch_size,
    reference_input_size=384,
    reference_scales=None,
):
    if reference_scales is None:
        reference_scales = [
            {"spatial_size": 16, "dim": 48, "depth": 4, "num_heads": 3},
            {"spatial_size": 12, "dim": 96, "depth": 4, "num_heads": 3},
            {"spatial_size": 6, "dim": 196, "depth": 4, "num_heads": 4},
        ]
    # Use the explicitly configured object pyramid when supplied. Otherwise
    # retain the original pretrained 16/12/6 resolutions for backward
    # compatibility with existing non-PushObj runs and checkpoints.
    used_sizes = set()
    scales = []
    for scale in reference_scales:
        spatial_size = int(scale["spatial_size"])
        if spatial_size in used_sizes:
            continue
        used_sizes.add(spatial_size)
        scales.append(
            {
                "spatial_size": spatial_size,
                "dim": int(scale["dim"]),
                "depth": int(scale.get("depth", 4)),
                "num_heads": int(scale.get("num_heads", 3)),
            }
        )
    scales.sort(key=lambda x: x["spatial_size"], reverse=True)
    return scales


def _resolve_repo_relative_path(path_str):
    candidate = os.path.abspath(path_str)
    if os.path.exists(candidate):
        return candidate
    if not os.path.isabs(path_str):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidate = os.path.abspath(os.path.join(repo_root, path_str))
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(path_str)


def _load_color_channel_weights_artifact(path, embed_dim):
    """Load the shared frozen encoder channel weights calibrated at 512px."""
    resolved_path = _resolve_repo_relative_path(path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(
            f"Color-channel weight artifact not found: {resolved_path}"
        )
    payload = torch.load(resolved_path, map_location="cpu")
    if not isinstance(payload, dict) or int(payload.get("format_version", -1)) != 1:
        raise ValueError(
            f"Unsupported color-channel artifact format in {resolved_path}"
        )
    if int(payload.get("encoder_block", -1)) != 12:
        raise ValueError("Planar color-channel weights must target encoder block 12")
    if int(payload.get("embed_dim", -1)) != int(embed_dim):
        raise ValueError(
            "Color-channel artifact embed_dim mismatch: "
            f"{payload.get('embed_dim')} != {embed_dim}"
        )
    if int(payload.get("calibration_input_size", -1)) != 512:
        raise ValueError("Planar color-channel weights must be calibrated at 512px")
    weights = payload.get("weights")
    if weights is None:
        raise ValueError("Color-channel artifact is missing `weights`")
    weights = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
    if weights.shape != (int(embed_dim),):
        raise ValueError(
            "Color-channel weight shape mismatch: "
            f"{tuple(weights.shape)} != ({embed_dim},)"
        )
    if not torch.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ValueError("Color-channel weights must be finite and strictly positive")
    return resolved_path, weights


class SlotStateAdapter(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, out_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, state):
        return self.net(state)


class SlotMaskWorldModel(nn.Module):
    is_slot_world_model = True

    def __init__(
        self,
        num_hist,
        num_pred,
        state_dim,
        action_dim,
        state_mean,
        state_std,
        action_mean,
        action_std,
        frameskip,
        env_name,
        realm_ckpt_path="artifacts/checkpoints/pretrain/slot_jepa_pretrained.pt",
        realm_config_path="artifacts/checkpoints/pretrain/slot_jepa_pretrained.yaml",
        input_size=256,
        patch_size=16,
        slot_spatial_size=None,
        num_objects=8,
        window_size=(1, 1),
        slots_per_window=4,
        visual_normalization="centered",
        encoder_contrast_scale=2.0,
        encoder_flip_tta=False,
        color_channel_weights_path=None,
        color_channel_preserve_token_norm=True,
        mask_softmax_temperature=1.0,
        mask_sharpen_gamma=1.0,
        predictor_state_dim=32,
        predictor_architecture="transition_memory",
        predictor_num_obj_tokens=1,
        predictor_object_branch_scales=None,
        predictor_internal_spatial_size=None,
        predictor_use_warp_concat=True,
        predictor_use_extrinsics=True,
        predictor_extrinsics_dim=32,
        predictor_transport_loss_weight=1.0,
        predictor_mask_l1_loss_weight=1.0,
        predictor_mask_kl_loss_weight=0.0,
        predictor_mask_sobel_loss_weight=0.0,
        predictor_recon_loss_weight=0.1,
        sinkhorn_iters=5,
        sinkhorn_entropy_reg=0.05,
        unbalanced_ot_rho=0.2,
        unbalanced_ot_eps=1.0e-8,
        uot_batch_size=128,
        mask_transport_area_normalize=False,
        mask_transport_area_normalize_eps=1.0e-3,
        slot_change_weighting=False,
        slot_change_min_weight=0.05,
        slot_change_max_weight=5.0,
        predictor_velocity_clip=None,
        predictor_proprio_indices=None,
        predictor_proprio_loss_weight=1.0,
        proprio_reconstruction_loss_weight=1.0,
        retrieval_num_context=5,
        retrieval_candidate_pool_size=None,
        training_retrieval_context=False,
        training_retrieval_exclude_same_layout=False,
        training_retrieval_same_layout_only=False,
        training_rollout_steps=1,
        training_teacher_force_weight=1.0,
        training_autoregressive_weight=0.0,
        training_context_generation_strategy=None,
        retrieval_bank_size=32768,
        retrieval_distance_chunk_size=32768,
        retrieval_visual_distance_weight=1.0,
        train_student=False,
        train_extractor=False,
        train_predictor=True,
        train_decoder=False,
        debug_timing=False,
        load_pretrained_weights=True,
        **_ignored,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.frameskip = frameskip
        self.env_name = env_name
        self.predictor_architecture = str(predictor_architecture)
        if self.predictor_architecture not in {
            "transition_memory",
            "sequential",
        }:
            raise ValueError(
                "predictor_architecture must be transition_memory or sequential, "
                f"got {self.predictor_architecture!r}"
            )
        self.train_encoder = bool(train_student or train_extractor)
        self.train_predictor = bool(train_predictor)
        self.train_decoder = bool(train_decoder)
        self.predictor_transport_loss_weight = float(predictor_transport_loss_weight)
        self.predictor_mask_l1_loss_weight = float(predictor_mask_l1_loss_weight)
        self.predictor_mask_kl_loss_weight = float(predictor_mask_kl_loss_weight)
        if self.predictor_mask_kl_loss_weight < 0.0:
            raise ValueError("predictor_mask_kl_loss_weight must be non-negative")
        self.predictor_mask_sobel_loss_weight = float(
            predictor_mask_sobel_loss_weight
        )
        if self.predictor_mask_sobel_loss_weight < 0.0:
            raise ValueError(
                "predictor_mask_sobel_loss_weight must be non-negative"
            )
        self.predictor_recon_loss_weight = float(predictor_recon_loss_weight)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_entropy_reg = float(sinkhorn_entropy_reg)
        self.unbalanced_ot_rho = float(unbalanced_ot_rho)
        self.unbalanced_ot_eps = float(unbalanced_ot_eps)
        self.uot_batch_size = max(1, int(uot_batch_size))
        self.mask_transport_area_normalize = bool(mask_transport_area_normalize)
        self.mask_transport_area_normalize_eps = float(
            mask_transport_area_normalize_eps
        )
        self.slot_change_weighting = bool(slot_change_weighting)
        self.slot_change_min_weight = float(slot_change_min_weight)
        self.slot_change_max_weight = float(slot_change_max_weight)
        # PointMaze clips qvel immediately before every primitive simulator
        # action. Canonicalize frame-boundary states to the state that the next
        # transition actually receives. The env-based fallback also fixes
        # inference from checkpoints whose saved Hydra config predates this
        # explicit option.
        if predictor_velocity_clip is None and self.env_name == "point_maze":
            predictor_velocity_clip = 5.0
        self.predictor_velocity_clip = (
            None
            if predictor_velocity_clip is None
            else float(predictor_velocity_clip)
        )
        if self.predictor_velocity_clip is not None and self.predictor_velocity_clip <= 0:
            raise ValueError("predictor_velocity_clip must be positive or null")
        if self.predictor_velocity_clip is not None:
            log.info(
                "Canonicalizing %s predictor velocity state[2:4] to [-%.3g, %.3g]",
                self.env_name,
                self.predictor_velocity_clip,
                self.predictor_velocity_clip,
            )
        self._uot_kernel_cache = {}
        self.predictor_proprio_loss_weight = float(predictor_proprio_loss_weight)
        self.proprio_reconstruction_loss_weight = float(
            proprio_reconstruction_loss_weight
        )
        if self.proprio_reconstruction_loss_weight < 0.0:
            raise ValueError("proprio_reconstruction_loss_weight must be non-negative")
        self.training_retrieval_context = bool(training_retrieval_context)
        self.training_retrieval_exclude_same_layout = bool(
            training_retrieval_exclude_same_layout
        )
        self.training_retrieval_same_layout_only = bool(
            training_retrieval_same_layout_only
        )
        self.training_rollout_steps = int(training_rollout_steps)
        self.training_teacher_force_weight = float(training_teacher_force_weight)
        self.training_autoregressive_weight = float(training_autoregressive_weight)
        self.training_context_generation_strategy = (
            None
            if training_context_generation_strategy is None
            else str(training_context_generation_strategy)
        )
        if self.training_context_generation_strategy not in (
            None,
            "pushobj_structured_torque",
        ):
            raise ValueError(
                "Unsupported training_context_generation_strategy: "
                f"{self.training_context_generation_strategy!r}"
            )
        if not 1 <= self.training_rollout_steps <= self.num_pred:
            raise ValueError(
                "training_rollout_steps must be in [1, num_pred], got "
                f"{self.training_rollout_steps} for num_pred={self.num_pred}"
            )
        if self.training_teacher_force_weight < 0.0:
            raise ValueError("training_teacher_force_weight must be non-negative")
        if self.training_autoregressive_weight < 0.0:
            raise ValueError("training_autoregressive_weight must be non-negative")
        if (
            self.training_teacher_force_weight
            + self.training_autoregressive_weight
            <= 0.0
        ):
            raise ValueError("At least one training rollout loss weight must be positive")
        if (
            self.training_retrieval_exclude_same_layout
            and self.training_retrieval_same_layout_only
        ):
            raise ValueError(
                "training retrieval cannot both exclude and require the query layout"
            )
        if (
            self.training_retrieval_exclude_same_layout
            or self.training_retrieval_same_layout_only
        ) and not self.training_retrieval_context:
            raise ValueError(
                "layout-filtered training retrieval requires "
                "training_retrieval_context=true"
            )
        self.retrieval_num_context = int(retrieval_num_context)
        self.retrieval_candidate_pool_size = int(
            self.retrieval_num_context
            if retrieval_candidate_pool_size is None
            else retrieval_candidate_pool_size
        )
        self.retrieval_bank_size = int(retrieval_bank_size)
        self.retrieval_distance_chunk_size = int(retrieval_distance_chunk_size)
        self.retrieval_visual_distance_weight = float(
            retrieval_visual_distance_weight
        )
        if self.retrieval_num_context < 1:
            raise ValueError("retrieval_num_context must be positive")
        if self.training_retrieval_context and self.num_hist != 1:
            raise ValueError(
                "training_retrieval_context requires num_hist=1: the two input "
                "frames are the query source and target"
            )
        if self.predictor_architecture == "sequential":
            if self.training_retrieval_context:
                raise ValueError(
                    "Sequential A1 training uses contiguous temporal context, "
                    "not retrieval-context training"
                )
            if self.num_hist < 2:
                raise ValueError("Sequential A1 requires num_hist >= 2")
            if self.training_autoregressive_weight != 0.0:
                raise ValueError(
                    "Sequential A1 currently requires "
                    "training_autoregressive_weight=0"
                )
            if (
                predictor_use_extrinsics
                and int(predictor_extrinsics_dim) != int(predictor_state_dim)
            ):
                raise ValueError(
                    "Sequential A1 reuses the observed frame-t state/action "
                    "context for both condition tokens, so "
                    "predictor_extrinsics_dim must equal predictor_state_dim"
                )
        if self.retrieval_bank_size < self.retrieval_num_context:
            raise ValueError("retrieval_bank_size must cover retrieval_num_context")
        if self.retrieval_candidate_pool_size < self.retrieval_num_context:
            raise ValueError(
                "retrieval_candidate_pool_size must cover retrieval_num_context"
            )
        if self.retrieval_distance_chunk_size < 1:
            raise ValueError("retrieval_distance_chunk_size must be positive")
        if self.retrieval_visual_distance_weight < 0.0:
            raise ValueError("retrieval_visual_distance_weight must be non-negative")
        self._retrieval_bank_path = None
        self._retrieval_source_states = None
        self._retrieval_target_states = None
        self._retrieval_source_proprio = None
        self._retrieval_target_proprio = None
        self._retrieval_actions = None
        self._retrieval_source_mask_logits = None
        self._retrieval_target_mask_logits = None
        self._retrieval_sample_to_bank_index = None
        self._retrieval_min_same_layout_count = None
        self._retrieval_min_other_layout_count = None
        self._retrieval_layout_ids = None
        self._retrieval_context_modes = None
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        if self.input_size % self.patch_size != 0:
            raise ValueError("input_size must be divisible by patch_size")
        self.encoder_spatial_size = self.input_size // self.patch_size
        self.spatial_size = (
            self.encoder_spatial_size
            if slot_spatial_size is None
            else int(slot_spatial_size)
        )
        if self.spatial_size != self.encoder_spatial_size:
            raise ValueError(
                "Released planar models require the native encoder mask grid: "
                "slot_spatial_size must equal input_size // patch_size, got "
                f"{self.spatial_size} != {self.input_size} // {self.patch_size} "
                f"({self.encoder_spatial_size})"
            )
        self.num_objects = int(num_objects)
        self.slot_window_size = tuple(int(value) for value in window_size)
        self.slots_per_window = int(slots_per_window)
        self.visual_normalization = str(visual_normalization)
        if self.visual_normalization not in ("centered", "imagenet"):
            raise ValueError(
                "visual_normalization must be 'centered' or 'imagenet', got "
                f"{self.visual_normalization!r}"
            )
        self.encoder_contrast_scale = float(encoder_contrast_scale)
        if self.encoder_contrast_scale <= 0.0:
            raise ValueError("encoder_contrast_scale must be positive")
        self.encoder_flip_tta = bool(encoder_flip_tta)
        self.color_channel_preserve_token_norm = bool(
            color_channel_preserve_token_norm
        )
        self.mask_softmax_temperature = float(mask_softmax_temperature)
        if self.mask_softmax_temperature <= 0.0:
            raise ValueError("mask_softmax_temperature must be positive")
        self.mask_sharpen_gamma = float(mask_sharpen_gamma)
        if self.mask_sharpen_gamma <= 0.0:
            raise ValueError("mask_sharpen_gamma must be positive")
        self.raw_state_dim = int(state_dim)
        if self.env_name == "pusht" and predictor_proprio_indices is None:
            raise RuntimeError(
                "PushT predictor_proprio_indices must explicitly select only "
                "controllable pusher state [x, y, vx, vy]. Existing checkpoints "
                "without this setting used privileged block pose and must be retrained."
            )
        if predictor_proprio_indices is None:
            proprio_indices = torch.arange(self.raw_state_dim, dtype=torch.long)
        else:
            proprio_indices = torch.as_tensor(
                predictor_proprio_indices, dtype=torch.long
            )
            if proprio_indices.ndim != 1 or proprio_indices.numel() == 0:
                raise ValueError("predictor_proprio_indices must be a non-empty 1-D list")
            if torch.unique(proprio_indices).numel() != proprio_indices.numel():
                raise ValueError("predictor_proprio_indices must not contain duplicates")
            if int(proprio_indices.min()) < 0 or int(proprio_indices.max()) >= self.raw_state_dim:
                raise ValueError(
                    "predictor_proprio_indices are outside raw state dimension "
                    f"{self.raw_state_dim}: {proprio_indices.tolist()}"
                )
        self.state_dim = int(proprio_indices.numel())
        self.action_chunk_dim = int(action_dim)
        self.action_dim = int(action_mean.numel())
        self.predictor_state_dim = int(predictor_state_dim)
        self.predictor_use_extrinsics = bool(predictor_use_extrinsics)
        self.predictor_extrinsics_dim = int(predictor_extrinsics_dim)
        self.realm_ckpt_path = _resolve_repo_relative_path(realm_ckpt_path)
        self.realm_config_path = _resolve_repo_relative_path(realm_config_path)
        self.debug_timing = bool(debug_timing)
        self.latest_timing = {}

        state_mean = torch.as_tensor(state_mean, dtype=torch.float32)
        state_std = torch.as_tensor(state_std, dtype=torch.float32).clamp_min(1.0e-6)
        if state_mean.numel() != self.raw_state_dim or state_std.numel() != self.raw_state_dim:
            raise ValueError(
                "state statistics must describe the raw environment state, got "
                f"mean={state_mean.numel()} std={state_std.numel()} raw={self.raw_state_dim}"
            )
        state_mean = state_mean.index_select(0, proprio_indices)
        state_std = state_std.index_select(0, proprio_indices)
        action_mean = torch.as_tensor(action_mean, dtype=torch.float32)
        action_std = torch.as_tensor(action_std, dtype=torch.float32).clamp_min(1.0e-6)
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std)
        self.register_buffer(
            "predictor_proprio_indices", proprio_indices, persistent=True
        )
        # Version 2 means predictor conditioning and predicted proprio targets
        # use an explicitly declared controllable-state projection.  PushT
        # checkpoints without this marker used privileged block pose state.
        self.register_buffer(
            "predictor_proprio_schema_version",
            torch.tensor(2 if predictor_proprio_indices is not None else 1, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)
        # Prevent older predictor/context semantics from being loaded silently.
        self.register_buffer(
            "transition_pair_schema_version",
            torch.tensor(6, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "training_retrieval_schema_version",
            torch.tensor(
                3
                if self.training_retrieval_same_layout_only
                else (2 if self.training_retrieval_exclude_same_layout else 1),
                dtype=torch.int64,
            ),
            persistent=True,
        )

        with open(self.realm_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg["model"]
        self.embed_dim = int(model_cfg.get("embed_dim", 768))
        self.color_channel_weights_path = None
        color_channel_weights = torch.ones(self.embed_dim, dtype=torch.float32)
        if color_channel_weights_path is not None:
            (
                self.color_channel_weights_path,
                color_channel_weights,
            ) = _load_color_channel_weights_artifact(
                color_channel_weights_path,
                embed_dim=self.embed_dim,
            )
            log.info(
                "Loaded frozen block-12 color-channel weights from %s "
                "(512px calibration shared at input=%d, min=%.4f, mean=%.4f, "
                "max=%.4f, preserve_norm=%s)",
                self.color_channel_weights_path,
                self.input_size,
                float(color_channel_weights.min()),
                float(color_channel_weights.mean()),
                float(color_channel_weights.max()),
                self.color_channel_preserve_token_norm,
            )
        self.color_channel_reweight_enabled = self.color_channel_weights_path is not None
        # The artifact is immutable external model data, not downstream state.
        # Keeping this buffer non-persistent prevents duplicating it in every
        # task checkpoint while still moving it with model.to(device).
        self.register_buffer(
            "color_channel_weights",
            color_channel_weights,
            persistent=False,
        )
        self.slot_feature_dim = int(model_cfg.get("slot_feature_dim", 192))
        self.predictor_embed_dim = int(model_cfg.get("pred_embed_dim", 192))
        self.predictor_depth = int(model_cfg.get("pred_depth", 12))
        self.predictor_num_heads = int(model_cfg.get("pred_num_heads", 3))
        self.predictor_internal_spatial_size = int(
            self.spatial_size
            if predictor_internal_spatial_size is None
            else predictor_internal_spatial_size
        )
        if self.predictor_internal_spatial_size > self.spatial_size:
            raise ValueError(
                "predictor_internal_spatial_size must not exceed "
                "slot_spatial_size, got "
                f"{self.spatial_size} and {self.predictor_internal_spatial_size}"
            )
        self.object_branch_scales = _build_scaled_object_branch_scales(
            input_size=self.spatial_size * self.patch_size,
            patch_size=self.patch_size,
            reference_scales=predictor_object_branch_scales,
        )
        self.autocast_dtype = torch.bfloat16 if torch.cuda.is_available() else None
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

        self.image_transform = transforms.Resize((self.input_size, self.input_size))

        self.student = _build_student_from_config(model_cfg, self.input_size, self.patch_size)
        self.extractor = TwoStageExtractor(
            window_size=list(self.slot_window_size),
            slots_per_window=self.slots_per_window,
            global_k=self.num_objects,
            D_in=self.embed_dim,
            mask_temp=model_cfg.get("mask_temp", 0.05),
            L_max=model_cfg.get("L_max", max(24, self.num_objects)),
            use_decoder_checkpoint=model_cfg.get("use_decoder_checkpoint", False),
        )
        predictor_class = (
            SequentialActionConditionalMaskPredictor
            if self.predictor_architecture == "sequential"
            else ActionConditionalMaskPredictor
        )
        self.predictor = predictor_class(
            embed_dim=self.embed_dim,
            feature_dim=self.slot_feature_dim,
            predictor_embed_dim=self.predictor_embed_dim,
            depth=self.predictor_depth,
            num_heads=self.predictor_num_heads,
            num_context_frames=max(self.num_hist, self.retrieval_num_context),
            spatial_size=self.spatial_size,
            num_objects=self.num_objects,
            state_dim=self.predictor_state_dim,
            use_extrinsics=self.predictor_use_extrinsics,
            extrinsics_dim=self.predictor_extrinsics_dim,
            num_obj_tokens=int(predictor_num_obj_tokens),
            concat_mask_input=bool(model_cfg.get("concat_mask_input", True)),
            use_warp_concat=bool(predictor_use_warp_concat),
            patch_size=self.patch_size,
            use_checkpoint=False,
            object_branch_scales=self.object_branch_scales,
            proprio_output_dim=self.predictor_state_dim,
            predictor_spatial_size=self.predictor_internal_spatial_size,
        )
        # The reference-derived ordered slot bank is part of the predictor's
        # persistent state. It is initialized once before a fresh training run
        # and thereafter restored directly from every Slot-AC checkpoint.
        self.predictor.register_parameter(
            "fixed_obj_latents",
            nn.Parameter(
                torch.zeros(
                    1, self.num_objects, self.embed_dim, dtype=torch.float32
                ),
                requires_grad=False,
            ),
        )
        self.predictor.register_buffer(
            "fixed_slots_initialized",
            torch.zeros((), dtype=torch.bool),
            persistent=True,
        )
        self.predictor.debug_timing = self.debug_timing
        self.proprio_encoder = SlotStateAdapter(
            self.state_dim,
            hidden_dim=128,
            out_dim=self.predictor_state_dim,
        )
        # Auxiliary inverse map used only for representation supervision.  A
        # real input embedding must reconstruct every normalized proprio
        # component, preventing the 32-D representation from retaining only
        # the visually useful position coordinates and discarding velocity.
        self.proprio_decoder = SlotStateAdapter(
            self.predictor_state_dim,
            hidden_dim=128,
            out_dim=self.state_dim,
        )
        self.action_encoder = SlotStateAdapter(
            self.action_chunk_dim,
            hidden_dim=128,
            out_dim=self.predictor_state_dim,
        )
        self.transition_adapter = SlotStateAdapter(
            2 * self.predictor_state_dim,
            hidden_dim=128,
            out_dim=self.predictor_state_dim,
        )
        # TMP needs a learned target-state token for its query pair. A1 has no
        # target/query pair construction, so omit this otherwise-dead parameter.
        if self.predictor_architecture == "sequential":
            self.register_parameter("query_state_embedding", None)
        else:
            self.query_state_embedding = nn.Parameter(
                torch.zeros(1, 1, self.predictor_state_dim)
            )
            nn.init.normal_(self.query_state_embedding, std=0.02)
        # Keep the old public name for checkpoint compatibility and external
        # callers, but all context/query tokens now share the same semantics.
        self.state_adapter = self.transition_adapter
        if self.predictor_use_extrinsics:
            # Retained only so older REALM/task state dictionaries remain
            # loadable. Transition-pair mode uses this condition-token slot
            # for observed/learned target proprio instead of camera pose.
            self.learned_extrinsics = nn.Parameter(
                torch.zeros(1, 1, self.predictor_extrinsics_dim),
                requires_grad=False,
            )
        else:
            self.learned_extrinsics = None

        if load_pretrained_weights:
            self._load_pretrained_weights()

        if not train_student:
            for param in self.student.parameters():
                param.requires_grad = False
        if not train_extractor:
            for param in self.extractor.parameters():
                param.requires_grad = False

    def split_pretrained_and_new_parameters(self):
        """Return disjoint trainable parameter groups for differential LR.

        The REALM checkpoint initializes ``student``, ``extractor`` and
        ``predictor``.  The state/action conditioning and proprio prediction
        modules below are task-specific additions and need a larger learning
        rate when adapting the pretrained predictor to control data.
        """
        new_modules = [
            self.proprio_encoder,
            self.proprio_decoder,
            self.action_encoder,
            self.transition_adapter,
            self.predictor.proprio_head,
            self.predictor.recon_branch.pair_fusion,
            *[branch.pair_fusion for branch in self.predictor.object_branch.branches],
        ]
        new_modules = [module for module in new_modules if module is not None]
        new_parameter_ids = {
            id(param)
            for module in new_modules
            for param in module.parameters()
            if param.requires_grad
        }
        for param in (self.query_state_embedding, self.learned_extrinsics):
            if param is not None and param.requires_grad:
                new_parameter_ids.add(id(param))

        pretrained_parameters = []
        new_parameters = []
        for param in self.parameters():
            if not param.requires_grad:
                continue
            if id(param) in new_parameter_ids:
                new_parameters.append(param)
            else:
                pretrained_parameters.append(param)

        grouped_ids = {id(param) for param in pretrained_parameters + new_parameters}
        expected_ids = {
            id(param) for param in self.parameters() if param.requires_grad
        }
        if grouped_ids != expected_ids or len(grouped_ids) != (
            len(pretrained_parameters) + len(new_parameters)
        ):
            raise RuntimeError("Slot optimizer parameter groups overlap or are incomplete")
        if not pretrained_parameters or not new_parameters:
            raise RuntimeError(
                "Expected non-empty pretrained and task-specific parameter groups"
            )
        return pretrained_parameters, new_parameters

    def validate_proprio_encoder_checkpoint(self, model_state):
        """Require every learned proprio-encoder tensor in a Slot checkpoint."""
        expected = {
            f"proprio_encoder.{key}" for key in self.proprio_encoder.state_dict()
        }
        missing = sorted(expected.difference(model_state))
        if missing:
            raise RuntimeError(
                "Slot checkpoint is missing learned proprio_encoder parameters: "
                + ", ".join(missing)
            )

    @staticmethod
    def _is_shared_pretrain_key(key):
        """Return whether ``key`` is restored from the shared pretrain file."""
        return key.startswith(("student.", "extractor."))

    def downstream_predictor_state_dict(self):
        """Return the task-specific state without duplicated frozen backbones.

        Besides ``predictor.*``, the planar predictor uses learned action,
        proprioception, and transition adapters plus task/schema buffers.  All
        of those values are part of the downstream predictor checkpoint.  The
        frozen student and slot extractor are deliberately omitted because the
        constructor restores them from the shared Slot-JEPA pretrain checkpoint.
        """
        return {
            key: value
            for key, value in self.state_dict().items()
            if not self._is_shared_pretrain_key(key)
        }

    def load_downstream_predictor_state_dict(
        self, predictor_state, merge_shared_pretrain=False
    ):
        """Merge shared pretraining and downstream state, then load strictly.

        During inference ``merge_shared_pretrain`` lets the constructor defer
        loading its shared student/extractor weights.  This method then builds
        one complete state dictionary in memory and performs one strict model
        load, instead of first loading the shared model and subsequently
        producing an ``_IncompatibleKeys`` result while overlaying the compact
        downstream checkpoint.
        """
        expected = set(self.downstream_predictor_state_dict())
        provided = set(predictor_state)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing[:20]}")
            if unexpected:
                details.append(f"unexpected={unexpected[:20]}")
            raise RuntimeError(
                "Invalid planar predictor checkpoint (" + "; ".join(details) + ")"
            )

        merged_state = self.state_dict()
        if merge_shared_pretrain:
            shared_checkpoint = torch.load(self.realm_ckpt_path, map_location="cpu")
            log.info(
                "Merging shared slot-ac pretrained weights from %s with the "
                "compact downstream checkpoint",
                self.realm_ckpt_path,
            )
            for module_name, checkpoint_key in (
                ("student", "student_state"),
                ("extractor", "extractor_state"),
            ):
                component_state = shared_checkpoint.get(checkpoint_key)
                if component_state is None:
                    raise KeyError(
                        f"Expected {checkpoint_key} in shared pretrain checkpoint"
                    )
                for key, value in component_state.items():
                    full_key = f"{module_name}.{key}"
                    if full_key not in merged_state:
                        # Preserve the former strict=False behavior for shared
                        # checkpoint entries belonging to an older architecture.
                        continue
                    if merged_state[full_key].shape != value.shape:
                        raise RuntimeError(
                            "Shared pretrain tensor shape mismatch for "
                            f"{full_key}: checkpoint={tuple(value.shape)}, "
                            f"model={tuple(merged_state[full_key].shape)}"
                        )
                    merged_state[full_key] = value

        merged_state.update(predictor_state)
        # The dictionary now covers the entire model. Shape mismatches, missing
        # model tensors, and unexpected downstream tensors are hard failures.
        return self.load_state_dict(merged_state, strict=True)

    def _autocast_context(self):
        if self.autocast_dtype is None or not torch.cuda.is_available():
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _fp32_context(self):
        if not torch.cuda.is_available():
            return nullcontext()
        return torch.autocast(device_type="cuda", enabled=False)

    def _slot_extract_context(self):
        if self.train_encoder:
            return nullcontext()
        return torch.no_grad()

    def _sync_device(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _cuda_mem_mb():
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 2)

    @staticmethod
    def _cuda_peak_mb():
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 2)

    def _reset_cuda_peak(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _tic(self):
        self._sync_device()
        return time.perf_counter()

    def _toc(self, start_time):
        self._sync_device()
        return time.perf_counter() - start_time

    def _load_pretrained_weights(self):
        ckpt = torch.load(self.realm_ckpt_path, map_location="cpu")
        log.info("Loading slot-ac pretrained weights from %s", self.realm_ckpt_path)
        student_result = self.student.load_state_dict(ckpt["student_state"], strict=False)
        extractor_result = self.extractor.load_state_dict(ckpt["extractor_state"], strict=False)
        log.info(
            "Loaded student_state: missing=%d unexpected=%d | extractor_state: missing=%d unexpected=%d",
            len(student_result.missing_keys),
            len(student_result.unexpected_keys),
            len(extractor_result.missing_keys),
            len(extractor_result.unexpected_keys),
        )

        pred_state = ckpt.get("mask_predictor_state")
        if pred_state is None:
            raise KeyError("Expected mask_predictor_state in REALM pretrain checkpoint")

        adapted_state = {}
        target_state = self.predictor.state_dict()
        direct_match_keys = []
        skipped_keys = []
        for key, value in pred_state.items():
            if key not in target_state:
                skipped_keys.append((key, tuple(value.shape), None, "missing_in_target"))
                continue
            tgt = target_state[key]
            if value.shape == tgt.shape:
                adapted_state[key] = value
                direct_match_keys.append(key)
                continue
            # A tensor's axes can contain concatenated semantic groups (for
            # example reconstruction features, masks, warped features, and
            # warped masks). Prefix-copying or interpolating a shape mismatch
            # silently moves those groups to the wrong channels. Keep the
            # target module's own initialization for every mismatched tensor.
            skipped_keys.append(
                (key, tuple(value.shape), tuple(tgt.shape), "shape_mismatch_reinitialized")
            )
        predictor_result = self.predictor.load_state_dict(adapted_state, strict=False)
        expected_task_specific_keys = {
            "fixed_obj_latents",
            "fixed_slots_initialized",
            "query_state_embedding",
        }
        reinitialized_keys = {
            item[0]
            for item in skipped_keys
            if item[3] == "shape_mismatch_reinitialized"
        }
        missing_predictor_keys = [
            key
            for key in predictor_result.missing_keys
            if key not in expected_task_specific_keys
            and key not in reinitialized_keys
            and not key.startswith("proprio_head.")
        ]
        log.info(
            "Loaded mask_predictor_state: matched=%d reinitialized_shape_mismatch=%d skipped=%d | missing_after_load=%d unexpected_after_load=%d",
            len(direct_match_keys),
            sum(item[3] == "shape_mismatch_reinitialized" for item in skipped_keys),
            len(skipped_keys),
            len(missing_predictor_keys),
            len(predictor_result.unexpected_keys),
        )
        if skipped_keys:
            log.warning(
                "Predictor skipped/reinitialized pretrained keys (first 20): %s",
                skipped_keys[:20],
            )
        if missing_predictor_keys:
            log.warning(
                "Predictor missing keys after load (first 50): %s",
                missing_predictor_keys[:50],
            )
        if predictor_result.unexpected_keys:
            log.warning(
                "Predictor unexpected keys after load (first 50): %s",
                predictor_result.unexpected_keys[:50],
            )

    def _select_predictor_proprio(self, state):
        if state.shape[-1] == self.state_dim and self.state_dim != self.raw_state_dim:
            return state
        if state.shape[-1] != self.raw_state_dim:
            raise ValueError(
                "Predictor state must be raw or already projected, got "
                f"last_dim={state.shape[-1]}, raw={self.raw_state_dim}, "
                f"projected={self.state_dim}"
            )
        return state.index_select(
            -1, self.predictor_proprio_indices.to(state.device)
        )

    def _canonicalize_predictor_state(self, state):
        state = self._select_predictor_proprio(state)
        if self.predictor_velocity_clip is None:
            return state
        if self.env_name != "point_maze":
            raise ValueError(
                "predictor_velocity_clip is currently defined only for point_maze"
            )
        if state.shape[-1] < 4:
            raise ValueError(
                f"PointMaze predictor state must contain [x,y,vx,vy], got {tuple(state.shape)}"
            )
        velocity = state[..., 2:4].clamp(
            -self.predictor_velocity_clip,
            self.predictor_velocity_clip,
        )
        return torch.cat([state[..., :2], velocity, state[..., 4:]], dim=-1)

    def normalize_states(self, state):
        state = self._canonicalize_predictor_state(state)
        return (state - self.state_mean) / self.state_std

    def _encode_proprio(self, state):
        return self.proprio_encoder(self.normalize_states(state.float()))

    @property
    def has_retrieval_bank(self):
        return self._retrieval_source_proprio is not None

    @property
    def loaded_retrieval_bank_size(self):
        """Number of training transitions available for inference context."""
        if not self.has_retrieval_bank:
            return 0
        return int(self._retrieval_source_proprio.shape[0])

    @staticmethod
    def _mask_retrieval_descriptor(mask_logits):
        """Compact layout descriptor: per-slot area and normalized centroid."""
        probs = F.softmax(mask_logits.float(), dim=-2)
        num_patches = probs.shape[-1]
        spatial_size = int(math.sqrt(num_patches))
        if spatial_size * spatial_size != num_patches:
            raise ValueError(f"Mask token count must be square, got {num_patches}")
        coords = torch.linspace(0.0, 1.0, spatial_size, device=probs.device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        xx = xx.flatten()
        yy = yy.flatten()
        mass = probs.sum(dim=-1).clamp_min(1.0e-6)
        area = mass / float(num_patches)
        centroid_x = (probs * xx).sum(dim=-1) / mass
        centroid_y = (probs * yy).sum(dim=-1) / mass
        return torch.stack([area, centroid_x, centroid_y], dim=-1).flatten(-2)

    @torch.no_grad()
    def load_retrieval_bank(self, bank_path):
        """Load transition contexts and index them with the final state encoder."""
        bank_path = Path(bank_path).expanduser().resolve()
        payload = torch.load(bank_path, map_location="cpu")
        expected_schema = (
            2 if self.training_context_generation_strategy is not None
            and self.state_dim != self.raw_state_dim
            else 1 if self.training_context_generation_strategy is not None
            else 4
        )
        if int(payload.get("schema_version", -1)) != expected_schema:
            raise RuntimeError(
                f"Retrieval bank {bank_path} has schema "
                f"{payload.get('schema_version')}, expected {expected_schema}"
            )
        if self.state_dim != self.raw_state_dim:
            stored_indices = payload.get("proprio_indices")
            expected_indices = self.predictor_proprio_indices.tolist()
            if stored_indices != expected_indices:
                raise RuntimeError(
                    f"Retrieval bank proprio indices {stored_indices} do not match "
                    f"model indices {expected_indices}"
                )
        required = {
            "source_states",
            "target_states",
            "actions",
            "source_mask_logits",
            "target_mask_logits",
        }
        if self.training_context_generation_strategy is None:
            required.update({"sample_indices", "transition_indices", "dataset_size"})
        missing = required.difference(payload)
        if missing:
            raise KeyError(f"Retrieval bank {bank_path} is missing {sorted(missing)}")
        if (
            self.training_retrieval_exclude_same_layout
            or self.training_retrieval_same_layout_only
        ) and "layout_ids" not in payload:
            raise RuntimeError(
                f"Retrieval bank {bank_path} has no layout_ids, but training is "
                "configured to filter contexts by query layout"
            )
        count = int(payload["source_states"].shape[0])
        if int(payload["source_states"].shape[-1]) != self.state_dim:
            raise RuntimeError(
                f"Retrieval bank state dimension {payload['source_states'].shape[-1]} "
                f"does not match predictor proprio dimension {self.state_dim}"
            )
        if count < self.retrieval_num_context:
            raise ValueError(
                f"Retrieval bank has {count} transitions, fewer than "
                f"retrieval_num_context={self.retrieval_num_context}"
            )
        device = next(self.parameters()).device
        self._retrieval_bank_path = str(bank_path)
        source_states_cpu = payload["source_states"].float().cpu()
        target_states_cpu = payload["target_states"].float().cpu()
        # Banks are small (1000 rows for PushObj). Keep state rows beside the
        # masks/actions so every model step avoids a GPU->CPU index transfer
        # followed by a CPU->GPU copy of five tiny context tensors.
        self._retrieval_source_states = source_states_cpu.to(device)
        self._retrieval_target_states = target_states_cpu.to(device)
        self._retrieval_actions = payload["actions"].to(device=device, dtype=torch.float32)
        self._retrieval_layout_ids = (
            None
            if "layout_ids" not in payload
            else payload["layout_ids"].to(device=device, dtype=torch.long).flatten()
        )
        self._retrieval_context_modes = (
            None
            if "context_modes" not in payload
            else payload["context_modes"].to(
                device=device, dtype=torch.long
            ).flatten()
        )
        if self.training_context_generation_strategy == "pushobj_structured_torque":
            if self._retrieval_context_modes is None:
                raise RuntimeError(
                    "Structured PushObj training requires context_modes in the retrieval bank"
                )
            if (
                torch.unique(self._retrieval_context_modes).numel()
                != len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES)
            ):
                raise RuntimeError(
                    "Structured PushObj retrieval bank must contain all five context modes"
                )
            if self._retrieval_layout_ids is not None:
                for layout_id in torch.unique(self._retrieval_layout_ids):
                    layout_modes = self._retrieval_context_modes[
                        self._retrieval_layout_ids == layout_id
                    ]
                    if (
                        torch.unique(layout_modes).numel()
                        != len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES)
                    ):
                        raise RuntimeError(
                            "Structured PushObj retrieval bank is missing a mode "
                            f"for layout {int(layout_id)}"
                        )
        if self._retrieval_layout_ids is not None and len(self._retrieval_layout_ids) != count:
            raise RuntimeError(
                f"Retrieval bank has {len(self._retrieval_layout_ids)} layout ids "
                f"for {count} transitions"
            )
        self._retrieval_sample_to_bank_index = None
        if "sample_indices" in payload:
            sample_indices = payload["sample_indices"].to(
                device=device, dtype=torch.long
            ).flatten()
            transition_indices = payload["transition_indices"].to(
                device=device, dtype=torch.long
            ).flatten()
            max_transitions = max(1, self.num_hist + self.num_pred - 1)
            self._retrieval_sample_to_bank_index = torch.full(
                (int(payload["dataset_size"]), max_transitions),
                -1,
                device=device,
                dtype=torch.long,
            )
            self._retrieval_sample_to_bank_index[
                sample_indices, transition_indices
            ] = torch.arange(count, device=device)
        if self._retrieval_layout_ids is None:
            self._retrieval_min_same_layout_count = count
            self._retrieval_min_other_layout_count = count
        else:
            _, layout_counts = torch.unique(
                self._retrieval_layout_ids, return_counts=True
            )
            self._retrieval_min_same_layout_count = int(layout_counts.min())
            self._retrieval_min_other_layout_count = int(
                count - layout_counts.max()
            )
        self._retrieval_source_mask_logits = payload["source_mask_logits"].to(
            device=device, dtype=torch.float16
        )
        self._retrieval_target_mask_logits = payload["target_mask_logits"].to(
            device=device, dtype=torch.float16
        )
        self._retrieval_visual_descriptor = self._mask_retrieval_descriptor(
            self._retrieval_source_mask_logits
        )
        self._retrieval_visual_mean = self._retrieval_visual_descriptor.mean(
            dim=0, keepdim=True
        )
        self._retrieval_visual_std = self._retrieval_visual_descriptor.std(
            dim=0, keepdim=True
        ).clamp_min(1.0e-4)
        self.refresh_retrieval_state_embeddings()
        log.info(
            "Loaded %d transition contexts from %s on %s",
            count,
            bank_path,
            device,
        )

    @torch.no_grad()
    def refresh_retrieval_state_embeddings(self):
        if self._retrieval_source_states is None:
            raise RuntimeError("No raw retrieval bank has been loaded")
        device = next(self.parameters()).device
        chunk_size = self.retrieval_distance_chunk_size

        def encode_chunks(states):
            return torch.cat(
                [
                    self._encode_proprio(states[start : start + chunk_size].to(device))
                    for start in range(0, states.shape[0], chunk_size)
                ],
                dim=0,
            ).detach()

        self._retrieval_source_proprio = encode_chunks(
            self._retrieval_source_states
        )
        self._retrieval_target_proprio = encode_chunks(
            self._retrieval_target_states
        )
        self._retrieval_proprio_mean = self._retrieval_source_proprio.mean(
            dim=0, keepdim=True
        )
        self._retrieval_proprio_std = self._retrieval_source_proprio.std(
            dim=0, keepdim=True
        ).clamp_min(1.0e-4)

    @staticmethod
    def _sample_retrieval_pool(best_dist, best_index, num_context):
        """Uniformly choose unique pool entries and retain distance-rank order."""
        num_context = min(int(num_context), best_index.shape[1])
        if best_index.shape[1] <= num_context:
            return best_index
        sampled_pool_ranks = torch.topk(
            torch.rand_like(best_dist),
            k=num_context,
            dim=1,
            largest=False,
            sorted=False,
        ).indices.sort(dim=1).values
        return torch.gather(best_index, 1, sampled_pool_ranks)

    @torch.no_grad()
    def _retrieve_transition_context(
        self,
        query_proprio,
        query_mask_logits,
        excluded_indices=None,
        excluded_layout_ids=None,
        required_layout_ids=None,
    ):
        """Retrieve a uniformly sampled subset of the nearest candidate pool."""
        if not self.has_retrieval_bank:
            raise RuntimeError(
                "Slot rollout requires a training transition retrieval bank"
            )
        if query_proprio.ndim != 3 or query_proprio.shape[1] != 1:
            raise ValueError(
                f"Expected query proprio [B,1,D], got {tuple(query_proprio.shape)}"
            )
        query = (query_proprio[:, 0].float() - self._retrieval_proprio_mean) / self._retrieval_proprio_std
        bank = (
            self._retrieval_source_proprio.float() - self._retrieval_proprio_mean
        ) / self._retrieval_proprio_std
        query_visual = self._mask_retrieval_descriptor(query_mask_logits[:, 0])
        query_visual = (
            query_visual - self._retrieval_visual_mean
        ) / self._retrieval_visual_std
        bank_visual = (
            self._retrieval_visual_descriptor - self._retrieval_visual_mean
        ) / self._retrieval_visual_std
        if excluded_indices is not None:
            excluded_indices = torch.as_tensor(
                excluded_indices, device=query.device, dtype=torch.long
            ).flatten()
            if excluded_indices.shape[0] != query.shape[0]:
                raise ValueError(
                    "excluded_indices must contain one bank index per query"
                )
        if excluded_layout_ids is not None and required_layout_ids is not None:
            raise ValueError("Cannot both exclude and require retrieval layout ids")
        if excluded_layout_ids is not None:
            if self._retrieval_layout_ids is None:
                raise RuntimeError(
                    "Cannot exclude query layouts because the retrieval bank has no layout_ids"
                )
            excluded_layout_ids = torch.as_tensor(
                excluded_layout_ids, device=query.device, dtype=torch.long
            ).flatten()
            if excluded_layout_ids.shape[0] != query.shape[0]:
                raise ValueError(
                    "excluded_layout_ids must contain one layout id per query"
                )
        if required_layout_ids is not None:
            if self._retrieval_layout_ids is None:
                raise RuntimeError(
                    "Cannot require query layouts because the retrieval bank has no layout_ids"
                )
            required_layout_ids = torch.as_tensor(
                required_layout_ids, device=query.device, dtype=torch.long
            ).flatten()
            if required_layout_ids.shape[0] != query.shape[0]:
                raise ValueError(
                    "required_layout_ids must contain one layout id per query"
                )
        if self.training_context_generation_strategy == "pushobj_structured_torque":
            pool_size = self.retrieval_num_context
        else:
            if required_layout_ids is not None:
                available = self._retrieval_min_same_layout_count
            elif excluded_layout_ids is not None:
                available = self._retrieval_min_other_layout_count
            else:
                available = bank.shape[0]
            pool_size = min(self.retrieval_candidate_pool_size, available)
            if pool_size < self.retrieval_num_context:
                raise ValueError(
                    "Retrieval bank has too few eligible transitions for the "
                    f"requested {self.retrieval_num_context} contexts"
                )
        best_dist = None
        best_index = None
        structured_best_dist = [None] * len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES)
        structured_best_index = [None] * len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES)
        chunk_size = self.retrieval_distance_chunk_size
        query_sq = query.square().sum(dim=-1, keepdim=True)
        for start in range(0, bank.shape[0], chunk_size):
            bank_chunk = bank[start : start + chunk_size]
            distances = (
                query_sq
                + bank_chunk.square().sum(dim=-1).unsqueeze(0)
                - 2.0 * query @ bank_chunk.transpose(0, 1)
            ).clamp_min_(0.0)
            if excluded_indices is not None:
                excluded_in_chunk = (
                    (excluded_indices >= start)
                    & (excluded_indices < start + bank_chunk.shape[0])
                )
                rows = torch.arange(query.shape[0], device=query.device)
                cols = (excluded_indices - start).clamp(
                    min=0, max=bank_chunk.shape[0] - 1
                )
                distances[rows, cols] = torch.where(
                    excluded_in_chunk,
                    torch.full_like(distances[rows, cols], torch.inf),
                    distances[rows, cols],
                )
            if excluded_layout_ids is not None:
                chunk_layout_ids = self._retrieval_layout_ids[
                    start : start + bank_chunk.shape[0]
                ]
                distances.masked_fill_(
                    chunk_layout_ids.unsqueeze(0)
                    == excluded_layout_ids.unsqueeze(1),
                    torch.inf,
                )
            if required_layout_ids is not None:
                chunk_layout_ids = self._retrieval_layout_ids[
                    start : start + bank_chunk.shape[0]
                ]
                distances.masked_fill_(
                    chunk_layout_ids.unsqueeze(0)
                    != required_layout_ids.unsqueeze(1),
                    torch.inf,
                )
            if self.retrieval_visual_distance_weight > 0.0:
                bank_visual_chunk = bank_visual[start : start + bank_chunk.shape[0]]
                visual_distances = (
                    query_visual.square().sum(dim=-1, keepdim=True)
                    + bank_visual_chunk.square().sum(dim=-1).unsqueeze(0)
                    - 2.0 * query_visual @ bank_visual_chunk.transpose(0, 1)
                ).clamp_min_(0.0)
                distances = distances + (
                    self.retrieval_visual_distance_weight * visual_distances
                )
            if self.training_context_generation_strategy == "pushobj_structured_torque":
                chunk_modes = self._retrieval_context_modes[
                    start : start + bank_chunk.shape[0]
                ]
                for mode_index in range(len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES)):
                    mode_distances = distances.masked_fill(
                        chunk_modes.unsqueeze(0) != mode_index, torch.inf
                    )
                    chunk_best_dist, chunk_position = mode_distances.min(dim=1)
                    chunk_best_index = chunk_position + start
                    if structured_best_dist[mode_index] is None:
                        structured_best_dist[mode_index] = chunk_best_dist
                        structured_best_index[mode_index] = chunk_best_index
                    else:
                        improve = (
                            chunk_best_dist < structured_best_dist[mode_index]
                        )
                        structured_best_dist[mode_index] = torch.where(
                            improve,
                            chunk_best_dist,
                            structured_best_dist[mode_index],
                        )
                        structured_best_index[mode_index] = torch.where(
                            improve,
                            chunk_best_index,
                            structured_best_index[mode_index],
                        )
            if self.training_context_generation_strategy != "pushobj_structured_torque":
                indices = torch.arange(
                    start, start + bank_chunk.shape[0], device=query.device
                ).unsqueeze(0).expand(query.shape[0], -1)
                if best_dist is not None:
                    distances = torch.cat([best_dist, distances], dim=1)
                    indices = torch.cat([best_index, indices], dim=1)
                best_dist, positions = torch.topk(
                    distances,
                    k=min(pool_size, distances.shape[1]),
                    dim=1,
                    largest=False,
                )
                best_index = torch.gather(indices, 1, positions)

        if self.training_context_generation_strategy == "pushobj_structured_torque":
            if self.retrieval_num_context != len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES):
                raise ValueError(
                    "pushobj_structured_torque training requires retrieval_num_context=5"
                )
            selected_distances = torch.stack(structured_best_dist, dim=1)
            best_index = torch.stack(structured_best_index, dim=1)
            if self.debug_timing and not torch.isfinite(selected_distances).all():
                raise RuntimeError(
                    "No eligible transition exists for at least one of the "
                    "five structured PushObj context modes"
                )
            best_index = torch.gather(
                best_index, 1, selected_distances.argsort(dim=1)
            )
        else:
            # Uniformly sample without replacement from the top candidate pool.
            # Sorting the sampled pool ranks afterwards preserves the predictor's
            # nearest-to-farthest temporal ordering while still diversifying which
            # transitions are used at each MPC retrieval boundary.
            best_index = self._sample_retrieval_pool(
                best_dist,
                best_index,
                self.retrieval_num_context,
            )
        if self.debug_timing and excluded_layout_ids is not None:
            selected_layout_ids = self._retrieval_layout_ids[best_index]
            if (
                selected_layout_ids
                == excluded_layout_ids.unsqueeze(1)
            ).any():
                raise RuntimeError(
                    "Same-layout transition survived retrieval layout exclusion"
                )
        if self.debug_timing and required_layout_ids is not None:
            selected_layout_ids = self._retrieval_layout_ids[best_index]
            if (
                selected_layout_ids
                != required_layout_ids.unsqueeze(1)
            ).any():
                raise RuntimeError(
                    "Different-layout transition survived same-layout-only retrieval"
                )

        source_logits = self._retrieval_source_mask_logits[best_index].float()
        target_logits = self._retrieval_target_mask_logits[best_index].float()
        source_proprio = self._retrieval_source_proprio[best_index]
        target_proprio = self._retrieval_target_proprio[best_index]
        actions = self._retrieval_actions[best_index]
        obj_latents = self.predictor.fixed_obj_latents.expand(
            query.shape[0], -1, -1
        )
        source_slots = {
            "obj_latents": obj_latents,
            "mask_logits": source_logits,
            "mask_probs": F.softmax(source_logits, dim=2),
        }
        target_slots = {
            "obj_latents": obj_latents,
            "mask_logits": target_logits,
            "mask_probs": F.softmax(target_logits, dim=2),
        }
        context_tokens = self.transition_adapter(
            torch.cat(
                [source_proprio, self.action_encoder(actions.float())], dim=-1
            )
        )
        return source_slots, target_slots, context_tokens, target_proprio, best_index

    def configure_inference_retrieval(self, num_context, candidate_pool_size):
        """Override retrieval breadth at inference without changing training length."""
        num_context = int(num_context)
        candidate_pool_size = int(candidate_pool_size)
        if num_context < 1:
            raise ValueError("Inference retrieval num_context must be positive")
        if candidate_pool_size < num_context:
            raise ValueError(
                "Inference retrieval candidate_pool_size must cover num_context"
            )
        if self._retrieval_source_states is not None:
            bank_size = int(self._retrieval_source_states.shape[0])
            if bank_size < candidate_pool_size:
                raise ValueError(
                    f"Retrieval bank has {bank_size} entries, fewer than requested "
                    f"candidate_pool_size={candidate_pool_size}"
                )
        self.retrieval_num_context = num_context
        self.retrieval_candidate_pool_size = candidate_pool_size
        # The predictor has no length-specific learned parameters; this field
        # is only its runtime upper-bound check. Training and inference may
        # therefore use different context lengths.
        self.predictor.num_context_frames = max(
            int(self.predictor.num_context_frames), num_context
        )
        log.info(
            "Inference retrieval: uniformly sample %d contexts from top-%d",
            num_context,
            candidate_pool_size,
        )

    @torch.no_grad()
    def prepare_rollout_retrieval_context(
        self, query_proprio, query_mask_logits, required_layout_ids=None
    ):
        """Materialize the real transition context used by one MPC decision.

        The returned flat tensor dictionary can be expanded across CEM samples
        without repeating nearest-neighbour search.  It is intentionally
        independent of candidate actions.
        """
        (
            source_slots,
            target_slots,
            context_tokens,
            target_proprio,
            indices,
        ) = self._retrieve_transition_context(
            query_proprio,
            query_mask_logits,
            required_layout_ids=required_layout_ids,
        )
        return {
            "obj_latents": source_slots["obj_latents"],
            "source_mask_logits": source_slots["mask_logits"],
            "source_mask_probs": source_slots["mask_probs"],
            "target_mask_logits": target_slots["mask_logits"],
            "target_mask_probs": target_slots["mask_probs"],
            "context_tokens": context_tokens,
            "target_proprio": target_proprio,
            "indices": indices,
        }

    @torch.no_grad()
    def prepare_random_bank_context(self, indices):
        """Materialize explicitly sampled bank rows without query-state retrieval.

        ``indices`` has shape ``[batch, num_context]``.  Sampling deliberately
        lives in the evaluator so each evaluation case has its own reproducible
        RNG stream; this method only gathers the selected training transitions.
        """
        if not self.has_retrieval_bank:
            raise RuntimeError("Random bank context requested before loading a bank")
        indices = torch.as_tensor(
            indices,
            device=self._retrieval_source_proprio.device,
            dtype=torch.long,
        )
        if indices.ndim != 2 or indices.shape[1] < 1:
            raise ValueError(
                "Random bank indices must have shape [batch, num_context], got "
                f"{tuple(indices.shape)}"
            )
        bank_size = self.loaded_retrieval_bank_size
        if torch.any(indices < 0) or torch.any(indices >= bank_size):
            raise IndexError(
                f"Random bank indices must lie in [0, {bank_size})"
            )

        source_logits = self._retrieval_source_mask_logits[indices].float()
        target_logits = self._retrieval_target_mask_logits[indices].float()
        source_proprio = self._retrieval_source_proprio[indices]
        target_proprio = self._retrieval_target_proprio[indices]
        actions = self._retrieval_actions[indices]
        obj_latents = self.predictor.fixed_obj_latents.expand(
            indices.shape[0], -1, -1
        )
        context_tokens = self.transition_adapter(
            torch.cat(
                [source_proprio, self.action_encoder(actions.float())], dim=-1
            )
        )
        self.predictor.num_context_frames = max(
            int(self.predictor.num_context_frames), indices.shape[1]
        )
        return {
            "obj_latents": obj_latents,
            "source_mask_logits": source_logits,
            "source_mask_probs": F.softmax(source_logits, dim=2),
            "target_mask_logits": target_logits,
            "target_mask_probs": F.softmax(target_logits, dim=2),
            "context_tokens": context_tokens,
            "target_proprio": target_proprio,
            "indices": indices,
        }

    @torch.no_grad()
    def prepare_observed_transition_context(
        self,
        source_visual,
        target_visual,
        source_states,
        target_states,
        actions,
    ):
        """Encode real test-time transitions as predictor ICL context."""
        if source_visual.shape[:2] != target_visual.shape[:2]:
            raise ValueError("Observed context source/target visuals must align")
        if source_states.shape[:2] != source_visual.shape[:2]:
            raise ValueError("Observed context source states must align with visuals")
        if target_states.shape[:2] != source_visual.shape[:2]:
            raise ValueError("Observed context target states must align with visuals")
        if actions.shape[:2] != source_visual.shape[:2]:
            raise ValueError("Observed context actions must align with visuals")
        batch_size, num_context = source_visual.shape[:2]
        if num_context < 1:
            raise ValueError("Observed transition context must not be empty")
        joint_visual = torch.cat([source_visual, target_visual], dim=1)
        # Bound encoder memory when many evaluation cases are batched. Five
        # contexts already mean ten full-resolution frames per trajectory.
        batch_chunk_size = max(1, 20 // (2 * num_context))
        mask_logits_parts = []
        with self._slot_extract_context():
            for start in range(0, batch_size, batch_chunk_size):
                slots = self._extract_slots(
                    joint_visual[start : start + batch_chunk_size]
                )
                mask_logits_parts.append(slots["mask_logits"])
        joint_mask_logits = torch.cat(mask_logits_parts, dim=0)
        source_logits = joint_mask_logits[:, :num_context]
        target_logits = joint_mask_logits[:, num_context:]
        source_proprio = self._encode_proprio(source_states.float())
        target_proprio = self._encode_proprio(target_states.float())
        context_tokens = self.transition_adapter(
            torch.cat(
                [source_proprio, self.action_encoder(actions.float())], dim=-1
            )
        )
        self.predictor.num_context_frames = max(
            int(self.predictor.num_context_frames), num_context
        )
        return {
            "obj_latents": self.predictor.fixed_obj_latents.expand(
                batch_size, -1, -1
            ),
            "source_mask_logits": source_logits,
            "source_mask_probs": F.softmax(source_logits, dim=2),
            "target_mask_logits": target_logits,
            "target_mask_probs": F.softmax(target_logits, dim=2),
            "context_tokens": context_tokens,
            "target_proprio": target_proprio,
            # Negative indices mark online observations rather than bank rows.
            "indices": torch.full(
                (batch_size, num_context),
                -1,
                device=source_visual.device,
                dtype=torch.long,
            ),
        }

    def _normalized_zero_action_chunk(self, batch_size, timesteps, device, dtype):
        primitive_zero = (-self.action_mean / self.action_std).to(
            device=device, dtype=dtype
        )
        chunk = primitive_zero.repeat(self.frameskip)
        return chunk.view(1, 1, -1).expand(batch_size, timesteps, -1)

    def _build_query_tokens(self, proprio_embedding, action_chunk):
        if proprio_embedding.shape[:2] != action_chunk.shape[:2]:
            raise ValueError(
                "Query proprio/action must align, got "
                f"{tuple(proprio_embedding.shape)} and {tuple(action_chunk.shape)}"
            )
        return self.transition_adapter(
            torch.cat(
                [proprio_embedding, self.action_encoder(action_chunk.float())], dim=-1
            )
        )

    def _build_transition_inputs(self, pre_state, act, proprio_embedding=None):
        if pre_state is None:
            raise ValueError("Transition-conditioned slot-ac requires raw pre-state input")
        if act is None:
            raise ValueError("Transition-conditioned slot-ac requires action input")
        if pre_state.ndim != 3 or act.ndim != 3:
            raise ValueError(
                f"Expected pre_state and act to be rank-3, got {tuple(pre_state.shape)} and {tuple(act.shape)}"
            )
        if pre_state.shape[:2] != act.shape[:2]:
            raise ValueError(
                f"Expected matching time dimensions for pre_state and act, got {tuple(pre_state.shape)} and {tuple(act.shape)}"
            )
        action_chunk = act.float()
        pre_state = pre_state.float()
        if proprio_embedding is None:
            proprio_embedding = self._encode_proprio(pre_state)
        # Markov transition semantics: row t is (visual_t, state_t, action_t)
        # and predicts row t+1. The final action row is the query action.
        context_action_embedding = self.action_encoder(action_chunk)
        transition_input = torch.cat(
            [proprio_embedding, context_action_embedding], dim=-1
        )
        transition_tokens = self.transition_adapter(transition_input)
        return {
            "pre_state": pre_state,
            "action_chunk": action_chunk,
            "proprio_embedding": proprio_embedding,
            "tokens": transition_tokens,
            "query_tokens": transition_tokens,
        }

    @staticmethod
    def _to_unit_range_visual(visual):
        if visual.dtype != torch.float32:
            visual = visual.float()
        if visual.max() > 1.0:
            visual = visual / 255.0
        if visual.min() < 0.0:
            visual = (visual + 1.0) / 2.0
        return visual.clamp(0.0, 1.0)

    def _imagenet_normalize_visual(self, visual):
        if self.visual_normalization == "centered":
            return visual - 0.5
        mean = self.imagenet_mean.to(device=visual.device, dtype=visual.dtype)
        std = self.imagenet_std.to(device=visual.device, dtype=visual.dtype)
        return (visual - mean) / std

    def _get_learned_extrinsics(self, batch_size, timesteps, device, dtype):
        if not self.predictor_use_extrinsics or self.learned_extrinsics is None:
            return None
        extrinsics = self.learned_extrinsics.expand(batch_size, timesteps, -1)
        return extrinsics.to(dtype=dtype)

    @staticmethod
    def _build_external_warped_mask_probs(context_mask_logits, num_query):
        mask_probs = torch.softmax(context_mask_logits, dim=2)
        return mask_probs[:, None].expand(-1, num_query, -1, -1, -1).contiguous()

    def _predict_outputs(
        self,
        context_source_slots,
        context_target_slots,
        query_source_slots,
        context_state_tokens,
        query_state_tokens,
        context_target_proprio,
        predictor_context_cache=None,
    ):
        if self.predictor_architecture == "sequential":
            t_query = query_state_tokens.shape[1]
            keep_context = max(
                0, int(self.predictor.num_context_frames) - t_query
            )
            sequence_masks = torch.cat(
                [
                    context_source_slots["mask_logits"][:, -keep_context:]
                    if keep_context
                    else context_source_slots["mask_logits"][:, :0],
                    query_source_slots["mask_logits"],
                ],
                dim=1,
            )
            sequence_tokens = torch.cat(
                [
                    context_state_tokens[:, -keep_context:]
                    if keep_context
                    else context_state_tokens[:, :0],
                    query_state_tokens,
                ],
                dim=1,
            )
            pred_logits, pred_proprio = self.predictor.forward_sequence(
                context_source_slots["obj_latents"],
                sequence_masks,
                sequence_tokens,
                return_proprio=True,
            )
            return pred_logits[:, -t_query:], pred_proprio[:, -t_query:]

        batch_size = context_state_tokens.shape[0]
        t_ctx = context_state_tokens.shape[1]
        t_query = query_state_tokens.shape[1]
        dtype = context_state_tokens.dtype
        device = context_state_tokens.device
        shared_target_state_tokens = None
        shared_query_target_tokens = None
        shared_external_warped_mask_probs = None
        if self.predictor_use_extrinsics:
            if self.predictor_extrinsics_dim != self.predictor_state_dim:
                raise ValueError(
                    "Markov transition target-state tokens require matching "
                    "predictor_state_dim and predictor_extrinsics_dim"
                )
            shared_target_state_tokens = context_target_proprio.to(dtype=dtype)
            shared_query_target_tokens = self.query_state_embedding.to(
                device=device, dtype=dtype
            ).expand(batch_size, t_query, -1)
            shared_external_warped_mask_probs = self._build_external_warped_mask_probs(
                context_source_slots["mask_logits"],
                num_query=t_query,
            )
        if predictor_context_cache is None:
            pred_logits, pred_proprio_embedding = self.predictor(
                context_source_slots["obj_latents"],
                context_source_slots["mask_logits"],
                context_state_tokens,
                query_state_tokens,
                context_target_masks=context_target_slots["mask_logits"],
                query_source_masks=query_source_slots["mask_logits"],
                extrinsics=shared_target_state_tokens,
                next_extrinsics=shared_query_target_tokens,
                external_warped_mask_probs=shared_external_warped_mask_probs,
                return_proprio=True,
            )
        else:
            pred_logits, pred_proprio_embedding = (
                self.predictor.forward_query_cached(
                    predictor_context_cache,
                    query_source_masks=query_source_slots["mask_logits"],
                    next_states=query_state_tokens,
                    next_extrinsics=shared_query_target_tokens,
                    return_proprio=True,
                )
            )
        return pred_logits, pred_proprio_embedding

    def _prepare_predictor_context_cache(
        self,
        context_source_slots,
        context_target_slots,
        context_state_tokens,
        context_target_proprio,
    ):
        """Cache query-independent predictor attention state for one rollout."""
        if self.predictor_architecture == "sequential":
            return None
        batch_size = context_state_tokens.shape[0]
        dtype = context_state_tokens.dtype
        device = context_state_tokens.device
        target_state_tokens = None
        query_target_tokens = None
        external_warped_mask_probs = None
        if self.predictor_use_extrinsics:
            if self.predictor_extrinsics_dim != self.predictor_state_dim:
                raise ValueError(
                    "Markov transition target-state tokens require matching "
                    "predictor_state_dim and predictor_extrinsics_dim"
                )
            target_state_tokens = context_target_proprio.to(dtype=dtype)
            query_target_tokens = self.query_state_embedding.to(
                device=device, dtype=dtype
            ).expand(batch_size, 1, -1)
            external_warped_mask_probs = (
                self._build_external_warped_mask_probs(
                    context_source_slots["mask_logits"], num_query=1
                )
            )
        return self.predictor.prepare_context_cache(
            context_source_slots["obj_latents"],
            context_source_slots["mask_logits"],
            context_state_tokens,
            context_target_masks=context_target_slots["mask_logits"],
            extrinsics=target_state_tokens,
            next_extrinsics=query_target_tokens,
            external_warped_mask_probs=external_warped_mask_probs,
        )

    def prepare_rollout_predictor_context_cache(self, retrieval_context):
        """Prepare one compact per-case cache for an entire CEM invocation."""
        context_source_slots = {
            "obj_latents": retrieval_context["obj_latents"],
            "mask_logits": retrieval_context["source_mask_logits"],
            "mask_probs": retrieval_context["source_mask_probs"],
        }
        context_target_slots = {
            "obj_latents": retrieval_context["obj_latents"],
            "mask_logits": retrieval_context["target_mask_logits"],
            "mask_probs": retrieval_context["target_mask_probs"],
        }
        with self._autocast_context():
            return self._prepare_predictor_context_cache(
                context_source_slots,
                context_target_slots,
                retrieval_context["context_tokens"],
                retrieval_context["target_proprio"],
            )

    def expand_rollout_predictor_context_cache(
        self, cache, sample_count, case_start=0, case_end=None
    ):
        return self.predictor.expand_context_cache(
            cache,
            sample_count=sample_count,
            case_start=case_start,
            case_end=case_end,
        )

    def _predict_mask_logits(self, *args, **kwargs):
        return self._predict_outputs(*args, **kwargs)[0]

    def _preprocess_visual(self, visual):
        visual = self._to_unit_range_visual(visual)
        visual = self._apply_encoder_contrast(visual)
        B, T = visual.shape[:2]
        flat = rearrange(visual, "b t c h w -> (b t) c h w")
        flat = self.image_transform(flat)
        flat = self._imagenet_normalize_visual(flat)
        return rearrange(flat, "(b t) c h w -> b t c h w", b=B, t=T)

    def _apply_encoder_contrast(self, visual):
        """Apply the shared train/inference RGB contrast preprocessing."""
        return (
            (visual - 0.5) * self.encoder_contrast_scale + 0.5
        ).clamp(0.0, 1.0)

    def _encode_frames(self, visual):
        visual = self._preprocess_visual(visual)
        B, T = visual.shape[:2]
        flat = rearrange(visual, "b t c h w -> (b t) c h w")
        flip_views = (
            ((), ()),
            ((-1,), (3,)),
            ((-2,), (2,)),
            ((-2, -1), (2, 3)),
        ) if self.encoder_flip_tta else (((), ()),)
        aligned_features = []
        for input_flip_dims, feature_flip_dims in flip_views:
            encoder_input = (
                torch.flip(flat, dims=input_flip_dims)
                if input_flip_dims
                else flat
            )
            with self._autocast_context():
                feats = self.student(encoder_input)
            feats = feats.float().view(
                B,
                T,
                self.encoder_spatial_size,
                self.encoder_spatial_size,
                self.embed_dim,
            )
            if feature_flip_dims:
                feats = torch.flip(feats, dims=feature_flip_dims)
            aligned_features.append(feats)
        features = torch.stack(aligned_features, dim=0).mean(dim=0)
        if not self.color_channel_reweight_enabled:
            return features
        original_norm = features.float().norm(dim=-1, keepdim=True)
        features = features.float() * self.color_channel_weights.view(
            1, 1, 1, 1, -1
        )
        if self.color_channel_preserve_token_norm:
            features = features * (
                original_norm
                / features.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
            )
        return features

    def initialize_fixed_slots(self, reference_visual):
        """Extract and persist one ordered slot bank from reference frames."""
        if reference_visual.ndim != 5 or reference_visual.shape[0] != 1:
            raise ValueError(
                "reference_visual must have shape (1, T, C, H, W), got "
                f"{tuple(reference_visual.shape)}"
            )
        with self._slot_extract_context():
            feats = self._encode_frames(reference_visual)
            with self._fp32_context():
                obj_latents, _ = self.extractor.merger(feats.float())
        if obj_latents.shape != self.predictor.fixed_obj_latents.shape:
            raise ValueError(
                "Reference slot shape mismatch: expected "
                f"{tuple(self.predictor.fixed_obj_latents.shape)}, got "
                f"{tuple(obj_latents.shape)}"
            )
        self.predictor.fixed_obj_latents.copy_(obj_latents.detach().float())
        self.predictor.fixed_slots_initialized.fill_(True)
        return self.predictor.fixed_obj_latents

    @property
    def has_fixed_slots(self):
        return bool(self.predictor.fixed_slots_initialized.item())

    def _resize_extracted_slots(self, slots):
        """Convert raw extractor masks to the predictor's patch grid."""
        mask_probs = slots["mask_probs"]
        source_size = int(math.isqrt(mask_probs.shape[-1]))
        if source_size * source_size != mask_probs.shape[-1]:
            raise ValueError(
                "Slot mask token count must be square, got "
                f"{mask_probs.shape[-1]}"
            )
        if source_size == self.spatial_size:
            return slots
        if source_size < self.spatial_size:
            raise ValueError(
                "Cached/extracted masks are lower resolution than the configured "
                f"slot grid ({source_size} < {self.spatial_size}); rebuild the "
                "slot cache instead of upsampling lost contour detail."
            )
        batch_size, timesteps, num_objects = mask_probs.shape[:3]
        mask_probs = F.interpolate(
            mask_probs.reshape(
                batch_size * timesteps, num_objects, source_size, source_size
            ),
            size=(self.spatial_size, self.spatial_size),
            mode="bilinear",
            align_corners=False,
            antialias=self.spatial_size < source_size,
        ).reshape(batch_size, timesteps, num_objects, self.spatial_size**2)
        mask_probs = mask_probs / mask_probs.sum(
            dim=2, keepdim=True
        ).clamp_min(1.0e-8)
        resized = dict(slots)
        resized["mask_probs"] = mask_probs
        resized["mask_logits"] = mask_probs.clamp_min(1.0e-8).log()
        return resized

    def _decoder_mask_probs(self, decoder_mask_logits):
        """Apply the configured temperature and gamma on the native mask grid."""
        mask_probs = F.softmax(
            decoder_mask_logits.float() / self.mask_softmax_temperature,
            dim=2,
        )
        if self.mask_sharpen_gamma != 1.0:
            mask_probs = mask_probs.pow(self.mask_sharpen_gamma)
            mask_probs = mask_probs / mask_probs.sum(
                dim=2, keepdim=True
            ).clamp_min(1.0e-8)
        return mask_probs

    def _extract_slots_raw(self, visual):
        """Extract slots and masks at the encoder's native resolution."""
        if not self.has_fixed_slots:
            raise RuntimeError(
                "Fixed predictor slots are not initialized. Only "
                "initialize_fixed_slots(reference_visual) may run the slot merger."
            )
        feats = self._encode_frames(visual)
        with self._fp32_context():
            obj_latents = self.predictor.fixed_obj_latents.expand(
                visual.shape[0], -1, -1
            )
            class_state = self.extractor.decode_with_latents(
                obj_latents,
                feats.float(),
                spatial_dims=(self.encoder_spatial_size, self.encoder_spatial_size),
                T_dim=visual.shape[1],
            )
        decoder_mask_logits = class_state.get_layer(-1)["mask_logits"].float()
        mask_probs = self._decoder_mask_probs(decoder_mask_logits)
        # Store log-probabilities as predictor-ready logits. This keeps cache
        # reloads from applying the extraction temperature a second time.
        mask_logits = mask_probs.clamp_min(1.0e-8).log()
        return {
            "features": feats,
            "obj_latents": obj_latents.float(),
            "mask_logits": mask_logits,
            "mask_probs": mask_probs,
        }

    def _extract_slots(self, visual):
        return self._resize_extracted_slots(self._extract_slots_raw(visual))

    def _cached_slots_from_obs(self, obs):
        if "slot_mask_logits" not in obs:
            return None
        mask_logits = obs["slot_mask_logits"].float()
        if self.has_fixed_slots:
            obj_latents = self.predictor.fixed_obj_latents.expand(
                mask_logits.shape[0], -1, -1
            )
        elif "slot_obj_latents" in obs:
            obj_latents = obs["slot_obj_latents"].float()
            if obj_latents.ndim == 4:
                obj_latents = obj_latents[:, 0]
            elif obj_latents.ndim != 3:
                raise ValueError(
                    f"Unexpected cached obj_latents shape: {tuple(obj_latents.shape)}"
                )
        else:
            raise RuntimeError(
                "Mask-only slot cache requires initialized fixed object slots"
            )
        if mask_logits.ndim != 4:
            raise ValueError(f"Unexpected cached mask_logits shape: {tuple(mask_logits.shape)}")
        raw_slots = {
            "features": None,
            "obj_latents": obj_latents,
            "mask_logits": mask_logits,
            "mask_probs": F.softmax(mask_logits, dim=2),
        }
        return self._resize_extracted_slots(raw_slots)

    def encode_obs(self, obs, state=None):
        slots = self._extract_slots(obs["visual"])
        out = {
            "visual": slots["mask_probs"],
            "mask_probs": slots["mask_probs"],
            "obj_latents": slots["obj_latents"],
        }
        if state is None:
            proprio = obs["proprio"]
            out["proprio"] = proprio.float()
        else:
            out["proprio"] = state.float()
        return out

    def initialize_planning_slots(
        self, obs_0, obs_g, state_g=None, context_visual=None
    ):
        """Decode initial/goal masks with the checkpoint's immutable slot bank."""
        if obs_0["visual"].shape[1] != 1 or obs_g["visual"].shape[1] != 1:
            raise ValueError("Slot-AC planning initialization expects one initial and one goal frame")
        joint_visual = torch.cat([obs_0["visual"], obs_g["visual"]], dim=1)
        with self._slot_extract_context():
            joint_slots = self._extract_slots(joint_visual)
        initial_slots = self._slice_shared_slots(joint_slots, [0])
        goal_slots = self._slice_shared_slots(joint_slots, [1])
        goal_proprio = (
            state_g[:, None].float()
            if state_g is not None
            else obs_g["proprio"].float()
        )
        goal_z = {
            "visual": goal_slots["mask_probs"],
            "mask_probs": goal_slots["mask_probs"],
            "proprio": self._encode_proprio(goal_proprio),
        }
        return initial_slots, initial_slots, goal_slots, goal_z

    def encode_with_object_latents(self, visual, obj_latents):
        """Decode frames with the persisted predictor slots; never reinitialize."""
        if not self.has_fixed_slots:
            raise RuntimeError("Cannot encode without initialized fixed predictor slots")
        obj_latents = self.predictor.fixed_obj_latents.expand(
            visual.shape[0], -1, -1
        )
        with self._slot_extract_context():
            feats = self._encode_frames(visual)
            with self._fp32_context():
                state = self.extractor.decode_with_latents(
                    obj_latents.float(),
                    feats.float(),
                    spatial_dims=(self.encoder_spatial_size, self.encoder_spatial_size),
                    T_dim=visual.shape[1],
                )
        decoder_mask_logits = state.get_layer(-1)["mask_logits"].float()
        mask_probs = self._decoder_mask_probs(decoder_mask_logits)
        # Keep the same predictor-ready representation as _extract_slots_raw:
        # Temperature and gamma are applied exactly once; downstream softmax
        # recovers these probabilities without sharpening them a second time.
        mask_logits = mask_probs.clamp_min(1.0e-8).log()
        raw_slots = {
            "features": feats,
            "obj_latents": obj_latents.float(),
            "mask_logits": mask_logits,
            "mask_probs": mask_probs,
        }
        return self._resize_extracted_slots(raw_slots)

    @staticmethod
    def _generalized_kl(x, y, eps):
        x_safe = x.clamp_min(eps)
        y_safe = y.clamp_min(eps)
        return (x_safe * (torch.log(x_safe) - torch.log(y_safe)) - x_safe + y_safe).sum(dim=-1)

    def uot_loss(self, pred_mask_probs, target_mask_probs, context_mask_probs=None):
        if pred_mask_probs.shape != target_mask_probs.shape:
            raise ValueError(
                f"Expected equal shapes for UOT: {tuple(pred_mask_probs.shape)} vs {tuple(target_mask_probs.shape)}"
            )
        if (
            self.mask_transport_area_normalize or self.slot_change_weighting
        ) and context_mask_probs is None:
            raise ValueError(
                "Area/change-weighted training UOT requires context mask probabilities"
            )
        slot_weights = None
        area_normalizer = None
        if self.slot_change_weighting:
            slot_weights = _slot_change_weights(
                context_mask_probs[:, -1],
                target_mask_probs.detach()[:, -1],
                min_weight=self.slot_change_min_weight,
                max_weight=self.slot_change_max_weight,
            )
        if self.mask_transport_area_normalize:
            area_normalizer = _slot_area_normalizer(
                context_mask_probs,
                target_mask_probs.detach(),
                eps=self.mask_transport_area_normalize_eps,
            )
        per_case_loss = _uot_mask_loss(
            pred_mask_probs,
            target_mask_probs,
            sinkhorn_iters=self.sinkhorn_iters,
            entropy_reg=self.sinkhorn_entropy_reg,
            unbalanced_ot_rho=self.unbalanced_ot_rho,
            eps=self.unbalanced_ot_eps,
            batch_size=self.uot_batch_size,
            transport_cache=self._uot_kernel_cache,
            slot_weights=slot_weights,
            area_normalizer=area_normalizer,
            area_normalize=self.mask_transport_area_normalize,
            area_normalize_eps=self.mask_transport_area_normalize_eps,
        )
        return per_case_loss.mean()

    def _get_spatial_transport_cost(self, spatial_size, device):
        ys, xs = torch.meshgrid(
            torch.linspace(0.0, 1.0, spatial_size, device=device),
            torch.linspace(0.0, 1.0, spatial_size, device=device),
            indexing="ij",
        )
        coords = torch.stack([ys, xs], dim=-1).reshape(-1, 2)
        return torch.cdist(coords, coords, p=2).pow(2)

    def mask_l1_loss(
        self, pred_mask_probs, target_mask_probs, context_mask_probs=None
    ):
        if (
            self.mask_transport_area_normalize or self.slot_change_weighting
        ) and context_mask_probs is None:
            raise ValueError(
                "Area/change-weighted training mask L1 requires context masks"
            )
        slot_weights = None
        area_normalizer = None
        if self.slot_change_weighting:
            slot_weights = _slot_change_weights(
                context_mask_probs[:, -1],
                target_mask_probs.detach()[:, -1],
                min_weight=self.slot_change_min_weight,
                max_weight=self.slot_change_max_weight,
            )
        if self.mask_transport_area_normalize:
            area_normalizer = _slot_area_normalizer(
                context_mask_probs,
                target_mask_probs.detach(),
                eps=self.mask_transport_area_normalize_eps,
            )
        return _mask_l1_loss(
            pred_mask_probs,
            target_mask_probs,
            slot_weights=slot_weights,
            area_normalizer=area_normalizer,
            area_normalize=self.mask_transport_area_normalize,
            area_normalize_eps=self.mask_transport_area_normalize_eps,
        ).mean()

    def mask_kl_loss(
        self, pred_mask_logits, target_mask_probs, context_mask_probs=None
    ):
        if (
            self.mask_transport_area_normalize or self.slot_change_weighting
        ) and context_mask_probs is None:
            raise ValueError(
                "Area/change-weighted training mask KL requires context masks"
            )
        slot_weights = None
        area_normalizer = None
        if self.slot_change_weighting:
            slot_weights = _slot_change_weights(
                context_mask_probs[:, -1],
                target_mask_probs.detach()[:, -1],
                min_weight=self.slot_change_min_weight,
                max_weight=self.slot_change_max_weight,
            )
        if self.mask_transport_area_normalize:
            area_normalizer = _slot_area_normalizer(
                context_mask_probs,
                target_mask_probs.detach(),
                eps=self.mask_transport_area_normalize_eps,
            )
        return _mask_kl_loss(
            pred_mask_logits,
            target_mask_probs,
            slot_weights=slot_weights,
            area_normalizer=area_normalizer,
            area_normalize=self.mask_transport_area_normalize,
            area_normalize_eps=self.mask_transport_area_normalize_eps,
        ).mean()

    def mask_sobel_loss(
        self, pred_mask_probs, target_mask_probs, context_mask_probs=None
    ):
        if (
            self.mask_transport_area_normalize or self.slot_change_weighting
        ) and context_mask_probs is None:
            raise ValueError(
                "Area/change-weighted training mask Sobel requires context masks"
            )
        slot_weights = None
        area_normalizer = None
        if self.slot_change_weighting:
            slot_weights = _slot_change_weights(
                context_mask_probs[:, -1],
                target_mask_probs.detach()[:, -1],
                min_weight=self.slot_change_min_weight,
                max_weight=self.slot_change_max_weight,
            )
        if self.mask_transport_area_normalize:
            area_normalizer = _slot_area_normalizer(
                context_mask_probs,
                target_mask_probs.detach(),
                eps=self.mask_transport_area_normalize_eps,
            )
        return _mask_sobel_loss(
            pred_mask_probs,
            target_mask_probs,
            slot_weights=slot_weights,
            area_normalizer=area_normalizer,
            area_normalize=self.mask_transport_area_normalize,
            area_normalize_eps=self.mask_transport_area_normalize_eps,
        ).mean()

    @staticmethod
    def reconstruct_from_slots(mask_probs, obj_latents):
        if mask_probs.ndim == 3:
            mask_probs = mask_probs.unsqueeze(1)
        return torch.einsum("btkn,bkd->btnd", mask_probs.to(obj_latents.dtype), obj_latents)

    @staticmethod
    def _slice_shared_slots(full_slots, frame_indices):
        index = torch.as_tensor(frame_indices, device=full_slots["mask_logits"].device, dtype=torch.long)
        sliced = {
            "obj_latents": full_slots["obj_latents"],
            "mask_logits": full_slots["mask_logits"].index_select(1, index),
            "mask_probs": full_slots["mask_probs"].index_select(1, index),
        }
        if full_slots.get("features", None) is not None:
            sliced["features"] = full_slots["features"].index_select(1, index)
        else:
            sliced["features"] = None
        return sliced

    @staticmethod
    def _gather_time(tensor, order):
        gather_index = order.view(
            order.shape[0], order.shape[1], *([1] * (tensor.ndim - 2))
        ).expand(-1, -1, *tensor.shape[2:])
        return torch.gather(tensor, 1, gather_index)

    def _prediction_losses(
        self,
        pred_logits,
        pred_proprio_embedding,
        target_slots,
        target_state,
        source_mask_probs,
        obj_latents,
    ):
        """Shared single-step loss used by TMP and sequential A1."""
        pred_mask_probs = F.softmax(pred_logits, dim=2)
        target_mask_probs = target_slots["mask_probs"].detach()
        source_mask_probs = source_mask_probs.detach()
        transport_loss = self.uot_loss(
            pred_mask_probs,
            target_mask_probs,
            context_mask_probs=source_mask_probs,
        )
        pred_recon = self.reconstruct_from_slots(pred_mask_probs, obj_latents)
        target_recon = self.reconstruct_from_slots(
            target_mask_probs, target_slots["obj_latents"].detach()
        )
        recon_loss = feature_recon_loss(
            pred_recon.reshape(
                pred_recon.shape[0] * pred_recon.shape[1],
                -1,
                pred_recon.shape[-1],
            ).float(),
            target_recon.reshape(
                target_recon.shape[0] * target_recon.shape[1],
                -1,
                target_recon.shape[-1],
            ).float(),
            loss_type="l1",
        )
        mask_l1_loss = self.mask_l1_loss(
            pred_mask_probs,
            target_mask_probs,
            context_mask_probs=source_mask_probs,
        )
        # Use per-slot Bernoulli KL (soft-target BCE minus target entropy), so
        # every slot divergence is non-negative before area/change weighting.
        mask_kl_loss = self.mask_kl_loss(
            pred_logits,
            target_mask_probs,
            context_mask_probs=source_mask_probs,
        )
        mask_sobel_loss = (
            self.mask_sobel_loss(
                pred_mask_probs,
                target_mask_probs,
                context_mask_probs=source_mask_probs,
            )
            if self.predictor_mask_sobel_loss_weight > 0.0
            else pred_mask_probs.new_zeros(())
        )
        target_proprio_embedding = self._encode_proprio(target_state)
        proprio_embedding_loss = F.mse_loss(
            pred_proprio_embedding.float(),
            target_proprio_embedding.detach().float(),
        )
        normalized_target_proprio = self.normalize_states(target_state).detach()
        reconstructed_target_proprio = self.proprio_decoder(
            target_proprio_embedding.float()
        )
        target_position_reconstruction_loss = F.mse_loss(
            reconstructed_target_proprio[..., :2].float(),
            normalized_target_proprio[..., :2].float(),
        )
        target_velocity_reconstruction_loss = F.mse_loss(
            reconstructed_target_proprio[..., 2:4].float(),
            normalized_target_proprio[..., 2:4].float(),
        )
        predicted_target_proprio = self.proprio_decoder(
            pred_proprio_embedding.float()
        )
        predictor_position_prediction_loss = F.mse_loss(
            predicted_target_proprio[..., :2].float(),
            normalized_target_proprio[..., :2].float(),
        )
        predictor_velocity_prediction_loss = F.mse_loss(
            predicted_target_proprio[..., 2:4].float(),
            normalized_target_proprio[..., 2:4].float(),
        )
        target_proprio_reconstruction_loss = 0.5 * (
            target_position_reconstruction_loss
            + target_velocity_reconstruction_loss
        )
        predictor_proprio_prediction_loss = 0.5 * (
            predictor_position_prediction_loss
            + predictor_velocity_prediction_loss
        )
        proprio_reconstruction_loss = 0.5 * (
            target_proprio_reconstruction_loss
            + predictor_proprio_prediction_loss
        )
        total_loss = (
            self.predictor_transport_loss_weight * transport_loss
            + self.predictor_mask_l1_loss_weight * mask_l1_loss
            + self.predictor_mask_kl_loss_weight * mask_kl_loss
            + self.predictor_mask_sobel_loss_weight * mask_sobel_loss
            + self.predictor_recon_loss_weight * recon_loss
            + self.predictor_proprio_loss_weight * proprio_embedding_loss
            + self.proprio_reconstruction_loss_weight
            * proprio_reconstruction_loss
        )
        return {
            "pred_logits": pred_logits,
            "pred_mask_probs": pred_mask_probs,
            "pred_recon": pred_recon,
            "target_state": target_state,
            "pred_proprio_embedding": pred_proprio_embedding,
            "proprio_embedding_loss": proprio_embedding_loss,
            "proprio_reconstruction_loss": proprio_reconstruction_loss,
            "target_proprio_reconstruction_loss": target_proprio_reconstruction_loss,
            "target_position_reconstruction_loss": target_position_reconstruction_loss,
            "target_velocity_reconstruction_loss": target_velocity_reconstruction_loss,
            "predictor_proprio_prediction_loss": predictor_proprio_prediction_loss,
            "predictor_position_prediction_loss": predictor_position_prediction_loss,
            "predictor_velocity_prediction_loss": predictor_velocity_prediction_loss,
            "transport_loss": transport_loss,
            "mask_l1_loss": mask_l1_loss,
            "mask_kl_loss": mask_kl_loss,
            "mask_sobel_loss": mask_sobel_loss,
            "recon_loss": recon_loss,
            "total_loss": total_loss,
        }

    def _compute_sequential_losses(self, transition_slots, transition_inputs, state):
        """Supervise every causal position t against the real frame t+1."""
        source_indices = list(range(self.num_hist))
        target_indices = [index + 1 for index in source_indices]
        source_slots = self._slice_shared_slots(transition_slots, source_indices)
        target_slots = self._slice_shared_slots(transition_slots, target_indices)
        state_tokens = transition_inputs["tokens"][:, source_indices]
        # If a future dataset/config provides fewer sources than the configured
        # causal window, apply the same left cold-start padding used during
        # online planning. Padded outputs are context-only and excluded from
        # the loss. Current A1 configs use four real source positions, so this
        # branch is normally inactive.
        pad_count = max(
            0, int(self.predictor.num_context_frames) - len(source_indices)
        )
        if pad_count:
            first_mask_logits = source_slots["mask_logits"][:, :1].expand(
                -1, pad_count, -1, -1
            )
            source_mask_logits = torch.cat(
                [first_mask_logits, source_slots["mask_logits"]], dim=1
            )
            first_proprio = transition_inputs["proprio_embedding"][:, :1].expand(
                -1, pad_count, -1
            )
            zero_actions = self._normalized_zero_action_chunk(
                first_proprio.shape[0],
                pad_count,
                first_proprio.device,
                first_proprio.dtype,
            )
            padding_tokens = self._build_query_tokens(first_proprio, zero_actions)
            state_tokens = torch.cat([padding_tokens, state_tokens], dim=1)
        else:
            source_mask_logits = source_slots["mask_logits"]
        pred_logits, pred_proprio = self.predictor.forward_sequence(
            source_slots["obj_latents"],
            source_mask_logits,
            state_tokens,
            return_proprio=True,
        )
        pred_logits = pred_logits[:, pad_count:]
        pred_proprio = pred_proprio[:, pad_count:]
        step_losses = []
        for step, target_index in enumerate(target_indices):
            step_losses.append(
                self._prediction_losses(
                    pred_logits[:, step : step + 1],
                    pred_proprio[:, step : step + 1],
                    self._slice_shared_slots(target_slots, [step]),
                    state[:, target_index : target_index + 1].float(),
                    source_slots["mask_probs"][:, step : step + 1],
                    source_slots["obj_latents"],
                )
            )
        scalar_keys = [
            key
            for key, value in step_losses[0].items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        ]
        result = {
            key: torch.stack([item[key] for item in step_losses]).mean()
            for key in scalar_keys
        }
        for key in (
            "pred_logits",
            "pred_mask_probs",
            "pred_recon",
            "target_state",
            "pred_proprio_embedding",
        ):
            result[key] = torch.cat([item[key] for item in step_losses], dim=1)
        return result

    def _compute_losses(
        self,
        transition_slots,
        transition_inputs,
        state,
        context_source_indices,
        context_target_indices,
        query_source_indices,
        target_indices,
        query_layout_ids=None,
        query_sample_indices=None,
        query_source_slots_override=None,
        query_source_proprio_override=None,
        exclude_exact_transition=True,
    ):
        query_source_slots = (
            self._slice_shared_slots(transition_slots, query_source_indices)
            if query_source_slots_override is None
            else query_source_slots_override
        )
        target_slots = self._slice_shared_slots(transition_slots, target_indices)
        query_source_state = state[:, query_source_indices].float()
        if self.training_retrieval_context:
            if not self.has_retrieval_bank:
                raise RuntimeError(
                    "training_retrieval_context requires a loaded retrieval bank"
                )
            # The query observes only (state_t, action_t).  state_{t+1} is not
            # encoded on this path: its predictor input position is occupied by
            # query_state_embedding in _predict_outputs, and the real value is
            # read below only as a supervision target.
            query_source_proprio = (
                self._encode_proprio(query_source_state)
                if query_source_proprio_override is None
                else query_source_proprio_override
            )
            if (
                (
                    self.training_retrieval_exclude_same_layout
                    or self.training_retrieval_same_layout_only
                )
                and query_layout_ids is None
            ):
                raise RuntimeError(
                    "Training retrieval must filter by query layout, but the "
                    "batch does not contain layout_id"
                )
            query_action = transition_inputs["action_chunk"][
                :, query_source_indices
            ]
            # Retrieval-context training intentionally keeps this dictionary
            # limited to query actions so the real target state cannot leak
            # into preprocessing. Encode the selected query action here.
            query_action_embedding = self.action_encoder(query_action.float())
            query_state_tokens = self.transition_adapter(
                torch.cat(
                    [query_source_proprio, query_action_embedding], dim=-1
                )
            )
            # The training query itself is also present in a bank built from
            # the training split. Exclude that exact transition before top-k
            # retrieval so its observed target cannot leak in as context.
            excluded_indices = None
            if (
                exclude_exact_transition
                and self.training_context_generation_strategy
                != "pushobj_structured_torque"
            ):
                if query_sample_indices is None:
                    raise RuntimeError(
                        "Training retrieval requires stable sample_index metadata"
                    )
                if self._retrieval_sample_to_bank_index is None:
                    raise RuntimeError(
                        "Retrieval bank has no cached sample-to-bank index"
                    )
                if len(query_source_indices) != 1:
                    raise ValueError(
                        "Exact-transition exclusion expects one query source index"
                    )
                excluded_indices = self._retrieval_sample_to_bank_index[
                    query_sample_indices.to(
                        device=query_source_state.device, dtype=torch.long
                    ).flatten(),
                    int(query_source_indices[0]),
                ]
            # Candidate ranking/sampling is discrete, but the selected raw
            # state/action rows are encoded again below with gradients enabled.
            with torch.no_grad():
                (
                    context_source_slots,
                    context_target_slots,
                    _,
                    _,
                    retrieval_indices,
                ) = self._retrieve_transition_context(
                    query_source_proprio.detach(),
                    query_source_slots["mask_logits"].detach(),
                    excluded_indices=excluded_indices,
                    excluded_layout_ids=(
                        query_layout_ids
                        if (
                            self.training_retrieval_exclude_same_layout
                        )
                        else None
                    ),
                    required_layout_ids=(
                        query_layout_ids
                        if (
                            self.training_retrieval_same_layout_only
                        )
                        else None
                    ),
                )
            source_states = self._retrieval_source_states[
                retrieval_indices
            ].float()
            target_states = self._retrieval_target_states[
                retrieval_indices
            ].float()
            selected_actions = self._retrieval_actions[retrieval_indices].float()
            source_proprio = self._encode_proprio(source_states)
            context_target_proprio = self._encode_proprio(target_states)
            context_state_tokens = self.transition_adapter(
                torch.cat(
                    [source_proprio, self.action_encoder(selected_actions)],
                    dim=-1,
                )
            )
        else:
            query_source_proprio = transition_inputs["proprio_embedding"][
                :, query_source_indices
            ]
            query_state_tokens = transition_inputs["query_tokens"][
                :, query_source_indices
            ]
            context_source_slots = self._slice_shared_slots(
                transition_slots, context_source_indices
            )
            context_target_slots = self._slice_shared_slots(
                transition_slots, context_target_indices
            )
            context_target_proprio = transition_inputs["proprio_embedding"][
                :, context_target_indices
            ]
            context_state_tokens = transition_inputs["tokens"][
                :, context_source_indices
            ]
            # Legacy contiguous-window training orders its contexts by
            # retrieval distance before applying temporal RoPE.
            source_state_norm = self.normalize_states(
                state[:, context_source_indices].float()
            )
            query_state_norm = self.normalize_states(
                state[:, query_source_indices].float()
            )
            context_distance = (
                source_state_norm - query_state_norm[:, :1]
            ).square().mean(dim=-1)
            if self.retrieval_visual_distance_weight > 0.0:
                context_visual_descriptor = self._mask_retrieval_descriptor(
                    context_source_slots["mask_logits"]
                )
                query_visual_descriptor = self._mask_retrieval_descriptor(
                    query_source_slots["mask_logits"]
                )
                context_distance = context_distance + self.retrieval_visual_distance_weight * (
                    context_visual_descriptor - query_visual_descriptor[:, :1]
                ).square().mean(dim=-1)
            context_order = context_distance.argsort(dim=1)
            for slots in (context_source_slots, context_target_slots):
                slots["mask_logits"] = self._gather_time(
                    slots["mask_logits"], context_order
                )
                slots["mask_probs"] = self._gather_time(
                    slots["mask_probs"], context_order
                )
            context_state_tokens = self._gather_time(
                context_state_tokens, context_order
            )
            context_target_proprio = self._gather_time(
                context_target_proprio, context_order
            )
        target_state = state[:, target_indices].float()
        pred_logits, pred_proprio_embedding = self._predict_outputs(
            context_source_slots,
            context_target_slots,
            query_source_slots,
            context_state_tokens,
            query_state_tokens,
            context_target_proprio,
        )
        return self._prediction_losses(
            pred_logits,
            pred_proprio_embedding,
            target_slots,
            target_state,
            query_source_slots["mask_probs"],
            context_source_slots["obj_latents"],
        )

    def forward(self, obs, act, state=None):
        if state is None:
            raise ValueError("SlotMaskWorldModel.forward requires raw state supervision")
        timing = {}
        t_all = self._tic() if self.debug_timing else None
        if self.debug_timing and torch.cuda.is_available():
            self._reset_cuda_peak()
            timing["mem_forward_start_mb"] = self._cuda_mem_mb()
        cached_slots_available = "slot_mask_logits" in obs
        if cached_slots_available:
            total_frames = obs["slot_mask_logits"].shape[1]
        elif "visual" in obs:
            total_frames = obs["visual"].shape[1]
        else:
            raise ValueError(
                "SlotMaskWorldModel.forward requires either cached slots or visual input"
            )
        expected_total_frames = self.num_hist + self.num_pred
        if total_frames != expected_total_frames:
            raise ValueError(
                f"Expected {expected_total_frames} transition observations for transition-conditioned slot-ac forward, got {total_frames}"
            )
        t = self._tic() if self.debug_timing else None
        full_slots = self._cached_slots_from_obs(obs)
        if full_slots is None:
            visual = obs["visual"]
            with self._slot_extract_context():
                full_slots = self._extract_slots(visual)
        if self.training_retrieval_context:
            # Keep the real next state out of query preprocessing altogether.
            # Only its frame/mask remains available as the visual target.
            transition_inputs = {"action_chunk": act.float()}
        else:
            transition_inputs = self._build_transition_inputs(state, act)
        if self.debug_timing:
            elapsed = self._toc(t)
            timing["context_extract_s"] = elapsed
            timing["target_extract_s"] = 0.0
            timing["mem_after_slot_extract_mb"] = self._cuda_mem_mb()
            timing["mem_peak_after_slot_extract_mb"] = self._cuda_peak_mb()
        t = self._tic() if self.debug_timing else None
        with self._autocast_context():
            context_source_indices = list(range(0, self.num_hist - 1))
            context_target_indices = [index + 1 for index in context_source_indices]
            if not context_source_indices and not self.training_retrieval_context:
                raise ValueError(
                    "Retrieval-context training requires num_hist >= 2"
                )
            if self.predictor_architecture == "sequential":
                context_source_indices = list(range(self.num_hist))
                teacher_forced_predictions = [
                    self._compute_sequential_losses(
                        full_slots, transition_inputs, state
                    )
                ]
            else:
                teacher_forced_predictions = []
                for rollout_step in range(self.training_rollout_steps):
                    source_index = self.num_hist - 1 + rollout_step
                    teacher_forced_predictions.append(
                        self._compute_losses(
                            full_slots,
                            transition_inputs,
                            state,
                            context_source_indices=context_source_indices,
                            context_target_indices=context_target_indices,
                            query_source_indices=[source_index],
                            target_indices=[source_index + 1],
                            query_layout_ids=obs.get("layout_id"),
                            query_sample_indices=obs.get("sample_index"),
                        )
                    )

            autoregressive_predictions = []
            if self.training_autoregressive_weight > 0.0:
                source_index = self.num_hist - 1
                rolling_slots = self._slice_shared_slots(full_slots, [source_index])
                rolling_proprio = self._encode_proprio(
                    state[:, [source_index]].float()
                )
                for rollout_step in range(self.training_rollout_steps):
                    target_index = self.num_hist + rollout_step
                    if (
                        rollout_step == 0
                        and self.training_context_generation_strategy
                        == "pushobj_structured_torque"
                    ):
                        # Both branches have the same observed source at step
                        # zero. Structured retrieval is deterministic, so this
                        # result is exactly the teacher-forced result and need
                        # not launch the predictor/UOT kernels twice.
                        prediction_step = teacher_forced_predictions[0]
                    else:
                        prediction_step = self._compute_losses(
                            full_slots,
                            transition_inputs,
                            state,
                            context_source_indices=context_source_indices,
                            context_target_indices=context_target_indices,
                            query_source_indices=[target_index - 1],
                            target_indices=[target_index],
                            query_layout_ids=obs.get("layout_id"),
                            query_sample_indices=obs.get("sample_index"),
                            query_source_slots_override=rolling_slots,
                            query_source_proprio_override=rolling_proprio,
                            exclude_exact_transition=False,
                        )
                    autoregressive_predictions.append(prediction_step)

                    # Truncated rollout training: the next step consumes the
                    # model prediction, but no gradient crosses the model-step
                    # boundary. This bounds BPTT while still exposing every
                    # step to prediction-distribution inputs.
                    rolling_mask_probs = prediction_step["pred_mask_probs"].detach()
                    rolling_slots = {
                        "obj_latents": full_slots["obj_latents"].detach(),
                        "mask_logits": prediction_step["pred_logits"].detach(),
                        "mask_probs": rolling_mask_probs,
                    }
                    rolling_proprio = prediction_step[
                        "pred_proprio_embedding"
                    ].detach()

            loss_names = (
                "transport_loss",
                "mask_l1_loss",
                "mask_kl_loss",
                "mask_sobel_loss",
                "recon_loss",
                "proprio_embedding_loss",
                "proprio_reconstruction_loss",
                "target_proprio_reconstruction_loss",
                "target_position_reconstruction_loss",
                "target_velocity_reconstruction_loss",
                "predictor_proprio_prediction_loss",
                "predictor_position_prediction_loss",
                "predictor_velocity_prediction_loss",
                "total_loss",
            )

            def mean_prediction_losses(predictions):
                return {
                    name: torch.stack([item[name] for item in predictions]).mean()
                    for name in loss_names
                }

            teacher_forced = mean_prediction_losses(teacher_forced_predictions)
            autoregressive = (
                mean_prediction_losses(autoregressive_predictions)
                if autoregressive_predictions
                else None
            )
            loss_weight_sum = self.training_teacher_force_weight
            if autoregressive is not None:
                loss_weight_sum += self.training_autoregressive_weight
            prediction = {
                name: (
                    self.training_teacher_force_weight * teacher_forced[name]
                    + (
                        0.0
                        if autoregressive is None
                        else self.training_autoregressive_weight
                        * autoregressive[name]
                    )
                )
                / loss_weight_sum
                for name in loss_names
            }
            output_predictions = (
                autoregressive_predictions
                if autoregressive_predictions
                else teacher_forced_predictions
            )
            prediction["pred_mask_probs"] = torch.cat(
                [item["pred_mask_probs"] for item in output_predictions], dim=1
            )
            prediction["pred_proprio_embedding"] = torch.cat(
                [
                    item["pred_proprio_embedding"]
                    for item in output_predictions
                ],
                dim=1,
            )
        if self.debug_timing:
            timing["predictor_s"] = self._toc(t)
            timing["mem_after_predictor_mb"] = self._cuda_mem_mb()
            timing["mem_peak_after_predictor_mb"] = self._cuda_peak_mb()
            predictor_timing = getattr(self.predictor, "latest_timing", {})
            if predictor_timing:
                timing.update(predictor_timing)
        t = self._tic() if self.debug_timing else None
        transport_loss = prediction["transport_loss"]
        mask_l1_loss = prediction["mask_l1_loss"]
        mask_kl_loss = prediction["mask_kl_loss"]
        mask_sobel_loss = prediction["mask_sobel_loss"]
        recon_loss = prediction["recon_loss"]
        total_loss = prediction["total_loss"]
        if self.debug_timing:
            timing["uot_s"] = self._toc(t)
            timing["mem_after_loss_mb"] = self._cuda_mem_mb()
            timing["mem_peak_after_loss_mb"] = self._cuda_peak_mb()
        if self.debug_timing:
            timing["forward_total_s"] = self._toc(t_all)
            self.latest_timing = timing
        loss_components = {
            "loss": total_loss.detach(),
            "transport_loss": transport_loss.detach(),
            "mask_l1_loss": mask_l1_loss.detach(),
            "mask_kl_loss": mask_kl_loss.detach(),
            "mask_sobel_loss": mask_sobel_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "proprio_embedding_loss": prediction["proprio_embedding_loss"].detach(),
            "proprio_reconstruction_loss": prediction[
                "proprio_reconstruction_loss"
            ].detach(),
            "target_proprio_reconstruction_loss": prediction[
                "target_proprio_reconstruction_loss"
            ].detach(),
            "target_position_reconstruction_loss": prediction[
                "target_position_reconstruction_loss"
            ].detach(),
            "target_velocity_reconstruction_loss": prediction[
                "target_velocity_reconstruction_loss"
            ].detach(),
            "predictor_proprio_prediction_loss": prediction[
                "predictor_proprio_prediction_loss"
            ].detach(),
            "predictor_position_prediction_loss": prediction[
                "predictor_position_prediction_loss"
            ].detach(),
            "predictor_velocity_prediction_loss": prediction[
                "predictor_velocity_prediction_loss"
            ].detach(),
            "context_length": torch.tensor(
                float(len(context_source_indices)), device=total_loss.device
            ),
            "teacher_forced_loss": teacher_forced["total_loss"].detach(),
        }
        for rollout_step, prediction_step in enumerate(
            teacher_forced_predictions, start=1
        ):
            loss_components[
                f"teacher_forced_loss_step_{rollout_step}"
            ] = prediction_step["total_loss"].detach()
            loss_components[
                f"teacher_forced_mask_l1_step_{rollout_step}"
            ] = prediction_step["mask_l1_loss"].detach()
            loss_components[
                f"teacher_forced_mask_kl_step_{rollout_step}"
            ] = prediction_step["mask_kl_loss"].detach()
            loss_components[
                f"teacher_forced_mask_sobel_step_{rollout_step}"
            ] = prediction_step["mask_sobel_loss"].detach()
        if autoregressive is not None:
            loss_components["autoregressive_rollout_loss"] = autoregressive[
                "total_loss"
            ].detach()
            for rollout_step, prediction_step in enumerate(
                autoregressive_predictions, start=1
            ):
                loss_components[
                    f"autoregressive_loss_step_{rollout_step}"
                ] = prediction_step["total_loss"].detach()
                loss_components[
                    f"autoregressive_mask_l1_step_{rollout_step}"
                ] = prediction_step["mask_l1_loss"].detach()
                loss_components[
                    f"autoregressive_mask_kl_step_{rollout_step}"
                ] = prediction_step["mask_kl_loss"].detach()
                loss_components[
                    f"autoregressive_mask_sobel_step_{rollout_step}"
                ] = prediction_step["mask_sobel_loss"].detach()
        z_out = {
            "visual": prediction["pred_mask_probs"],
            "mask_probs": prediction["pred_mask_probs"],
            "proprio": prediction["pred_proprio_embedding"],
        }
        return z_out, None, None, total_loss, loss_components

    def rollout(
        self,
        obs_0,
        act,
        state_0=None,
        context_slots=None,
        context_proprio_embeddings=None,
        context_actions=None,
        context_state_tokens=None,
        return_retrieval_trace=False,
        retrieval_context=None,
        retrieval_layout_ids=None,
        predictor_context_cache=None,
    ):
        if state_0 is None:
            raise ValueError("SlotMaskWorldModel.rollout requires raw initial state")
        if context_slots is None:
            with self._slot_extract_context():
                context_slots = self._extract_slots(obs_0["visual"])
        rolling_slots = {
            "obj_latents": context_slots["obj_latents"],
            "mask_logits": context_slots["mask_logits"][:, -1:],
            "mask_probs": context_slots["mask_probs"][:, -1:],
        }
        if context_proprio_embeddings is None:
            context_proprio_embeddings = self._encode_proprio(state_0[:, None].float())
        rolling_proprio = context_proprio_embeddings[:, -1:]

        # Retrieval belongs to the unique current state of one MPC decision,
        # not to a CEM candidate. Planning supplies a context cached before
        # CEM starts; standalone callers retrieve once here. It remains fixed
        # only for this CEM model evaluation. After an action is selected, the
        # predicted mask/proprio becomes the next decision's retrieval query;
        # no intermediate environment observation is required.
        if retrieval_context is None:
            if self.predictor_architecture == "sequential":
                count = int(self.predictor.num_context_frames)
                source_logits = rolling_slots["mask_logits"].expand(
                    -1, count, -1, -1
                )
                source_probs = rolling_slots["mask_probs"].expand(
                    -1, count, -1, -1
                )
                source_proprio = rolling_proprio.expand(-1, count, -1)
                zero_actions = self._normalized_zero_action_chunk(
                    source_logits.shape[0],
                    count,
                    source_logits.device,
                    source_proprio.dtype,
                )
                retrieval_context = {
                    "obj_latents": rolling_slots["obj_latents"],
                    "source_mask_logits": source_logits,
                    "source_mask_probs": source_probs,
                    "target_mask_logits": source_logits,
                    "target_mask_probs": source_probs,
                    "context_tokens": self._build_query_tokens(
                        source_proprio, zero_actions
                    ),
                    "target_proprio": source_proprio,
                    "indices": torch.full(
                        (source_logits.shape[0], count),
                        -1,
                        device=source_logits.device,
                        dtype=torch.long,
                    ),
                }
            else:
                retrieval_context = self.prepare_rollout_retrieval_context(
                    rolling_proprio,
                    rolling_slots["mask_logits"],
                    required_layout_ids=retrieval_layout_ids,
                )
        context_source_slots = {
            "obj_latents": retrieval_context["obj_latents"],
            "mask_logits": retrieval_context["source_mask_logits"],
            "mask_probs": retrieval_context["source_mask_probs"],
        }
        context_target_slots = {
            "obj_latents": retrieval_context["obj_latents"],
            "mask_logits": retrieval_context["target_mask_logits"],
            "mask_probs": retrieval_context["target_mask_probs"],
        }
        context_tokens = retrieval_context["context_tokens"]
        context_target_proprio = retrieval_context["target_proprio"]
        rollout_retrieval_indices = retrieval_context["indices"]
        if (
            predictor_context_cache is None
            and not self.training
            and not torch.is_grad_enabled()
        ):
            with self._autocast_context():
                predictor_context_cache = self._prepare_predictor_context_cache(
                    context_source_slots,
                    context_target_slots,
                    context_tokens,
                    context_target_proprio,
                )

        predicted_masks, predicted_proprio = [], []
        # Keep an explicit trace of the inference path for diagnostics and
        # training-time visualization.  After step 0 these query masks are
        # predictions, not teacher-forced observations.  The recalled real
        # context above remains fixed throughout the rollout.
        query_masks = [] if return_retrieval_trace else None
        retrieved_source_masks = [] if return_retrieval_trace else None
        retrieved_target_masks = [] if return_retrieval_trace else None
        retrieval_indices = [] if return_retrieval_trace else None
        for step in range(act.shape[1]):
            if return_retrieval_trace:
                query_masks.append(rolling_slots["mask_probs"][:, -1:])
                retrieved_source_masks.append(context_source_slots["mask_probs"])
                retrieved_target_masks.append(context_target_slots["mask_probs"])
                retrieval_indices.append(rollout_retrieval_indices[:, None])
            query_source_slots = {
                "obj_latents": rolling_slots["obj_latents"],
                "mask_logits": rolling_slots["mask_logits"][:, -1:],
                "mask_probs": rolling_slots["mask_probs"][:, -1:],
            }
            query_tokens = self._build_query_tokens(
                rolling_proprio[:, -1:], act[:, step : step + 1]
            )
            with self._autocast_context():
                step_logits, step_proprio = self._predict_outputs(
                    context_source_slots,
                    context_target_slots,
                    query_source_slots,
                    context_tokens,
                    query_tokens,
                    context_target_proprio,
                    predictor_context_cache=predictor_context_cache,
                )
            step_mask = F.softmax(step_logits, dim=2)
            step_context_logits = step_mask.clamp_min(1.0e-8).log()
            predicted_masks.append(step_mask)
            predicted_proprio.append(step_proprio)
            if self.predictor_architecture == "sequential":
                max_history = int(self.predictor.num_context_frames)
                context_source_slots = {
                    "obj_latents": context_source_slots["obj_latents"],
                    "mask_logits": torch.cat(
                        [
                            context_source_slots["mask_logits"],
                            query_source_slots["mask_logits"],
                        ],
                        dim=1,
                    )[:, -max_history:],
                    "mask_probs": torch.cat(
                        [
                            context_source_slots["mask_probs"],
                            query_source_slots["mask_probs"],
                        ],
                        dim=1,
                    )[:, -max_history:],
                }
                context_tokens = torch.cat(
                    [context_tokens, query_tokens], dim=1
                )[:, -max_history:]
            rolling_slots = {
                "obj_latents": rolling_slots["obj_latents"],
                "mask_logits": step_context_logits,
                "mask_probs": step_mask,
            }
            rolling_proprio = step_proprio

        pred_mask_probs = torch.cat(predicted_masks, dim=1)
        pred_proprio_embeddings = torch.cat(predicted_proprio, dim=1)
        rollout_output = {
            "visual": pred_mask_probs,
            "mask_probs": pred_mask_probs,
            "proprio": pred_proprio_embeddings,
        }
        rollout_trace = {"proprio_embeddings": pred_proprio_embeddings}
        if return_retrieval_trace:
            rollout_trace.update(
                {
                    "query_mask_probs": torch.cat(query_masks, dim=1),
                    "retrieved_source_mask_probs": torch.stack(
                        retrieved_source_masks, dim=1
                    ),
                    "retrieved_target_mask_probs": torch.stack(
                        retrieved_target_masks, dim=1
                    ),
                    "retrieval_indices": torch.cat(retrieval_indices, dim=1),
                }
            )
        return rollout_output, rollout_trace

    def receding_rollout(
        self,
        obs_0,
        act,
        state_0=None,
        context_slots=None,
        context_proprio_embeddings=None,
        return_retrieval_trace=False,
        retrieval_layout_ids=None,
    ):
        """Roll out repeated MPC decisions without observation feedback.

        Each action is treated as the action selected by one MPC decision.
        Its predicted next mask/proprio becomes the current state for the next
        decision, which performs a fresh retrieval before predicting again.
        Online planning uses this predictor/retrieval boundary inside an
        ``n_taken_actions`` block.  Environment feedback is consumed only once,
        after the complete block has been executed.
        """
        if state_0 is None:
            raise ValueError("SlotMaskWorldModel.receding_rollout requires state_0")
        if context_slots is None:
            with self._slot_extract_context():
                context_slots = self._extract_slots(obs_0["visual"])
        rolling_slots = {
            "obj_latents": context_slots["obj_latents"],
            "mask_logits": context_slots["mask_logits"][:, -1:],
            "mask_probs": context_slots["mask_probs"][:, -1:],
        }
        if context_proprio_embeddings is None:
            context_proprio_embeddings = self._encode_proprio(
                state_0[:, None].float()
            )
        rolling_proprio = context_proprio_embeddings[:, -1:]

        output_parts = []
        proprio_parts = []
        trace_parts = []
        for step in range(act.shape[1]):
            step_output, step_trace = self.rollout(
                obs_0=obs_0,
                act=act[:, step : step + 1],
                state_0=state_0,
                context_slots=rolling_slots,
                context_proprio_embeddings=rolling_proprio,
                return_retrieval_trace=return_retrieval_trace,
                retrieval_layout_ids=retrieval_layout_ids,
                # Deliberately do not pass retrieval_context: this is a new
                # MPC decision at the newly predicted current state.
            )
            output_parts.append(step_output["mask_probs"])
            proprio_parts.append(step_output["proprio"])
            step_mask = step_output["mask_probs"][:, -1:]
            rolling_slots = {
                "obj_latents": rolling_slots["obj_latents"],
                "mask_logits": step_mask.clamp_min(1.0e-8).log(),
                "mask_probs": step_mask,
            }
            rolling_proprio = step_output["proprio"][:, -1:]
            if return_retrieval_trace:
                trace_parts.append(step_trace)

        pred_masks = torch.cat(output_parts, dim=1)
        pred_proprio = torch.cat(proprio_parts, dim=1)
        output = {
            "visual": pred_masks,
            "mask_probs": pred_masks,
            "proprio": pred_proprio,
        }
        trace = {"proprio_embeddings": pred_proprio}
        if return_retrieval_trace:
            for key in (
                "query_mask_probs",
                "retrieved_source_mask_probs",
                "retrieved_target_mask_probs",
                "retrieval_indices",
            ):
                trace[key] = torch.cat([part[key] for part in trace_parts], dim=1)
        return output, trace
