import io
import argparse
import math
import os
import sys
import time
import zlib
from pathlib import Path
from collections import deque
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import zmq
from scipy.spatial.transform import Rotation

_RELEASE_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _RELEASE_SRC not in sys.path:
    sys.path.insert(0, _RELEASE_SRC)

from slot_jepa.predictors import ActionConditionalMaskPredictor
from slot_jepa.slots import TwoStageExtractor
from slot_jepa.encoders import vision_transformer as vit
from slot_jepa.utils.mpc import compute_new_pose


DEFAULT_SLOT_WARMUP_XY_RADIUS = 0.2
DEFAULT_SLOT_WARMUP_Z_OFFSETS = [-0.15, -0.25]
DEFAULT_SLOT_WARMUP_POINTS_PER_Z = 6
DEFAULT_SLOT_WARMUP_SETTLE_STEPS = 8
DEFAULT_SLOT_STEERING_MEMORY_TOPK = 12
DEFAULT_SLOT_STEERING_SAMPLE_FRACTION = 0.0
DEFAULT_SLOT_STEERING_LOSS_RANGE_EPS = 1.0e-5
DEFAULT_OBSERVED_MOTION_WINDOW = 5
MEMORY_SELECTION_MODES = ("relevance", "recent", "in_trajectory_random")
_GPU_MEMORY_RESERVATIONS = []
_GPU_MEMORY_RESERVED_IN_CACHE = False


def _batch_candidate_history_tensors(selected_items_by_candidate, context_frames, view_name):
    """Batch candidate histories without repeatedly concatenating shared frames."""
    if not selected_items_by_candidate:
        raise ValueError(f"No selected history items for {view_name}.")

    selected_count = len(selected_items_by_candidate[0])
    if selected_count <= 0:
        raise ValueError(f"No selected history frames for {view_name}.")
    if any(len(selected) != selected_count for selected in selected_items_by_candidate):
        lengths = [len(selected) for selected in selected_items_by_candidate]
        raise ValueError(f"Candidate history lengths differ for {view_name}: {lengths}")

    context_count = min(int(context_frames), selected_count)
    unique_items = []
    unique_index_by_storage = {}
    candidate_indices = []
    for selected_items in selected_items_by_candidate:
        missing_indices = [
            int(item["original_index"])
            for item in selected_items
            if item.get("context") is None
        ]
        if missing_indices:
            raise ValueError(
                f"Missing cached SLOT context for {view_name} candidate history indices "
                f"{missing_indices}"
            )
        tail_items = selected_items[-context_count:]
        tail_indices = []
        for item in tail_items:
            context = item.get("context")
            storage_key = (id(context), id(item["state"]), id(item["extrinsics"]))
            pool_index = unique_index_by_storage.get(storage_key)
            if pool_index is None:
                pool_index = len(unique_items)
                unique_index_by_storage[storage_key] = pool_index
                unique_items.append(item)
            tail_indices.append(pool_index)
        candidate_indices.append(tail_indices)

    mask_logits_pool = torch.cat(
        [item["context"]["decoded_mask_logits"] for item in unique_items], dim=0
    ).float()
    mask_probs_pool = torch.cat(
        [item["context"]["decoded_mask_probs"] for item in unique_items], dim=0
    ).float()
    states_pool = torch.cat([item["state"] for item in unique_items], dim=0).float()
    extrinsics_pool = torch.cat(
        [item["extrinsics"] for item in unique_items], dim=0
    ).float()

    batch_size = len(candidate_indices)
    flat_indices = torch.as_tensor(
        candidate_indices,
        dtype=torch.long,
        device=mask_logits_pool.device,
    ).reshape(-1)
    context_masks = mask_logits_pool.index_select(0, flat_indices).reshape(
        batch_size, context_count, *mask_logits_pool.shape[1:]
    )
    poses_view = states_pool.index_select(
        0, flat_indices.to(device=states_pool.device)
    ).reshape(batch_size, context_count, *states_pool.shape[1:])
    extr_view = extrinsics_pool.index_select(
        0, flat_indices.to(device=extrinsics_pool.device)
    ).reshape(batch_size, context_count, *extrinsics_pool.shape[1:])
    current_indices = flat_indices.reshape(batch_size, context_count)[:, -1]
    current_mask_probs = mask_probs_pool.index_select(
        0, current_indices.to(device=mask_probs_pool.device)
    )
    return context_masks, current_mask_probs, poses_view, extr_view


def reserve_visible_gpu_memory(fraction=0.85, margin_mb=4096, chunk_mb=256):
    """Reserve free VRAM in this process' PyTorch cache while keeping it reusable."""
    global _GPU_MEMORY_RESERVED_IN_CACHE
    if not torch.cuda.is_available():
        print("[gpu_reserve] CUDA is unavailable; skipping GPU memory reservation.", flush=True)
        return []
    fraction = float(np.clip(float(fraction), 0.0, 1.0))
    margin_bytes = int(max(float(margin_mb), 0.0) * 1024 * 1024)
    chunk_bytes = int(max(float(chunk_mb), 1.0) * 1024 * 1024)
    summaries = []
    for device_idx in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{device_idx}")
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except Exception as exc:
            print(f"[gpu_reserve] cuda:{device_idx} mem_get_info failed: {exc}", flush=True)
            continue
        # Keep a real headroom for future PyTorch allocations. Filling to 100%
        # makes the next large workspace allocation fail even if the cached
        # blocks are theoretically reusable.
        reservable_bytes = max(0, int((int(free_bytes) - margin_bytes) * fraction))
        reserved_bytes = 0
        device_chunks = []
        while reserved_bytes < reservable_bytes:
            alloc_bytes = min(chunk_bytes, reservable_bytes - reserved_bytes)
            if alloc_bytes <= 0:
                break
            try:
                # uint8 gives direct byte-level control over the reservation size.
                chunk = torch.empty((alloc_bytes,), dtype=torch.uint8, device=device)
                device_chunks.append(chunk)
                reserved_bytes += int(alloc_bytes)
            except RuntimeError as exc:
                print(
                    f"[gpu_reserve] cuda:{device_idx} stopped after "
                    f"{reserved_bytes / 1024 ** 3:.2f} GiB due to allocation error. "
                    "This is non-fatal, but reduce --gpu-memory-reserve-fraction or increase "
                    f"--gpu-memory-reserve-margin-mb if inference OOM follows: {exc}",
                    flush=True,
                )
                break
        if device_chunks:
            # Drop Python references but do NOT call torch.cuda.empty_cache().
            # PyTorch keeps these blocks in its caching allocator, so other
            # processes cannot claim them while this server can still reuse them.
            del device_chunks
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
        try:
            free_after_bytes, _ = torch.cuda.mem_get_info(device)
            torch_reserved_bytes = torch.cuda.memory_reserved(device)
            torch_allocated_bytes = torch.cuda.memory_allocated(device)
        except Exception:
            free_after_bytes = None
            torch_reserved_bytes = None
            torch_allocated_bytes = None
        summary = {
            "device": device_idx,
            "total_bytes": int(total_bytes),
            "free_before_bytes": int(free_bytes),
            "reserved_request_bytes": int(reserved_bytes),
            "free_after_bytes": None if free_after_bytes is None else int(free_after_bytes),
            "torch_reserved_bytes": None if torch_reserved_bytes is None else int(torch_reserved_bytes),
            "torch_allocated_bytes": None if torch_allocated_bytes is None else int(torch_allocated_bytes),
        }
        summaries.append(summary)
        print(
            f"[gpu_reserve] cuda:{device_idx}: total={total_bytes / 1024 ** 3:.2f} GiB "
            f"free_before={free_bytes / 1024 ** 3:.2f} GiB "
            f"reserved_in_cache={reserved_bytes / 1024 ** 3:.2f} GiB "
            f"margin={margin_bytes / 1024 ** 3:.2f} GiB fraction={fraction:.3f}",
            flush=True,
        )
    _GPU_MEMORY_RESERVED_IN_CACHE = bool(summaries)
    return summaries


class MsgSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        import msgpack

        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        import msgpack

        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if "__ndarray_class__" in obj:
            obj = np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class SlotWorldModelAgent:
    VIEW_ORDER = ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]
    # DEFAULT_RUN_DIR = "outputs/v2.0-warp-both-disbale_shared_obj_latent"
    DEFAULT_RUN_DIR = "outputs/slot-jepa"
    DEFAULT_CKPT_PATH = os.path.join(DEFAULT_RUN_DIR, "downstream", "run", "latest.pt")

    @staticmethod
    def _build_student_from_config(model_cfg, crop_size, patch_size):
        img_as_video_nframes = int(model_cfg.get("vjepa_img_as_video_nframes", 2))
        model_name = model_cfg.get("vjepa_model_name", model_cfg.get("model_name", "vit_base"))
        model = vit.__dict__[model_name](
            img_size=crop_size,
            num_frames=img_as_video_nframes,
            patch_size=patch_size,
            tubelet_size=2,
            uniform_power=True,
            use_rope=True,
            interpolate_rope=True,
        )
        rope_reference_grid_size = int(crop_size) // int(patch_size)
        for block in model.blocks:
            attention = getattr(block, "attn", None)
            if attention is not None and hasattr(attention, "interpolate_rope"):
                attention.interpolate_rope = True
                attention.pretrained_grid_size = rope_reference_grid_size

        def forward_prehook(_module, input):
            image = input[0]
            return image.unsqueeze(2).repeat(1, 1, img_as_video_nframes, 1, 1)

        model.register_forward_pre_hook(forward_prehook)
        return model

    def __init__(
        self,
        run_dir=DEFAULT_RUN_DIR,
        ckpt_path=None,
        config_path=None,
        history_len=8,
        device="cuda",
        transport_loss_mix=0.0,
        use_goal_pose_prior=False,
        sinkhorn_iters=5,
        sinkhorn_entropy_reg=0.05,
        mask_centralize_gamma=2.0,
        unbalanced_ot_rho=0.2,
        unbalanced_ot_eps=1.0e-8,
        mask_transport_area_normalize=True,
        mask_transport_area_normalize_eps=1.0e-3,
        use_shared_visible_wrist_ot=True,
        shared_visible_sim_threshold=0.55,
        shared_visible_temperature=0.05,
        shared_visible_min_mass=0.02,
        factorized_pose_cem=True,
        factorized_cem_steps_per_phase=None,
        gripper_cem_steps=1,
        gripper_samples=5,
        cem_samples=80,
        cem_elite_topk=10,
        wrist_rerank_topk=10,
        wrist_refine_momentum_mean=0.75,
        wrist_refine_momentum_std=0.90,
        cem_steps=3,
        fixed_loss_search_min_scale=0.35,
        pose_rotation_cap=0.15,
        pose_rotation_std=0.05,
        pose_translation_xy_cap=0.03,
        pose_translation_z_cap=0.3,
        pose_translation_xy_std=0.01,
        pose_translation_z_std=0.1,
        execution_translation_cap=None,
        execution_rotation_cap=None,
        slot_action_range_scale=1.0,
        action_l2_weight=0.0,
        view_loss_mode="gated",
        fixed_view_loss_gate_threshold=0.001,
        optimize_gripper_action=True,
        gripper_position_max=1.0,
        fixed_gripper_target=0.0,
        gripper_min_open_gap=0.05,
        gripper_downpress_max_z=0.05,
        gripper_delta_std=0.03,
        gripper_delta_cap=0.08,
        gripper_cem_skip_translation_threshold=0.03,
        gripper_cem_skip_rotation_threshold=0.04,
        slot_warmup_xy_radius=DEFAULT_SLOT_WARMUP_XY_RADIUS,
        slot_warmup_z_offsets=None,
        slot_warmup_settle_steps=DEFAULT_SLOT_WARMUP_SETTLE_STEPS,
        fixed_slot_min_area_ratio=0.01,
        fixed_slot_max_area_ratio=0.20,
        fixed_slot_entropy_weight=0.7,
        fixed_slot_centroid_weight=0.3,
        slot_steering_memory_topk=DEFAULT_SLOT_STEERING_MEMORY_TOPK,
        slot_steering_sample_fraction=DEFAULT_SLOT_STEERING_SAMPLE_FRACTION,
        slot_steering_loss_range_eps=DEFAULT_SLOT_STEERING_LOSS_RANGE_EPS,
        history_pose_sampling_temperature=0.05,
        history_pose_uniform_mix=0.15,
        memory_selection_mode="relevance",
        memory_random_seed=0,
        always_execute_best_candidate=False,
    ):
        self.device = device
        self.history_len = history_len
        self.use_goal_pose_prior = bool(use_goal_pose_prior)
        self.model_context_frames = max(1, history_len - 1)
        self.gripper_position_min = 0.0
        # DROID/model convention used by this server: normalized gripper state
        # is 0=open and 1=closed. Simulator command conversion happens only at
        # the client/controller boundary.
        self.gripper_position_max = float(max(gripper_position_max, self.gripper_position_min))
        self.fixed_gripper_target = float(
            np.clip(fixed_gripper_target, self.gripper_position_min, self.gripper_position_max)
        )
        self.gripper_min_open_gap = float(max(gripper_min_open_gap, 0.0))
        self.gripper_downpress_max_z = float(max(gripper_downpress_max_z, 0.0))
        # Keep gripper proposals local. Large gripper jumps are valid in DROID,
        # but they often destabilize the predictor during visual MPC.
        self.gripper_delta_std = float(max(gripper_delta_std, 0.0))
        self.gripper_delta_cap = float(max(gripper_delta_cap, 0.0))
        self.history_pose_sampling_temperature = float(max(history_pose_sampling_temperature, 1.0e-6))
        self.history_pose_uniform_mix = float(np.clip(history_pose_uniform_mix, 0.0, 1.0))
        if memory_selection_mode not in MEMORY_SELECTION_MODES:
            raise ValueError(
                f"Unsupported memory_selection_mode={memory_selection_mode!r}; "
                f"expected one of {MEMORY_SELECTION_MODES}."
            )
        self.memory_selection_mode = str(memory_selection_mode)
        self.always_execute_best_candidate = bool(always_execute_best_candidate)
        self.memory_random_seed = int(memory_random_seed)
        self.in_trajectory_random_effective_seed = int(self.memory_random_seed) & 0xFFFFFFFF
        self.in_trajectory_random_rng = np.random.default_rng(self.in_trajectory_random_effective_seed)
        self.in_trajectory_random_index_cache = {}
        self.gripper_history_pose_distance_weight = 0.2
        self.wrist_refine_momentum_mean = float(np.clip(wrist_refine_momentum_mean, 0.0, 1.0))
        self.wrist_refine_momentum_std = float(np.clip(wrist_refine_momentum_std, 0.0, 1.0))
        self.crop_size = 384
        self.patch_size = 16
        self.tokens_per_frame = (self.crop_size // self.patch_size) * (self.crop_size // self.patch_size)
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.autocast_dtype = torch.bfloat16 if (torch.cuda.is_available() and "cuda" in str(device)) else None

        ckpt_path, config_path = self._resolve_model_paths(run_dir, ckpt_path, config_path)
        if config_path is None:
            config_path = self._infer_config_path(ckpt_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg["model"]
        data_cfg = cfg.get("data", {})

        self.crop_size = int(data_cfg.get("crop_size", self.crop_size))
        self.patch_size = int(data_cfg.get("patch_size", self.patch_size))
        dataset_fpcs = data_cfg.get("dataset_fpcs")
        if dataset_fpcs:
            self.history_len = int(max(dataset_fpcs))
            self.model_context_frames = max(1, self.history_len - 1)
        self.tokens_per_frame = (self.crop_size // self.patch_size) * (self.crop_size // self.patch_size)
        self.frame_buffer = {view_name: deque() for view_name in self.VIEW_ORDER}
        self.state_buffer = {view_name: deque() for view_name in self.VIEW_ORDER}
        self.extrinsics_buffer = {view_name: deque() for view_name in self.VIEW_ORDER}
        self.slot_context_buffer = {view_name: deque() for view_name in self.VIEW_ORDER}
        self.history_goal_loss_buffer = {view_name: deque() for view_name in self.VIEW_ORDER}
        self.slot_area_normalizer_by_view = {}
        self.embed_dim = int(model_cfg.get("embed_dim", 768))
        self.slot_feature_dim = int(model_cfg.get("slot_feature_dim", model_cfg.get("feature_dim", 196)))
        self.predictor_embed_dim = int(model_cfg.get("predictor_embed_dim", model_cfg.get("pred_embed_dim", 196)))
        self.predictor_depth = int(model_cfg.get("predictor_depth", model_cfg.get("pred_depth", 8)))
        self.predictor_num_heads = int(model_cfg.get("predictor_num_heads", model_cfg.get("pred_num_heads", 4)))
        self.num_objects = int(model_cfg.get("num_objects", model_cfg.get("global_k", 24)))
        self.use_extrinsics = bool(model_cfg.get("use_extrinsics", True))
        self.concat_mask_input = bool(model_cfg.get("concat_mask_input", True))
        self.num_obj_tokens = int(model_cfg.get("num_obj_tokens", 1))
        self.num_extrinsics_tokens = int(model_cfg.get("num_extrinsics_tokens", 1))
        self.mask_temp = float(model_cfg.get("mask_temp", 0.05))
        self.use_wrist_warp_concat = bool(model_cfg.get("use_wrist_warp_concat", True))
        self.warp_hfov_deg = float(model_cfg.get("warp_hfov_deg", 60.0))
        self.warp_extrinsics_format = model_cfg.get("warp_extrinsics_format", "c2w")
        self.warp_rotation_mode = model_cfg.get("warp_rotation_mode", "euler_xyz")
        self.warp_depth_bins = model_cfg.get("warp_depth_bins", [0.15, 0.25, 0.4, 0.7, 1.2, 2.0])
        self.object_branch_scales = model_cfg.get("object_branch_scales", None)
        self.transport_loss_mix = float(np.clip(transport_loss_mix, 0.0, 1.0))
        # The public CLI binds both flags to --disable-uot-goal-energy. Keep an
        # explicit mode bit for safeguards that remain specific to A2.
        self.a2_no_uot_mode = bool(self.use_goal_pose_prior and self.transport_loss_mix <= 0.0)
        self.fixed_view_noop_fallback_enabled = bool(
            not self.always_execute_best_candidate
            and (
                self.a2_no_uot_mode
                or self.memory_selection_mode in {"recent", "in_trajectory_random"}
            )
        )
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_entropy_reg = float(sinkhorn_entropy_reg)
        self.mask_centralize_gamma = float(max(mask_centralize_gamma, 1.0))
        self.unbalanced_ot_rho = float(max(unbalanced_ot_rho, 1.0e-8))
        self.unbalanced_ot_eps = float(max(unbalanced_ot_eps, 1.0e-12))
        self.mask_transport_area_normalize = bool(mask_transport_area_normalize)
        self.mask_transport_area_normalize_eps = float(max(mask_transport_area_normalize_eps, self.unbalanced_ot_eps))
        self.use_shared_visible_wrist_ot = bool(use_shared_visible_wrist_ot)
        self.shared_visible_sim_threshold = float(shared_visible_sim_threshold)
        self.shared_visible_temperature = float(max(shared_visible_temperature, 1.0e-4))
        self.shared_visible_min_mass = float(max(shared_visible_min_mass, 0.0))
        self.factorized_pose_cem = bool(factorized_pose_cem)
        self.factorized_cem_steps_per_phase = (
            None
            if factorized_cem_steps_per_phase is None or int(factorized_cem_steps_per_phase) <= 0
            else int(factorized_cem_steps_per_phase)
        )
        self.gripper_cem_steps = max(1, int(gripper_cem_steps))
        self.gripper_samples = max(2, int(gripper_samples))
        self._ot_cost_cache = {}

        print(f"Loading SLOT-AC World Model from {ckpt_path}...", flush=True)
        print(f"Using SLOT-AC config from {config_path}", flush=True)
        self.student = self._build_student_from_config(model_cfg, self.crop_size, self.patch_size).to(device)
        self.extractor = TwoStageExtractor(
            window_size=model_cfg.get("window_size", [2, 4]),
            slots_per_window=model_cfg.get("slots_per_window", 4),
            global_k=self.num_objects,
            D_in=self.embed_dim,
            mask_temp=self.mask_temp,
            L_max=model_cfg.get("L_max", self.num_objects),
            use_decoder_checkpoint=model_cfg.get("use_decoder_checkpoint", False),
        ).to(device)
        self.predictor = ActionConditionalMaskPredictor(
            embed_dim=self.embed_dim,
            feature_dim=self.slot_feature_dim,
            predictor_embed_dim=self.predictor_embed_dim,
            depth=self.predictor_depth,
            num_heads=self.predictor_num_heads,
            num_context_frames=self.model_context_frames,
            spatial_size=self.crop_size // self.patch_size,
            num_objects=self.num_objects,
            state_dim=int(model_cfg.get("state_dim", 7)),
            use_extrinsics=self.use_extrinsics,
            extrinsics_dim=int(model_cfg.get("extrinsics_dim", 6)),
            num_obj_tokens=self.num_obj_tokens,
            concat_mask_input=self.concat_mask_input,
            use_warp_concat=self.use_wrist_warp_concat,
            warp_hfov_deg=self.warp_hfov_deg,
            warp_extrinsics_format=self.warp_extrinsics_format,
            warp_rotation_mode=self.warp_rotation_mode,
            warp_depth_bins=self.warp_depth_bins,
            patch_size=self.patch_size,
            use_checkpoint=False,
            object_branch_scales=self.object_branch_scales,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        student_msg = self.student.load_state_dict(ckpt["student_state"], strict=False)
        extractor_msg = self.extractor.load_state_dict(ckpt["extractor_state"], strict=False)
        print(f"Loaded SLOT student from latest.pt with msg: {student_msg}", flush=True)
        print(f"Loaded SLOT extractor from latest.pt with msg: {extractor_msg}", flush=True)
        pred_state = ckpt.get("slot_ac_predictor_state", ckpt.get("predictor_state"))
        if pred_state is None:
            raise KeyError("Checkpoint is missing slot_ac_predictor_state/predictor_state")
        if any(key.startswith("default.") or key.startswith("wrist.") for key in pred_state.keys()):
            remapped_state = {}
            for key, value in pred_state.items():
                if key.startswith("default."):
                    remapped_state[key[len("default."):]] = value
                elif key.startswith("wrist."):
                    continue
                else:
                    remapped_state[key] = value
            pred_state = remapped_state
        msg = self.predictor.load_state_dict(pred_state, strict=False)
        print(f"Loaded SLOT-AC predictor with msg: {msg}", flush=True)

        self.student.eval()
        self.extractor.eval()
        self.predictor.eval()
        for module in (self.student, self.extractor, self.predictor):
            for param in module.parameters():
                param.requires_grad = False

        self.mpc_args = {
            "rollout": 1,
            "samples": int(max(1, cem_samples)),
            "topk": int(max(1, cem_elite_topk)),
            "wrist_rerank_topk": int(max(1, wrist_rerank_topk)),
            "cem_steps": int(max(1, cem_steps)),
            "momentum_mean": 0.15,
            "momentum_mean_gripper": 0.15,
            "momentum_std": 0.75,
            "momentum_std_gripper": 0.15,
            "maxnorm": 0.05,
            "verbose": False,
        }
        self.mpc_eval_batch_size = 80
        self.fixed_loss_search_min_scale = float(np.clip(fixed_loss_search_min_scale, 0.01, 1.0))
        self.slot_action_range_scale = float(max(slot_action_range_scale, 0.0))
        self.pose_rotation_std = float(pose_rotation_std) * self.slot_action_range_scale
        self.pose_rotation_cap = float(pose_rotation_cap) * self.slot_action_range_scale
        self.pose_translation_xy_cap = float(pose_translation_xy_cap) * self.slot_action_range_scale
        self.pose_translation_z_cap = float(pose_translation_z_cap) * self.slot_action_range_scale
        self.pose_translation_xy_std = float(pose_translation_xy_std) * self.slot_action_range_scale
        self.pose_translation_z_std = float(pose_translation_z_std) * self.slot_action_range_scale
        self.execution_translation_cap = (
            float(execution_translation_cap)
            if execution_translation_cap is not None and float(execution_translation_cap) > 0.0
            else float(np.max(self._configured_translation_cap_xyz()))
        )
        self.execution_rotation_cap = (
            float(execution_rotation_cap)
            if execution_rotation_cap is not None and float(execution_rotation_cap) > 0.0
            else float(self.pose_rotation_cap)
        )
        self.action_l2_weight = float(max(action_l2_weight, 0.0))
        if view_loss_mode not in {"gated", "fixed_only", "wrist_only", "all"}:
            raise ValueError(
                f"Unsupported view_loss_mode={view_loss_mode!r}; "
                "expected one of gated, fixed_only, wrist_only, all."
            )
        self.view_loss_mode = view_loss_mode
        self.fixed_view_loss_gate_threshold = float(max(fixed_view_loss_gate_threshold, 0.0))
        self.optimize_gripper_action = bool(optimize_gripper_action)
        self.gripper_cem_skip_translation_threshold = float(max(gripper_cem_skip_translation_threshold, 0.0))
        self.gripper_cem_skip_rotation_threshold = float(max(gripper_cem_skip_rotation_threshold, 0.0))
        self.slot_warmup_xy_radius = float(max(slot_warmup_xy_radius, 0.0))
        if slot_warmup_z_offsets is None:
            slot_warmup_z_offsets = DEFAULT_SLOT_WARMUP_Z_OFFSETS
        self.slot_warmup_z_offsets = [float(x) for x in np.asarray(slot_warmup_z_offsets, dtype=np.float32).reshape(-1)]
        if len(self.slot_warmup_z_offsets) < 1:
            raise ValueError("Expected at least one SLOT warmup z offset.")
        self.slot_warmup_settle_steps = max(1, int(slot_warmup_settle_steps))
        self.fixed_slot_min_area_ratio = float(max(fixed_slot_min_area_ratio, 0.0))
        self.fixed_slot_max_area_ratio = float(max(fixed_slot_max_area_ratio, self.fixed_slot_min_area_ratio))
        self.fixed_slot_entropy_weight = float(max(fixed_slot_entropy_weight, 0.0))
        self.fixed_slot_centroid_weight = float(max(fixed_slot_centroid_weight, 0.0))
        print(
            "[SLOT-AC] Effective action range: "
            f"translation_cap_xyz={self._configured_translation_cap_xyz().tolist()} "
            f"translation_std_xyz={self._configured_translation_std_xyz().tolist()} "
            f"rotation_cap={self.pose_rotation_cap:g} rotation_std={self.pose_rotation_std:g} "
            f"execution_translation_cap={self.execution_translation_cap:g} "
            f"execution_rotation_cap={self.execution_rotation_cap:g} "
            f"scale={self.slot_action_range_scale:g}",
            flush=True,
        )
        # Broad DROID camera-frame training ranges from exported multiview labels.
        # The penalty is only a guardrail against CEM exploiting OOD states.
        self._state_xyz_bounds = {
            "exterior_image_1_left": (
                np.array([-0.50, -0.62, -0.05], dtype=np.float32),
                np.array([0.65, 0.45, 1.15], dtype=np.float32),
            ),
            "exterior_image_2_left": (
                np.array([-0.55, -0.60, -0.05], dtype=np.float32),
                np.array([0.65, 0.65, 1.15], dtype=np.float32),
            ),
            "wrist_image_left": (
                np.array([0.00, 0.04, -0.06], dtype=np.float32),
                np.array([0.07, 0.12, 0.07], dtype=np.float32),
            ),
        }
        self._extrinsics_xyz_bounds = {
            "exterior_image_1_left": (
                np.array([-0.25, -0.75, 0.10], dtype=np.float32),
                np.array([0.50, 1.00, 0.80], dtype=np.float32),
            ),
            "exterior_image_2_left": (
                np.array([-0.25, -0.90, 0.10], dtype=np.float32),
                np.array([0.70, 0.80, 0.80], dtype=np.float32),
            ),
            "wrist_image_left": (
                np.array([0.10, -0.35, -0.10], dtype=np.float32),
                np.array([0.85, 0.35, 0.75], dtype=np.float32),
            ),
        }
        self.goal_features = None
        self.goal_feature_full = None
        self.goal_mask_probs = None
        self.goal_view_names = []
        self.goal_image_maps = None
        self.fixed_obj_latents = {}
        self.fixed_slot_index_by_view = {}
        self.fixed_slot_debug_by_view = {}
        self.gripper_slot_index_by_view = {}
        self.gripper_slot_debug_by_view = {}
        self.fixed_latent_initialized = False
        self.goal_state = None
        self.last_goal_pose_prior_debug = None
        self.current_goal_idx = 0
        self.predict_step_idx = 0
        self.slot_planning_phase_cursor = "pose"
        self.last_obj_latent_refresh_step = -1
        self.last_obj_latent_refresh_reason = None
        self.last_obj_latent_refresh_mode = None
        self.recent_selected_pose_deltas = deque(maxlen=3)
        self.last_observed_robot_pose6 = None
        self.observed_motion_window = int(DEFAULT_OBSERVED_MOTION_WINDOW)
        self.recent_observed_robot_poses = deque(maxlen=self.observed_motion_window + 1)
        self.recent_observed_pose_deltas = deque(maxlen=self.observed_motion_window)
        self.history_z_context = []
        self.history_z_warp = []
        self.history_z_pred = []
        self.history_mask_context = []
        self.history_mask_warp = []
        self.history_mask_pred = []
        self.history_debug = []
        self.last_predictor_debug = None
        self.last_history_selection_debug = {}
        self.pending_one_step_predictions = {}
        self.next_pending_prediction_id = 0
        self.pose_memory_by_view = {view_name: [] for view_name in self.VIEW_ORDER}
        self.gripper_history_bank_by_view = {view_name: [] for view_name in self.VIEW_ORDER}
        self.fixed_loss_search_reference_by_goal = {}
        self.warmup_fixed_anchor_by_goal = {}
        self.pose_view_best_losses_by_goal = {}
        self.pose_view_no_improve_steps_by_goal = {}
        self.pose_view_active_group_by_goal = {}
        self.pose_view_loss_history_by_goal = {}
        self.warmup_score_obs_sequence = None
        self.pose_memory_topk = max(1, int(slot_steering_memory_topk))
        self.steering_sample_fraction = float(np.clip(slot_steering_sample_fraction, 0.0, 1.0))
        self.steering_loss_range_eps = float(max(slot_steering_loss_range_eps, 0.0))

    def _prepare_in_trajectory_random_memory(self):
        case_id = "unknown_case"
        if isinstance(self.goal_state, dict):
            case_id = str(
                self.goal_state.get("success_case_dir")
                or self.goal_state.get("task_cfg_path")
                or case_id
            )
        case_hash = zlib.crc32(case_id.encode("utf-8")) & 0xFFFFFFFF
        seed_sequence = np.random.SeedSequence(
            [int(self.memory_random_seed) & 0xFFFFFFFF, case_hash]
        )
        self.in_trajectory_random_effective_seed = int(seed_sequence.generate_state(1)[0])
        self.in_trajectory_random_rng = np.random.default_rng(seed_sequence)
        self.in_trajectory_random_index_cache = {}

    def _in_trajectory_random_indices(self, num_items, max_items):
        num_items = int(num_items)
        sample_count = min(num_items, int(max_items), int(self.history_len))
        cache_key = (
            int(self.predict_step_idx),
            num_items,
            sample_count,
        )
        cached = self.in_trajectory_random_index_cache.get(cache_key)
        if cached is None:
            if sample_count <= 0:
                cached = []
            else:
                # Keep the current observation as the query source and sample
                # the remaining memory uniformly from the complete preceding
                # in-trajectory history. Warmup observations remain eligible.
                latest_idx = num_items - 1
                history_count = max(0, sample_count - 1)
                history_pool = np.arange(0, latest_idx, dtype=np.int64)
                chosen_count = min(history_count, int(history_pool.size))
                if chosen_count > 0:
                    chosen = self.in_trajectory_random_rng.choice(
                        history_pool,
                        size=chosen_count,
                        replace=False,
                    )
                    cached = sorted(int(idx) for idx in chosen.tolist())
                else:
                    cached = []
                cached.append(int(latest_idx))
            self.in_trajectory_random_index_cache[cache_key] = cached
        return list(cached)

    def get_warmup_config(self):
        gripper_warmup_max = min(
            self.gripper_position_max,
            max(self.gripper_position_min, self.gripper_position_max - self.gripper_min_open_gap),
        )
        return {
            "status": "success",
            "planning_ablation": {
                "uot_goal_energy_enabled": bool(self.transport_loss_mix > 0.0),
                "goal_energy_mode": (
                    "mask_l1"
                    if self.transport_loss_mix <= 0.0
                    else "mask_l1_uot_mix"
                ),
                "goal_pose_prior_enabled": bool(self.use_goal_pose_prior),
                "warmup_anchor_guidance_enabled": bool(not self.a2_no_uot_mode),
                "fixed_view_noop_fallback_enabled": bool(
                    self.fixed_view_noop_fallback_enabled
                ),
                "always_execute_best_candidate": bool(
                    self.always_execute_best_candidate
                ),
                "transport_loss_mix": float(self.transport_loss_mix),
                "memory_selection_mode": self.memory_selection_mode,
                "memory_random_seed": int(self.memory_random_seed),
                "memory_sampling_scope": (
                    "current_trajectory_history"
                    if self.memory_selection_mode == "in_trajectory_random"
                    else None
                ),
                "in_trajectory_random_effective_seed": int(
                    self.in_trajectory_random_effective_seed
                ),
            },
            "mode": "z_levels_xy_circle",
            "xy_radius": float(self.slot_warmup_xy_radius),
            "z_offsets": list(self.slot_warmup_z_offsets),
            "points_per_z": int(DEFAULT_SLOT_WARMUP_POINTS_PER_Z),
            "num_frames": int(len(self.slot_warmup_z_offsets) * DEFAULT_SLOT_WARMUP_POINTS_PER_Z + 1),
            "fixed_initial_gripper": True,
            "fixed_gripper_state": float(self.gripper_position_min),
            "random_gripper": False,
            "gripper_min": float(self.gripper_position_min),
            "gripper_max": float(gripper_warmup_max),
            "gripper_downpress_max_z": float(self.gripper_downpress_max_z),
            "settle_steps": int(self.slot_warmup_settle_steps),
        }

    @staticmethod
    def _resolve_model_paths(run_dir, ckpt_path=None, config_path=None):
        if run_dir is not None:
            run_dir = os.path.abspath(run_dir)
        if ckpt_path is None:
            if run_dir is None:
                raise ValueError("Expected either run_dir or ckpt_path")
            ckpt_path = os.path.join(run_dir, "downstream", "run", "latest.pt")
        if config_path is None and run_dir is not None:
            candidates = [
                os.path.join(run_dir, "downstream.yaml"),
                os.path.join(run_dir, "train_camera_base.yaml"),
                os.path.join(run_dir, "downstream", "train_camera_base.yaml"),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    config_path = candidate
                    break
        return ckpt_path, config_path

    @staticmethod
    def _infer_config_path(ckpt_path):
        ckpt_path = os.path.abspath(ckpt_path)
        run_dir = os.path.dirname(ckpt_path)
        parent_dir = os.path.dirname(run_dir)
        exp_dir = os.path.dirname(parent_dir)
        candidates = [
            os.path.join(parent_dir, "train_camera_base.yaml"),
            os.path.join(exp_dir, "downstream.yaml"),
            os.path.join(exp_dir, "train_camera_base.yaml"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            "Could not infer SLOT-AC config path. Tried: " + ", ".join(candidates)
        )

    def _autocast_context(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _get_fixed_view_prediction_streams(self):
        """Lazily create the two streams used by same-GPU fixed-view inference."""
        device = torch.device(self.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return ()
        streams = getattr(self, "_fixed_view_prediction_stream_cache", None)
        stream_device = getattr(self, "_fixed_view_prediction_stream_device", None)
        device_key = (device.type, device.index)
        if streams is None or stream_device != device_key:
            streams = (
                torch.cuda.Stream(device=device),
                torch.cuda.Stream(device=device),
            )
            self._fixed_view_prediction_stream_cache = streams
            self._fixed_view_prediction_stream_device = device_key
        return streams

    def _get_spatial_transport_cost(self, spatial_size, device):
        cache_key = (int(spatial_size), device.type, device.index)
        if cache_key in self._ot_cost_cache:
            return self._ot_cost_cache[cache_key]
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0.0, 1.0, spatial_size, device=device),
                torch.linspace(0.0, 1.0, spatial_size, device=device),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        cost = torch.cdist(coords, coords, p=2).pow(2)
        self._ot_cost_cache[cache_key] = cost
        return cost

    @staticmethod
    def _weighted_slot_loss_mean(loss_per_slot, slot_weights=None):
        if slot_weights is None:
            return loss_per_slot.reshape(loss_per_slot.shape[0], -1).mean(dim=1)
        weights = slot_weights.to(loss_per_slot.device, dtype=loss_per_slot.dtype).clamp_min(0.0)
        if weights.ndim == 1:
            weights = weights.view(*([1] * (loss_per_slot.ndim - 1)), weights.shape[0])
        elif weights.ndim == 2 and loss_per_slot.ndim > 2:
            weights = weights.view(weights.shape[0], *([1] * (loss_per_slot.ndim - 2)), weights.shape[1])
        if weights.shape[0] == 1 and loss_per_slot.shape[0] != 1:
            weights = weights.expand(loss_per_slot.shape[0], *weights.shape[1:])
        weights = torch.broadcast_to(weights, loss_per_slot.shape)
        weighted = (loss_per_slot * weights).reshape(loss_per_slot.shape[0], -1).sum(dim=1)
        denom = weights.reshape(weights.shape[0], -1).sum(dim=1).clamp_min(1.0e-6)
        return weighted / denom

    @staticmethod
    def _slot_change_weights(reference_mask_probs, target_mask_probs, min_weight=0.05, max_weight=5.0):
        reference = reference_mask_probs.detach().float()
        target = target_mask_probs.detach().float()
        if reference.ndim == 2:
            reference = reference.unsqueeze(0)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        if target.shape[0] == 1 and reference.shape[0] != 1:
            target = target.expand(reference.shape[0], -1, -1)
        if reference.shape != target.shape:
            raise ValueError(
                f"Expected reference and target masks to align, got {tuple(reference.shape)} vs {tuple(target.shape)}"
            )
        weights = torch.mean(torch.abs(target - reference), dim=-1)
        mean = weights.mean(dim=-1, keepdim=True)
        uniform = torch.ones_like(weights)
        weights = torch.where(mean > 1.0e-8, weights / mean.clamp_min(1.0e-8), uniform)
        return weights.clamp(float(min_weight), float(max_weight)).float()

    def _slot_area_normalizer_for_view(self, view_name, device=None, dtype=torch.float32):
        normalizer = self.slot_area_normalizer_by_view.get(view_name)
        if normalizer is None:
            return None
        return normalizer.to(device or self.device, dtype=dtype).clamp_min(self.mask_transport_area_normalize_eps)

    def _centralize_mask_probs(self, mask_probs):
        mask = mask_probs.float().clamp_min(0.0)
        if self.mask_centralize_gamma <= 1.0:
            return mask
        area = mask.sum(dim=-1, keepdim=True)
        concentrated = mask.pow(self.mask_centralize_gamma)
        concentrated_sum = concentrated.sum(dim=-1, keepdim=True)
        restored = area * concentrated / concentrated_sum.clamp_min(self.unbalanced_ot_eps)
        return torch.where(area > self.unbalanced_ot_eps, restored, mask)

    def _select_min_variation_slot_from_warmup_masks(self, view_name, mask_list):
        mask_list = list(mask_list or [])
        if len(mask_list) < 2:
            return None, {
                "available": False,
                "reason": f"insufficient_{view_name}_warmup_masks",
                "num_masks": int(len(mask_list)),
            }
        raw_masks = torch.stack(
            [mask.detach().to(self.device, dtype=torch.float32) for mask in mask_list],
            dim=0,
        )
        masks = self._centralize_mask_probs(raw_masks)
        if masks.ndim != 3:
            return None, {
                "available": False,
                "reason": f"unexpected_mask_shape_{tuple(masks.shape)}",
                "num_masks": int(len(mask_list)),
            }
        mean_mask = masks.mean(dim=0, keepdim=True)
        spatial_size = int(math.isqrt(raw_masks.size(-1)))
        if spatial_size * spatial_size != raw_masks.size(-1):
            return None, {
                "available": False,
                "reason": f"non_square_mask_patch_count_{int(raw_masks.size(-1))}",
                "num_masks": int(len(mask_list)),
            }
        per_frame_l1 = torch.sum(torch.abs(masks - mean_mask), dim=2)
        per_frame_area = masks.sum(dim=2).clamp_min(self.unbalanced_ot_eps)
        normalized_l1_variation = (per_frame_l1 / per_frame_area).mean(dim=0)
        area_ratio = masks.sum(dim=2).mean(dim=0) / float(masks.shape[-1])

        flat = masks.clamp_min(0.0)
        mass = flat.sum(dim=2).clamp_min(self.unbalanced_ot_eps)
        prob = flat / mass.unsqueeze(-1)
        entropy = -(prob * prob.clamp_min(self.unbalanced_ot_eps).log()).sum(dim=2)
        entropy = entropy / math.log(float(raw_masks.size(-1)))
        mean_entropy = entropy.mean(dim=0)

        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, spatial_size, device=self.device, dtype=torch.float32),
            torch.linspace(0.0, 1.0, spatial_size, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        xx = xx.reshape(1, 1, -1)
        yy = yy.reshape(1, 1, -1)
        cx = (flat * xx).sum(dim=2) / mass
        cy = (flat * yy).sum(dim=2) / mass
        centroid_std = torch.sqrt(cx.var(dim=0, unbiased=False) + cy.var(dim=0, unbiased=False))

        def _rank01(values):
            order = torch.argsort(values)
            ranks = torch.empty_like(values, dtype=torch.float32)
            ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
            denom = float(max(values.numel() - 1, 1))
            return ranks / denom

        score = (
            _rank01(normalized_l1_variation)
            + self.fixed_slot_entropy_weight * _rank01(mean_entropy)
            + self.fixed_slot_centroid_weight * _rank01(centroid_std)
        )
        valid = (area_ratio >= self.fixed_slot_min_area_ratio) & (area_ratio <= self.fixed_slot_max_area_ratio)
        if bool(valid.any().detach().cpu().item()):
            masked_score = torch.where(valid, score, torch.full_like(score, float("inf")))
            slot_idx = int(torch.argmin(masked_score).detach().cpu().item())
            fallback_reason = None
        else:
            slot_idx = int(torch.argmin(score).detach().cpu().item())
            fallback_reason = "no_slot_satisfies_area_ratio_range"
        return slot_idx, {
            "available": True,
            "method": "warmup_area_normalized_l1_stable_compact_slot_score_with_area_range",
            "num_masks": int(len(mask_list)),
            "slot_index": int(slot_idx),
            "min_area_ratio": float(self.fixed_slot_min_area_ratio),
            "max_area_ratio": float(self.fixed_slot_max_area_ratio),
            "selected_area_ratio": float(area_ratio[slot_idx].detach().cpu().item()),
            "selected_mean_entropy": float(mean_entropy[slot_idx].detach().cpu().item()),
            "selected_centroid_std": float(centroid_std[slot_idx].detach().cpu().item()),
            "selected_score": float(score[slot_idx].detach().cpu().item()),
            "fallback_reason": fallback_reason,
            "valid_slot_indices": torch.nonzero(valid, as_tuple=False).reshape(-1).detach().cpu().numpy().astype(np.int64).tolist(),
            "mean_area_ratio_by_slot": area_ratio.detach().cpu().numpy().astype(np.float32).tolist(),
            "mean_entropy_by_slot": mean_entropy.detach().cpu().numpy().astype(np.float32).tolist(),
            "centroid_std_by_slot": centroid_std.detach().cpu().numpy().astype(np.float32).tolist(),
            "score_by_slot": score.detach().cpu().numpy().astype(np.float32).tolist(),
            "mean_area_normalized_l1_to_mean_by_slot": normalized_l1_variation.detach().cpu().numpy().astype(np.float32).tolist(),
        }

    def _update_fixed_slots_from_warmup_masks(self, masks_by_view):
        self.fixed_slot_index_by_view = {}
        self.fixed_slot_debug_by_view = {}
        view_name = "wrist_image_left"
        if view_name not in masks_by_view:
            self.fixed_slot_debug_by_view[view_name] = {
                "available": False,
                "reason": "missing_wrist_warmup_masks",
            }
            return
        slot_idx, debug = self._select_min_variation_slot_from_warmup_masks(
            view_name,
            masks_by_view.get(view_name) or [],
        )
        self.fixed_slot_debug_by_view[view_name] = debug
        if slot_idx is not None:
            self.fixed_slot_index_by_view[view_name] = int(slot_idx)

    def _update_gripper_slot_from_warmup_masks(self, masks_by_view):
        view_name = "wrist_image_left"
        slot_idx, debug = self._select_min_variation_slot_from_warmup_masks(
            view_name,
            masks_by_view.get(view_name) or [],
        )
        self.gripper_slot_debug_by_view[view_name] = debug
        if slot_idx is None:
            return
        self.gripper_slot_index_by_view[view_name] = int(slot_idx)

    @staticmethod
    def _generalized_kl(x, y, eps):
        x_safe = x.clamp_min(eps)
        y_safe = y.clamp_min(eps)
        return (x_safe * (torch.log(x_safe) - torch.log(y_safe)) - x_safe + y_safe).sum(dim=-1)

    def _unbalanced_sinkhorn_mask_loss(
        self,
        pred_mask_probs,
        target_mask_probs,
        spatial_size,
        patch_weights=None,
        chunk_size=32,
        slot_weights=None,
        area_normalizer=None,
        return_per_slot=False,
    ):
        if pred_mask_probs.shape != target_mask_probs.shape:
            raise ValueError(
                f"Expected pred and target mask probs to share shape, got {tuple(pred_mask_probs.shape)} vs {tuple(target_mask_probs.shape)}"
            )
        if pred_mask_probs.ndim == 2:
            pred_mask_probs = pred_mask_probs.unsqueeze(0)
            target_mask_probs = target_mask_probs.unsqueeze(0)
        if pred_mask_probs.ndim < 3:
            raise ValueError(f"Expected mask probs [...,K,N], got {tuple(pred_mask_probs.shape)}")

        pred_mass = self._centralize_mask_probs(pred_mask_probs)
        target_mass = self._centralize_mask_probs(target_mask_probs)
        if patch_weights is not None:
            patch_weights = patch_weights.to(pred_mass.device, dtype=torch.float32)
            if patch_weights.ndim == 1:
                patch_weights = patch_weights.view(1, 1, -1)
            elif patch_weights.ndim == 2:
                patch_weights = patch_weights.unsqueeze(1)
            else:
                raise ValueError(f"Expected patch_weights [N] or [B,N], got {tuple(patch_weights.shape)}")
            if patch_weights.size(-1) != pred_mass.size(-1):
                raise ValueError(
                    f"Patch weight size {patch_weights.size(-1)} does not match mask patches {pred_mass.size(-1)}"
                )
            if patch_weights.size(0) == 1 and pred_mass.size(0) != 1:
                patch_weights = patch_weights.expand(pred_mass.size(0), -1, -1)
            patch_weights = patch_weights.clamp(0.0, 1.0)
            pred_mass = pred_mass * patch_weights
            target_mass = target_mass * patch_weights

        pred_sum = pred_mass.sum(dim=-1)
        target_sum = target_mass.sum(dim=-1)
        effective_mass = torch.minimum(pred_sum, target_sum)
        pred_flat = pred_mass.reshape(-1, pred_mass.size(-1))
        target_flat = target_mass.reshape(-1, target_mass.size(-1))
        num_distributions, num_patches = pred_flat.shape
        if num_distributions == 0:
            zeros = pred_mask_probs.new_zeros((pred_mask_probs.shape[0],), dtype=torch.float32)
            if return_per_slot:
                return zeros, effective_mass.reshape(pred_mask_probs.shape[0], -1).mean(dim=1).float(), pred_mask_probs.new_zeros(pred_mask_probs.shape[:-1])
            return zeros, effective_mass.reshape(pred_mask_probs.shape[0], -1).mean(dim=1).float()

        cost = self._get_spatial_transport_cost(spatial_size, pred_mask_probs.device).float()
        kernel = torch.exp(-cost / self.sinkhorn_entropy_reg).clamp_min(self.unbalanced_ot_eps)
        tau = self.unbalanced_ot_rho / (self.unbalanced_ot_rho + self.sinkhorn_entropy_reg)
        out = pred_flat.new_zeros((num_distributions,), dtype=torch.float32)
        offset = 0
        for start in range(0, num_distributions, chunk_size):
            end = min(start + chunk_size, num_distributions)
            a = pred_flat[start:end].clamp_min(self.unbalanced_ot_eps)
            b = target_flat[start:end].clamp_min(self.unbalanced_ot_eps)
            u = torch.ones_like(a)
            v = torch.ones_like(b)
            for _ in range(self.sinkhorn_iters):
                kv = torch.matmul(v, kernel.transpose(0, 1)).clamp_min(self.unbalanced_ot_eps)
                u = (a / kv).pow(tau)
                ktu = torch.matmul(u, kernel).clamp_min(self.unbalanced_ot_eps)
                v = (b / ktu).pow(tau)
            transport_plan = u.unsqueeze(-1) * kernel.unsqueeze(0) * v.unsqueeze(-2)
            transported_pred = transport_plan.sum(dim=2)
            transported_target = transport_plan.sum(dim=1)
            transport_cost = (transport_plan * cost.unsqueeze(0)).sum(dim=(1, 2))
            mass_penalty = self.unbalanced_ot_rho * (
                self._generalized_kl(transported_pred, a, self.unbalanced_ot_eps)
                + self._generalized_kl(transported_target, b, self.unbalanced_ot_eps)
            )
            out[offset : offset + (end - start)] = transport_cost + mass_penalty
            offset += end - start

        loss_per_slot_raw = out.view(*pred_mask_probs.shape[:-1])
        if self.mask_transport_area_normalize:
            # Raw UOT scales with mask mass, so large background slots can
            # dominate candidate ranking. Normalize by target-slot area to
            # compare per-unit-mask transport while preserving UOT's mass
            # creation / deletion penalty when an object disappears. Prefer a
            # per-view max area computed from the frames used to extract fixed
            # obj latents, so a slot that is tiny / absent in one goal does not
            # get an artificially huge scale.
            if area_normalizer is None:
                denom = target_sum.reshape_as(loss_per_slot_raw)
            else:
                denom = area_normalizer.to(loss_per_slot_raw.device, dtype=loss_per_slot_raw.dtype)
                if denom.ndim == 1:
                    denom = denom.view(*([1] * (loss_per_slot_raw.ndim - 1)), denom.shape[0])
                elif denom.ndim == 2 and loss_per_slot_raw.ndim > 2:
                    denom = denom.view(denom.shape[0], *([1] * (loss_per_slot_raw.ndim - 2)), denom.shape[1])
                denom = torch.broadcast_to(denom, loss_per_slot_raw.shape)
            num_slots = float(pred_mask_probs.shape[-2])
            # Convert per-unit-area UOT back to a per-slot comparable scale:
            # downstream aggregation still averages slots, so multiplying by
            # K keeps the final mean close to the summed slot-loss magnitude.
            loss_per_slot = (
                loss_per_slot_raw / denom.clamp_min(self.mask_transport_area_normalize_eps)
            ) * num_slots
        else:
            loss_per_slot = loss_per_slot_raw
        loss = self._weighted_slot_loss_mean(loss_per_slot, slot_weights=slot_weights).float()
        if return_per_slot:
            return loss, effective_mass.reshape(pred_mask_probs.shape[0], -1).mean(dim=1).float(), loss_per_slot.float()
        return loss, effective_mass.reshape(pred_mask_probs.shape[0], -1).mean(dim=1).float()

    def _sinkhorn_mask_transport_loss(
        self,
        pred_mask_probs,
        target_mask_probs,
        spatial_size,
        chunk_size=32,
        slot_weights=None,
        area_normalizer=None,
    ):
        loss, _ = self._unbalanced_sinkhorn_mask_loss(
            pred_mask_probs,
            target_mask_probs,
            spatial_size=spatial_size,
            patch_weights=None,
            chunk_size=chunk_size,
            slot_weights=slot_weights,
            area_normalizer=area_normalizer,
        )
        return loss

    def _sinkhorn_mask_transport_loss_per_slot(
        self,
        pred_mask_probs,
        target_mask_probs,
        spatial_size,
        chunk_size=32,
        area_normalizer=None,
    ):
        _, _, loss_per_slot = self._unbalanced_sinkhorn_mask_loss(
            pred_mask_probs,
            target_mask_probs,
            spatial_size=spatial_size,
            patch_weights=None,
            chunk_size=chunk_size,
            slot_weights=None,
            area_normalizer=area_normalizer,
            return_per_slot=True,
        )
        return loss_per_slot.float()

    def _soft_iou_mask_loss(self, pred_mask_probs, target_mask_probs, min_target_mass=0.0):
        if pred_mask_probs.shape != target_mask_probs.shape:
            raise ValueError(
                f"Expected pred and target mask probs to share shape, got {tuple(pred_mask_probs.shape)} vs {tuple(target_mask_probs.shape)}"
            )
        pred = pred_mask_probs.float().clamp_min(0.0)
        target = target_mask_probs.float().clamp_min(0.0)
        intersection = (pred * target).sum(dim=-1)
        pred_mass = pred.sum(dim=-1)
        target_mass = target.sum(dim=-1)
        union = (pred_mass + target_mass - intersection).clamp_min(1.0e-6)
        loss_per_slot = 1.0 - intersection / union
        valid = target_mass >= float(min_target_mass)
        if bool(valid.any()):
            valid_f = valid.float()
            loss = (loss_per_slot * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp_min(1.0)
            invalid_batch = valid_f.sum(dim=1) < 0.5
            if bool(invalid_batch.any()):
                fallback = loss_per_slot.mean(dim=1)
                loss = torch.where(invalid_batch, fallback, loss)
            return loss.float(), target_mass.mean(dim=1).float()
        return loss_per_slot.mean(dim=1).float(), target_mass.mean(dim=1).float()

    def _soft_iou_mask_loss_masked(self, pred_mask_probs, target_mask_probs, patch_weights):
        patch_weights = patch_weights.to(pred_mask_probs.device, dtype=torch.float32)
        if patch_weights.ndim == 1:
            patch_weights = patch_weights.view(1, 1, -1)
        elif patch_weights.ndim == 2:
            patch_weights = patch_weights.unsqueeze(1)
        else:
            raise ValueError(f"Expected patch_weights [N] or [B,N], got {tuple(patch_weights.shape)}")
        if patch_weights.size(-1) != pred_mask_probs.size(-1):
            raise ValueError(
                f"Shared-visible weight size {patch_weights.size(-1)} does not match mask patches {pred_mask_probs.size(-1)}"
            )
        if patch_weights.size(0) == 1 and pred_mask_probs.size(0) != 1:
            patch_weights = patch_weights.expand(pred_mask_probs.size(0), -1, -1)
        pred_masked = pred_mask_probs.float() * patch_weights.clamp(0.0, 1.0)
        target_masked = target_mask_probs.float() * patch_weights.clamp(0.0, 1.0)
        masked_loss, effective_mass = self._soft_iou_mask_loss(
            pred_masked,
            target_masked,
            min_target_mass=self.shared_visible_min_mass,
        )
        if bool((effective_mass < self.shared_visible_min_mass).all()):
            fallback_loss, fallback_mass = self._soft_iou_mask_loss(pred_mask_probs, target_mask_probs)
            return fallback_loss, fallback_mass
        return masked_loss, effective_mass

    def _shared_visible_weights_from_recon(self, current_recon, goal_recon):
        current_recon = current_recon.detach().float()
        goal_recon = goal_recon.detach().float()
        if current_recon.ndim == 3:
            current_recon = current_recon[0]
        if goal_recon.ndim == 3:
            goal_recon = goal_recon[0]
        if current_recon.ndim != 2 or goal_recon.ndim != 2:
            raise ValueError(
                f"Expected recon features [N,D], got {tuple(current_recon.shape)} and {tuple(goal_recon.shape)}"
            )
        current_norm = F.normalize(current_recon, dim=-1, eps=1.0e-6)
        goal_norm = F.normalize(goal_recon, dim=-1, eps=1.0e-6)
        # Score each goal patch by how confidently it can be matched from the
        # current frame. This estimates shared visibility from reconstructed
        # features, not from object masks.
        goal_best_sim = torch.matmul(goal_norm, current_norm.transpose(0, 1)).amax(dim=1)
        weights = torch.sigmoid(
            (goal_best_sim - self.shared_visible_sim_threshold) / self.shared_visible_temperature
        )
        return weights.clamp(0.0, 1.0).float()

    def _sinkhorn_mask_transport_loss_masked(
        self,
        pred_mask_probs,
        target_mask_probs,
        spatial_size,
        patch_weights,
        chunk_size=32,
        slot_weights=None,
        area_normalizer=None,
    ):
        return self._unbalanced_sinkhorn_mask_loss(
            pred_mask_probs,
            target_mask_probs,
            spatial_size=spatial_size,
            patch_weights=patch_weights,
            chunk_size=chunk_size,
            slot_weights=slot_weights,
            area_normalizer=area_normalizer,
        )

    @staticmethod
    def _pose6_to_transform(pose6):
        pose6 = np.asarray(pose6, dtype=np.float64).reshape(6)
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = Rotation.from_euler("xyz", pose6[3:6], degrees=False).as_matrix()
        tf[:3, 3] = pose6[:3]
        return tf

    @staticmethod
    def _transform_to_pose6(tf):
        pos = tf[:3, 3]
        rpy = Rotation.from_matrix(tf[:3, :3]).as_euler("xyz", degrees=False)
        return np.concatenate([pos, rpy], axis=0).astype(np.float32)

    def _compose_pose6(self, parent_pose6, child_pose6):
        tf = self._pose6_to_transform(parent_pose6) @ self._pose6_to_transform(child_pose6)
        return self._transform_to_pose6(tf)

    def _pose_in_parent_frame(self, child_pose6, parent_pose6):
        tf = np.linalg.inv(self._pose6_to_transform(parent_pose6)) @ self._pose6_to_transform(child_pose6)
        return self._transform_to_pose6(tf)

    def _pose_delta6(self, start_pose6, end_pose6):
        start_pose6 = np.asarray(start_pose6, dtype=np.float32).reshape(6)
        end_pose6 = np.asarray(end_pose6, dtype=np.float32).reshape(6)
        xyz_diff = end_pose6[:3] - start_pose6[:3]
        s_rot = Rotation.from_euler("xyz", start_pose6[3:6], degrees=False)
        e_rot = Rotation.from_euler("xyz", end_pose6[3:6], degrees=False)
        rot_diff = (e_rot * s_rot.inv()).as_euler("xyz", degrees=False).astype(np.float32)
        return np.concatenate([xyz_diff.astype(np.float32), rot_diff], axis=0)

    def _apply_pose_delta6(self, base_pose6, delta_pose6, rotation_delta_frame="parent"):
        base_pose6 = np.asarray(base_pose6, dtype=np.float32).reshape(6)
        delta_pose6 = np.asarray(delta_pose6, dtype=np.float32).reshape(6)
        base_rot = Rotation.from_euler("xyz", base_pose6[3:6], degrees=False)
        delta_rot = Rotation.from_euler("xyz", delta_pose6[3:6], degrees=False)
        if rotation_delta_frame == "ee":
            next_rot = base_rot * delta_rot
        elif rotation_delta_frame == "parent":
            next_rot = delta_rot * base_rot
        else:
            raise ValueError(f"Unsupported rotation_delta_frame={rotation_delta_frame!r}")
        next_xyz = base_pose6[:3] + delta_pose6[:3]
        next_rpy = next_rot.as_euler("xyz", degrees=False).astype(np.float32)
        return np.concatenate([next_xyz.astype(np.float32), next_rpy], axis=0)

    @staticmethod
    def _cosine_similarity(a, b):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-8:
            return None
        return float(np.dot(a, b) / denom)

    def _current_goal_target(self):
        if self.goal_state is None:
            return None
        goal_targets = self.goal_state.get("goal_targets")
        if goal_targets is None or not (0 <= self.current_goal_idx < len(goal_targets)):
            return None
        return goal_targets[self.current_goal_idx]

    def _configured_translation_cap_xyz(self):
        return np.array(
            [
                self.pose_translation_xy_cap,
                self.pose_translation_xy_cap,
                self.pose_translation_z_cap,
            ],
            dtype=np.float32,
        )

    def _configured_translation_std_xyz(self):
        return np.array(
            [
                self.pose_translation_xy_std,
                self.pose_translation_xy_std,
                self.pose_translation_z_std,
            ],
            dtype=np.float32,
        )

    def _reference_view(self, view_names):
        for view_name in self.VIEW_ORDER:
            if view_name in view_names:
                return view_name
        return view_names[0]

    def _goal_pose_prior_task_name(self):
        if self.goal_state is None:
            return None
        success_case_dir = self.goal_state.get("success_case_dir")
        if success_case_dir:
            return os.path.basename(str(success_case_dir).rstrip("/"))
        task_cfg_path = self.goal_state.get("task_cfg_path")
        if task_cfg_path:
            return str(task_cfg_path).replace("\\", "/").rstrip("/").split("/")[-2]
        return None

    def _goal_pose_prior_task_enabled(self):
        task_name = self._goal_pose_prior_task_name()
        if not task_name:
            return False, None
        task_name = str(task_name)
        enabled = (
            task_name.startswith("reach_")
            or task_name.startswith("put_")
            or task_name.startswith("rotate_")
        )
        return bool(enabled), task_name

    def _build_pose_delta_init(self, current_robot_pose6, current_state7, current_goal_target):
        goal_idx = int(self.current_goal_idx)
        task_enabled, task_name = self._goal_pose_prior_task_enabled()
        debug = {
            "active": False,
            "enabled_for_variant": bool(self.use_goal_pose_prior),
            "goal_idx": goal_idx,
            "task_name": task_name,
            "reason": None,
            "current_goal_name": None if current_goal_target is None else current_goal_target.get("name"),
            "current_goal_stage": None if current_goal_target is None else current_goal_target.get("stage"),
            "mean_delta_pose6_robot": None,
        }

        if not self.use_goal_pose_prior:
            debug["reason"] = "disabled_for_non_a2_variant"
            self.last_goal_pose_prior_debug = debug
            return np.zeros(6, dtype=np.float32), 0.0
        if not task_enabled:
            debug["reason"] = "task_not_reach_put_or_rotate"
            self.last_goal_pose_prior_debug = debug
            return np.zeros(6, dtype=np.float32), 0.0
        if current_goal_target is None or current_goal_target.get("ee_state7_robot") is None:
            debug["reason"] = "missing_goal_pose"
            self.last_goal_pose_prior_debug = debug
            return np.zeros(6, dtype=np.float32), 0.0

        goal_state7_robot = np.asarray(current_goal_target["ee_state7_robot"], dtype=np.float32).reshape(-1)
        if goal_state7_robot.shape[0] < 6:
            debug["reason"] = "invalid_goal_pose_shape"
            self.last_goal_pose_prior_debug = debug
            return np.zeros(6, dtype=np.float32), 0.0

        target_pose6_robot = goal_state7_robot[:6].astype(np.float32).copy()
        goal_delta_pose6 = self._pose_delta6(
            np.asarray(current_robot_pose6, dtype=np.float32).reshape(6),
            target_pose6_robot,
        ).astype(np.float32)
        goal_delta_xyz = goal_delta_pose6[:3].astype(np.float32).copy()
        goal_distance_xyz = float(np.linalg.norm(goal_delta_xyz))
        # Match V-JEPA target guidance: enable only while farther than 30 cm,
        # and initialize translation with 20% of the goal displacement.
        far_translation_threshold = 0.3
        common_debug = {
            "target_pose6_robot": target_pose6_robot,
            "current_pose6_robot": np.asarray(current_robot_pose6, dtype=np.float32).reshape(6).copy(),
            "current_state7": np.asarray(current_state7, dtype=np.float32).reshape(-1).copy(),
            "goal_delta_pose6_robot": goal_delta_pose6,
            "goal_distance_xyz": goal_distance_xyz,
            "far_translation_threshold": far_translation_threshold,
            "bias_ratio": 0.2,
        }
        if goal_distance_xyz <= far_translation_threshold:
            debug.update(common_debug)
            debug.update(
                {
                    "reason": "close_enough_use_pure_l1_search",
                    "mean_delta_pose6_robot": np.zeros(6, dtype=np.float32),
                }
            )
            self.last_goal_pose_prior_debug = debug
            return np.zeros(6, dtype=np.float32), 0.0

        translation_cap_xyz = self._configured_translation_cap_xyz()
        mean_delta_pose6 = np.zeros(6, dtype=np.float32)
        mean_delta_pose6[:3] = np.clip(
            0.2 * goal_delta_xyz,
            -0.2 * translation_cap_xyz,
            0.2 * translation_cap_xyz,
        ).astype(np.float32)
        debug.update(common_debug)
        debug.update(
            {
                "active": True,
                "reason": "far_goal_xyz_bias_only",
                "mean_delta_pose6_robot": mean_delta_pose6.copy(),
            }
        )
        self.last_goal_pose_prior_debug = debug
        return mean_delta_pose6, 0.0

    def _build_goal_distance_filter(self, current_robot_pose6, current_goal_target):
        task_enabled, task_name = self._goal_pose_prior_task_enabled()
        debug = {
            "active": False,
            "enabled_for_variant": bool(self.a2_no_uot_mode),
            "task_name": task_name,
            "current_goal_name": None if current_goal_target is None else current_goal_target.get("name"),
            "current_goal_stage": None if current_goal_target is None else current_goal_target.get("stage"),
            "goal_xyz_robot": None,
            "current_distance_xyz": None,
            "far_translation_threshold": 0.3,
            "max_distance_increase_tolerance": 0.005,
            "applied": False,
            "reason": None,
        }
        if not self.a2_no_uot_mode:
            debug["reason"] = "disabled_for_non_a2_variant"
            return debug
        if not task_enabled:
            debug["reason"] = "task_not_reach_put_or_rotate"
            return debug
        goal_stage = str(current_goal_target.get("stage") or "") if current_goal_target is not None else ""
        # The historical V-JEPA filter covered MOVE_CLOSE/PLACE/SUCCESS. A2
        # additionally supports the REACH targets requested for this ablation.
        if goal_stage not in {"REACH", "MOVE_CLOSE", "SUCCESS", "PLACE_INTO", "PLACE_ONTO"}:
            debug["reason"] = "goal_stage_not_reach_move_close_place_or_success"
            return debug
        if current_goal_target is None or current_goal_target.get("ee_state7_robot") is None:
            debug["reason"] = "missing_goal_pose"
            return debug
        goal_state7_robot = np.asarray(current_goal_target["ee_state7_robot"], dtype=np.float32).reshape(-1)
        if goal_state7_robot.shape[0] < 3:
            debug["reason"] = "invalid_goal_pose_shape"
            return debug
        goal_xyz_robot = goal_state7_robot[:3].astype(np.float32).copy()
        current_robot_pose6 = np.asarray(current_robot_pose6, dtype=np.float32).reshape(-1)
        current_distance_xyz = float(np.linalg.norm(goal_xyz_robot - current_robot_pose6[:3]))
        debug["goal_xyz_robot"] = goal_xyz_robot
        debug["current_distance_xyz"] = current_distance_xyz
        if current_distance_xyz <= float(debug["far_translation_threshold"]):
            debug["reason"] = "close_enough_use_pure_l1_search"
            return debug
        debug["active"] = True
        debug["reason"] = "filter_nonimproving_goal_distance"
        return debug

    @staticmethod
    def _apply_goal_distance_filter(losses, target_robot_pose6_all, goal_distance_filter):
        filtered_losses = np.asarray(losses, dtype=np.float32).copy()
        debug = dict(goal_distance_filter or {})
        if not debug.get("active", False):
            return filtered_losses, debug
        if target_robot_pose6_all is None:
            debug["reason"] = "missing_target_robot_pose6_all"
            return filtered_losses, debug
        target_robot_pose6_all = np.asarray(target_robot_pose6_all, dtype=np.float32).reshape(-1, 6)
        goal_xyz_robot = np.asarray(debug.get("goal_xyz_robot"), dtype=np.float32).reshape(3)
        candidate_distances_xyz = np.linalg.norm(
            target_robot_pose6_all[:, :3] - goal_xyz_robot[None, :], axis=1
        )
        current_distance_xyz = float(debug.get("current_distance_xyz", float("inf")))
        tolerance = float(debug.get("max_distance_increase_tolerance", 0.005))
        valid_mask = np.isfinite(candidate_distances_xyz) & (
            candidate_distances_xyz <= current_distance_xyz + tolerance
        )
        debug["num_candidates"] = int(target_robot_pose6_all.shape[0])
        debug["num_valid_candidates"] = int(valid_mask.sum())
        debug["best_candidate_distance_xyz"] = (
            None if candidate_distances_xyz.size == 0 else float(np.min(candidate_distances_xyz))
        )
        if bool(np.any(valid_mask)):
            filtered_losses[~valid_mask] = np.inf
            debug["applied"] = True
        else:
            debug["reason"] = "no_candidate_nonincreasing_distance"
        return filtered_losses, debug

    def _add_pose_memory_entry(
        self,
        pose6_robot,
        gripper_state,
        per_view_losses,
        source,
        stage=None,
        step=None,
        extra=None,
    ):
        pose6_robot = np.asarray(pose6_robot, dtype=np.float32).reshape(6)
        gripper_state = float(gripper_state)
        per_view_losses = {
            str(view_name): float(loss)
            for view_name, loss in (per_view_losses or {}).items()
            if loss is not None and np.isfinite(float(loss))
        }
        if not per_view_losses:
            return
        for view_name, loss in per_view_losses.items():
            if view_name not in self.pose_memory_by_view:
                self.pose_memory_by_view[view_name] = []
            entry = {
                "pose6_robot": pose6_robot.astype(np.float32).copy(),
                "gripper_state": float(gripper_state),
                "per_view_losses": dict(per_view_losses),
                "view_loss": float(loss),
                "source": str(source),
                "stage": None if stage is None else str(stage),
                "step": None if step is None else int(step),
                "extra": dict(extra or {}),
            }
            memory = self.pose_memory_by_view[view_name]
            memory.append(entry)
            memory.sort(key=lambda item: float(item["view_loss"]))
            del memory[self.pose_memory_topk :]

    def _pose_memory_summary(self):
        return {
            view_name: [
                {
                    "loss": float(entry["view_loss"]),
                    "source": entry.get("source"),
                    "stage": entry.get("stage"),
                    "step": entry.get("step"),
                    "pose6_robot": np.asarray(entry["pose6_robot"], dtype=np.float32).tolist(),
                    "gripper_state": float(entry["gripper_state"]),
                }
                for entry in self.pose_memory_by_view.get(view_name, [])
            ]
            for view_name in self.VIEW_ORDER
        }

    def _update_warmup_fixed_anchor(self, goal_idx, pose6_robot, gripper_state, per_view_losses, step=None):
        fixed_losses = [
            float(loss)
            for view_name, loss in (per_view_losses or {}).items()
            if "wrist" not in str(view_name).lower() and loss is not None and np.isfinite(float(loss))
        ]
        if not fixed_losses:
            return
        goal_idx = int(goal_idx)
        fixed_loss = float(np.mean(fixed_losses))
        prev = self.warmup_fixed_anchor_by_goal.get(goal_idx)
        if prev is not None and fixed_loss >= float(prev["fixed_loss"]):
            return
        self.warmup_fixed_anchor_by_goal[goal_idx] = {
            "goal_idx": int(goal_idx),
            "fixed_loss": float(fixed_loss),
            "pose6_robot": np.asarray(pose6_robot, dtype=np.float32).reshape(6).copy(),
            "gripper_state": float(gripper_state),
            "per_view_losses": {
                str(view_name): float(loss)
                for view_name, loss in (per_view_losses or {}).items()
                if loss is not None and np.isfinite(float(loss))
            },
            "step": None if step is None else int(step),
            "source": "warmup_fixed_view_best",
        }

    def _get_warmup_fixed_anchor_steering(
        self,
        goal_idx,
        current_fixed_view_loss,
        current_robot_pose6,
        current_gripper_state,
    ):
        del current_gripper_state
        if self.a2_no_uot_mode:
            return {
                "active": False,
                "goal_idx": int(goal_idx),
                "reason": "disabled_a2_uses_goal_pose_init_only",
            }
        anchor = self.warmup_fixed_anchor_by_goal.get(int(goal_idx))
        if anchor is None:
            return {
                "active": False,
                "goal_idx": int(goal_idx),
                "reason": "missing_warmup_fixed_anchor",
            }
        anchor_loss = float(anchor["fixed_loss"])
        current_fixed_view_loss = float(current_fixed_view_loss)
        active = bool(np.isfinite(current_fixed_view_loss) and current_fixed_view_loss > anchor_loss)
        delta6 = self._pose_delta6(
            np.asarray(current_robot_pose6, dtype=np.float32).reshape(6),
            np.asarray(anchor["pose6_robot"], dtype=np.float32).reshape(6),
        ).astype(np.float32)
        return {
            "active": active,
            "goal_idx": int(goal_idx),
            "reason": "current_fixed_loss_above_anchor" if active else "current_fixed_loss_reached_anchor",
            "anchor_fixed_loss": float(anchor_loss),
            "current_fixed_view_loss": float(current_fixed_view_loss),
            "pose6_robot": np.asarray(anchor["pose6_robot"], dtype=np.float32).tolist(),
            "gripper_state": float(anchor["gripper_state"]),
            "per_view_losses": dict(anchor.get("per_view_losses", {})),
            "step": anchor.get("step"),
            "source": anchor.get("source", "warmup_fixed_view_best"),
            "delta6_robot": delta6.tolist(),
            "delta_translation_norm": float(np.linalg.norm(delta6[:3])),
            "delta_rotation_norm": float(np.linalg.norm(delta6[3:6])),
            "stop_condition": "current_fixed_view_loss <= anchor_fixed_loss",
        }

    def _update_pose_view_switch(
        self,
        goal_idx,
        current_loss_by_view,
        fixed_view_names,
        wrist_view_names,
        patience=5,
        update=True,
        step_idx=None,
    ):
        goal_idx = int(goal_idx)
        patience = max(1, int(patience))
        update = bool(update)
        step_idx = int(self.predict_step_idx if step_idx is None else step_idx)
        fixed_view_names = [view_name for view_name in fixed_view_names if view_name in current_loss_by_view]
        wrist_view_names = [view_name for view_name in wrist_view_names if view_name in current_loss_by_view]
        group_views = {}
        if fixed_view_names:
            group_views["fixed"] = list(fixed_view_names)
        if wrist_view_names:
            group_views["wrist"] = list(wrist_view_names)
        best_losses = self.pose_view_best_losses_by_goal.setdefault(goal_idx, {})
        no_improve_steps_by_group = self.pose_view_no_improve_steps_by_goal.setdefault(goal_idx, {})
        current_group_losses = {}
        for group_name, names in group_views.items():
            values = [
                float(current_loss_by_view[name])
                for name in names
                if name in current_loss_by_view and np.isfinite(float(current_loss_by_view[name]))
            ]
            if not values:
                continue
            value = float(np.mean(values))
            current_group_losses[group_name] = value

        fixed_groups = [group_name for group_name in group_views if group_name == "fixed"]
        wrist_group = "wrist" if "wrist" in group_views else None
        active_group = self.pose_view_active_group_by_goal.get(goal_idx)
        active_group_initialized = False
        if active_group not in group_views:
            active_group = fixed_groups[0] if fixed_groups else wrist_group
            active_group_initialized = True

        loss_window = self.pose_view_loss_history_by_goal.get(goal_idx)
        if not isinstance(loss_window, deque):
            loss_window = deque(maxlen=patience)
        if active_group_initialized:
            loss_window.clear()

        window_reset_on_switch = False
        improved_groups = []
        if update and active_group is not None and current_group_losses:
            loss_window.append(
                {
                    "step_idx": int(step_idx),
                    "active_group": active_group,
                    "group_losses": {
                        group_name: float(loss)
                        for group_name, loss in current_group_losses.items()
                    },
                }
            )

        window_records = list(loss_window)
        window_group_values = {}
        for group_name in group_views:
            values = [
                float(record["group_losses"][group_name])
                for record in window_records
                if group_name in record.get("group_losses", {})
                and np.isfinite(float(record["group_losses"][group_name]))
            ]
            window_group_values[group_name] = values
            if values:
                best_losses[group_name] = float(np.min(values))
                if len(values) == 1 or values[-1] <= float(np.min(values[:-1])) - 1.0e-8:
                    improved_groups.append(group_name)
            else:
                best_losses.pop(group_name, None)

        active_values = window_group_values.get(active_group, []) if active_group is not None else []
        active_no_improve_count = 0
        active_stale = False
        if active_values:
            best_index = int(np.argmin(active_values))
            active_no_improve_count = int(len(active_values) - 1 - best_index)
            active_stale = len(active_values) >= patience and best_index == 0
        for group_name in group_views:
            if group_name == active_group:
                no_improve_steps_by_group[group_name] = int(active_no_improve_count)
            else:
                no_improve_steps_by_group[group_name] = 0

        fixed_groups_with_loss = [group_name for group_name in fixed_groups if group_name in current_group_losses]
        all_fixed_stale = bool(active_group == "fixed" and active_stale)
        switch_reason = None
        prev_active_group = active_group
        rising_other_groups = []
        active_no_improve_switch = None
        if update and active_group is not None:
            for group_name, values in window_group_values.items():
                if group_name == active_group or group_name not in current_group_losses:
                    continue
                if len(values) < patience:
                    continue
                first = float(values[0])
                last = float(values[-1])
                midpoint = max(1, len(values) // 2)
                early_mean = float(np.mean(values[:midpoint]))
                late_mean = float(np.mean(values[midpoint:]))
                increase = last - first
                if increase > 1.0e-6 and late_mean > early_mean + 1.0e-6:
                    denom = max(abs(first), 1.0e-6)
                    rising_other_groups.append(
                        {
                            "group": group_name,
                            "first": first,
                            "last": last,
                            "increase": float(increase),
                            "relative_increase": float(increase / denom),
                            "early_mean": early_mean,
                            "late_mean": late_mean,
                            "history": values,
                        }
                    )
            if active_stale:
                candidates = [group_name for group_name in current_group_losses if group_name != active_group]
                if candidates:
                    active_no_improve_switch = {
                        "from_group": active_group,
                        "to_group": min(candidates, key=lambda name: current_group_losses.get(name, float("inf"))),
                        "no_improve_steps": int(no_improve_steps_by_group.get(active_group, 0)),
                        "current_loss": float(current_group_losses.get(active_group, float("nan"))),
                    }
        if not update:
            switch_reason = "pose_view_switch_update_skipped_non_pose_request"
        elif active_no_improve_switch is not None:
            active_group = active_no_improve_switch["to_group"]
            switch_reason = "active_view_no_new_min_for_patience_steps"
        elif rising_other_groups:
            rising_other_groups = sorted(
                rising_other_groups,
                key=lambda item: (item["relative_increase"], item["increase"]),
                reverse=True,
            )
            active_group = rising_other_groups[0]["group"]
            switch_reason = "other_view_observed_loss_rising_for_patience_steps"

        switched_group = active_group is not None and active_group != prev_active_group
        if switched_group:
            # View-switch decisions use one common window that starts at the
            # moment the current view group becomes active. Old fixed/wrist
            # histories must not immediately switch the controller back.
            loss_window = deque(maxlen=patience)
            window_records = []
            window_group_values = {group_name: [] for group_name in group_views}
            best_losses.clear()
            no_improve_steps_by_group.clear()
            for group_name in group_views:
                no_improve_steps_by_group[group_name] = 0
            window_reset_on_switch = True

        if update and active_group is not None:
            self.pose_view_active_group_by_goal[goal_idx] = active_group
            self.pose_view_loss_history_by_goal[goal_idx] = loss_window
        selected_view_names = list(group_views.get(active_group, [])) if active_group is not None else []
        use_wrist_pose = active_group == wrist_group and wrist_group is not None
        return {
            "goal_idx": int(goal_idx),
            "patience": int(patience),
            "updated_from_current_request": bool(update),
            "step_idx": int(step_idx),
            "sticky_until_goal_advance": False,
            "active_group": active_group,
            "selected_view_names": selected_view_names,
            "current_group_losses": dict(current_group_losses),
            "best_group_losses": {group_name: float(loss) for group_name, loss in best_losses.items()},
            "no_improve_steps_by_group": {
                group_name: int(no_improve_steps_by_group.get(group_name, 0))
                for group_name in group_views
            },
            "observed_loss_history_by_group": {
                group_name: [float(x) for x in window_group_values.get(group_name, [])]
                for group_name in group_views
            },
            "observed_loss_window_since_switch": window_records,
            "observed_loss_window_len": int(len(window_records)),
            "window_reset_on_switch": bool(window_reset_on_switch),
            "rising_other_groups": rising_other_groups,
            "active_no_improve_switch": active_no_improve_switch,
            "improved_groups": list(improved_groups),
            "fixed_groups": list(fixed_groups),
            "wrist_group": wrist_group,
            "all_fixed_groups_stale": bool(all_fixed_stale),
            "active_group_stale": bool(active_stale),
            "switch_reason": switch_reason,
            "switched_group": bool(switched_group),
            "previous_active_group": prev_active_group,
            "new_active_group": active_group,
            "use_wrist_pose_search": bool(use_wrist_pose),
            "switch_condition": (
                "switch when the active view group has no new minimum for 5 pose steps, "
                "or when the other view group's observed loss rises over the same recent "
                "window; reset that common window whenever the active view group switches"
            ),
        }

    def _reset_goal_control_loss_tracking(self, goal_idx):
        goal_idx = int(goal_idx)
        self.fixed_loss_search_reference_by_goal.pop(goal_idx, None)
        self.warmup_fixed_anchor_by_goal.pop(goal_idx, None)
        self.pose_view_best_losses_by_goal.pop(goal_idx, None)
        self.pose_view_no_improve_steps_by_goal.pop(goal_idx, None)
        self.pose_view_active_group_by_goal.pop(goal_idx, None)
        self.pose_view_loss_history_by_goal.pop(goal_idx, None)

    def _copy_parsed_obs_sequence(self, obs_sequence):
        copied = []
        for obs_views in obs_sequence:
            copied_views = {}
            for view_name, view_payload in obs_views.items():
                copied_views[view_name] = {
                    "image": np.asarray(view_payload["image"], dtype=np.uint8).copy(),
                    "state7": np.asarray(view_payload["state7"], dtype=np.float32).copy(),
                    "extrinsics6": np.asarray(view_payload["extrinsics6"], dtype=np.float32).copy(),
                    "camera_frame_convention": view_payload.get("camera_frame_convention"),
                }
            copied.append(copied_views)
        return copied

    @torch.no_grad()
    def _refresh_warmup_memory_for_goal(self, goal_idx):
        goal_idx = int(goal_idx)
        self.pose_memory_by_view = {view_name: [] for view_name in self.VIEW_ORDER}
        self._reset_goal_control_loss_tracking(goal_idx)
        if not self.warmup_score_obs_sequence:
            return {
                "status": "skipped",
                "reason": "no_warmup_score_obs_sequence",
                "goal_idx": int(goal_idx),
            }
        obs_sequence = self._copy_parsed_obs_sequence(self.warmup_score_obs_sequence)
        available_views = self._filter_available_views(obs_sequence[-1])
        result = self._score_parsed_observations_to_goal(
            obs_sequence=obs_sequence,
            available_views=available_views,
            goal_idx=goal_idx,
            update_memory=True,
            update_warmup_anchor=False,
            memory_source="warmup_observed_goal_score_after_goal_advance",
        )
        result["refreshed_from_warmup_sequence"] = True
        return result

    def _select_steering_direction(self, phase_view_names, phase_dims, current_robot_pose6):
        phase_view_names = [view_name for view_name in phase_view_names if view_name in self.pose_memory_by_view]
        if not phase_view_names:
            return None
        phase_dims = np.asarray(phase_dims, dtype=np.int64).reshape(-1)
        phase_dims = phase_dims[(phase_dims >= 0) & (phase_dims < 6)]
        if phase_dims.size <= 0:
            return None
        proposals = []
        seen = set()
        for view_name in phase_view_names:
            for entry in self.pose_memory_by_view.get(view_name, []):
                losses = entry.get("per_view_losses", {})
                if any(view_name not in losses for view_name in phase_view_names):
                    continue
                pose_key = tuple(np.round(np.asarray(entry["pose6_robot"], dtype=np.float32), 5).tolist())
                grip_key = round(float(entry["gripper_state"]), 5)
                key = (pose_key, grip_key, tuple(phase_view_names))
                if key in seen:
                    continue
                seen.add(key)
                proposals.append(entry)
        if not proposals:
            return None

        norm_losses_by_view = {}
        view_weights = {}
        for view_name in phase_view_names:
            values = np.asarray([entry["per_view_losses"][view_name] for entry in proposals], dtype=np.float32)
            if values.size < 2:
                lo = float(values.min())
                hi = float(values.max())
            else:
                lo = float(np.percentile(values, 10))
                hi = float(np.percentile(values, 90))
            span = hi - lo
            if span < self.steering_loss_range_eps:
                view_weights[view_name] = 0.0
                norm_losses_by_view[view_name] = np.zeros(values.shape, dtype=np.float32)
            else:
                view_weights[view_name] = 1.0
                norm_losses_by_view[view_name] = np.clip((values - lo) / (span + 1.0e-8), 0.0, 2.0).astype(np.float32)
        weight_sum = float(sum(view_weights.values()))
        if weight_sum <= 0.0:
            return None
        scores = np.zeros((len(proposals),), dtype=np.float32)
        for view_name in phase_view_names:
            scores += float(view_weights[view_name]) * norm_losses_by_view[view_name]
        scores /= weight_sum

        current_robot_pose6 = np.asarray(current_robot_pose6, dtype=np.float32).reshape(6)
        proposal_delta6 = np.stack(
            [
                self._pose_delta6(current_robot_pose6, np.asarray(entry["pose6_robot"], dtype=np.float32))
                for entry in proposals
            ],
            axis=0,
        ).astype(np.float32)
        steering_direction6 = np.zeros(6, dtype=np.float32)
        dim_stats = {}
        delta_eps = 1.0e-5
        for dim in phase_dims.tolist():
            dim = int(dim)
            dim_deltas = proposal_delta6[:, dim]
            pos_mask = dim_deltas > delta_eps
            neg_mask = dim_deltas < -delta_eps
            pos_count = int(np.count_nonzero(pos_mask))
            neg_count = int(np.count_nonzero(neg_mask))
            pos_mean = float(np.mean(scores[pos_mask])) if pos_count > 0 else None
            neg_mean = float(np.mean(scores[neg_mask])) if neg_count > 0 else None
            selected_direction = 0.0
            selected_side = "none"
            reason = "insufficient_two_sided_samples"
            if pos_mean is not None and neg_mean is not None:
                if pos_mean + self.steering_loss_range_eps < neg_mean:
                    selected_direction = 1.0
                    selected_side = "positive"
                    reason = "positive_side_lower_loss"
                elif neg_mean + self.steering_loss_range_eps < pos_mean:
                    selected_direction = -1.0
                    selected_side = "negative"
                    reason = "negative_side_lower_loss"
                else:
                    reason = "two_sided_loss_too_close"
            steering_direction6[dim] = float(selected_direction)
            dim_stats[str(dim)] = {
                "positive_count": pos_count,
                "negative_count": neg_count,
                "positive_mean_score": pos_mean,
                "negative_mean_score": neg_mean,
                "selected_direction": float(selected_direction),
                "selected_side": selected_side,
                "reason": reason,
            }
        active_norm = float(np.linalg.norm(steering_direction6[phase_dims]))
        if active_norm < 1.0e-8:
            return None
        best_idx = int(np.argmin(scores))
        best_entry = proposals[best_idx]
        return {
            "entry": best_entry,
            "score": float(scores[best_idx]),
            "phase_view_names": list(phase_view_names),
            "steering_direction6": steering_direction6,
            "active_norm": float(active_norm),
            "num_proposals": int(len(proposals)),
            "view_weights": {view_name: float(weight) for view_name, weight in view_weights.items()},
            "raw_losses": {
                view_name: float(best_entry["per_view_losses"][view_name])
                for view_name in phase_view_names
            },
            "dim_stats": dim_stats,
        }

    def _get_action_limits(self):
        translation_cap_xyz = self._configured_translation_cap_xyz()
        translation_std_xyz = self._configured_translation_std_xyz()
        return {
            "translation_cap": float(np.max(translation_cap_xyz)),
            "translation_cap_xyz": translation_cap_xyz.astype(np.float32),
            "rotation_cap": float(self.pose_rotation_cap),
            "translation_std": float(np.max(translation_std_xyz)),
            "translation_std_xyz": translation_std_xyz.astype(np.float32),
            "rotation_std": float(self.pose_rotation_std),
            "translation_scale": 1.0,
            "rotation_scale": 1.0,
        }

    def _fixed_loss_to_search_scale(self, fixed_loss, goal_idx):
        fixed_loss = float(fixed_loss)
        goal_idx = int(goal_idx)
        ref_loss = self.fixed_loss_search_reference_by_goal.get(goal_idx)
        if ref_loss is None or not np.isfinite(float(ref_loss)) or float(ref_loss) <= 1.0e-8:
            ref_loss = max(fixed_loss, 1.0e-8)
            self.fixed_loss_search_reference_by_goal[goal_idx] = float(ref_loss)
        ratio = float(np.clip(fixed_loss / max(float(ref_loss), 1.0e-8), 0.0, 1.0))
        scale = self.fixed_loss_search_min_scale + (1.0 - self.fixed_loss_search_min_scale) * ratio
        return {
            "scale": float(np.clip(scale, self.fixed_loss_search_min_scale, 1.0)),
            "reference_fixed_loss": float(ref_loss),
            "input_fixed_loss": float(fixed_loss),
            "loss_ratio": float(ratio),
            "min_scale": float(self.fixed_loss_search_min_scale),
        }

    @staticmethod
    def _cap_pose_delta_norm(delta_pose6, translation_cap, rotation_cap):
        delta_pose6 = np.asarray(delta_pose6, dtype=np.float32).copy()
        translation_cap_arr = np.asarray(translation_cap, dtype=np.float32).reshape(-1)
        if translation_cap_arr.size == 3:
            delta_pose6[..., :3] = np.clip(
                delta_pose6[..., :3],
                -translation_cap_arr.reshape((1,) * (delta_pose6[..., :3].ndim - 1) + (3,)),
                translation_cap_arr.reshape((1,) * (delta_pose6[..., :3].ndim - 1) + (3,)),
            )
        else:
            trans_norm = np.linalg.norm(delta_pose6[..., :3], axis=-1, keepdims=True)
            trans_scale = np.minimum(1.0, float(translation_cap_arr[0]) / np.maximum(trans_norm, 1.0e-8))
            delta_pose6[..., :3] *= trans_scale
        rot_norm = np.linalg.norm(delta_pose6[..., 3:6], axis=-1, keepdims=True)
        rot_scale = np.minimum(1.0, float(rotation_cap) / np.maximum(rot_norm, 1.0e-8))
        delta_pose6[..., 3:6] *= rot_scale
        return delta_pose6.astype(np.float32)

    @staticmethod
    def _cap_pose_delta_total_norm(delta_pose6, translation_norm_cap, rotation_norm_cap):
        delta_pose6 = np.asarray(delta_pose6, dtype=np.float32).copy()
        translation_norm_cap = float(translation_norm_cap)
        rotation_norm_cap = float(rotation_norm_cap)
        if translation_norm_cap > 0.0:
            trans_norm = np.linalg.norm(delta_pose6[..., :3], axis=-1, keepdims=True)
            trans_scale = np.minimum(1.0, translation_norm_cap / np.maximum(trans_norm, 1.0e-8))
            delta_pose6[..., :3] *= trans_scale
        if rotation_norm_cap > 0.0:
            rot_norm = np.linalg.norm(delta_pose6[..., 3:6], axis=-1, keepdims=True)
            rot_scale = np.minimum(1.0, rotation_norm_cap / np.maximum(rot_norm, 1.0e-8))
            delta_pose6[..., 3:6] *= rot_scale
        return delta_pose6.astype(np.float32)

    def _train_aligned_center_crop(self, tensor):
        _, _, h, w = tensor.shape
        in_ratio = float(w) / float(h)
        min_ratio, max_ratio = 0.75, 1.35
        if in_ratio < min_ratio:
            crop_w = w
            crop_h = int(round(crop_w / min_ratio))
        elif in_ratio > max_ratio:
            crop_h = h
            crop_w = int(round(crop_h * max_ratio))
        else:
            crop_h = h
            crop_w = w
        top = max((h - crop_h) // 2, 0)
        left = max((w - crop_w) // 2, 0)
        return tensor[:, :, top : top + crop_h, left : left + crop_w]

    def _normalize_tensor(self, tensor):
        return (tensor - self._mean.to(tensor.device)) / self._std.to(tensor.device)

    def _to_uint8_rgb(self, image):
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {image.shape}")
        return image

    def _ensure_state7(self, robot_state, gripper_state=None):
        state = np.asarray(robot_state, dtype=np.float32).reshape(-1)
        if state.shape[0] == 7:
            return state
        if state.shape[0] == 6:
            if gripper_state is None:
                raise ValueError("6D robot_state provided without gripper_state; expected ee_state7.")
            grip = np.atleast_1d(np.array(gripper_state, dtype=np.float32)).reshape(-1)
            return np.concatenate([state, grip[:1]], axis=0).astype(np.float32)
        raise ValueError(f"Expected robot_state with 6 or 7 dims, got shape {state.shape}")

    def _ensure_extrinsics6(self, camera_extrinsics):
        extrinsics = np.asarray(camera_extrinsics, dtype=np.float32).reshape(-1)
        if extrinsics.shape[0] != 6:
            raise ValueError(f"Expected camera_extrinsics with 6 dims, got shape {extrinsics.shape}")
        return extrinsics

    def _preprocess_image_tensor(self, rgb_image):
        rgb_image = self._to_uint8_rgb(rgb_image)
        tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        tensor = self._train_aligned_center_crop(tensor.unsqueeze(0)).squeeze(0)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(self.crop_size, self.crop_size),
            mode="bilinear",
            align_corners=False,
        )
        return self._normalize_tensor(tensor)

    def _poses_to_diffs(self, poses):
        poses = np.asarray(poses, dtype=np.float32)
        if len(poses) < 2:
            return np.zeros((0, 7), dtype=np.float32)
        xyz = poses[:, :3]
        thetas = poses[:, 3:6]
        matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
        xyz_diff = xyz[1:] - xyz[:-1]
        angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
        angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
        angle_diff = np.stack(angle_diff, axis=0).astype(np.float32)
        gripper = poses[:, -1:]
        gripper_diff = gripper[1:] - gripper[:-1]
        return np.concatenate([xyz_diff, angle_diff, gripper_diff], axis=1).astype(np.float32)

    def _transform_action_between_extrinsics(self, action_seq, src_ext6, dst_ext6):
        action_seq_np = action_seq.detach().float().cpu().numpy()
        src_rot = Rotation.from_euler("xyz", np.asarray(src_ext6, dtype=np.float32)[3:6], degrees=False)
        dst_rot = Rotation.from_euler("xyz", np.asarray(dst_ext6, dtype=np.float32)[3:6], degrees=False)
        transformed = []
        for action in action_seq_np.reshape(-1, 7):
            delta_xyz_world = src_rot.apply(action[:3])
            delta_xyz_dst = dst_rot.inv().apply(delta_xyz_world)
            delta_rot_src = Rotation.from_euler("xyz", action[3:6], degrees=False)
            delta_rot_world = src_rot * delta_rot_src * src_rot.inv()
            delta_rot_dst = dst_rot.inv() * delta_rot_world * dst_rot
            delta_rpy_dst = delta_rot_dst.as_euler("xyz", degrees=False).astype(np.float32)
            transformed.append(
                np.concatenate([delta_xyz_dst.astype(np.float32), delta_rpy_dst, action[6:7].astype(np.float32)], axis=0)
            )
        transformed = np.stack(transformed, axis=0).reshape(action_seq_np.shape)
        return torch.as_tensor(transformed, dtype=action_seq.dtype, device=action_seq.device)

    def _parse_observation_views(self, data):
        default_camera_frame_convention = data.get("camera_frame_convention", None)
        if "views" in data:
            payload = data["views"]
            parsed = {}
            for view_name, view_payload in payload.items():
                parsed[view_name] = {
                    "image": self._to_uint8_rgb(view_payload["image"]),
                    "state7": self._ensure_state7(
                        view_payload.get("cartesian_position", view_payload["robot_state"]),
                        data.get("gripper_state"),
                    ),
                    "extrinsics6": self._ensure_extrinsics6(view_payload["camera_extrinsics"]),
                    "camera_frame_convention": view_payload.get(
                        "camera_frame_convention",
                        default_camera_frame_convention,
                    ),
                }
            return parsed
        base_im = self._to_uint8_rgb(data["base_im"])
        state7 = self._ensure_state7(data.get("cartesian_position", data["robot_state"]), data.get("gripper_state"))
        extrinsics6 = self._ensure_extrinsics6(data["camera_extrinsics"])
        parsed = {"exterior_image_1_left": {"image": base_im, "state7": state7, "extrinsics6": extrinsics6}}
        if data.get("base_im_second") is not None:
            parsed["exterior_image_2_left"] = {
                "image": self._to_uint8_rgb(data["base_im_second"]),
                "state7": state7.copy(),
                "extrinsics6": extrinsics6.copy(),
                "camera_frame_convention": default_camera_frame_convention,
            }
        if data.get("wrist_im") is not None:
            parsed["wrist_image_left"] = {
                "image": self._to_uint8_rgb(data["wrist_im"]),
                "state7": state7.copy(),
                "extrinsics6": extrinsics6.copy(),
                "camera_frame_convention": default_camera_frame_convention,
            }
        parsed["exterior_image_1_left"]["camera_frame_convention"] = default_camera_frame_convention
        return parsed

    def _filter_available_views(self, obs_views):
        available_views = [view_name for view_name in self.VIEW_ORDER if view_name in obs_views]
        if self.goal_view_names:
            available_views = [view_name for view_name in available_views if view_name in self.goal_view_names]
        if not available_views:
            raise ValueError("No overlapping observation views are available for the current goal.")
        return available_views

    def _append_obs_views_to_buffers(self, obs_views, available_views):
        for view_name in available_views:
            current = obs_views[view_name]
            current_state = torch.as_tensor(current["state7"], dtype=torch.float32, device=self.device).view(1, 7)
            current_extrinsics = torch.as_tensor(current["extrinsics6"], dtype=torch.float32, device=self.device).view(1, 6)
            current_frame = self._preprocess_image_tensor(current["image"])
            self.frame_buffer[view_name].append(current_frame)
            self.state_buffer[view_name].append(current_state)
            self.extrinsics_buffer[view_name].append(current_extrinsics)
            if view_name in self.fixed_obj_latents:
                self.slot_context_buffer[view_name].append(
                    self._decode_single_frame_context(current_frame, self.fixed_obj_latents[view_name])
                )
            self.history_goal_loss_buffer[view_name].append(None)

    def _append_obs_sequence_to_buffers(self, obs_views_sequence, available_views):
        if not obs_views_sequence:
            raise ValueError("Expected at least one observation in pre_obs_sequence.")
        for obs_views in obs_views_sequence:
            missing = [view_name for view_name in available_views if view_name not in obs_views]
            if missing:
                raise ValueError(f"History observation is missing views: {missing}")
            for view_name in available_views:
                current = obs_views[view_name]
                current_state = torch.as_tensor(current["state7"], dtype=torch.float32, device=self.device).view(1, 7)
                current_extrinsics = torch.as_tensor(current["extrinsics6"], dtype=torch.float32, device=self.device).view(1, 6)
                current_frame = self._preprocess_image_tensor(current["image"])
                self.frame_buffer[view_name].append(current_frame)
                self.state_buffer[view_name].append(current_state)
                self.extrinsics_buffer[view_name].append(current_extrinsics)
                if view_name in self.fixed_obj_latents:
                    self.slot_context_buffer[view_name].append(
                        self._decode_single_frame_context(current_frame, self.fixed_obj_latents[view_name])
                    )
                self.history_goal_loss_buffer[view_name].append(None)

    def _clear_temporal_buffers(self):
        for view_name in self.VIEW_ORDER:
            self.frame_buffer[view_name].clear()
            self.state_buffer[view_name].clear()
            self.extrinsics_buffer[view_name].clear()
            self.slot_context_buffer[view_name].clear()
            self.history_goal_loss_buffer[view_name].clear()

    def _clear_gripper_history_bank(self):
        self.gripper_history_bank_by_view = {view_name: [] for view_name in self.VIEW_ORDER}

    def _make_history_item_from_obs_view(self, view_name, obs_view, original_index, source=None, goal_loss=None):
        current_state = torch.as_tensor(obs_view["state7"], dtype=torch.float32, device=self.device).view(1, 7)
        current_extrinsics = torch.as_tensor(obs_view["extrinsics6"], dtype=torch.float32, device=self.device).view(1, 6)
        current_frame = self._preprocess_image_tensor(obs_view["image"])
        context = None
        if view_name in self.fixed_obj_latents:
            context = self._decode_single_frame_context(current_frame, self.fixed_obj_latents[view_name])
        state_np = current_state.detach().float().cpu().numpy().reshape(-1)
        return {
            "frame": current_frame,
            "state": current_state,
            "extrinsics": current_extrinsics,
            "context": context,
            "goal_loss": goal_loss,
            "robot_pose6": self._state_extrinsics_to_robot_pose6(current_state, current_extrinsics),
            "gripper_state": None if state_np.shape[0] < 7 else float(state_np[6]),
            "original_index": int(original_index),
            "source": source,
        }

    def _append_obs_views_to_gripper_history_bank(self, obs_views, available_views, source="gripper_calibration"):
        # This bank is intentionally separate from pose history and is only
        # appended by explicit calibration records. Normal planning never evicts
        # or updates it, so gripper search keeps a stable open/closed reference set.
        bank_sizes = {}
        for view_name in available_views:
            current = obs_views[view_name]
            bank = self.gripper_history_bank_by_view.setdefault(view_name, [])
            original_index = 1_000_000 + len(bank)
            bank.append(
                self._make_history_item_from_obs_view(
                    view_name,
                    current,
                    original_index=original_index,
                    source=source,
                )
            )
            bank_sizes[view_name] = int(len(bank))
        return bank_sizes

    @torch.no_grad()
    def _context_goal_objective_loss(self, view_name, context_entry, goal_idx):
        if self.goal_features is None or self.goal_mask_probs is None:
            return float("inf")
        if view_name not in self.goal_view_names:
            return float("inf")
        goal_idx = int(np.clip(goal_idx, 0, int(self.goal_features.shape[0] - 1)))
        view_idx = self.goal_view_names.index(view_name)
        mask_probs = context_entry["decoded_mask_probs"].squeeze(0).to(self.device, dtype=torch.float32)
        goal_mask = self.goal_mask_probs[goal_idx, view_idx].to(self.device, dtype=torch.float32)
        mask_l1_loss = torch.mean(torch.abs(mask_probs - goal_mask))
        if self.transport_loss_mix > 0.0:
            slot_weights = self._slot_change_weights(mask_probs.unsqueeze(0), goal_mask.unsqueeze(0))
            transport_loss = self._sinkhorn_mask_transport_loss(
                mask_probs.unsqueeze(0),
                goal_mask.unsqueeze(0),
                spatial_size=self.crop_size // self.patch_size,
                slot_weights=slot_weights,
                area_normalizer=self._slot_area_normalizer_for_view(view_name, device=self.device),
            ).mean()
            loss = (
                (1.0 - self.transport_loss_mix) * mask_l1_loss
                + self.transport_loss_mix * transport_loss
            )
        else:
            loss = mask_l1_loss
        return float(loss.detach().cpu().item())

    def _set_view_history_items(self, view_name, items):
        self.frame_buffer[view_name].clear()
        self.state_buffer[view_name].clear()
        self.extrinsics_buffer[view_name].clear()
        self.slot_context_buffer[view_name].clear()
        self.history_goal_loss_buffer[view_name].clear()
        for item in items:
            self.frame_buffer[view_name].append(item["frame"])
            self.state_buffer[view_name].append(item["state"])
            self.extrinsics_buffer[view_name].append(item["extrinsics"])
            if item.get("context") is not None:
                self.slot_context_buffer[view_name].append(item["context"])
            self.history_goal_loss_buffer[view_name].append(item.get("goal_loss"))

    def _pad_history_left(self, view_name):
        if not self.frame_buffer[view_name]:
            return
        while len(self.frame_buffer[view_name]) < self.history_len:
            self.frame_buffer[view_name].appendleft(self.frame_buffer[view_name][0])
            self.state_buffer[view_name].appendleft(self.state_buffer[view_name][0])
            self.extrinsics_buffer[view_name].appendleft(self.extrinsics_buffer[view_name][0])
            if self.slot_context_buffer[view_name]:
                self.slot_context_buffer[view_name].appendleft(self.slot_context_buffer[view_name][0])
            if self.history_goal_loss_buffer[view_name]:
                self.history_goal_loss_buffer[view_name].appendleft(self.history_goal_loss_buffer[view_name][0])
            else:
                self.history_goal_loss_buffer[view_name].appendleft(None)

    def _state_extrinsics_to_robot_pose6(self, state, extrinsics):
        state_np = state.detach().float().cpu().numpy().reshape(-1)
        extr_np = extrinsics.detach().float().cpu().numpy().reshape(-1)
        if state_np.shape[0] < 6 or extr_np.shape[0] < 6:
            return None
        return self._compose_pose6(extr_np[:6], state_np[:6]).astype(np.float32)

    def _pose_history_distance(self, pose6_a, pose6_b):
        pose6_a = np.asarray(pose6_a, dtype=np.float32).reshape(6)
        pose6_b = np.asarray(pose6_b, dtype=np.float32).reshape(6)
        delta6 = self._pose_delta6(pose6_a, pose6_b)
        # Use translation as the primary nearest-history criterion. Rotation
        # breaks ties without letting Euler noise dominate frame selection.
        return float(np.linalg.norm(delta6[:3]) + 0.05 * np.linalg.norm(delta6[3:6]))

    @staticmethod
    def _pose_history_distance_prepared(
        target_pose6,
        target_rotation_inverse,
        history_pose6,
        history_rotation,
    ):
        """Equivalent to _pose_history_distance with reusable rotations."""
        target_pose6 = np.asarray(target_pose6, dtype=np.float32).reshape(6)
        history_pose6 = np.asarray(history_pose6, dtype=np.float32).reshape(6)
        xyz_diff = history_pose6[:3] - target_pose6[:3]
        rot_diff = (history_rotation * target_rotation_inverse).as_euler(
            "xyz", degrees=False
        ).astype(np.float32)
        return float(
            np.linalg.norm(xyz_diff) + 0.05 * np.linalg.norm(rot_diff)
        )

    def _all_history_items_for_view(self, view_name):
        frame_items = list(self.frame_buffer[view_name])
        state_items = list(self.state_buffer[view_name])
        extr_items = list(self.extrinsics_buffer[view_name])
        context_items = list(self.slot_context_buffer[view_name])
        loss_items = list(self.history_goal_loss_buffer[view_name])
        num_items = min(len(frame_items), len(state_items), len(extr_items))
        items = []
        for idx in range(num_items):
            context = context_items[idx] if idx < len(context_items) else None
            if context is None and view_name in self.fixed_obj_latents:
                context = self._decode_single_frame_context(frame_items[idx], self.fixed_obj_latents[view_name])
            pose6 = self._state_extrinsics_to_robot_pose6(state_items[idx], extr_items[idx])
            state_np = state_items[idx].detach().float().cpu().numpy().reshape(-1)
            goal_loss = loss_items[idx] if idx < len(loss_items) else None
            items.append(
                {
                    "frame": frame_items[idx],
                    "state": state_items[idx],
                    "extrinsics": extr_items[idx],
                    "context": context,
                    "goal_loss": goal_loss,
                    "robot_pose6": pose6,
                    "_robot_rotation": (
                        None
                        if pose6 is None
                        else Rotation.from_euler("xyz", pose6[3:6], degrees=False)
                    ),
                    "gripper_state": None if state_np.shape[0] < 7 else float(state_np[6]),
                    "original_index": idx,
                }
            )
        return items

    def _pad_selected_history_items(self, selected, max_items):
        selected = list(selected)
        if not selected:
            return selected, 0
        padded_left_count = 0
        while len(selected) < int(max_items):
            selected.insert(0, selected[0])
            padded_left_count += 1
        return selected[-int(max_items) :], padded_left_count

    def _weighted_pose_history_sample(self, items, target_count):
        target_count = int(target_count)
        if target_count <= 0 or not items:
            return []
        if len(items) <= target_count:
            return list(items)
        distances = np.asarray(
            [
                float(item.get("candidate_pose_distance", float("inf")))
                for item in items
            ],
            dtype=np.float64,
        )
        finite = np.isfinite(distances)
        if not finite.any():
            probs = np.full(len(items), 1.0 / len(items), dtype=np.float64)
        else:
            finite_distances = distances[finite]
            d_min = float(np.min(finite_distances))
            tau = max(
                float(self.history_pose_sampling_temperature),
                float(np.median(finite_distances) - d_min),
                1.0e-6,
            )
            weights = np.zeros(len(items), dtype=np.float64)
            weights[finite] = np.exp(-(distances[finite] - d_min) / tau)
            if float(weights.sum()) <= 0.0:
                weights[finite] = 1.0
            uniform = np.full(len(items), 1.0 / len(items), dtype=np.float64)
            probs = weights / float(weights.sum())
            probs = (1.0 - self.history_pose_uniform_mix) * probs + self.history_pose_uniform_mix * uniform
            probs = probs / float(probs.sum())
        chosen = np.random.choice(len(items), size=target_count, replace=False, p=probs)
        return [items[int(idx)] for idx in chosen.tolist()]

    def _select_balanced_gripper_history_items(self, items, latest_item, max_items):
        max_items = int(max_items)
        if max_items <= 0:
            return []
        latest_idx = int(latest_item["original_index"])
        selected_by_idx = {latest_idx: latest_item}
        split = self.gripper_position_min + 0.5 * (self.gripper_position_max - self.gripper_position_min)

        def _valid_with_gripper(item):
            return item.get("gripper_state") is not None and np.isfinite(float(item["gripper_state"]))

        open_items = [
            item
            for item in items
            if int(item["original_index"]) != latest_idx
            and _valid_with_gripper(item)
            and float(item["gripper_state"]) < split
        ]
        closed_items = [
            item
            for item in items
            if int(item["original_index"]) != latest_idx
            and _valid_with_gripper(item)
            and float(item["gripper_state"]) >= split
        ]

        def _sort_for_gripper_group(group):
            return sorted(
                group,
                key=lambda item: (
                    float(item["candidate_pose_distance"]),
                    float(item["candidate_gripper_distance"]),
                    -int(item["original_index"]),
                ),
            )

        open_sorted = _sort_for_gripper_group(open_items)
        closed_sorted = _sort_for_gripper_group(closed_items)
        latest_is_closed = (
            latest_item.get("gripper_state") is not None
            and float(latest_item["gripper_state"]) >= split
        )
        closed_quota_total = max_items // 2
        open_quota_total = max_items - closed_quota_total
        closed_needed = max(0, closed_quota_total - (1 if latest_is_closed else 0))
        open_needed = max(0, open_quota_total - (0 if latest_is_closed else 1))

        for item in closed_sorted[:closed_needed]:
            selected_by_idx[int(item["original_index"])] = item
        for item in open_sorted[:open_needed]:
            selected_by_idx[int(item["original_index"])] = item

        if len(selected_by_idx) < max_items:
            all_remaining = sorted(
                [
                    item
                    for item in items
                    if int(item["original_index"]) not in selected_by_idx
                ],
                key=lambda item: (
                    float(item["candidate_pose_distance"]),
                    float(item["candidate_gripper_distance"]),
                    -int(item["original_index"]),
                ),
            )
            for item in all_remaining[: max_items - len(selected_by_idx)]:
                selected_by_idx[int(item["original_index"])] = item
        return list(selected_by_idx.values())

    def _select_gripper_pose_weighted_history_items(self, items, latest_item, max_items):
        max_items = int(max_items)
        if max_items <= 0:
            return []
        latest_idx = int(latest_item["original_index"])
        selected_by_idx = {latest_idx: latest_item}
        ranked = sorted(
            [
                item
                for item in items
                if int(item["original_index"]) != latest_idx
            ],
            key=lambda item: (
                float(item["candidate_gripper_pose_distance"]),
                float(item["candidate_gripper_distance"]),
                float(item["candidate_pose_distance"]),
                -int(item["original_index"]),
            ),
        )
        for item in ranked[: max(0, max_items - 1)]:
            selected_by_idx[int(item["original_index"])] = item
        if len(selected_by_idx) < max_items:
            fallback = sorted(
                [
                    item
                    for item in items
                    if int(item["original_index"]) not in selected_by_idx
                ],
                key=lambda item: (
                    float(item["candidate_gripper_pose_distance"]),
                    -int(item["original_index"]),
                ),
            )
            for item in fallback[: max_items - len(selected_by_idx)]:
                selected_by_idx[int(item["original_index"])] = item
        return list(selected_by_idx.values())

    def _select_candidate_history_items(
        self,
        view_name,
        target_pose6_robot,
        target_gripper,
        max_items,
        mode,
        history_items=None,
    ):
        # Materializing history items performs several tiny device-to-host
        # copies per frame. CEM calls this method once per candidate, while the
        # underlying history is immutable for the whole optimize_action call.
        # Reuse the materialized metadata, but copy the dictionaries because
        # candidate-specific distances below are intentionally written into
        # each item and are included in diagnostics.
        if history_items is None:
            normal_items = self._all_history_items_for_view(view_name)
        else:
            normal_items = [dict(item) for item in history_items]
        items = normal_items
        if not items:
            return [], {
                "mode": "empty",
                "num_frames_total": 0,
                "selected_original_indices": [],
            }
        # A3/A4 replace relevance retrieval for every predictor query,
        # including gripper queries, and do not use the calibration bank.
        if self.memory_selection_mode in {"recent", "in_trajectory_random"}:
            if self.memory_selection_mode == "recent":
                selected = list(normal_items[-int(max_items) :])
                target_count = int(max_items)
            else:
                target_count = min(int(max_items), int(self.history_len))
                selected_indices = self._in_trajectory_random_indices(
                    len(normal_items),
                    target_count,
                )
                selected = [normal_items[idx] for idx in selected_indices]
            padded_left_count = 0
            # Cold start is defined using frame 0, including its state,
            # extrinsics, source mask and target/context mask representation.
            while selected and len(selected) < target_count:
                selected.insert(0, normal_items[0])
                padded_left_count += 1
            return selected, {
                "mode": f"{self.memory_selection_mode}_context",
                "memory_selection_mode": self.memory_selection_mode,
                "num_frames_total": int(len(normal_items)),
                "num_frames_selected": int(len(selected)),
                "padded_left_count": int(padded_left_count),
                "cold_start_padding_source_index": 0 if padded_left_count else None,
                "random_seed": self.memory_random_seed
                if self.memory_selection_mode == "in_trajectory_random"
                else None,
                "effective_random_seed": self.in_trajectory_random_effective_seed
                if self.memory_selection_mode == "in_trajectory_random"
                else None,
                "sampling_scope": "current_trajectory_history"
                if self.memory_selection_mode == "in_trajectory_random"
                else None,
                "sampling_pool": "full_prior_history_plus_latest"
                if self.memory_selection_mode == "in_trajectory_random"
                else None,
                "random_history_window": None,
                "random_pool_start_original_index": 0
                if self.memory_selection_mode == "in_trajectory_random"
                else None,
                "random_pool_end_original_index": int(len(normal_items)) - 2
                if self.memory_selection_mode == "in_trajectory_random"
                else None,
                "forced_latest_original_index": int(len(normal_items)) - 1
                if self.memory_selection_mode == "in_trajectory_random"
                else None,
                "selected_original_indices": [int(item["original_index"]) for item in selected],
                "selected_goal_losses": [
                    None if item.get("goal_loss") is None else float(item["goal_loss"])
                    for item in selected
                ],
                "selected_sources": [item.get("source") for item in selected],
            }

        gripper_bank = [
            dict(item) for item in self.gripper_history_bank_by_view.get(view_name, [])
        ]
        use_frozen_gripper_bank = bool(mode == "gripper_pose_weighted" and gripper_bank)
        if use_frozen_gripper_bank:
            latest_normal_item = normal_items[-1]
            latest_item = dict(latest_normal_item)
            latest_item["original_index"] = 2_000_000 + int(latest_normal_item["original_index"])
            latest_item["source"] = "latest_pose_history"
            items = list(gripper_bank) + [latest_item]
        else:
            latest_item = items[-1]
        target_pose6_robot = np.asarray(target_pose6_robot, dtype=np.float32).reshape(6)
        target_rotation_inverse = Rotation.from_euler(
            "xyz", target_pose6_robot[3:6], degrees=False
        ).inv()
        latest_idx = int(latest_item["original_index"])
        target_gripper = float(target_gripper)

        for item in items:
            pose6 = item.get("robot_pose6")
            history_rotation = item.get("_robot_rotation")
            if pose6 is not None and history_rotation is None:
                history_rotation = Rotation.from_euler(
                    "xyz", np.asarray(pose6, dtype=np.float32).reshape(6)[3:6], degrees=False
                )
            item["candidate_pose_distance"] = (
                float("inf")
                if pose6 is None
                else self._pose_history_distance_prepared(
                    target_pose6_robot,
                    target_rotation_inverse,
                    pose6,
                    history_rotation,
                )
            )
            grip = item.get("gripper_state")
            item["candidate_gripper_distance"] = (
                float("inf") if grip is None else abs(float(grip) - target_gripper)
            )
            item["candidate_gripper_pose_distance"] = (
                float(item["candidate_gripper_distance"])
                + self.gripper_history_pose_distance_weight * float(item["candidate_pose_distance"])
            )

        if use_frozen_gripper_bank:
            selected = list(items)
            selected = sorted(selected, key=lambda item: int(item["original_index"]))
        elif mode == "gripper_pose_weighted":
            selected = self._select_gripper_pose_weighted_history_items(
                items,
                latest_item=latest_item,
                max_items=max_items,
            )
            selected = sorted(selected, key=lambda item: int(item["original_index"]))
        elif mode in {"gripper_nearest", "gripper_balanced"}:
            # Preserve both open and closed gripper appearances. Selecting
            # purely by target gripper can collapse the context to open-only
            # history, making closed candidates look out-of-distribution.
            selected = self._select_balanced_gripper_history_items(
                items,
                latest_item=latest_item,
                max_items=max_items,
            )
            selected = sorted(selected, key=lambda item: int(item["original_index"]))
        else:
            # Keep the latest observation, but sample the rest by distance
            # rather than taking hard nearest neighbors. This maintains
            # pose diversity for the warped-history predictor input.
            selected_by_idx = {latest_idx: latest_item}
            remaining = [item for item in items if int(item["original_index"]) != latest_idx]
            sampled = self._weighted_pose_history_sample(
                remaining,
                max(0, int(max_items) - 1),
            )
            for item in sampled:
                selected_by_idx[int(item["original_index"])] = item
            if len(selected_by_idx) < int(max_items):
                fallback = sorted(
                    [
                        item
                        for item in items
                        if int(item["original_index"]) not in selected_by_idx
                    ],
                    key=lambda item: (
                        float(item["candidate_pose_distance"]),
                        float(item["candidate_gripper_distance"]),
                        -int(item["original_index"]),
                    ),
                )
                for item in fallback[: int(max_items) - len(selected_by_idx)]:
                    selected_by_idx[int(item["original_index"])] = item
            selected = list(selected_by_idx.values())
            selected = sorted(selected, key=lambda item: int(item["original_index"]))

        selected, padded_left_count = self._pad_selected_history_items(selected, max_items)
        split = self.gripper_position_min + 0.5 * (self.gripper_position_max - self.gripper_position_min)
        selected_gripper_states = [
            None if item.get("gripper_state") is None else float(item["gripper_state"])
            for item in selected
        ]
        available_gripper_states = [
            None if item.get("gripper_state") is None else float(item["gripper_state"])
            for item in items
        ]
        return selected, {
            "mode": mode,
            "num_frames_total": int(len(items)),
            "num_pose_history_frames_total": int(len(normal_items)),
            "gripper_history_bank_size": int(len(gripper_bank)),
            "gripper_history_bank_used": bool(use_frozen_gripper_bank),
            "gripper_history_selection_policy": (
                "frozen_bank_all_plus_latest" if use_frozen_gripper_bank else mode
            ),
            "num_frames_selected": int(len(selected)),
            "padded_left_count": int(padded_left_count),
            "forced_latest_original_index": (
                int(latest_idx)
                if mode in {"gripper_nearest", "gripper_balanced", "gripper_pose_weighted"}
                else None
            ),
            "pose_sampling_temperature": float(self.history_pose_sampling_temperature),
            "pose_uniform_mix": float(self.history_pose_uniform_mix),
            "gripper_history_pose_distance_weight": float(self.gripper_history_pose_distance_weight),
            "gripper_open_closed_split": float(split),
            "selected_open_count": int(
                sum(1 for grip in selected_gripper_states if grip is not None and float(grip) < split)
            ),
            "selected_closed_count": int(
                sum(1 for grip in selected_gripper_states if grip is not None and float(grip) >= split)
            ),
            "available_open_count": int(
                sum(1 for grip in available_gripper_states if grip is not None and float(grip) < split)
            ),
            "available_closed_count": int(
                sum(1 for grip in available_gripper_states if grip is not None and float(grip) >= split)
            ),
            "target_gripper": float(target_gripper),
            "selected_original_indices": [int(item["original_index"]) for item in selected],
            "selected_pose_distances": [
                None
                if not np.isfinite(float(item["candidate_pose_distance"]))
                else float(item["candidate_pose_distance"])
                for item in selected
            ],
            "selected_gripper_distances": [
                None
                if not np.isfinite(float(item["candidate_gripper_distance"]))
                else float(item["candidate_gripper_distance"])
                for item in selected
            ],
            "selected_gripper_pose_distances": [
                None
                if not np.isfinite(float(item["candidate_gripper_pose_distance"]))
                else float(item["candidate_gripper_pose_distance"])
                for item in selected
            ],
            "selected_gripper_states": selected_gripper_states,
            "selected_goal_losses": [
                None if item.get("goal_loss") is None else float(item["goal_loss"])
                for item in selected
            ],
            "selected_sources": [item.get("source") for item in selected],
        }

    def _build_candidate_predictor_cache_for_view(self, view_name, selected_items_by_candidate, base_cache):
        context_masks, current_mask_probs, poses_view, extr_view = (
            _batch_candidate_history_tensors(
                selected_items_by_candidate,
                context_frames=self.model_context_frames,
                view_name=view_name,
            )
        )

        obj_latents = self.fixed_obj_latents[view_name].float().to(self.device)
        batch_size = context_masks.size(0)
        goal_mask_probs = base_cache["goal_mask_probs"].to(self.device, dtype=torch.float32)
        current_mask_probs = current_mask_probs.to(self.device, dtype=torch.float32)
        slot_change_weights = self._slot_change_weights(
            current_mask_probs,
            goal_mask_probs.unsqueeze(0).expand_as(current_mask_probs),
        )
        shared_visible_weights = base_cache["shared_visible_weights"]
        return {
            "obj_latents": obj_latents.expand(batch_size, -1, -1),
            "context_masks": context_masks.to(self.device, dtype=torch.float32),
            "poses_view": poses_view.to(self.device, dtype=torch.float32),
            "extr_view": extr_view.to(self.device, dtype=torch.float32),
            "goal_feat": base_cache["goal_feat"],
            "goal_mask_probs": goal_mask_probs,
            "slot_change_weights": slot_change_weights,
            "shared_visible_weights": shared_visible_weights,
        }

    def _select_uniform_history_items(self, view_name, max_items):
        frame_items = list(self.frame_buffer[view_name])
        state_items = list(self.state_buffer[view_name])
        extr_items = list(self.extrinsics_buffer[view_name])
        context_items = list(self.slot_context_buffer[view_name])
        loss_items = list(self.history_goal_loss_buffer[view_name])
        num_items = min(len(frame_items), len(state_items), len(extr_items))
        if num_items <= 0:
            return [], {
                "mode": "empty",
                "num_frames_total": 0,
                "selected_original_indices": [],
            }
        frame_items = frame_items[:num_items]
        state_items = state_items[:num_items]
        extr_items = extr_items[:num_items]
        latest_idx = num_items - 1
        current_pose6 = self._state_extrinsics_to_robot_pose6(state_items[latest_idx], extr_items[latest_idx])
        items = []
        for idx in range(num_items):
            context = context_items[idx] if idx < len(context_items) else None
            if context is None and view_name in self.fixed_obj_latents:
                context = self._decode_single_frame_context(frame_items[idx], self.fixed_obj_latents[view_name])
            pose6 = self._state_extrinsics_to_robot_pose6(state_items[idx], extr_items[idx])
            distance = float("inf") if current_pose6 is None or pose6 is None else self._pose_history_distance(current_pose6, pose6)
            goal_loss = loss_items[idx] if idx < len(loss_items) else None
            items.append(
                {
                    "frame": frame_items[idx],
                    "state": state_items[idx],
                    "extrinsics": extr_items[idx],
                    "context": context,
                    "goal_loss": goal_loss,
                    "robot_pose6": pose6,
                    "pose_distance": distance,
                    "original_index": idx,
                }
            )
        if num_items > max_items:
            selected_indices = np.linspace(0, latest_idx, int(max_items), dtype=np.int64)
            selected_indices[-1] = latest_idx
            selected_indices = np.unique(selected_indices)
            if selected_indices.shape[0] < int(max_items):
                missing = int(max_items) - int(selected_indices.shape[0])
                remaining = [idx for idx in range(num_items) if idx not in set(selected_indices.tolist())]
                selected_indices = np.asarray(
                    sorted(list(selected_indices.tolist()) + remaining[-missing:]),
                    dtype=np.int64,
                )
            selected = [items[int(idx)] for idx in selected_indices[-int(max_items) :]]
        else:
            selected = items
        selected_before_padding = list(selected)
        padded_left_count = 0
        if selected:
            while len(selected) < int(max_items):
                selected.insert(0, selected[0])
                padded_left_count += 1
        debug = {
            "mode": "uniform_history_with_latest",
            "num_frames_total": int(num_items),
            "num_frames_selected": int(len(selected)),
            "num_frames_selected_before_padding": int(len(selected_before_padding)),
            "padded_left_count": int(padded_left_count),
            "forced_latest_original_index": int(latest_idx),
            "selected_original_indices": [int(item["original_index"]) for item in selected],
            "selected_pose_distances": [
                None if not np.isfinite(item["pose_distance"]) else float(item["pose_distance"])
                for item in selected
            ],
            "selected_goal_losses": [
                None if item.get("goal_loss") is None else float(item["goal_loss"])
                for item in selected
            ],
        }
        return selected, debug

    @torch.no_grad()
    def _select_history_by_goal_loss(self, available_views, goal_idx=0):
        selection_debug = {
            "goal_idx": int(goal_idx),
            "history_len": int(self.history_len),
            "views": {},
        }
        if self.goal_features is None or self.goal_mask_probs is None:
            for view_name in available_views:
                self._pad_history_left(view_name)
                selection_debug["views"][view_name] = {
                    "mode": "pad_without_goal_features",
                    "num_frames_after": int(len(self.frame_buffer[view_name])),
                }
            self.last_history_selection_debug = selection_debug
            return
        for view_name in available_views:
            frame_items = list(self.frame_buffer[view_name])
            state_items = list(self.state_buffer[view_name])
            extr_items = list(self.extrinsics_buffer[view_name])
            context_items = list(self.slot_context_buffer[view_name])
            loss_items = list(self.history_goal_loss_buffer[view_name])
            if not frame_items:
                continue
            items = []
            for idx, (frame, state, extrinsics) in enumerate(zip(frame_items, state_items, extr_items)):
                context = context_items[idx] if idx < len(context_items) else None
                if context is None and view_name in self.fixed_obj_latents:
                    context = self._decode_single_frame_context(frame, self.fixed_obj_latents[view_name])
                goal_loss = loss_items[idx] if idx < len(loss_items) else None
                if goal_loss is None and context is not None:
                    goal_loss = self._context_goal_objective_loss(view_name, context, goal_idx)
                items.append(
                    {
                        "frame": frame,
                        "state": state,
                        "extrinsics": extrinsics,
                        "context": context,
                        "goal_loss": goal_loss,
                        "original_index": idx,
                    }
                )
            if len(items) > self.history_len:
                newest_item = items[-1]
                history_slots = max(0, self.history_len - 1)
                ranked_old = sorted(
                    items[:-1],
                    key=lambda item: (
                        float("inf") if item["goal_loss"] is None else float(item["goal_loss"]),
                        item["original_index"],
                    ),
                )
                selected = sorted(
                    ranked_old[:history_slots] + [newest_item],
                    key=lambda item: item["original_index"],
                )
            else:
                selected = items
            self._set_view_history_items(view_name, selected)
            self._pad_history_left(view_name)
            selection_debug["views"][view_name] = {
                "mode": "latest_plus_goal_loss_topk",
                "num_frames_before": int(len(items)),
                "num_frames_after": int(len(self.frame_buffer[view_name])),
                "forced_latest_original_index": int(items[-1]["original_index"]),
                "forced_latest_goal_loss": None
                if items[-1]["goal_loss"] is None
                else float(items[-1]["goal_loss"]),
                "selected_original_indices": [int(item["original_index"]) for item in selected],
                "selected_goal_losses": [
                    None if item["goal_loss"] is None else float(item["goal_loss"])
                    for item in selected
                ],
            }
        self.last_history_selection_debug = selection_debug

    @torch.no_grad()
    def _encode_student_sequence(self, frames):
        frame_list = list(frames)
        if not frame_list:
            raise ValueError("Expected at least one frame for SLOT encoding.")
        first = frame_list[0]
        if first.ndim == 4:
            clip = torch.cat(frame_list, dim=0).to(self.device)
        elif first.ndim == 3:
            clip = torch.stack(frame_list, dim=0).to(self.device)
        else:
            raise ValueError(f"Unsupported frame tensor shape for SLOT encoding: {tuple(first.shape)}")
        clip = clip.unsqueeze(0)  # [1,T,C,H,W]
        B, T, C, H, W = clip.shape
        flat = clip.view(B * T, C, H, W)
        with self._autocast_context():
            student_feats = self.student(flat)
        student_feats = student_feats.float()
        D = student_feats.size(-1)
        spatial_size = self.crop_size // self.patch_size
        student_5d = student_feats.view(B, T, spatial_size, spatial_size, D)
        return student_5d.float()

    @torch.no_grad()
    def _decode_sequence_with_fixed_latents(self, frames, obj_latents):
        student_5d = self._encode_student_sequence(frames)
        B, T, H, W, _ = student_5d.shape
        class_state_all = self.extractor.decode_with_latents(
            obj_latents.to(self.device, dtype=torch.float32),
            student_5d.to(self.device, dtype=torch.float32),
            spatial_dims=(H, W),
            T_dim=T,
        )
        decoded_mask_logits = class_state_all.get_layer(-1)["mask_logits"]
        decoded_mask_probs = F.softmax(decoded_mask_logits, dim=2)
        decoder_recon_all = torch.einsum("btkn,bkd->btnd", decoded_mask_probs.to(obj_latents.dtype), obj_latents)
        return {
            "obj_latents": obj_latents.float(),
            "decoded_mask_logits": decoded_mask_logits.float(),
            "decoded_mask_probs": decoded_mask_probs.float(),
            "decoder_recon_all": decoder_recon_all.float(),
        }

    @torch.no_grad()
    def _decode_single_frame_context(self, frame, obj_latents):
        decoded = self._decode_sequence_with_fixed_latents([frame], obj_latents)
        return {
            "decoded_mask_logits": decoded["decoded_mask_logits"][:, 0].detach().clone(),
            "decoded_mask_probs": decoded["decoded_mask_probs"][:, 0].detach().clone(),
            "decoder_recon": decoded["decoder_recon_all"][:, 0].detach().clone(),
        }

    @torch.no_grad()
    def _rebuild_slot_context_buffer_for_view(self, view_name):
        if view_name not in self.fixed_obj_latents:
            return
        frames = list(self.frame_buffer[view_name])
        self.slot_context_buffer[view_name].clear()
        if not frames:
            return
        decoded = self._decode_sequence_with_fixed_latents(frames, self.fixed_obj_latents[view_name])
        for frame_idx in range(decoded["decoded_mask_logits"].size(1)):
            self.slot_context_buffer[view_name].append(
                {
                    "decoded_mask_logits": decoded["decoded_mask_logits"][:, frame_idx].detach().clone(),
                    "decoded_mask_probs": decoded["decoded_mask_probs"][:, frame_idx].detach().clone(),
                    "decoder_recon": decoded["decoder_recon_all"][:, frame_idx].detach().clone(),
                }
            )

    @torch.no_grad()
    def _encode_frame_sequence(self, frames):
        student_5d = self._encode_student_sequence(frames)
        B, T, H, W, _ = student_5d.shape
        class_state_all, obj_latents = self.extractor(
            student_5d.to(self.device, dtype=torch.float32),
            spatial_dims=(H, W),
            T_dim=T,
        )
        decoded_mask_logits = class_state_all.get_layer(-1)["mask_logits"]
        decoded_mask_probs = F.softmax(decoded_mask_logits, dim=2)
        decoder_recon_all = torch.einsum("btkn,bkd->btnd", decoded_mask_probs.to(obj_latents.dtype), obj_latents)
        return {
            "obj_latents": obj_latents.float(),
            "decoded_mask_logits": decoded_mask_logits.float(),
            "decoded_mask_probs": decoded_mask_probs.float(),
            "decoder_recon_all": decoder_recon_all.float(),
        }

    @torch.no_grad()
    def _encode_multiview_obj_latents(self, view_frame_sequences, view_names):
        if not view_names:
            raise ValueError("Expected at least one view for multiview SLOT latent initialization.")
        frame_counts = [len(view_frame_sequences[view_name]) for view_name in view_names]
        if len(set(frame_counts)) != 1:
            raise ValueError(f"Expected equal frame counts for all views, got {dict(zip(view_names, frame_counts))}")
        per_view_student = [
            self._encode_student_sequence(view_frame_sequences[view_name]).squeeze(0)
            for view_name in view_names
        ]
        joint_student = torch.stack(per_view_student, dim=0).unsqueeze(0)
        _, num_views, total_frames, H, W, D = joint_student.shape
        joint_student = joint_student.reshape(1, num_views * total_frames, H, W, D)
        _, obj_latents = self.extractor(
            joint_student.to(self.device, dtype=torch.float32),
            spatial_dims=(H, W),
            T_dim=num_views * total_frames,
        )
        return obj_latents.float()

    @torch.no_grad()
    def _encode_single_goal_frame(self, rgb_image, obj_latents=None):
        frame = self._preprocess_image_tensor(rgb_image)
        if obj_latents is None:
            encoded = self._encode_frame_sequence([frame])
        else:
            encoded = self._decode_sequence_with_fixed_latents([frame], obj_latents)
        return encoded["decoder_recon_all"][:, 0]

    @torch.no_grad()
    def _refresh_fixed_latents(self, view_names, reason):
        if not self.goal_image_maps or not self.goal_view_names:
            raise ValueError("Goal images must be set before SLOT latent initialization.")
        init_view_names = [view_name for view_name in self.goal_view_names if view_name in view_names and len(self.frame_buffer[view_name]) > 0]
        if not init_view_names:
            raise ValueError("Failed to refresh fixed SLOT latents: no overlapping view history.")
        view_frame_sequences = {}
        for view_name in init_view_names:
            view_frame_sequences[view_name] = list(self.frame_buffer[view_name])
            view_frame_sequences[view_name].extend(
                self._preprocess_image_tensor(goal_view_map[view_name])
                for goal_view_map in self.goal_image_maps
                if view_name in goal_view_map
            )
        per_view_obj_latents = {}
        for view_name in init_view_names:
            encoded = self._encode_frame_sequence(view_frame_sequences[view_name])
            per_view_obj_latents[view_name] = encoded["obj_latents"].detach().clone()
        self.fixed_obj_latents = per_view_obj_latents
        self.goal_view_names = [view_name for view_name in self.goal_view_names if view_name in self.fixed_obj_latents]
        for view_name in self.goal_view_names:
            self._rebuild_slot_context_buffer_for_view(view_name)
        goal_features = []
        goal_mask_probs = []
        for goal_view_map in self.goal_image_maps:
            per_view = []
            per_view_mask = []
            for view_name in self.goal_view_names:
                goal_encoded = self._decode_sequence_with_fixed_latents(
                    [self._preprocess_image_tensor(goal_view_map[view_name])],
                    self.fixed_obj_latents[view_name],
                )
                per_view.append(goal_encoded["decoder_recon_all"][:, 0].squeeze(0))
                per_view_mask.append(goal_encoded["decoded_mask_probs"][:, 0].squeeze(0))
            goal_features.append(torch.stack(per_view, dim=0))
            goal_mask_probs.append(torch.stack(per_view_mask, dim=0))
        self.goal_features = torch.stack(goal_features, dim=0)
        self.goal_feature_full = self.goal_features
        self.goal_mask_probs = torch.stack(goal_mask_probs, dim=0)
        self.slot_area_normalizer_by_view = {}
        for view_idx, view_name in enumerate(self.goal_view_names):
            masks_for_area = [
                entry["decoded_mask_probs"].squeeze(0).detach().float().to(self.device)
                for entry in self.slot_context_buffer[view_name]
                if "decoded_mask_probs" in entry
            ]
            masks_for_area.extend(
                self.goal_mask_probs[goal_idx, view_idx].detach().float().to(self.device)
                for goal_idx in range(int(self.goal_mask_probs.shape[0]))
            )
            if masks_for_area:
                mask_area_stack = torch.stack(masks_for_area, dim=0).sum(dim=-1)
                self.slot_area_normalizer_by_view[view_name] = (
                    mask_area_stack.amax(dim=0)
                    .clamp_min(self.mask_transport_area_normalize_eps)
                    .detach()
                    .cpu()
                )
        self.fixed_latent_initialized = True
        self.last_obj_latent_refresh_step = int(self.predict_step_idx)
        self.last_obj_latent_refresh_reason = str(reason)
        self.last_obj_latent_refresh_mode = "warmup_goal_per_view_independent"

    @torch.no_grad()
    def _initialize_fixed_latents_if_needed(self, obs_views):
        if self.fixed_latent_initialized:
            return
        init_view_names = [view_name for view_name in self.goal_view_names if view_name in obs_views]
        self._refresh_fixed_latents(init_view_names, reason="initial")

    def _build_context_from_buffers(self, available_views):
        context_by_view = {}
        states_by_view = {}
        extrinsics_by_view = {}
        current_latent_by_view = {}
        selection_debug = {
            "mode": "build_context_uniform_history_with_latest",
            "history_len": int(self.history_len),
            "model_context_frames": int(self.model_context_frames),
            "views": {},
        }
        for view_name in available_views:
            if view_name not in self.fixed_obj_latents:
                raise ValueError(f"Missing fixed SLOT latent for view {view_name}")
            if len(self.slot_context_buffer[view_name]) != len(self.frame_buffer[view_name]):
                raise ValueError(
                    f"SLOT context cache mismatch for {view_name}: "
                    f"context={len(self.slot_context_buffer[view_name])}, frames={len(self.frame_buffer[view_name])}. "
                    "Historical masks/recons are no longer rebuilt after warmup initialization."
                )
            selected_items, view_selection_debug = self._select_uniform_history_items(
                view_name,
                max_items=self.history_len,
            )
            selection_debug["views"][view_name] = view_selection_debug
            if not selected_items:
                raise ValueError(f"Missing history frames for view {view_name}")
            context_entries = [item.get("context") for item in selected_items]
            if any(entry is None for entry in context_entries):
                missing_indices = [
                    int(item["original_index"])
                    for item, entry in zip(selected_items, context_entries)
                    if entry is None
                ]
                raise ValueError(
                    f"Missing cached SLOT context for view {view_name} "
                    f"at original history indices {missing_indices}"
                )
            context_by_view[view_name] = {
                "obj_latents": self.fixed_obj_latents[view_name].float(),
                "decoded_mask_logits": torch.stack(
                    [entry["decoded_mask_logits"] for entry in context_entries],
                    dim=1,
                ).float(),
                "decoded_mask_probs": torch.stack(
                    [entry["decoded_mask_probs"] for entry in context_entries],
                    dim=1,
                ).float(),
                "decoder_recon_all": torch.stack(
                    [entry["decoder_recon"] for entry in context_entries],
                    dim=1,
                ).float(),
            }
            states_by_view[view_name] = torch.stack([item["state"] for item in selected_items], dim=1)
            extrinsics_by_view[view_name] = torch.stack([item["extrinsics"] for item in selected_items], dim=1)
            current_latent_by_view[view_name] = context_by_view[view_name]["decoder_recon_all"][:, -1]
        self.last_history_selection_debug = selection_debug
        return context_by_view, states_by_view, extrinsics_by_view, current_latent_by_view

    @torch.no_grad()
    def _prepare_predictor_cache(self, context_by_view, states_by_view, extrinsics_by_view, view_names):
        cache_by_view = {}
        goal_view_indices = {view_name: idx for idx, view_name in enumerate(self.goal_view_names)}
        for view_name in view_names:
            context = context_by_view[view_name]
            obj_latents = context["obj_latents"][:, :, :].to(self.device, dtype=torch.float32)
            context_masks = context["decoded_mask_logits"][:, -self.model_context_frames :].to(self.device, dtype=torch.float32)
            poses_view = states_by_view[view_name][:, -self.model_context_frames :].to(self.device, dtype=torch.float32)
            extr_view = extrinsics_by_view[view_name][:, -self.model_context_frames :].to(self.device, dtype=torch.float32)
            goal_feat = self.goal_features[self.current_goal_idx, goal_view_indices[view_name]].to(
                self.device, dtype=torch.float32
            )
            goal_mask_probs = self.goal_mask_probs[self.current_goal_idx, goal_view_indices[view_name]].to(
                self.device, dtype=torch.float32
            )
            current_recon = context["decoder_recon_all"][:, -1].to(self.device, dtype=torch.float32)
            shared_visible_weights = self._shared_visible_weights_from_recon(current_recon, goal_feat)
            current_mask_probs = context["decoded_mask_probs"][:, -1].to(self.device, dtype=torch.float32)
            slot_change_weights = self._slot_change_weights(
                current_mask_probs,
                goal_mask_probs.unsqueeze(0),
            )

            cache_by_view[view_name] = {
                "obj_latents": obj_latents,
                "context_masks": context_masks,
                "poses_view": poses_view,
                "extr_view": extr_view,
                "goal_feat": goal_feat,
                "goal_mask_probs": goal_mask_probs,
                "slot_change_weights": slot_change_weights,
                "slot_change_weight_mean": float(slot_change_weights.mean().detach().cpu().item()),
                "slot_change_weight_max": float(slot_change_weights.max().detach().cpu().item()),
                "shared_visible_weights": shared_visible_weights,
                "shared_visible_weight_mean": float(shared_visible_weights.mean().detach().cpu().item()),
                "shared_visible_weight_mass": float(shared_visible_weights.sum().detach().cpu().item()),
            }
        return cache_by_view

    @torch.no_grad()
    def _build_copied_warped_masks(
        self, context_masks, num_queries, context_mask_probs=None
    ):
        if context_mask_probs is None:
            context_mask_probs = F.softmax(context_masks, dim=2)
        return context_mask_probs.unsqueeze(1).expand(-1, num_queries, -1, -1, -1).contiguous()

    def _mask_grid_image(self, mask_probs, grid_cols=4, normalize_each=True, norm_min=None, norm_max=None):
        mask_probs = self._centralize_mask_probs(mask_probs.detach().float()).cpu()
        if mask_probs.ndim != 2:
            raise ValueError(f"Expected mask_probs [K, N], got {tuple(mask_probs.shape)}")
        num_masks, num_patches = mask_probs.shape
        spatial_size = int(math.isqrt(num_patches))
        if spatial_size * spatial_size != num_patches:
            raise ValueError(f"Mask patch count {num_patches} is not square.")
        grid_rows = int(math.ceil(num_masks / grid_cols))
        masks = mask_probs.reshape(num_masks, spatial_size, spatial_size)
        if norm_min is not None and norm_max is not None:
            norm_min = norm_min.detach().float().cpu().reshape(num_masks, 1, 1)
            norm_max = norm_max.detach().float().cpu().reshape(num_masks, 1, 1)
            masks = (masks - norm_min) / (norm_max - norm_min + 1.0e-6)
        elif normalize_each:
            m_min = masks.amin(dim=(1, 2), keepdim=True)
            m_max = masks.amax(dim=(1, 2), keepdim=True)
            masks = (masks - m_min) / (m_max - m_min + 1.0e-6)
        else:
            masks = masks.clamp(0.0, 1.0)
        tile = 96
        pad = 6
        label_h = 16
        canvas_h = grid_rows * (tile + label_h + pad) + pad
        canvas_w = grid_cols * (tile + pad) + pad
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)
        for mask_idx in range(num_masks):
            row = mask_idx // grid_cols
            col = mask_idx % grid_cols
            x = pad + col * (tile + pad)
            y = pad + row * (tile + label_h + pad)
            mask_u8 = (masks[mask_idx].numpy() * 255.0).clip(0, 255).astype(np.uint8)
            nearest_resample = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
            tile_img = Image.fromarray(mask_u8).resize((tile, tile), resample=nearest_resample)
            tile_rgb = Image.merge("RGB", (tile_img, tile_img, tile_img))
            canvas.paste(tile_rgb, (x, y))
            draw.text((x + 2, y + tile + 1), f"mask {mask_idx:02d}", fill=(0, 0, 0))
        return canvas

    def _model_input_to_pil(self, frame_tensor, tile_size):
        frame = frame_tensor.detach().float().cpu()
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim != 3 or frame.size(0) != 3:
            raise ValueError(f"Expected model input frame [3,H,W], got {tuple(frame.shape)}")
        mean = self._mean.view(3, 1, 1).cpu()
        std = self._std.view(3, 1, 1).cpu()
        frame = (frame * std + mean).clamp(0.0, 1.0)
        frame_u8 = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        from PIL import Image

        nearest_resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
        return Image.fromarray(frame_u8).resize((tile_size, tile_size), resample=nearest_resample)

    @staticmethod
    def _slot_color(mask_idx):
        palette = [
            (230, 25, 75),
            (60, 180, 75),
            (255, 225, 25),
            (0, 130, 200),
            (245, 130, 48),
            (145, 30, 180),
            (70, 240, 240),
            (240, 50, 230),
            (210, 245, 60),
            (250, 190, 190),
            (0, 128, 128),
            (230, 190, 255),
            (170, 110, 40),
            (255, 250, 200),
            (128, 0, 0),
            (170, 255, 195),
        ]
        return palette[mask_idx % len(palette)]

    def _mask_overlay_grid_image(
        self,
        mask_probs,
        base_frame,
        grid_cols=4,
        norm_min=None,
        norm_max=None,
        highlight_mask_idx=None,
        highlight_label="FIXED",
    ):
        mask_probs = self._centralize_mask_probs(mask_probs.detach().float()).cpu()
        if mask_probs.ndim != 2:
            raise ValueError(f"Expected mask_probs [K, N], got {tuple(mask_probs.shape)}")
        num_masks, num_patches = mask_probs.shape
        spatial_size = int(math.isqrt(num_patches))
        if spatial_size * spatial_size != num_patches:
            raise ValueError(f"Mask patch count {num_patches} is not square.")

        masks = mask_probs.reshape(num_masks, spatial_size, spatial_size)
        if norm_min is not None and norm_max is not None:
            norm_min = norm_min.detach().float().cpu().reshape(num_masks, 1, 1)
            norm_max = norm_max.detach().float().cpu().reshape(num_masks, 1, 1)
            masks = (masks - norm_min) / (norm_max - norm_min + 1.0e-6)
        else:
            masks = masks.clamp(0.0, 1.0)
        masks = masks.clamp(0.0, 1.0)

        from PIL import Image, ImageDraw

        tile = 112
        pad = 6
        label_h = 18
        grid_rows = int(math.ceil(num_masks / grid_cols))
        canvas_h = grid_rows * (tile + label_h + pad) + pad
        canvas_w = grid_cols * (tile + pad) + pad
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)
        base_img = self._model_input_to_pil(base_frame, tile)
        bilinear_resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR

        for mask_idx in range(num_masks):
            row = mask_idx // grid_cols
            col = mask_idx % grid_cols
            x = pad + col * (tile + pad)
            y = pad + row * (tile + label_h + pad)

            mask_u8 = (masks[mask_idx].numpy() * 255.0).clip(0, 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_u8).resize((tile, tile), resample=bilinear_resample)
            color = self._slot_color(mask_idx)
            color_img = Image.new("RGB", (tile, tile), color)
            alpha = np.asarray(mask_img, dtype=np.float32) / 255.0
            alpha = (alpha * 0.65).clip(0.0, 0.65)
            alpha_img = Image.fromarray((alpha * 255.0).astype(np.uint8))
            overlay = Image.composite(color_img, base_img, alpha_img)
            canvas.paste(overlay, (x, y))
            outline = (255, 0, 0) if highlight_mask_idx is not None and int(mask_idx) == int(highlight_mask_idx) else color
            outline_width = 4 if highlight_mask_idx is not None and int(mask_idx) == int(highlight_mask_idx) else 2
            draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=outline, width=outline_width)
            label = f"mask {mask_idx:02d}"
            if highlight_mask_idx is not None and int(mask_idx) == int(highlight_mask_idx):
                label += f" {highlight_label}"
            draw.text((x + 2, y + tile + 2), label, fill=(0, 0, 0))
        return canvas

    def _mask_overlay_grid_with_distances_image(
        self,
        mask_probs,
        target_mask_probs,
        base_frame,
        slot_distances,
        grid_cols=4,
        highlight_mask_idx=None,
        highlight_label="FIXED",
        target_label=None,
    ):
        mask_probs = self._centralize_mask_probs(mask_probs.detach().float()).cpu()
        target_mask_probs = self._centralize_mask_probs(target_mask_probs.detach().float()).cpu()
        if mask_probs.ndim != 2:
            raise ValueError(f"Expected mask_probs [K, N], got {tuple(mask_probs.shape)}")
        if target_mask_probs.shape != mask_probs.shape:
            raise ValueError(
                f"Target mask shape mismatch: current={tuple(mask_probs.shape)} target={tuple(target_mask_probs.shape)}"
            )
        num_masks, num_patches = mask_probs.shape
        spatial_size = int(math.isqrt(num_patches))
        if spatial_size * spatial_size != num_patches:
            raise ValueError(f"Mask patch count {num_patches} is not square.")

        # Use the same per-slot normalization for current and target masks so
        # faint / missing slots are visually comparable in the diagnostic panel.
        norm_stack = torch.stack([mask_probs, target_mask_probs], dim=0)
        norm_min = norm_stack.amin(dim=(0, 2)).reshape(num_masks, 1, 1)
        norm_max = norm_stack.amax(dim=(0, 2)).reshape(num_masks, 1, 1)
        masks = mask_probs.reshape(num_masks, spatial_size, spatial_size)
        masks = ((masks - norm_min) / (norm_max - norm_min + 1.0e-6)).clamp(0.0, 1.0)
        distances = np.asarray(slot_distances, dtype=np.float32).reshape(-1)

        from PIL import Image, ImageDraw

        tile = 112
        pad = 6
        label_h = 32
        grid_rows = int(math.ceil(num_masks / grid_cols))
        canvas_h = grid_rows * (tile + label_h + pad) + pad + 24
        canvas_w = grid_cols * (tile + pad) + pad
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)
        base_img = self._model_input_to_pil(base_frame, tile)
        nearest_resample = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST

        loss_name = "UOT/max-slot-area" if self.mask_transport_area_normalize else "UOT"
        if target_label is None:
            target_label = f"goal[{int(self.current_goal_idx)}]"
        draw.text((pad, 4), f"slot masks vs {target_label} {loss_name}", fill=(0, 0, 0))
        y_offset = 24
        for mask_idx in range(num_masks):
            row = mask_idx // grid_cols
            col = mask_idx % grid_cols
            x = pad + col * (tile + pad)
            y = y_offset + pad + row * (tile + label_h + pad)

            mask_u8 = (masks[mask_idx].numpy() * 255.0).clip(0, 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_u8).resize((tile, tile), resample=nearest_resample)
            color = self._slot_color(mask_idx)
            color_img = Image.new("RGB", (tile, tile), color)
            alpha = np.asarray(mask_img, dtype=np.float32) / 255.0
            alpha_img = Image.fromarray((alpha * 180.0).clip(0, 180).astype(np.uint8))
            overlay = Image.composite(color_img, base_img, alpha_img)
            canvas.paste(overlay, (x, y))

            outline = (255, 0, 0) if highlight_mask_idx is not None and int(mask_idx) == int(highlight_mask_idx) else color
            outline_width = 4 if highlight_mask_idx is not None and int(mask_idx) == int(highlight_mask_idx) else 2
            draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=outline, width=outline_width)
            label = f"{mask_idx:02d} d={float(distances[mask_idx]):.3f}" if mask_idx < distances.size else f"{mask_idx:02d}"
            if highlight_mask_idx is not None and int(mask_idx) == int(highlight_mask_idx):
                label += f" {highlight_label}"
            draw.text((x + 2, y + tile + 2), label, fill=(0, 0, 0))
        return canvas

    def _slot_mask_highlight_for_view(self, view_name):
        fixed_idx = self.fixed_slot_index_by_view.get(view_name)
        gripper_idx = self.gripper_slot_index_by_view.get(view_name)
        if fixed_idx is not None:
            label = "FIXED"
            if gripper_idx is not None and int(gripper_idx) == int(fixed_idx):
                label = "FIXED/GRIPPER"
            return int(fixed_idx), label
        if gripper_idx is not None:
            return int(gripper_idx), "GRIPPER"
        return None, "FIXED"

    def _render_initial_goal_mask_figure(self):
        if self.goal_mask_probs is None or not self.goal_view_names:
            raise ValueError("Goal masks are not available yet. Run at least one get_action call first.")
        if not any(len(self.slot_context_buffer[view_name]) > 0 for view_name in self.goal_view_names):
            raise ValueError("Initial frame masks are not available yet. Run at least one get_action call first.")

        from PIL import Image, ImageDraw

        view_panels = []
        goal_idx = int(self.current_goal_idx)
        goal_view_indices = {view_name: idx for idx, view_name in enumerate(self.goal_view_names)}
        for view_name in self.goal_view_names:
            context_entries = list(self.slot_context_buffer[view_name])
            if not context_entries:
                continue
            context_frames = list(self.frame_buffer[view_name])
            if len(context_frames) != len(context_entries):
                raise ValueError(
                    f"SLOT mask visualization mismatch for {view_name}: "
                    f"context={len(context_entries)}, frames={len(context_frames)}"
                )
            view_idx = goal_view_indices[view_name]
            mask_stack = [
                self._centralize_mask_probs(entry["decoded_mask_probs"][0].detach().float()).cpu()
                for entry in context_entries
            ]
            mask_stack.extend(
                self._centralize_mask_probs(self.goal_mask_probs[goal_i, view_idx].detach().float()).cpu()
                for goal_i in range(len(self.goal_image_maps))
            )
            norm_stack = torch.stack(mask_stack, dim=0)
            norm_min = norm_stack.amin(dim=(0, 2))
            norm_max = norm_stack.amax(dim=(0, 2))

            rows = []
            highlight_idx, highlight_label = self._slot_mask_highlight_for_view(view_name)
            for frame_idx, (entry, frame) in enumerate(zip(context_entries, context_frames)):
                grid = self._mask_overlay_grid_image(
                    entry["decoded_mask_probs"][0],
                    frame,
                    norm_min=norm_min,
                    norm_max=norm_max,
                    highlight_mask_idx=highlight_idx,
                    highlight_label=highlight_label,
                )
                rows.append((f"warmup/context[{frame_idx:02d}] masks", grid))
            for goal_i, goal_view_map in enumerate(self.goal_image_maps):
                goal_masks = self.goal_mask_probs[goal_i, view_idx]
                goal_frame = self._preprocess_image_tensor(goal_view_map[view_name])
                grid = self._mask_overlay_grid_image(
                    goal_masks,
                    goal_frame,
                    norm_min=norm_min,
                    norm_max=norm_max,
                    highlight_mask_idx=highlight_idx,
                    highlight_label=highlight_label,
                )
                current_marker = " current" if goal_i == goal_idx else ""
                rows.append((f"goal[{goal_i}]{current_marker} masks", grid))

            title_h = 26
            gap = 16
            panel_w = max(grid.width for _, grid in rows)
            panel_h = title_h + sum(title_h + grid.height for _, grid in rows) + gap * max(len(rows) - 1, 0)
            panel = Image.new("RGB", (panel_w, panel_h), "white")
            draw = ImageDraw.Draw(panel)
            y = 0
            draw.text((4, y + 5), f"{view_name} | per-view obj latents | centralized masks", fill=(0, 0, 0))
            y += title_h
            for row_idx, (label, grid) in enumerate(rows):
                draw.text((4, y + 4), label, fill=(0, 0, 0))
                y += title_h
                panel.paste(grid, (0, y))
                y += grid.height
                if row_idx != len(rows) - 1:
                    y += gap
            view_panels.append(panel)

        if not view_panels:
            raise ValueError("No view masks are available for visualization.")

        margin = 16
        header_h = 30
        out_w = sum(panel.width for panel in view_panels) + margin * (len(view_panels) + 1)
        out_h = max(panel.height for panel in view_panels) + margin * 2 + header_h
        out = Image.new("RGB", (out_w, out_h), "white")
        draw = ImageDraw.Draw(out)
        draw.text(
            (margin, 6),
            f"slot_mask_overlay_v2 | goal_idx={goal_idx} | latent_refresh_step={self.last_obj_latent_refresh_step} | reason={self.last_obj_latent_refresh_reason} | centralized_gamma={self.mask_centralize_gamma:g}",
            fill=(0, 0, 0),
        )
        x = margin
        for panel in view_panels:
            out.paste(panel, (x, margin + header_h))
            x += panel.width + margin
        return out

    @torch.no_grad()
    def render_current_slot_mask_panels(self, data):
        if self.goal_mask_probs is None or not self.goal_view_names:
            raise ValueError("Goal masks are not available yet. Run at least one get_action call first.")
        if not self.fixed_obj_latents:
            obs_views_for_init = self._parse_observation_views(data)
            self._initialize_fixed_latents_if_needed(obs_views_for_init)
        obs_views = self._parse_observation_views(data)
        available_views = self._filter_available_views(obs_views)
        goal_view_indices = {view_name: idx for idx, view_name in enumerate(self.goal_view_names)}
        panels_by_view = {}
        slot_distances_by_view = {}
        current_goal_idx = int(self.current_goal_idx)
        spatial_size = self.crop_size // self.patch_size

        for view_name in available_views:
            if view_name not in self.fixed_obj_latents or view_name not in goal_view_indices:
                continue
            frame = self._preprocess_image_tensor(obs_views[view_name]["image"])
            decoded = self._decode_single_frame_context(frame, self.fixed_obj_latents[view_name])
            current_mask = decoded["decoded_mask_probs"].squeeze(0).to(self.device, dtype=torch.float32)
            target_mask = self.goal_mask_probs[current_goal_idx, goal_view_indices[view_name]].to(
                self.device,
                dtype=torch.float32,
            )
            distances = self._sinkhorn_mask_transport_loss_per_slot(
                current_mask.unsqueeze(0),
                target_mask.unsqueeze(0),
                spatial_size=spatial_size,
                area_normalizer=self._slot_area_normalizer_for_view(view_name, device=self.device),
            ).squeeze(0)
            highlight_idx, highlight_label = self._slot_mask_highlight_for_view(view_name)
            panel = self._mask_overlay_grid_with_distances_image(
                current_mask,
                target_mask,
                frame,
                distances.detach().cpu().numpy(),
                highlight_mask_idx=highlight_idx,
                highlight_label=highlight_label,
            )
            panels_by_view[view_name] = np.asarray(panel, dtype=np.uint8)
            slot_distances_by_view[view_name] = distances.detach().cpu().numpy().astype(np.float32)

        if not panels_by_view:
            raise ValueError("No SLOT mask panels could be rendered for the provided observation.")
        return {
            "status": "success",
            "visualization": "current_slot_mask_distances",
            "goal_idx": current_goal_idx,
            "view_names": list(panels_by_view.keys()),
            "mask_panels_by_view": panels_by_view,
            "slot_distances_by_view": slot_distances_by_view,
        }

    @torch.no_grad()
    def render_goal_slot_mask_panels(self, data=None):
        if self.goal_mask_probs is None or not self.goal_view_names or not self.goal_image_maps:
            raise ValueError("Goal masks are not available yet. Run at least one get_action call first.")
        goal_view_indices = {view_name: idx for idx, view_name in enumerate(self.goal_view_names)}
        current_goal_idx = int(self.current_goal_idx)
        spatial_size = self.crop_size // self.patch_size
        panels_by_goal = []
        slot_distances_by_goal = []

        for goal_idx, goal_view_map in enumerate(self.goal_image_maps):
            panels_by_view = {}
            distances_by_view = {}
            for view_name in self.goal_view_names:
                if view_name not in goal_view_map or view_name not in goal_view_indices:
                    continue
                view_idx = goal_view_indices[view_name]
                current_mask = self.goal_mask_probs[goal_idx, view_idx].to(self.device, dtype=torch.float32)
                target_mask = self.goal_mask_probs[goal_idx, view_idx].to(self.device, dtype=torch.float32)
                distances = self._sinkhorn_mask_transport_loss_per_slot(
                    current_mask.unsqueeze(0),
                    target_mask.unsqueeze(0),
                    spatial_size=spatial_size,
                    area_normalizer=self._slot_area_normalizer_for_view(view_name, device=self.device),
                ).squeeze(0)
                frame = self._preprocess_image_tensor(goal_view_map[view_name])
                highlight_idx, highlight_label = self._slot_mask_highlight_for_view(view_name)
                panel = self._mask_overlay_grid_with_distances_image(
                    current_mask,
                    target_mask,
                    frame,
                    distances.detach().cpu().numpy(),
                    highlight_mask_idx=highlight_idx,
                    highlight_label=highlight_label,
                    target_label=f"self goal[{goal_idx}]",
                )
                panels_by_view[view_name] = np.asarray(panel, dtype=np.uint8)
                distances_by_view[view_name] = distances.detach().cpu().numpy().astype(np.float32)
            panels_by_goal.append(panels_by_view)
            slot_distances_by_goal.append(distances_by_view)

        return {
            "status": "success",
            "visualization": "goal_slot_mask_distances",
            "goal_idx": current_goal_idx,
            "num_goals": len(panels_by_goal),
            "view_names": list(self.goal_view_names),
            "mask_panels_by_goal": panels_by_goal,
            "slot_distances_by_goal": slot_distances_by_goal,
        }

    @torch.no_grad()
    def _predict_from_cache(self, cache, next_states, next_extrinsics=None, use_copied_warp=False):
        batch_size = next_states.size(0)
        context_masks = cache["context_masks"].expand(batch_size, -1, -1, -1)
        obj_latents = cache["obj_latents"].expand(batch_size, -1, -1)
        context_states = cache["poses_view"].expand(batch_size, -1, -1)
        context_extrinsics = cache["extr_view"].expand(batch_size, -1, -1) if self.use_extrinsics else None
        if self.use_extrinsics and next_extrinsics is None:
            raise ValueError("next_extrinsics is required when use_extrinsics=True")
        warp_result = self._build_warped_masks_from_cache(
            cache,
            next_states=next_states,
            next_extrinsics=next_extrinsics,
            use_copied_warp=use_copied_warp,
            expanded_context_masks=context_masks,
            return_precomputed=True,
        )
        if isinstance(warp_result, tuple):
            external_warped, context_mask_probs, warp_obj_latent_low = warp_result
        else:
            # Keep test doubles and external callers that implement the old
            # tensor-only helper contract working.
            external_warped = warp_result
            context_mask_probs = None
            warp_obj_latent_low = None
        with self._autocast_context():
            pred_logits = self.predictor(
                obj_latent=obj_latents,
                context_masks=context_masks,
                states=context_states,
                next_states=next_states,
                extrinsics=context_extrinsics,
                next_extrinsics=next_extrinsics,
                external_warped_mask_probs=external_warped,
                external_warp_matches_context=use_copied_warp,
                precomputed_mask_probs=context_mask_probs,
                # The warp helper runs outside autocast to preserve its
                # established float32 geometry. Reuse its projection only
                # when predictor autocast is disabled, otherwise recomputing
                # preserves the previous BF16 predictor numerics.
                precomputed_obj_latent_low=(
                    warp_obj_latent_low
                    if getattr(self, "autocast_dtype", None) is None
                    else None
                ),
            )
        return pred_logits.float(), external_warped

    @torch.no_grad()
    def _build_warped_masks_from_cache(
        self,
        cache,
        next_states,
        next_extrinsics=None,
        use_copied_warp=False,
        expanded_context_masks=None,
        return_precomputed=False,
    ):
        if not getattr(self.predictor, "use_warp_concat", False):
            return (None, None, None) if return_precomputed else None
        batch_size = next_states.size(0)
        context_masks = (
            expanded_context_masks
            if expanded_context_masks is not None
            else cache["context_masks"].expand(batch_size, -1, -1, -1)
        )
        mask_probs = torch.softmax(context_masks, dim=2)
        if use_copied_warp:
            warped = self._build_copied_warped_masks(
                context_masks,
                next_states.size(1),
                context_mask_probs=mask_probs,
            )
            return (warped, mask_probs, None) if return_precomputed else warped
        if not self.use_extrinsics:
            return (None, mask_probs, None) if return_precomputed else None
        if next_extrinsics is None:
            raise ValueError("next_extrinsics is required to solve geometric warped masks")
        obj_latents = cache["obj_latents"].expand(batch_size, -1, -1).to(self.device, dtype=torch.float32)
        context_states = cache["poses_view"].expand(batch_size, -1, -1).to(self.device, dtype=torch.float32)
        context_extrinsics = cache["extr_view"].expand(batch_size, -1, -1).to(self.device, dtype=torch.float32)
        del context_states  # States are not used by the geometric warp solver.
        obj_latent_low = self.predictor.recon_branch.obj_latent_reduce(obj_latents)
        warped = self.predictor._solve_warped_context_masks(
            obj_latent_low.detach(),
            mask_probs.detach(),
            context_extrinsics.detach(),
            next_extrinsics.detach(),
        )
        if return_precomputed:
            return warped, mask_probs, obj_latent_low
        return warped

    @torch.no_grad()
    def _warped_recon_from_masks(self, cache, warped_mask_probs, source_index=-1):
        if warped_mask_probs is None:
            return None, None
        if warped_mask_probs.ndim != 5:
            raise ValueError(f"Expected warped_mask_probs [B,Q,T,K,N], got {tuple(warped_mask_probs.shape)}")
        source_index = int(source_index)
        warped_last = warped_mask_probs[:, -1, source_index].to(self.device, dtype=torch.float32)
        obj_latents = cache["obj_latents"].expand(warped_last.size(0), -1, -1).to(self.device, dtype=torch.float32)
        warped_recon = torch.einsum("bkn,bkd->bnd", warped_last.to(obj_latents.dtype), obj_latents)
        return warped_recon.float(), warped_last.float()

    def reset(self):
        self._clear_temporal_buffers()
        self._clear_gripper_history_bank()
        self.fixed_obj_latents = {}
        self.fixed_latent_initialized = False
        self.goal_state = None
        self.last_goal_pose_prior_debug = None
        self.goal_features = None
        self.goal_mask_probs = None
        self.goal_view_names = []
        self.goal_view_indices = {}
        self.fixed_slot_index_by_view = {}
        self.fixed_slot_debug_by_view = {}
        self.gripper_slot_index_by_view = {}
        self.gripper_slot_debug_by_view = {}
        self.predict_step_idx = 0
        self.last_obj_latent_refresh_step = -1
        self.last_obj_latent_refresh_reason = None
        self.last_obj_latent_refresh_mode = None
        self.history_z_context.clear()
        self.history_z_warp.clear()
        self.history_z_pred.clear()
        self.history_mask_context.clear()
        self.history_mask_warp.clear()
        self.history_mask_pred.clear()
        self.history_debug.clear()
        self.pending_one_step_predictions.clear()
        self.next_pending_prediction_id = 0
        self.recent_selected_pose_deltas.clear()
        self.last_observed_robot_pose6 = None
        self.recent_observed_robot_poses.clear()
        self.recent_observed_pose_deltas.clear()
        self.current_completed_stages = set()
        self.slot_planning_phase_cursor = "pose"
        self.pose_memory_by_view = {view_name: [] for view_name in self.VIEW_ORDER}
        self.gripper_history_bank_by_view = {view_name: [] for view_name in self.VIEW_ORDER}
        self.fixed_loss_search_reference_by_goal = {}
        self.warmup_fixed_anchor_by_goal = {}
        self.pose_view_best_losses_by_goal = {}
        self.pose_view_no_improve_steps_by_goal = {}
        self.pose_view_active_group_by_goal = {}
        self.pose_view_loss_history_by_goal = {}
        self.warmup_score_obs_sequence = None
        self.last_predictor_debug = {}
        self.in_trajectory_random_effective_seed = int(self.memory_random_seed) & 0xFFFFFFFF
        self.in_trajectory_random_rng = np.random.default_rng(self.in_trajectory_random_effective_seed)
        self.in_trajectory_random_index_cache = {}
        if torch.cuda.is_available() and not _GPU_MEMORY_RESERVED_IN_CACHE:
            torch.cuda.empty_cache()

    def set_goal_image(self, goal_images):
        with torch.no_grad():
            if not isinstance(goal_images, list):
                goal_images = [goal_images]
            common_views = None
            encoded_goals = []
            for goal_item in goal_images:
                if isinstance(goal_item, dict):
                    goal_view_map = {k: self._to_uint8_rgb(v) for k, v in goal_item.items()}
                else:
                    goal_view_map = {"exterior_image_1_left": self._to_uint8_rgb(goal_item)}
                ordered_views = [view_name for view_name in self.VIEW_ORDER if view_name in goal_view_map]
                if common_views is None:
                    common_views = ordered_views
                else:
                    common_views = [view_name for view_name in common_views if view_name in goal_view_map]
                encoded_goals.append(goal_view_map)
            if not common_views:
                raise ValueError("No common views found across goal targets.")
            self.goal_image_maps = encoded_goals
            self.goal_features = None
            self.goal_feature_full = None
            self.goal_mask_probs = None
            self.goal_view_names = list(common_views)
            self.fixed_obj_latents = {}
            self.slot_area_normalizer_by_view = {}
            self._clear_gripper_history_bank()
            self.fixed_slot_index_by_view = {}
            self.fixed_slot_debug_by_view = {}
            self.gripper_slot_index_by_view = {}
            self.gripper_slot_debug_by_view = {}
            self.fixed_latent_initialized = False
            self.current_goal_idx = 0
            self.last_goal_pose_prior_debug = None
            self.recent_selected_pose_deltas.clear()
            self.last_observed_robot_pose6 = None
            self.recent_observed_robot_poses.clear()
            self.recent_observed_pose_deltas.clear()
            self.slot_planning_phase_cursor = "pose"
            self.fixed_loss_search_reference_by_goal = {}
            self.warmup_fixed_anchor_by_goal = {}
            self.pose_view_best_losses_by_goal = {}
            self.pose_view_no_improve_steps_by_goal = {}
            self.pose_view_active_group_by_goal = {}
            self.pose_view_loss_history_by_goal = {}
            self.warmup_score_obs_sequence = None
            self.in_trajectory_random_index_cache = {}

    @torch.no_grad()
    def optimize_action(
        self,
        context_by_view,
        states_by_view,
        extrinsics_by_view,
        view_names,
        current_obs_views=None,
        planning_request="joint",
    ):
        timing = {
            "goal_prep_s": 0.0,
            "cem_sampling_s": 0.0,
            "cem_eval_total_s": 0.0,
            "cem_sort_update_s": 0.0,
            "evaluate_cpu_pose_build_s": 0.0,
            "evaluate_history_materialize_s": 0.0,
            "evaluate_history_select_s": 0.0,
            "evaluate_view_prep_s": 0.0,
            "evaluate_query_tensor_to_gpu_s": 0.0,
            "evaluate_predictor_forward_s": 0.0,
            "evaluate_pred_extract_s": 0.0,
            "evaluate_loss_total_s": 0.0,
            "predictor_calls": 0,
            "predictor_views_evaluated": 0,
            "predictor_candidates_evaluated": 0,
            "transport_solves_reused": 0,
            "fixed_view_parallel_pairs": 0,
            "fixed_view_parallel_warmups": 0,
        }
        t_goal_prep_start = time.perf_counter()
        ref_view = self._reference_view(view_names)
        goal_view_indices = {view_name: idx for idx, view_name in enumerate(self.goal_view_names)}
        current_goal_target = self._current_goal_target()
        current_state7_by_view_np = {}
        current_extrinsics_by_view_np = {}
        for view_name in view_names:
            if current_obs_views is not None and view_name in current_obs_views:
                current_state7_by_view_np[view_name] = np.asarray(
                    current_obs_views[view_name]["state7"],
                    dtype=np.float32,
                ).reshape(7)
                current_extrinsics_by_view_np[view_name] = np.asarray(
                    current_obs_views[view_name]["extrinsics6"],
                    dtype=np.float32,
                ).reshape(6)
            else:
                current_state7_by_view_np[view_name] = (
                    states_by_view[view_name][0, -1].detach().cpu().numpy().astype(np.float32)
                )
                current_extrinsics_by_view_np[view_name] = (
                    extrinsics_by_view[view_name][0, -1].detach().cpu().numpy().astype(np.float32)
                )
        ref_state7_np = current_state7_by_view_np[ref_view].astype(np.float32)
        ref_current_pose6 = ref_state7_np[:6]
        ref_extrinsics6 = current_extrinsics_by_view_np[ref_view].astype(np.float32)
        current_robot_pose6 = self._compose_pose6(ref_extrinsics6, ref_current_pose6)
        observed_pose_delta6 = None
        if self.last_observed_robot_pose6 is not None:
            observed_pose_delta6 = self._pose_delta6(
                np.asarray(self.last_observed_robot_pose6, dtype=np.float32),
                current_robot_pose6.astype(np.float32),
            ).astype(np.float32)
            self.recent_observed_pose_deltas.append(observed_pose_delta6.copy())
        self.last_observed_robot_pose6 = current_robot_pose6.astype(np.float32).copy()
        self.recent_observed_robot_poses.append(current_robot_pose6.astype(np.float32).copy())
        recent_observed_pose_deltas = list(self.recent_observed_pose_deltas)
        if recent_observed_pose_deltas:
            recent_observed_pose_delta_arr = np.stack(recent_observed_pose_deltas, axis=0).astype(np.float32)
            observed_translation_norms = np.linalg.norm(recent_observed_pose_delta_arr[:, :3], axis=1)
            observed_rotation_norms = np.linalg.norm(recent_observed_pose_delta_arr[:, 3:6], axis=1)
            observed_step_mean_translation_norm = float(observed_translation_norms.mean())
            observed_step_mean_rotation_norm = float(observed_rotation_norms.mean())
        else:
            recent_observed_pose_delta_arr = np.zeros((0, 6), dtype=np.float32)
            observed_translation_norms = np.zeros((0,), dtype=np.float32)
            observed_rotation_norms = np.zeros((0,), dtype=np.float32)
            observed_step_mean_translation_norm = 0.0
            observed_step_mean_rotation_norm = 0.0
        recent_observed_robot_poses = list(self.recent_observed_robot_poses)
        observed_motion_window_ready = len(recent_observed_robot_poses) >= int(self.observed_motion_window + 1)
        if observed_motion_window_ready:
            observed_window_delta6 = self._pose_delta6(
                np.asarray(recent_observed_robot_poses[0], dtype=np.float32),
                np.asarray(recent_observed_robot_poses[-1], dtype=np.float32),
            ).astype(np.float32)
            observed_window_step_count = max(1, len(recent_observed_robot_poses) - 1)
            observed_translation_norm = float(np.linalg.norm(observed_window_delta6[:3]) / observed_window_step_count)
            observed_rotation_norm = float(np.linalg.norm(observed_window_delta6[3:6]) / observed_window_step_count)
        else:
            observed_window_delta6 = np.zeros((6,), dtype=np.float32)
            observed_window_step_count = max(0, len(recent_observed_robot_poses) - 1)
            observed_translation_norm = 0.0
            observed_rotation_norm = 0.0
        goal_robot_pose6 = None
        current_goal_pos_l2 = None
        current_goal_rot_l2 = None
        if current_goal_target is not None and current_goal_target.get("ee_state7_robot") is not None:
            goal_state7_robot = np.asarray(current_goal_target["ee_state7_robot"], dtype=np.float32).reshape(-1)
            if goal_state7_robot.shape[0] >= 6:
                goal_robot_pose6 = goal_state7_robot[:6].astype(np.float32)
                current_goal_delta6 = self._pose_delta6(current_robot_pose6, goal_robot_pose6)
                current_goal_pos_l2 = float(np.linalg.norm(current_goal_delta6[:3]))
                current_goal_rot_l2 = float(np.linalg.norm(current_goal_delta6[3:6]))
        mean_delta_pose6, mean_gripper_delta = self._build_pose_delta_init(
            current_robot_pose6=current_robot_pose6,
            current_state7=ref_state7_np,
            current_goal_target=current_goal_target,
        )
        goal_distance_filter = self._build_goal_distance_filter(
            current_robot_pose6=current_robot_pose6,
            current_goal_target=current_goal_target,
        )
        action_limits = self._get_action_limits()
        mean_delta_pose6 = self._cap_pose_delta_norm(
            mean_delta_pose6,
            action_limits["translation_cap_xyz"],
            action_limits["rotation_cap"],
        )
        timing["goal_prep_s"] = float(time.perf_counter() - t_goal_prep_start)

        predictor_cache_by_view = self._prepare_predictor_cache(
            context_by_view=context_by_view,
            states_by_view=states_by_view,
            extrinsics_by_view=extrinsics_by_view,
            view_names=view_names,
        )
        # Candidate selection repeatedly reads the same trajectory history.
        # Keep one CPU-materialized metadata snapshot per view for this entire
        # planning request instead of synchronizing tiny CUDA tensors for every
        # candidate. _select_candidate_history_items copies the dictionaries
        # before attaching candidate-specific distances, preserving selection
        # and diagnostic semantics.
        t_history_materialize_start = time.perf_counter()
        candidate_history_items_by_view = {
            view_name: self._all_history_items_for_view(view_name)
            for view_name in view_names
        }
        timing["evaluate_history_materialize_s"] = float(
            time.perf_counter() - t_history_materialize_start
        )
        fixed_view_names = [view_name for view_name in view_names if "wrist" not in view_name.lower()]
        wrist_view_names = [view_name for view_name in view_names if "wrist" in view_name.lower()]
        if not fixed_view_names:
            fixed_view_names = list(view_names)
        if not wrist_view_names:
            wrist_view_names = list(view_names)
        planning_request = str(planning_request or "joint").lower()
        if planning_request not in {"joint", "pose_only", "gripper_only"}:
            planning_request = "joint"

        current_loss_by_view = {}
        for view_name in view_names:
            current_mask = context_by_view[view_name]["decoded_mask_probs"][:, -1].to(
                self.device, dtype=torch.float32
            )
            goal_mask = predictor_cache_by_view[view_name]["goal_mask_probs"].unsqueeze(0)
            current_loss = torch.mean(torch.abs(current_mask - goal_mask), dim=(1, 2))
            if self.transport_loss_mix > 0.0:
                current_probs = context_by_view[view_name]["decoded_mask_probs"][:, -1].to(self.device, dtype=torch.float32)
                goal_mask_probs = predictor_cache_by_view[view_name]["goal_mask_probs"].unsqueeze(0).expand_as(current_probs)
                current_transport = self._sinkhorn_mask_transport_loss(
                    current_probs,
                    goal_mask_probs,
                    spatial_size=self.crop_size // self.patch_size,
                    slot_weights=predictor_cache_by_view[view_name]["slot_change_weights"],
                    area_normalizer=self._slot_area_normalizer_for_view(view_name, device=self.device),
                )
                current_loss = (1.0 - self.transport_loss_mix) * current_loss + self.transport_loss_mix * current_transport
            current_loss_by_view[view_name] = float(current_loss.mean().detach().cpu().item())
        current_fixed_view_loss = float(np.mean([current_loss_by_view[v] for v in fixed_view_names]))
        pose_switch_updates_this_request = planning_request in {"joint", "pose_only"}
        pose_view_switch = self._update_pose_view_switch(
            goal_idx=int(self.current_goal_idx),
            current_loss_by_view=current_loss_by_view,
            fixed_view_names=fixed_view_names,
            wrist_view_names=wrist_view_names,
            patience=5,
            update=pose_switch_updates_this_request,
            step_idx=int(self.predict_step_idx),
        )
        warmup_fixed_anchor_steering = self._get_warmup_fixed_anchor_steering(
            goal_idx=int(self.current_goal_idx),
            current_fixed_view_loss=current_fixed_view_loss,
            current_robot_pose6=current_robot_pose6,
            current_gripper_state=float(ref_state7_np[6]),
        )
        if (
            pose_switch_updates_this_request
            and warmup_fixed_anchor_steering.get("active", False)
        ):
            pre_anchor_pose_view_switch = dict(pose_view_switch)
            pose_view_switch = dict(pose_view_switch)
            pose_view_switch.update(
                {
                    "warmup_fixed_anchor_steering_active": True,
                    "pre_warmup_anchor_pose_view_switch": pre_anchor_pose_view_switch,
                    "switch_reason": (
                        pose_view_switch.get("switch_reason")
                        or "warmup_fixed_anchor_steering_active_no_view_override"
                    ),
                    "switch_condition": (
                        "warmup fixed anchor steering only biases CEM samples; "
                        "view switching and complement rerank remain active"
                    ),
                }
            )
        fixed_loss_search_scale_info = self._fixed_loss_to_search_scale(
            current_fixed_view_loss,
            goal_idx=int(self.current_goal_idx),
        )
        self._add_pose_memory_entry(
            pose6_robot=current_robot_pose6,
            gripper_state=float(ref_state7_np[6]),
            per_view_losses=current_loss_by_view,
            source="current_observed_goal_score",
            stage=None if current_goal_target is None else current_goal_target.get("stage"),
            step=int(self.predict_step_idx),
            extra={"goal_idx": int(self.current_goal_idx)},
        )

        def compose_gated_candidate_losses(
            per_view_losses_np,
            delta_pose6_batch_np,
            forced_view_loss_mode=None,
        ):
            available_view_names = [view_name for view_name in view_names if view_name in per_view_losses_np]
            available_fixed_view_names = [view_name for view_name in fixed_view_names if view_name in per_view_losses_np]
            available_wrist_view_names = [view_name for view_name in wrist_view_names if view_name in per_view_losses_np]
            if not available_view_names:
                raise ValueError("No per-view losses were computed for this CEM phase.")
            if not available_fixed_view_names:
                available_fixed_view_names = available_view_names
            if not available_wrist_view_names:
                available_wrist_view_names = available_view_names
            all_loss = np.mean(
                np.stack([per_view_losses_np[view_name] for view_name in available_view_names], axis=0),
                axis=0,
            ).astype(np.float32)
            fixed_loss = np.mean(
                np.stack([per_view_losses_np[view_name] for view_name in available_fixed_view_names], axis=0),
                axis=0,
            ).astype(np.float32)
            wrist_loss = np.mean(
                np.stack([per_view_losses_np[view_name] for view_name in available_wrist_view_names], axis=0),
                axis=0,
            ).astype(np.float32)
            current_all_loss = float(np.mean([current_loss_by_view[view_name] for view_name in available_view_names]))
            current_fixed_loss = float(np.mean([current_loss_by_view[view_name] for view_name in available_fixed_view_names]))
            current_wrist_loss = float(np.mean([current_loss_by_view[view_name] for view_name in available_wrist_view_names]))
            fixed_improvement = float(current_fixed_view_loss - float(np.min(fixed_loss)))
            view_loss_mode = forced_view_loss_mode or self.view_loss_mode
            if view_loss_mode == "fixed_only":
                losses = fixed_loss.copy()
                loss_selection_mode = "fixed_only"
                current_selected_loss = current_fixed_loss
            elif view_loss_mode == "wrist_only":
                losses = wrist_loss.copy()
                loss_selection_mode = "wrist_only"
                current_selected_loss = current_wrist_loss
            elif view_loss_mode == "all":
                losses = all_loss.copy()
                loss_selection_mode = "all_views"
                current_selected_loss = current_all_loss
            elif fixed_improvement >= self.fixed_view_loss_gate_threshold:
                losses = fixed_loss.copy()
                loss_selection_mode = "fixed_views"
                current_selected_loss = current_fixed_loss
            else:
                losses = wrist_loss.copy()
                loss_selection_mode = "wrist_views"
                current_selected_loss = current_wrist_loss
            if self.action_l2_weight > 0.0:
                trans_norm = np.linalg.norm(
                    delta_pose6_batch_np[:, :3] / np.maximum(action_limits["translation_cap_xyz"][None, :], 1.0e-6),
                    axis=1,
                )
                rot_norm = np.linalg.norm(
                    delta_pose6_batch_np[:, 3:6] / max(action_limits["rotation_cap"], 1.0e-6),
                    axis=1,
                )
                action_penalty = 0.5 * (trans_norm + rot_norm)
                losses += self.action_l2_weight * action_penalty.astype(np.float32)
            return losses.astype(np.float32), {
                "loss_selection_mode": loss_selection_mode,
                "current_fixed_view_loss": float(current_fixed_view_loss),
                "current_all_view_loss": float(current_all_loss),
                "current_wrist_view_loss": float(current_wrist_loss),
                "current_selected_view_loss": float(current_selected_loss),
                "min_candidate_fixed_view_loss": float(np.min(fixed_loss)),
                "fixed_view_improvement": float(fixed_improvement),
                "all_view_loss": all_loss,
                "fixed_view_loss": fixed_loss,
                "wrist_view_loss": wrist_loss,
            }

        def evaluate_pose_delta_batch(
            delta_pose6_batch_np,
            gripper_delta_batch_np,
            eval_view_names=None,
            rotation_delta_frame="parent",
        ):
            t_eval_start = time.perf_counter()
            if eval_view_names is None:
                eval_view_names = view_names
            eval_view_names = [view_name for view_name in eval_view_names if view_name in view_names]
            if not eval_view_names:
                raise ValueError("evaluate_pose_delta_batch received no valid eval_view_names.")
            batch_size = delta_pose6_batch_np.shape[0]
            pred_by_view = {}
            warp_by_view = {}
            next_state_by_view = {}
            action_by_view = {}
            target_robot_pose6_all = []
            t_cpu_pose_build_start = time.perf_counter()
            for idx in range(batch_size):
                target_robot_pose6_all.append(
                    self._apply_pose_delta6(
                        current_robot_pose6,
                        delta_pose6_batch_np[idx],
                        rotation_delta_frame=rotation_delta_frame,
                    ).astype(np.float32)
                )
            target_robot_pose6_all = np.stack(target_robot_pose6_all, axis=0).astype(np.float32)
            timing["evaluate_cpu_pose_build_s"] += float(time.perf_counter() - t_cpu_pose_build_start)

            t_view_prep_start = time.perf_counter()
            per_view_pred_recon = {}
            per_view_pred_probs = {}
            per_view_slot_weights = {}
            fixed_pair_eligible = bool(
                phase_name == "fixed_pose_coarse"
                and len(eval_view_names) == 2
                and all("wrist" not in name.lower() for name in eval_view_names)
                and torch.device(self.device).type == "cuda"
                and torch.cuda.is_available()
            )
            use_fixed_pair_streams = bool(
                fixed_pair_eligible
                and getattr(self, "_fixed_view_parallel_cache_warmed", False)
            )
            fixed_pair_streams = (
                self._get_fixed_view_prediction_streams()
                if use_fixed_pair_streams
                else ()
            )
            default_stream = (
                torch.cuda.current_stream(device=torch.device(self.device))
                if fixed_pair_streams
                else None
            )
            active_prediction_streams = []
            # Keep cross-stream inputs alive until the default stream has
            # acquired dependencies on both predictor streams.
            parallel_input_refs = []
            for view_index, view_name in enumerate(eval_view_names):
                base_view_cache = predictor_cache_by_view[view_name]
                current_view_state7 = current_state7_by_view_np[view_name]
                current_view_pose6 = current_view_state7[:6].astype(np.float32)
                current_view_gripper = float(current_view_state7[6])
                view_extrinsics_np = current_extrinsics_by_view_np[view_name].astype(np.float32)
                is_wrist_view = "wrist" in view_name.lower()
                wrist_ee_to_camera_pose6 = None
                if is_wrist_view:
                    wrist_ee_to_camera_pose6 = self._transform_to_pose6(
                        np.linalg.inv(self._pose6_to_transform(current_view_pose6))
                    )
                target_state7_batch = []
                target_extrinsics_batch = []
                action_view_batch_np = []
                selected_history_by_candidate = []
                candidate_history_debug = []
                history_select_mode = "gripper_pose_weighted" if phase_name == "wrist_gripper" else "pose_weighted_random"
                history_max_items = self.history_len
                if history_select_mode == "gripper_pose_weighted":
                    history_max_items = max(
                        int(self.history_len),
                        int(len(self.gripper_history_bank_by_view.get(view_name, []))) + 1,
                    )
                t_history_select_start = time.perf_counter()
                for idx in range(batch_size):
                    target_robot_pose6 = target_robot_pose6_all[idx]
                    if is_wrist_view:
                        camera_to_ee_pose6 = current_view_pose6
                        target_view_extrinsics = self._compose_pose6(
                            target_robot_pose6,
                            wrist_ee_to_camera_pose6,
                        )
                        target_view_pose6 = camera_to_ee_pose6
                    else:
                        target_view_extrinsics = view_extrinsics_np
                        target_view_pose6 = self._pose_in_parent_frame(target_robot_pose6, target_view_extrinsics)
                    action_pose6 = self._pose_delta6(current_view_pose6, target_view_pose6)
                    target_view_gripper = float(
                        np.clip(
                            current_view_gripper + gripper_delta_batch_np[idx],
                            self.gripper_position_min,
                            self.gripper_position_max,
                        )
                    )
                    gripper_delta = np.array([target_view_gripper - current_view_gripper], dtype=np.float32)
                    action_view = np.concatenate([action_pose6, gripper_delta], axis=0).astype(np.float32)
                    target_state7_batch.append(
                        np.concatenate(
                            [
                                target_view_pose6.astype(np.float32),
                                np.array([target_view_gripper], dtype=np.float32),
                            ],
                            axis=0,
                        ).astype(np.float32)
                    )
                    target_extrinsics_batch.append(target_view_extrinsics.astype(np.float32))
                    action_view_batch_np.append(action_view)
                    selected_history, history_debug = self._select_candidate_history_items(
                        view_name,
                        target_pose6_robot=target_robot_pose6,
                        target_gripper=target_view_gripper,
                        max_items=history_max_items,
                        mode=history_select_mode,
                        history_items=candidate_history_items_by_view[view_name],
                    )
                    selected_history_by_candidate.append(selected_history)
                    candidate_history_debug.append(history_debug)
                timing["evaluate_history_select_s"] += float(
                    time.perf_counter() - t_history_select_start
                )
                next_state_by_view[view_name] = np.stack(target_state7_batch, axis=0).astype(np.float32)
                next_extrinsics_np = np.stack(target_extrinsics_batch, axis=0).astype(np.float32)
                action_view_batch_np = np.stack(action_view_batch_np, axis=0).astype(np.float32)
                action_by_view[view_name] = action_view_batch_np
                view_cache = self._build_candidate_predictor_cache_for_view(
                    view_name,
                    selected_history_by_candidate,
                    base_view_cache,
                )
                per_view_slot_weights[view_name] = view_cache["slot_change_weights"]
                t_query_to_gpu_start = time.perf_counter()
                next_states = torch.from_numpy(next_state_by_view[view_name]).to(self.device, dtype=torch.float32).unsqueeze(1)
                next_extrinsics = torch.from_numpy(next_extrinsics_np).to(self.device, dtype=torch.float32).unsqueeze(1)
                timing["evaluate_query_tensor_to_gpu_s"] += float(time.perf_counter() - t_query_to_gpu_start)

                prediction_stream = (
                    fixed_pair_streams[view_index]
                    if fixed_pair_streams
                    else None
                )
                if prediction_stream is not None:
                    prediction_stream.wait_stream(default_stream)
                    prediction_context = torch.cuda.stream(prediction_stream)
                    active_prediction_streams.append(prediction_stream)
                    parallel_input_refs.append(
                        (view_cache, next_states, next_extrinsics)
                    )
                else:
                    prediction_context = nullcontext()
                with prediction_context:
                    t_forward_start = time.perf_counter()
                    pred_logits, warped_masks = self._predict_from_cache(
                        view_cache,
                        next_states=next_states,
                        next_extrinsics=next_extrinsics,
                        use_copied_warp=not is_wrist_view,
                    )
                    timing["evaluate_predictor_forward_s"] += float(
                        time.perf_counter() - t_forward_start
                    )
                    timing["predictor_calls"] += 1
                    timing["predictor_views_evaluated"] += 1
                    timing["predictor_candidates_evaluated"] += int(batch_size)

                    t_extract_start = time.perf_counter()
                    pred_probs = F.softmax(pred_logits[:, -1], dim=1)
                    obj_latents = view_cache["obj_latents"].expand(
                        batch_size, -1, -1
                    )
                    pred_recon = torch.einsum(
                        "bkn,bkd->bnd",
                        pred_probs.to(obj_latents.dtype),
                        obj_latents,
                    )
                    warped_recon, warped_last_masks = (
                        self._warped_recon_from_masks(
                            view_cache, warped_masks
                        )
                    )
                    timing["evaluate_pred_extract_s"] += float(
                        time.perf_counter() - t_extract_start
                    )
                per_view_pred_recon[view_name] = pred_recon
                per_view_pred_probs[view_name] = pred_probs
                pred_by_view[view_name] = pred_recon
                pred_by_view[f"{view_name}__mask"] = pred_probs
                if warped_recon is not None:
                    warp_by_view[view_name] = warped_recon
                    warp_by_view[f"{view_name}__mask"] = warped_last_masks
                # Store compact per-candidate history selection metadata next
                # to the action cache. This is intentionally CPU-only and small.
                action_by_view[f"{view_name}__history"] = candidate_history_debug

            if active_prediction_streams:
                for prediction_stream in active_prediction_streams:
                    default_stream.wait_stream(prediction_stream)
                timing["fixed_view_parallel_pairs"] += 1
            elif fixed_pair_eligible:
                # The first pair stays on the default stream so all lazy RoPE
                # caches are materialized before two streams read them.
                self._fixed_view_parallel_cache_warmed = True
                timing["fixed_view_parallel_warmups"] += 1

            timing["evaluate_view_prep_s"] += float(time.perf_counter() - t_view_prep_start)
            t_loss_start = time.perf_counter()
            per_view_losses_np = {}
            per_view_shared_mass_np = {}
            per_view_gripper_slot_loss_np = {}
            per_view_other_slot_loss_np = {}
            per_view_gripper_mixed_loss_np = {}
            per_view_without_fixed_slot_loss_np = {}
            for view_name in eval_view_names:
                slot_transport_loss_view = None
                gripper_slot_idx = self.gripper_slot_index_by_view.get(view_name)
                fixed_slot_idx = self.fixed_slot_index_by_view.get(view_name, gripper_slot_idx)
                goal_mask_probs = predictor_cache_by_view[view_name]["goal_mask_probs"].unsqueeze(0).expand_as(
                    per_view_pred_probs[view_name]
                )
                mask_l1_loss_view = torch.mean(
                    torch.abs(per_view_pred_probs[view_name] - goal_mask_probs), dim=(1, 2)
                )
                # Per-slot transport diagnostics are also used as wrist-specific
                # ranking overrides. Do not compute or expose them when A2 has
                # disabled UOT; otherwise the nominal mask-L1 branch would still
                # rank some pose candidates with a transport loss.
                if self.transport_loss_mix > 0.0 and (
                    gripper_slot_idx is not None or fixed_slot_idx is not None
                ):
                    slot_transport_loss_view = self._sinkhorn_mask_transport_loss_per_slot(
                        per_view_pred_probs[view_name],
                        goal_mask_probs,
                        spatial_size=self.crop_size // self.patch_size,
                        area_normalizer=self._slot_area_normalizer_for_view(view_name, device=self.device),
                    )
                    if gripper_slot_idx is not None and 0 <= int(gripper_slot_idx) < int(slot_transport_loss_view.shape[1]):
                        gripper_slot_loss = slot_transport_loss_view[:, int(gripper_slot_idx)]
                        if slot_transport_loss_view.shape[1] > 1:
                            other_slot_loss = (
                                slot_transport_loss_view.sum(dim=1) - gripper_slot_loss
                            ) / float(slot_transport_loss_view.shape[1] - 1)
                        else:
                            other_slot_loss = gripper_slot_loss
                        per_view_gripper_slot_loss_np[view_name] = gripper_slot_loss.detach().cpu().numpy().astype(np.float32)
                        per_view_other_slot_loss_np[view_name] = other_slot_loss.detach().cpu().numpy().astype(np.float32)
                        per_view_gripper_mixed_loss_np[view_name] = (
                            0.5 * per_view_gripper_slot_loss_np[view_name]
                            + 0.5 * per_view_other_slot_loss_np[view_name]
                        ).astype(np.float32)
                    if fixed_slot_idx is not None and 0 <= int(fixed_slot_idx) < int(slot_transport_loss_view.shape[1]):
                        fixed_slot_loss = slot_transport_loss_view[:, int(fixed_slot_idx)]
                        if slot_transport_loss_view.shape[1] > 1:
                            without_fixed_slot_loss = (
                                slot_transport_loss_view.sum(dim=1) - fixed_slot_loss
                            ) / float(slot_transport_loss_view.shape[1] - 1)
                        else:
                            without_fixed_slot_loss = fixed_slot_loss
                        per_view_without_fixed_slot_loss_np[view_name] = (
                            without_fixed_slot_loss.detach().cpu().numpy().astype(np.float32)
                        )
                if self.transport_loss_mix > 0.0:
                    if slot_transport_loss_view is None:
                        transport_loss_view = self._sinkhorn_mask_transport_loss(
                            per_view_pred_probs[view_name],
                            goal_mask_probs,
                            spatial_size=self.crop_size // self.patch_size,
                            slot_weights=per_view_slot_weights[view_name],
                            area_normalizer=self._slot_area_normalizer_for_view(view_name, device=self.device),
                        )
                    else:
                        # The per-slot diagnostic call above already computed
                        # the complete UOT matrix. Re-aggregate that exact
                        # result with the same weights instead of running the
                        # identical Sinkhorn solve a second time.
                        transport_loss_view = self._weighted_slot_loss_mean(
                            slot_transport_loss_view,
                            slot_weights=per_view_slot_weights[view_name],
                        ).float()
                        timing["transport_solves_reused"] += 1
                    per_view_shared_mass_np[view_name] = np.ones(
                        per_view_pred_recon[view_name].shape[0],
                        dtype=np.float32,
                    )
                    loss_view = (
                        (1.0 - self.transport_loss_mix) * mask_l1_loss_view
                        + self.transport_loss_mix * transport_loss_view
                    )
                else:
                    loss_view = mask_l1_loss_view
                    per_view_shared_mass_np[view_name] = np.ones(
                        per_view_pred_recon[view_name].shape[0],
                        dtype=np.float32,
                    )
                per_view_losses_np[view_name] = loss_view.detach().cpu().numpy().astype(np.float32)
            timing["evaluate_loss_total_s"] += float(time.perf_counter() - t_loss_start)
            timing["cem_eval_total_s"] += float(time.perf_counter() - t_eval_start)
            return (
                per_view_losses_np,
                per_view_gripper_slot_loss_np,
                per_view_other_slot_loss_np,
                per_view_gripper_mixed_loss_np,
                per_view_without_fixed_slot_loss_np,
                pred_by_view,
                warp_by_view,
                next_state_by_view,
                action_by_view,
                target_robot_pose6_all,
                per_view_shared_mass_np,
            )

        samples = int(self.mpc_args["samples"])
        topk = max(1, min(samples, int(self.mpc_args["topk"])))
        wrist_rerank_topk = max(1, min(samples, int(self.mpc_args.get("wrist_rerank_topk", topk))))
        cem_steps = int(self.mpc_args["cem_steps"])
        mean_delta_pose6_cem = mean_delta_pose6.copy()
        std_delta_pose6_cem = np.array(
            [
                action_limits["translation_std_xyz"][0],
                action_limits["translation_std_xyz"][1],
                action_limits["translation_std_xyz"][2],
                action_limits["rotation_std"],
                action_limits["rotation_std"],
                action_limits["rotation_std"],
            ],
            dtype=np.float32,
        )
        mean_gripper_delta_cem = mean_gripper_delta
        std_gripper_delta_cem = self.gripper_delta_std if self.optimize_gripper_action else 0.0

        best_pred_by_view = None
        best_pred_mask_by_view = None
        best_warp_by_view = {}
        best_warp_mask_by_view = {}
        best_next_state_by_view = None
        best_action_by_view = None
        best_target_robot_pose6 = current_robot_pose6.astype(np.float32).copy()
        candidate_diagnostics = []
        loss_gate_history = []
        steering_history = []
        skipped_gripper_cem = False
        gripper_cem_skip_reason = None
        cem_phase_outputs = []

        def _motion_skip_label(skip_by_insufficient_window, skip_by_xyz, skip_by_rot):
            if skip_by_insufficient_window:
                return "insufficient_observed_window"
            parts = []
            if skip_by_xyz:
                parts.append("xyz")
            if skip_by_rot:
                parts.append("rot")
            if parts:
                return "observed_motion_large_" + "_".join(parts)
            return "observed_motion_large"

        def _motion_skip_detail(skip_by_insufficient_window, skip_by_xyz, skip_by_rot, translation_threshold, rotation_threshold):
            return {
                "skip_by_insufficient_window": bool(skip_by_insufficient_window),
                "skip_by_xyz": bool(skip_by_xyz),
                "skip_by_rotation": bool(skip_by_rot),
                "observed_translation_norm": float(observed_translation_norm),
                "observed_rotation_norm": float(observed_rotation_norm),
                "translation_threshold": float(translation_threshold),
                "rotation_threshold": float(rotation_threshold),
                "observed_motion_metric": "net_pose_delta_over_window_per_step",
                "observed_window_size": int(observed_window_step_count),
                "required_observed_window_size": int(self.observed_motion_window),
                "observed_motion_window_ready": bool(observed_motion_window_ready),
                "observed_window_delta6_robot": observed_window_delta6.astype(np.float32).tolist(),
                "step_mean_translation_norm": float(observed_step_mean_translation_norm),
                "step_mean_rotation_norm": float(observed_step_mean_rotation_norm),
            }

        if self.factorized_pose_cem:
            steps_per_phase = self.factorized_cem_steps_per_phase or cem_steps
            skip_gripper_after_grasp = "GRASP" in getattr(self, "current_completed_stages", set())
            gripper_phase_enabled = bool(self.optimize_gripper_action and not skip_gripper_after_grasp)
            current_planning_phase = str(getattr(self, "slot_planning_phase_cursor", "pose"))
            if current_planning_phase in {"fixed_pose", "wrist_pose"}:
                current_planning_phase = "pose"
            valid_planning_phases = {"pose", "wrist_gripper"}
            if current_planning_phase not in valid_planning_phases:
                current_planning_phase = "pose"
            # Gripper search runs on every outer planning step before GRASP.
            # After GRASP, client-side post-grasp hold/tighten owns the
            # gripper to avoid reopening or weakening the grasp.
            effective_planning_phase = "pose"
            phase_redirect_reason = None
            phase_plan = []
            if planning_request in {"joint", "pose_only"}:
                warmup_anchor_steering_active = bool(
                    pose_view_switch.get("warmup_fixed_anchor_steering_active", False)
                )
                phase_plan.append(
                    {
                        "name": "fixed_pose_coarse",
                        "dims": np.arange(6, dtype=np.int64),
                        "view_loss_mode": "fixed_only",
                        "primary_view_loss_mode": "fixed_only",
                        "eval_view_names": list(fixed_view_names),
                        "rerank_view_names": [],
                        "rerank_view_loss_mode": None,
                        "rerank_topk": 0,
                        "sample_gripper": False,
                        "num_steps": int(steps_per_phase),
                        "sample_scale": 1.0,
                        "planning_cursor": current_planning_phase,
                        "effective_planning_cursor": "fixed_pose_coarse",
                        "fixed_view_stagnation_triggered": False,
                        "warmup_anchor_steering_active": bool(warmup_anchor_steering_active),
                        "pose_eval_uses_fixed_views": True,
                        "pose_eval_uses_wrist_only": False,
                        "pose_view_switch": dict(pose_view_switch),
                        "rotation_delta_frame": "ee",
                        "updates_execution_target": False,
                        "next_cursor_on_success": None,
                    }
                )
                phase_plan.append(
                    {
                        "name": "wrist_pose_refine",
                        "dims": np.arange(6, dtype=np.int64),
                        "view_loss_mode": "wrist_only",
                        "primary_view_loss_mode": "wrist_only",
                        "eval_view_names": list(wrist_view_names),
                        "rerank_view_names": [],
                        "rerank_view_loss_mode": None,
                        "rerank_topk": 0,
                        "sample_gripper": False,
                        "num_steps": int(steps_per_phase),
                        "sample_scale": 1.0,
                        "planning_cursor": current_planning_phase,
                        "effective_planning_cursor": "wrist_pose_refine",
                        "fixed_view_stagnation_triggered": False,
                        "warmup_anchor_steering_active": False,
                        "pose_eval_uses_fixed_views": False,
                        "pose_eval_uses_wrist_only": True,
                        "pose_view_switch": dict(pose_view_switch),
                        "rotation_delta_frame": "ee",
                        "momentum_mean": float(self.wrist_refine_momentum_mean),
                        "momentum_std": float(self.wrist_refine_momentum_std),
                        "top_pose_log_count": 5,
                        "updates_execution_target": True,
                        "next_cursor_on_success": "pose",
                    }
                )
            if planning_request in {"joint", "gripper_only"} and gripper_phase_enabled:
                current_goal_stage = None
                if isinstance(current_goal_target, dict):
                    current_goal_stage = current_goal_target.get("stage")
                phase_plan.append(
                    {
                        "name": "wrist_gripper",
                        "dims": np.array([], dtype=np.int64),
                        "view_loss_mode": "wrist_only",
                        "eval_view_names": list(wrist_view_names),
                        "sample_gripper": True,
                        "num_candidates": int(self.gripper_samples),
                        "num_steps": int(self.gripper_cem_steps),
                        "gripper_search_mode": "uniform_absolute",
                        "sample_scale": 1.0,
                        "z_offset": 0.0,
                        "gripper_downpress_max_z": float(self.gripper_downpress_max_z),
                        "z_offset_stage": current_goal_stage,
                        "planning_cursor": current_planning_phase,
                        "effective_planning_cursor": "wrist_gripper",
                        "rotation_delta_frame": "parent",
                        "updates_execution_target": True,
                        "next_cursor_on_success": "pose",
                    }
                )
            elif planning_request in {"joint", "gripper_only"} and self.optimize_gripper_action and skip_gripper_after_grasp:
                loss_gate_history.append(
                    {
                        "cem_phase": "wrist_gripper",
                        "phase_idx": len(phase_plan),
                        "cem_iter": None,
                        "loss_selection_mode": "skipped_gripper_after_grasp_hold_current",
                        "completed_stages": sorted(getattr(self, "current_completed_stages", set())),
                    }
                )
                cem_phase_outputs.append(
                    {
                        "cem_phase": "wrist_gripper",
                        "phase_idx": len(phase_plan),
                        "planning_cursor": current_planning_phase,
                        "effective_planning_cursor": "wrist_gripper",
                        "skipped": True,
                        "skip_reason": "grasp_already_reached_hold_current",
                        "completed_stages": sorted(getattr(self, "current_completed_stages", set())),
                        "target_robot_pose6": best_target_robot_pose6.astype(np.float32).tolist(),
                        "delta_pose6_robot": mean_delta_pose6_cem.astype(np.float32).tolist(),
                        "gripper_delta": float(mean_gripper_delta_cem),
                    }
                )
            if not phase_plan:
                phase_plan = [
                    {
                        "name": "noop",
                        "dims": np.array([], dtype=np.int64),
                        "view_loss_mode": None,
                        "eval_view_names": list(view_names),
                        "sample_gripper": False,
                        "num_steps": 1,
                    }
                ]
            next_planning_phase_cursor = "pose"
        else:
            current_planning_phase = "joint"
            effective_planning_phase = "joint"
            phase_redirect_reason = None
            next_planning_phase_cursor = "joint"
            phase_plan = [
                {
                    "name": "joint",
                    "dims": np.arange(6, dtype=np.int64),
                    "view_loss_mode": None,
                    "eval_view_names": list(view_names),
                    "sample_gripper": bool(self.optimize_gripper_action),
                    "num_steps": int(cem_steps),
                }
            ]

        fixed_view_fallback_payload = None
        fixed_view_fallback_debug = None
        for phase_idx, phase in enumerate(phase_plan):
            phase_best_loss = float("inf")
            phase_best_payload = None
            samples = max(1, int(phase.get("num_candidates", self.mpc_args["samples"])))
            phase_name = phase["name"]
            phase_dims = phase["dims"]
            phase_view_loss_mode = phase["view_loss_mode"]
            phase_eval_view_names = [view_name for view_name in phase.get("eval_view_names", view_names) if view_name in view_names]
            if not phase_eval_view_names:
                phase_eval_view_names = list(view_names)
            phase_sample_gripper = bool(phase["sample_gripper"])
            phase_sample_scale = float(np.clip(phase.get("sample_scale", 1.0), 0.0, 1.0))
            phase_sample_scale_by_dim = phase.get("sample_scale_by_dim")
            if phase_sample_scale_by_dim is not None:
                phase_sample_scale_by_dim = np.asarray(phase_sample_scale_by_dim, dtype=np.float32).reshape(-1)
                if phase_sample_scale_by_dim.shape[0] != phase_dims.size:
                    raise ValueError(
                        f"phase {phase_name} sample_scale_by_dim has {phase_sample_scale_by_dim.shape[0]} "
                        f"values for {phase_dims.size} dims"
                    )
                phase_sample_scale_by_dim = np.clip(phase_sample_scale_by_dim, 0.0, 1.0)
            else:
                phase_sample_scale_by_dim = np.full(phase_dims.size, phase_sample_scale, dtype=np.float32)
            phase_steps = max(1, int(phase["num_steps"]))
            phase_rotation_delta_frame = str(phase.get("rotation_delta_frame", "parent"))
            primary_view_loss_mode = str(phase.get("primary_view_loss_mode", phase_view_loss_mode))
            rerank_view_loss_mode = str(phase.get("rerank_view_loss_mode", "wrist_only"))
            phase_updates_execution_target = bool(phase.get("updates_execution_target", True))
            phase_momentum_mean = float(
                np.clip(phase.get("momentum_mean", self.mpc_args["momentum_mean"]), 0.0, 1.0)
            )
            phase_momentum_std = float(
                np.clip(phase.get("momentum_std", self.mpc_args["momentum_std"]), 0.0, 1.0)
            )
            topk_then_rerank_phase = phase_view_loss_mode == "topk_then_complement_rerank"
            primary_is_fixed_view = primary_view_loss_mode == "fixed_only"
            primary_fixed_phase_for_steering = primary_is_fixed_view
            anchor_steering_active = bool(
                primary_fixed_phase_for_steering
                and phase_dims.size > 0
                and warmup_fixed_anchor_steering.get("active", False)
            )
            # Keep only warmup fixed-anchor steering. Historical pose-memory steering
            # remains disabled because it can inject weak or stale directions into CEM.
            steering_info = None
            steering_delta_center = None
            steering_force_count = 0
            if anchor_steering_active:
                anchor_delta6 = np.asarray(warmup_fixed_anchor_steering["delta6_robot"], dtype=np.float32)
                anchor_delta6 = self._cap_pose_delta_norm(
                    anchor_delta6[None, :],
                    action_limits["translation_cap_xyz"],
                    action_limits["rotation_cap"],
                )[0].astype(np.float32)
                mean_delta_pose6_cem[phase_dims] = anchor_delta6[phase_dims]
                steering_delta_center = mean_delta_pose6_cem.copy()
                steering_force_count = 1
                steering_history.append(
                    {
                        "cem_phase": phase_name,
                        "phase_idx": int(phase_idx),
                        "selected": True,
                        "source": "warmup_fixed_anchor",
                        "phase_view_names": list(phase_eval_view_names),
                        "anchor": dict(warmup_fixed_anchor_steering),
                        "steering_delta_center": steering_delta_center.astype(np.float32).tolist(),
                        "forced_candidate_count": int(steering_force_count),
                    }
                )
            if steering_info is not None:
                steering_delta_center = mean_delta_pose6_cem.copy()
                steering_direction6 = steering_info["steering_direction6"].astype(np.float32)
                if phase_dims.size > 0:
                    if np.any(phase_dims < 3):
                        xyz_dims = phase_dims[phase_dims < 3]
                        steering_delta_center[xyz_dims] = (
                            mean_delta_pose6_cem[xyz_dims]
                            + steering_direction6[xyz_dims]
                            * std_delta_pose6_cem[xyz_dims]
                            * phase_sample_scale_by_dim[np.searchsorted(phase_dims, xyz_dims)]
                        )
                    if np.any(phase_dims >= 3):
                        rot_dims = phase_dims[phase_dims >= 3]
                        steering_delta_center[rot_dims] = (
                            mean_delta_pose6_cem[rot_dims]
                            + steering_direction6[rot_dims]
                            * std_delta_pose6_cem[rot_dims]
                            * phase_sample_scale_by_dim[np.searchsorted(phase_dims, rot_dims)]
                        )
                    steering_delta_center = self._cap_pose_delta_norm(
                        steering_delta_center[None, :],
                        action_limits["translation_cap_xyz"],
                        action_limits["rotation_cap"],
                    )[0].astype(np.float32)
                steering_history.append(
                    {
                        "cem_phase": phase_name,
                        "phase_idx": int(phase_idx),
                        "selected": True,
                        "score": float(steering_info["score"]),
                        "num_proposals": int(steering_info["num_proposals"]),
                        "phase_view_names": list(steering_info["phase_view_names"]),
                        "raw_losses": dict(steering_info["raw_losses"]),
                        "view_weights": dict(steering_info["view_weights"]),
                        "source": steering_info["entry"].get("source"),
                        "stage": steering_info["entry"].get("stage"),
                        "memory_step": steering_info["entry"].get("step"),
                        "memory_pose6_robot": np.asarray(steering_info["entry"]["pose6_robot"], dtype=np.float32).tolist(),
                        "steering_direction6_robot": steering_direction6.astype(np.float32).tolist(),
                        "steering_delta_center": None
                        if steering_delta_center is None
                        else steering_delta_center.astype(np.float32).tolist(),
                        "steering_gripper_center": None,
                        "dim_stats": steering_info.get("dim_stats", {}),
                    }
                )
            elif not anchor_steering_active:
                steering_history.append(
                    {
                        "cem_phase": phase_name,
                        "phase_idx": int(phase_idx),
                        "selected": False,
                        "disabled_reason": "pose_memory_steering_disabled_keep_warmup_fixed_anchor_only",
                        "phase_view_names": list(phase_eval_view_names),
                        "memory_sizes": {
                            view_name: int(len(self.pose_memory_by_view.get(view_name, [])))
                            for view_name in phase_eval_view_names
                        },
                    }
                )
            if phase_sample_gripper:
                phase_z_offset = float(phase.get("z_offset", 0.0))
                if phase_z_offset != 0.0:
                    mean_delta_pose6_cem[2] = float(mean_delta_pose6_cem[2] + phase_z_offset)
                    loss_gate_history.append(
                        {
                            "cem_phase": phase_name,
                            "phase_idx": int(phase_idx),
                            "cem_iter": None,
                            "loss_selection_mode": "applied_gripper_z_offset",
                            "z_offset": float(phase_z_offset),
                            "mean_delta_pose6_after_offset": mean_delta_pose6_cem.astype(np.float32).tolist(),
                        }
                    )
            phase_search_scale_info = dict(fixed_loss_search_scale_info)
            phase_top_pose_candidates = []
            phase_goal_distance_filter_debug = dict(goal_distance_filter)
            for cem_iter in range(phase_steps):
                t_sampling_start = time.perf_counter()
                sample_sources = np.full(samples, "local_cem", dtype=object)
                delta_pose6_batch = np.repeat(mean_delta_pose6_cem[None, :], samples, axis=0).astype(np.float32)
                iter_search_scale = (
                    float(phase_search_scale_info["scale"])
                    if primary_is_fixed_view and phase_dims.size > 0 and not phase_sample_gripper
                    else 1.0
                )
                iter_phase_sample_scale_by_dim = (phase_sample_scale_by_dim * iter_search_scale).astype(np.float32)
                iter_translation_cap_xyz = action_limits["translation_cap_xyz"].astype(np.float32) * iter_search_scale
                iter_rotation_cap = float(action_limits["rotation_cap"]) * iter_search_scale
                if phase_dims.size > 0:
                    sampled_dims = (
                        np.random.randn(samples, phase_dims.size).astype(np.float32)
                        * (std_delta_pose6_cem[phase_dims][None, :] * iter_phase_sample_scale_by_dim[None, :])
                        + mean_delta_pose6_cem[phase_dims][None, :]
                    )
                    if np.any(iter_phase_sample_scale_by_dim < 1.0):
                        local_caps = []
                        for dim, dim_scale in zip(phase_dims, iter_phase_sample_scale_by_dim):
                            if int(dim) < 3:
                                local_caps.append(float(action_limits["translation_cap_xyz"][int(dim)]) * float(dim_scale))
                            else:
                                local_caps.append(float(action_limits["rotation_cap"]) * float(dim_scale))
                        local_caps = np.asarray(local_caps, dtype=np.float32)
                        sampled_dims = np.clip(
                            sampled_dims,
                            mean_delta_pose6_cem[phase_dims][None, :] - local_caps[None, :],
                            mean_delta_pose6_cem[phase_dims][None, :] + local_caps[None, :],
                        )
                    delta_pose6_batch[:, phase_dims] = sampled_dims
                    if steering_delta_center is not None and (self.steering_sample_fraction > 0.0 or steering_force_count > 0):
                        steer_count = max(
                            int(steering_force_count),
                            int(round(float(samples) * self.steering_sample_fraction)),
                        )
                        steer_count = max(0, min(samples - 1, steer_count))
                        if steer_count > 0:
                            steer_slice = slice(1, 1 + steer_count)
                            delta_pose6_batch[steer_slice, phase_dims] = (
                                np.random.randn(steer_count, phase_dims.size).astype(np.float32)
                                * (std_delta_pose6_cem[phase_dims][None, :] * iter_phase_sample_scale_by_dim[None, :])
                                + steering_delta_center[phase_dims][None, :]
                            )
                            sample_sources[steer_slice] = "steering"
                delta_pose6_batch = self._cap_pose_delta_norm(
                    delta_pose6_batch,
                    iter_translation_cap_xyz,
                    iter_rotation_cap,
                )
                if phase_sample_gripper:
                    gripper_closed_upper = min(
                        self.gripper_position_max,
                        max(self.gripper_position_min, self.gripper_position_max - self.gripper_min_open_gap),
                    )
                    target_gripper_batch = np.linspace(
                        self.gripper_position_min,
                        gripper_closed_upper,
                        samples,
                        dtype=np.float32,
                    )
                    gripper_delta_batch = (
                        target_gripper_batch - float(ref_state7_np[6])
                    ).astype(np.float32)
                    # Gripper scoring should isolate open/close appearance.
                    # Downpress is an execution-side heuristic and would bias
                    # the visual loss against closed-gripper candidates.
                    gripper_downpress_z_batch = np.zeros(samples, dtype=np.float32)
                    sample_sources[:] = "uniform_gripper"
                else:
                    if self.optimize_gripper_action:
                        gripper_delta_batch = np.full(samples, mean_gripper_delta_cem, dtype=np.float32)
                    else:
                        fixed_gripper_delta = self.fixed_gripper_target - float(ref_state7_np[6])
                        gripper_delta_batch = np.full(samples, fixed_gripper_delta, dtype=np.float32)
                if samples > 0 and phase_dims.size > 0 and not self.always_execute_best_candidate:
                    # Always include a no-op for the current phase. This gives
                    # CEM a stable "do not change this component" option once a
                    # stage is already visually close to its target.
                    delta_pose6_batch[0, phase_dims] = mean_delta_pose6_cem[phase_dims]
                    sample_sources[0] = "noop"
                timing["cem_sampling_s"] += float(time.perf_counter() - t_sampling_start)

                per_view_loss_cache = []
                pred_cache = []
                warp_cache = []
                state_cache = []
                action_cache = []
                target_robot_cache = []
                shared_mass_cache = []
                gripper_slot_loss_cache = []
                other_slot_loss_cache = []
                gripper_mixed_loss_cache = []
                without_fixed_slot_loss_cache = []
                for start_idx in range(0, samples, self.mpc_eval_batch_size):
                    end_idx = min(start_idx + self.mpc_eval_batch_size, samples)
                    (
                        per_view_losses_chunk,
                        gripper_slot_losses_chunk,
                        other_slot_losses_chunk,
                        gripper_mixed_losses_chunk,
                        without_fixed_slot_losses_chunk,
                        pred_chunk,
                        warp_chunk,
                        state_chunk,
                        action_chunk,
                        target_robot_chunk,
                        shared_mass_chunk,
                    ) = evaluate_pose_delta_batch(
                        delta_pose6_batch[start_idx:end_idx],
                        gripper_delta_batch[start_idx:end_idx],
                        eval_view_names=phase_eval_view_names,
                        rotation_delta_frame=phase_rotation_delta_frame,
                    )
                    per_view_loss_cache.append(per_view_losses_chunk)
                    gripper_slot_loss_cache.append(gripper_slot_losses_chunk)
                    other_slot_loss_cache.append(other_slot_losses_chunk)
                    gripper_mixed_loss_cache.append(gripper_mixed_losses_chunk)
                    without_fixed_slot_loss_cache.append(without_fixed_slot_losses_chunk)
                    pred_cache.append(pred_chunk)
                    warp_cache.append(warp_chunk)
                    state_cache.append(state_chunk)
                    action_cache.append(action_chunk)
                    target_robot_cache.append(target_robot_chunk)
                    shared_mass_cache.append(shared_mass_chunk)

                t_sort_update_start = time.perf_counter()
                per_view_losses_iter = {
                    view_name: np.concatenate([chunk[view_name] for chunk in per_view_loss_cache], axis=0)
                    for view_name in phase_eval_view_names
                }
                gripper_slot_losses_iter = {
                    view_name: np.concatenate([chunk[view_name] for chunk in gripper_slot_loss_cache if view_name in chunk], axis=0)
                    for view_name in phase_eval_view_names
                    if any(view_name in chunk for chunk in gripper_slot_loss_cache)
                }
                other_slot_losses_iter = {
                    view_name: np.concatenate([chunk[view_name] for chunk in other_slot_loss_cache if view_name in chunk], axis=0)
                    for view_name in phase_eval_view_names
                    if any(view_name in chunk for chunk in other_slot_loss_cache)
                }
                gripper_mixed_losses_iter = {
                    view_name: np.concatenate([chunk[view_name] for chunk in gripper_mixed_loss_cache if view_name in chunk], axis=0)
                    for view_name in phase_eval_view_names
                    if any(view_name in chunk for chunk in gripper_mixed_loss_cache)
                }
                without_fixed_slot_losses_iter = {
                    view_name: np.concatenate([chunk[view_name] for chunk in without_fixed_slot_loss_cache if view_name in chunk], axis=0)
                    for view_name in phase_eval_view_names
                    if any(view_name in chunk for chunk in without_fixed_slot_loss_cache)
                }
                if (
                    phase_name != "wrist_gripper"
                    and "wrist_image_left" in phase_eval_view_names
                    and "wrist_image_left" in without_fixed_slot_losses_iter
                ):
                    per_view_losses_iter["wrist_image_left"] = without_fixed_slot_losses_iter["wrist_image_left"]
                per_view_shared_mass_iter = {
                    view_name: np.concatenate([chunk[view_name] for chunk in shared_mass_cache], axis=0)
                    for view_name in phase_eval_view_names
                }
                target_robot_pose6_iter = np.concatenate(target_robot_cache, axis=0)
                topk_then_rerank = phase_view_loss_mode == "topk_then_complement_rerank"
                is_final_cem_iter = cem_iter == phase_steps - 1
                losses, loss_gate_info = compose_gated_candidate_losses(
                    per_view_losses_iter,
                    delta_pose6_batch,
                    forced_view_loss_mode=primary_view_loss_mode if topk_then_rerank else phase_view_loss_mode,
                )
                if not phase_sample_gripper and phase_dims.size > 0:
                    losses, phase_goal_distance_filter_debug = self._apply_goal_distance_filter(
                        losses,
                        target_robot_pose6_iter,
                        goal_distance_filter,
                    )
                fixed_selection_losses = losses.copy()
                fixed_sorted_indices = np.argsort(fixed_selection_losses)
                fixed_top_indices = fixed_sorted_indices[:topk]
                next_phase_search_scale_info = (
                    self._fixed_loss_to_search_scale(
                        float(np.min(fixed_selection_losses)),
                        goal_idx=int(self.current_goal_idx),
                    )
                    if primary_is_fixed_view
                    else phase_search_scale_info
                )
                phase_rerank_topk = max(1, min(samples, int(phase.get("rerank_topk", wrist_rerank_topk))))
                wrist_rerank_indices = fixed_sorted_indices[:phase_rerank_topk]
                rerank_view_names = [
                    view_name for view_name in phase.get("rerank_view_names", []) if view_name in view_names
                ]
                rerank_losses_full = np.full(samples, np.nan, dtype=np.float32)
                rerank_shared_mass_full = np.full(samples, np.nan, dtype=np.float32)
                rerank_per_view_losses_iter = {}
                rerank_pred_cache = []
                rerank_warp_cache = []
                rerank_state_cache = []
                rerank_action_cache = []
                rerank_target_robot_cache = []
                rerank_shared_mass_cache = []
                rerank_best_local_idx = None
                if topk_then_rerank and is_final_cem_iter and rerank_view_names and wrist_rerank_indices.size > 0:
                    for start_idx in range(0, wrist_rerank_indices.size, self.mpc_eval_batch_size):
                        end_idx = min(start_idx + self.mpc_eval_batch_size, wrist_rerank_indices.size)
                        candidate_indices = wrist_rerank_indices[start_idx:end_idx]
                        (
                            rerank_losses_chunk,
                            _rerank_gripper_slot_losses_chunk,
                            _rerank_other_slot_losses_chunk,
                            _rerank_gripper_mixed_losses_chunk,
                            _rerank_without_fixed_slot_losses_chunk,
                            rerank_pred_chunk,
                            rerank_warp_chunk,
                            rerank_state_chunk,
                            rerank_action_chunk,
                            rerank_target_robot_chunk,
                            rerank_shared_mass_chunk,
                        ) = evaluate_pose_delta_batch(
                            delta_pose6_batch[candidate_indices],
                            gripper_delta_batch[candidate_indices],
                            eval_view_names=rerank_view_names,
                            rotation_delta_frame=phase_rotation_delta_frame,
                        )
                        if (
                            phase_name != "wrist_gripper"
                            and "wrist_image_left" in rerank_view_names
                            and "wrist_image_left" in _rerank_without_fixed_slot_losses_chunk
                        ):
                            rerank_losses_chunk["wrist_image_left"] = _rerank_without_fixed_slot_losses_chunk["wrist_image_left"]
                        rerank_pred_cache.append(rerank_pred_chunk)
                        rerank_warp_cache.append(rerank_warp_chunk)
                        rerank_state_cache.append(rerank_state_chunk)
                        rerank_action_cache.append(rerank_action_chunk)
                        rerank_target_robot_cache.append(rerank_target_robot_chunk)
                        rerank_shared_mass_cache.append(rerank_shared_mass_chunk)
                        for view_name, values in rerank_losses_chunk.items():
                            rerank_per_view_losses_iter.setdefault(view_name, []).append(values)
                    rerank_per_view_losses_iter = {
                        view_name: np.concatenate(chunks, axis=0)
                        for view_name, chunks in rerank_per_view_losses_iter.items()
                    }
                    rerank_per_view_shared_mass_iter = {
                        view_name: np.concatenate(
                            [chunk[view_name] for chunk in rerank_shared_mass_cache if view_name in chunk],
                            axis=0,
                        )
                        for view_name in rerank_view_names
                        if any(view_name in chunk for chunk in rerank_shared_mass_cache)
                    }
                    rerank_losses, rerank_loss_gate_info = compose_gated_candidate_losses(
                        rerank_per_view_losses_iter,
                        delta_pose6_batch[wrist_rerank_indices],
                        forced_view_loss_mode=rerank_view_loss_mode,
                    )
                    if not phase_sample_gripper and phase_dims.size > 0:
                        rerank_target_robot_pose6_all = np.concatenate(
                            rerank_target_robot_cache, axis=0
                        ).astype(np.float32)
                        rerank_losses, phase_goal_distance_filter_debug = self._apply_goal_distance_filter(
                            rerank_losses,
                            rerank_target_robot_pose6_all,
                            goal_distance_filter,
                        )
                    rerank_losses_full[wrist_rerank_indices] = rerank_losses
                    if rerank_per_view_shared_mass_iter:
                        rerank_shared_mass = np.mean(
                            np.stack(
                                [
                                    rerank_per_view_shared_mass_iter[view_name]
                                    for view_name in rerank_view_names
                                    if view_name in rerank_per_view_shared_mass_iter
                                ],
                                axis=0,
                            ),
                            axis=0,
                        ).astype(np.float32)
                        rerank_shared_mass_full[wrist_rerank_indices] = rerank_shared_mass
                    rerank_best_local_idx = int(np.argmin(rerank_losses))
                    best_idx = int(wrist_rerank_indices[rerank_best_local_idx])
                    # Only primary-view top-k candidates are eligible for execution;
                    # complement-view loss is used solely to rerank within that set.
                    losses = np.full(samples, 1.0e9, dtype=np.float32)
                    losses[wrist_rerank_indices] = rerank_losses.astype(np.float32)
                    if primary_view_loss_mode == "fixed_only":
                        fixed_view_loss_full = fixed_selection_losses
                        wrist_view_loss_full = rerank_losses_full
                    elif primary_view_loss_mode == "wrist_only":
                        fixed_view_loss_full = rerank_losses_full
                        wrist_view_loss_full = fixed_selection_losses
                    else:
                        fixed_view_loss_full = loss_gate_info["fixed_view_loss"]
                        wrist_view_loss_full = loss_gate_info["wrist_view_loss"]
                    all_view_loss_full = np.nanmean(
                        np.stack([fixed_view_loss_full, wrist_view_loss_full], axis=0),
                        axis=0,
                    ).astype(np.float32)
                    finite_fixed_losses = fixed_view_loss_full[np.isfinite(fixed_view_loss_full)]
                    min_fixed_view_loss = (
                        float(np.min(finite_fixed_losses))
                        if finite_fixed_losses.size > 0
                        else float(np.min(loss_gate_info["fixed_view_loss"]))
                    )
                    loss_gate_info = {
                        "loss_selection_mode": "topk_then_complement_rerank",
                        "primary_view_loss_mode": primary_view_loss_mode,
                        "rerank_view_loss_mode": rerank_view_loss_mode,
                        "current_fixed_view_loss": float(current_fixed_view_loss),
                        "current_all_view_loss": float(
                            np.mean([current_loss_by_view[view_name] for view_name in view_names])
                        ),
                        "current_wrist_view_loss": float(
                            np.mean([current_loss_by_view[view_name] for view_name in wrist_view_names])
                        )
                        if wrist_view_names
                        else float(current_fixed_view_loss),
                        "current_selected_view_loss": (
                            float(np.mean([current_loss_by_view[view_name] for view_name in wrist_view_names]))
                            if rerank_view_loss_mode == "wrist_only" and wrist_view_names
                            else (
                                float(np.mean([current_loss_by_view[view_name] for view_name in fixed_view_names]))
                                if rerank_view_loss_mode == "fixed_only" and fixed_view_names
                                else float(
                                    np.mean([current_loss_by_view[view_name] for view_name in view_names])
                                )
                            )
                        ),
                        "min_candidate_fixed_view_loss": min_fixed_view_loss,
                        "fixed_view_improvement": float(current_fixed_view_loss - min_fixed_view_loss),
                        "all_view_loss": all_view_loss_full,
                        "fixed_view_loss": fixed_view_loss_full,
                        "wrist_view_loss": wrist_view_loss_full,
                        "primary_top_indices": fixed_top_indices.astype(np.int64).tolist(),
                        "fixed_top_indices": fixed_top_indices.astype(np.int64).tolist(),
                        "wrist_rerank_indices": wrist_rerank_indices.astype(np.int64).tolist(),
                        "rerank_indices": wrist_rerank_indices.astype(np.int64).tolist(),
                        "wrist_selected_candidate_index": int(best_idx),
                        "wrist_selected_fixed_topk_rank": int(rerank_best_local_idx),
                        "rerank_selected_candidate_index": int(best_idx),
                        "rerank_selected_primary_topk_rank": int(rerank_best_local_idx),
                    }
                else:
                    best_idx = int(np.argmin(losses))
                    rerank_per_view_shared_mass_iter = {}
                    if topk_then_rerank:
                        loss_gate_info["loss_selection_mode"] = "primary_only_until_final_complement_rerank"
                        loss_gate_info["primary_view_loss_mode"] = primary_view_loss_mode
                        loss_gate_info["rerank_view_loss_mode"] = rerank_view_loss_mode
                        loss_gate_info["primary_top_indices"] = fixed_top_indices.astype(np.int64).tolist()
                        loss_gate_info["fixed_top_indices"] = fixed_top_indices.astype(np.int64).tolist()
                        loss_gate_info["wrist_rerank_indices"] = None
                        loss_gate_info["rerank_indices"] = None
                loss_gate_history.append(
                    {
                        "cem_phase": phase_name,
                        "phase_idx": int(phase_idx),
                        "cem_iter": int(cem_iter),
                        "is_final_cem_iter": bool(is_final_cem_iter),
                        "search_range_scale": float(iter_search_scale),
                        "next_search_range_scale": float(next_phase_search_scale_info["scale"]),
                        "search_scale_reference_fixed_loss": float(
                            phase_search_scale_info["reference_fixed_loss"]
                        ),
                        "search_scale_input_fixed_loss": float(phase_search_scale_info["input_fixed_loss"]),
                        "effective_translation_cap_xyz": iter_translation_cap_xyz.astype(np.float32).tolist(),
                        "effective_rotation_cap": float(iter_rotation_cap),
                        "loss_selection_mode": loss_gate_info["loss_selection_mode"],
                        "primary_view_loss_mode": loss_gate_info.get("primary_view_loss_mode", primary_view_loss_mode),
                        "rerank_view_loss_mode": loss_gate_info.get("rerank_view_loss_mode"),
                        "cem_update_momentum_mean": float(phase_momentum_mean),
                        "cem_update_momentum_std": float(phase_momentum_std),
                        "current_fixed_view_loss": float(loss_gate_info["current_fixed_view_loss"]),
                        "current_all_view_loss": float(loss_gate_info.get("current_all_view_loss", float("nan"))),
                        "current_wrist_view_loss": float(loss_gate_info.get("current_wrist_view_loss", float("nan"))),
                        "current_selected_view_loss": float(
                            loss_gate_info.get("current_selected_view_loss", float("nan"))
                        ),
                        "min_candidate_fixed_view_loss": float(loss_gate_info["min_candidate_fixed_view_loss"]),
                        "fixed_view_improvement": float(loss_gate_info["fixed_view_improvement"]),
                        "primary_top_indices": loss_gate_info.get("primary_top_indices"),
                        "fixed_top_indices": loss_gate_info.get("fixed_top_indices"),
                        "wrist_rerank_indices": loss_gate_info.get("wrist_rerank_indices"),
                        "rerank_indices": loss_gate_info.get("rerank_indices"),
                        "wrist_selected_candidate_index": loss_gate_info.get("wrist_selected_candidate_index"),
                        "wrist_selected_fixed_topk_rank": loss_gate_info.get("wrist_selected_fixed_topk_rank"),
                        "rerank_selected_candidate_index": loss_gate_info.get("rerank_selected_candidate_index"),
                        "rerank_selected_primary_topk_rank": loss_gate_info.get("rerank_selected_primary_topk_rank"),
                        "goal_distance_filter": dict(phase_goal_distance_filter_debug),
                        "mean_wrist_shared_visible_mass": float(
                            np.mean(
                                [
                                    per_view_shared_mass_iter[view_name]
                                    for view_name in wrist_view_names
                                    if view_name in per_view_shared_mass_iter
                                ]
                            )
                        )
                        if any(view_name in per_view_shared_mass_iter for view_name in wrist_view_names)
                        else (
                            float(np.nanmean(rerank_shared_mass_full))
                            if topk_then_rerank and np.any(np.isfinite(rerank_shared_mass_full))
                            else None
                        ),
                        "mean_rerank_shared_visible_mass": (
                            float(np.nanmean(rerank_shared_mass_full))
                            if topk_then_rerank and np.any(np.isfinite(rerank_shared_mass_full))
                            else None
                        ),
                    }
                )
                should_update_phase_best = (not topk_then_rerank) or is_final_cem_iter
                if should_update_phase_best and float(losses[best_idx]) < phase_best_loss:
                    phase_best_loss = float(losses[best_idx])
                    if topk_then_rerank and rerank_best_local_idx is not None:
                        local_chunk_idx = rerank_best_local_idx // self.mpc_eval_batch_size
                        local_idx = rerank_best_local_idx % self.mpc_eval_batch_size
                        payload_pred_cache = rerank_pred_cache
                        payload_warp_cache = rerank_warp_cache
                        payload_state_cache = rerank_state_cache
                        payload_action_cache = rerank_action_cache
                        payload_target_robot_cache = rerank_target_robot_cache
                        payload_view_names = rerank_view_names
                    else:
                        local_chunk_idx = best_idx // self.mpc_eval_batch_size
                        local_idx = best_idx % self.mpc_eval_batch_size
                        payload_pred_cache = pred_cache
                        payload_warp_cache = warp_cache
                        payload_state_cache = state_cache
                        payload_action_cache = action_cache
                        payload_target_robot_cache = target_robot_cache
                        payload_view_names = phase_eval_view_names
                    phase_best_payload = {
                        "pred_by_view": {
                            view_name: payload_pred_cache[local_chunk_idx][view_name][local_idx : local_idx + 1]
                            for view_name in payload_view_names
                        },
                        "pred_mask_by_view": {
                            view_name: payload_pred_cache[local_chunk_idx][f"{view_name}__mask"][
                                local_idx : local_idx + 1
                            ]
                            for view_name in payload_view_names
                        },
                        "warp_by_view": {
                            view_name: payload_warp_cache[local_chunk_idx][view_name][local_idx : local_idx + 1]
                            for view_name in view_names
                            if view_name in payload_warp_cache[local_chunk_idx]
                        },
                        "warp_mask_by_view": {
                            view_name: payload_warp_cache[local_chunk_idx][f"{view_name}__mask"][
                                local_idx : local_idx + 1
                            ]
                            for view_name in view_names
                            if f"{view_name}__mask" in payload_warp_cache[local_chunk_idx]
                        },
                        "next_state_by_view": {
                            view_name: payload_state_cache[local_chunk_idx][view_name][local_idx]
                            for view_name in payload_view_names
                        },
                        "action_by_view": {
                            view_name: payload_action_cache[local_chunk_idx][view_name][local_idx]
                            for view_name in payload_view_names
                        },
                        "history_selection_by_view": {
                            view_name: payload_action_cache[local_chunk_idx].get(f"{view_name}__history", [])[local_idx]
                            for view_name in payload_view_names
                            if local_idx < len(payload_action_cache[local_chunk_idx].get(f"{view_name}__history", []))
                        },
	                        "target_robot_pose6": payload_target_robot_cache[local_chunk_idx][local_idx].copy(),
	                        "delta_pose6": delta_pose6_batch[best_idx].copy(),
	                        "gripper_delta": float(gripper_delta_batch[best_idx]),
	                        "best_loss": float(losses[best_idx]),
	                        "current_selection_loss": float(
	                            loss_gate_info.get("current_selected_view_loss", float("nan"))
	                        ),
	                        "selection_mode": loss_gate_info["loss_selection_mode"],
	                        "candidate_index": int(best_idx),
	                        "fixed_topk_rank": (
	                            int(rerank_best_local_idx)
                            if topk_then_rerank and rerank_best_local_idx is not None
                            else None
                        ),
                        "primary_topk_rank": (
                            int(rerank_best_local_idx)
                            if topk_then_rerank and rerank_best_local_idx is not None
                            else None
                        ),
                    }

                sorted_indices = np.argsort(losses)
                top_indices = fixed_top_indices if topk_then_rerank else sorted_indices[:topk]
                memory_candidate_indices = set()
                for view_name in phase_eval_view_names:
                    view_sorted = np.argsort(per_view_losses_iter[view_name])[: self.pose_memory_topk]
                    memory_candidate_indices.update(int(idx) for idx in view_sorted.tolist())
                if topk_then_rerank:
                    memory_candidate_indices.update(int(idx) for idx in fixed_top_indices[: self.pose_memory_topk].tolist())
                for cand_idx in sorted(memory_candidate_indices):
                    target_gripper = float(
                        np.clip(
                            float(ref_state7_np[6]) + float(gripper_delta_batch[cand_idx]),
                            self.gripper_position_min,
                            self.gripper_position_max,
                        )
                    )
                    self._add_pose_memory_entry(
                        pose6_robot=target_robot_pose6_iter[cand_idx],
                        gripper_state=target_gripper,
                        per_view_losses={
                            view_name: float(per_view_losses_iter[view_name][cand_idx])
                            for view_name in phase_eval_view_names
                        },
                        source="candidate_predicted_goal_score",
                        stage=None if current_goal_target is None else current_goal_target.get("stage"),
                        step=int(self.predict_step_idx),
                        extra={
                            "cem_phase": phase_name,
                            "phase_idx": int(phase_idx),
                            "cem_iter": int(cem_iter),
                            "is_final_cem_iter": bool(is_final_cem_iter),
                            "candidate_index": int(cand_idx),
                            "sample_source": str(sample_sources[cand_idx]),
                            "rotation_delta_frame": phase_rotation_delta_frame,
                        },
                    )
                rank_by_idx = np.empty_like(sorted_indices)
                rank_by_idx[sorted_indices] = np.arange(sorted_indices.shape[0], dtype=sorted_indices.dtype)
                top_index_set = set(int(x) for x in top_indices.tolist())
                for cand_idx in range(samples):
                    target_robot_pose6 = target_robot_pose6_iter[cand_idx].astype(np.float32)
                    goal_pos_l2 = None
                    goal_rot_l2 = None
                    goal_pos_delta = None
                    goal_rot_delta = None
                    if goal_robot_pose6 is not None:
                        cand_goal_delta6 = self._pose_delta6(target_robot_pose6, goal_robot_pose6)
                        goal_pos_l2 = float(np.linalg.norm(cand_goal_delta6[:3]))
                        goal_rot_l2 = float(np.linalg.norm(cand_goal_delta6[3:6]))
                        goal_pos_delta = None if current_goal_pos_l2 is None else float(goal_pos_l2 - current_goal_pos_l2)
                        goal_rot_delta = None if current_goal_rot_l2 is None else float(goal_rot_l2 - current_goal_rot_l2)
                    target_gripper = float(
                        np.clip(
                            float(ref_state7_np[6]) + float(gripper_delta_batch[cand_idx]),
                            self.gripper_position_min,
                            self.gripper_position_max,
                        )
                    )
                    candidate_diagnostics.append(
                        {
                            "cem_phase": phase_name,
                            "phase_idx": int(phase_idx),
                            "cem_iter": int(cem_iter),
                            "candidate_index": int(cand_idx),
                            "is_phase_noop_candidate": bool(
                                cand_idx == 0 and not self.always_execute_best_candidate
                            ),
                            "sample_source": str(sample_sources[cand_idx]),
                            "rotation_delta_frame": phase_rotation_delta_frame,
                            "is_steering_candidate": bool(sample_sources[cand_idx] == "steering"),
                            "search_range_scale": float(iter_search_scale),
                            "rank": int(rank_by_idx[cand_idx]),
                            "is_topk": bool(cand_idx in top_index_set),
                            "is_iter_best": bool(cand_idx == best_idx),
                            "predicted_loss": float(losses[cand_idx]),
                            "loss_selection_mode": loss_gate_info["loss_selection_mode"],
                            "all_view_loss": float(loss_gate_info["all_view_loss"][cand_idx]),
                            "fixed_view_loss": float(loss_gate_info["fixed_view_loss"][cand_idx]),
                            "wrist_view_loss": (
                                float(loss_gate_info["wrist_view_loss"][cand_idx])
                                if np.isfinite(loss_gate_info["wrist_view_loss"][cand_idx])
                                else None
                            ),
                            "wrist_gripper_slot_index": (
                                None
                                if "wrist_image_left" not in self.gripper_slot_index_by_view
                                else int(self.gripper_slot_index_by_view["wrist_image_left"])
                            ),
                            "wrist_gripper_slot_transport_loss": (
                                float(gripper_slot_losses_iter["wrist_image_left"][cand_idx])
                                if "wrist_image_left" in gripper_slot_losses_iter
                                else None
                            ),
                            "wrist_gripper_loss_used": (
                                "wrist_objective_loss"
                                if phase_name == "wrist_gripper" and "wrist_image_left" in per_view_losses_iter
                                else None
                            ),
                            "wrist_gripper_loss_used_value": (
                                float(per_view_losses_iter["wrist_image_left"][cand_idx])
                                if phase_name == "wrist_gripper" and "wrist_image_left" in per_view_losses_iter
                                else None
                            ),
                            "wrist_other_slot_mean_transport_loss": (
                                float(other_slot_losses_iter["wrist_image_left"][cand_idx])
                                if "wrist_image_left" in other_slot_losses_iter
                                else None
                            ),
                            "wrist_gripper_mixed_transport_loss": (
                                float(gripper_mixed_losses_iter["wrist_image_left"][cand_idx])
                                if "wrist_image_left" in gripper_mixed_losses_iter
                                else None
                            ),
                            "wrist_pose_without_fixed_slot_transport_loss": (
                                float(without_fixed_slot_losses_iter["wrist_image_left"][cand_idx])
                                if "wrist_image_left" in without_fixed_slot_losses_iter
                                else None
                            ),
                            "wrist_pose_loss_used": (
                                "without_fixed_slot_transport_loss"
                                if phase_name != "wrist_gripper"
                                and "wrist_image_left" in phase_eval_view_names
                                and "wrist_image_left" in without_fixed_slot_losses_iter
                                else None
                            ),
                            "fixed_topk_rank": (
                                int(np.where(fixed_top_indices == cand_idx)[0][0])
                                if topk_then_rerank and cand_idx in top_index_set
                                else None
                            ),
                            "reranked_by_wrist": bool(
                                topk_then_rerank
                                and is_final_cem_iter
                                and rerank_view_loss_mode == "wrist_only"
                                and cand_idx in set(int(x) for x in wrist_rerank_indices.tolist())
                            ),
                            "selected_by_wrist_rerank": bool(
                                topk_then_rerank
                                and rerank_view_loss_mode == "wrist_only"
                                and cand_idx == loss_gate_info.get("wrist_selected_candidate_index")
                            ),
                            "reranked_by_complement": bool(
                                topk_then_rerank
                                and is_final_cem_iter
                                and cand_idx in set(int(x) for x in wrist_rerank_indices.tolist())
                            ),
                            "selected_by_complement_rerank": bool(
                                topk_then_rerank and cand_idx == loss_gate_info.get("rerank_selected_candidate_index")
                            ),
                            "evaluated_view_names": list(phase_eval_view_names),
                            "rerank_view_names": list(rerank_view_names) if topk_then_rerank else [],
                            "primary_view_loss_mode": primary_view_loss_mode,
                            "rerank_view_loss_mode": rerank_view_loss_mode if topk_then_rerank else None,
                            "wrist_shared_visible_mass": float(
                                np.mean(
                                    [
                                        per_view_shared_mass_iter[view_name][cand_idx]
                                        for view_name in wrist_view_names
                                        if view_name in per_view_shared_mass_iter
                                    ]
                                )
                            )
                            if any(view_name in per_view_shared_mass_iter for view_name in wrist_view_names)
                            else (
                                float(rerank_shared_mass_full[cand_idx])
                                if topk_then_rerank and np.isfinite(rerank_shared_mass_full[cand_idx])
                                else None
                            ),
                            "fixed_view_improvement": float(loss_gate_info["fixed_view_improvement"]),
                            "delta_pose6_robot": delta_pose6_batch[cand_idx].astype(np.float32).tolist(),
                            "target_robot_pose6": target_robot_pose6.tolist(),
                            "current_gripper_model": float(ref_state7_np[6]),
                            "gripper_delta_model": float(gripper_delta_batch[cand_idx]),
                            "target_gripper_model": target_gripper,
                            "gripper_downpress_z_scoring": float(gripper_downpress_z_batch[cand_idx])
                            if phase_sample_gripper
                            else 0.0,
                            "goal_pos_l2_before": current_goal_pos_l2,
                            "goal_pos_l2_after": goal_pos_l2,
                            "goal_pos_l2_delta": goal_pos_delta,
                            "goal_rot_l2_before": current_goal_rot_l2,
                            "goal_rot_l2_after": goal_rot_l2,
                            "goal_rot_l2_delta": goal_rot_delta,
                        }
                    )
                selected_delta = delta_pose6_batch[top_indices]
                selected_gripper = gripper_delta_batch[top_indices]
                top_pose_log_count = max(0, int(phase.get("top_pose_log_count", 5)))
                if top_pose_log_count > 0 and phase_dims.size > 0:
                    phase_top_pose_candidates = []
                    for rank, cand_idx in enumerate(sorted_indices[:top_pose_log_count]):
                        cand_idx = int(cand_idx)
                        phase_top_pose_candidates.append(
                            {
                                "rank": int(rank),
                                "candidate_index": int(cand_idx),
                                "sample_source": str(sample_sources[cand_idx]),
                                "selected_for_execution": bool(cand_idx == best_idx),
                                "predicted_loss": float(losses[cand_idx]),
                                "all_view_loss": float(loss_gate_info["all_view_loss"][cand_idx]),
                                "fixed_view_loss": float(loss_gate_info["fixed_view_loss"][cand_idx]),
                                "wrist_view_loss": (
                                    float(loss_gate_info["wrist_view_loss"][cand_idx])
                                    if np.isfinite(loss_gate_info["wrist_view_loss"][cand_idx])
                                    else None
                                ),
                                "delta_pose6_robot": delta_pose6_batch[cand_idx].astype(np.float32).tolist(),
                                "target_robot_pose6": target_robot_pose6_iter[cand_idx].astype(np.float32).tolist(),
                            }
                        )
                if phase_dims.size > 0:
                    mean_delta_pose6_cem[phase_dims] = (
                        selected_delta[:, phase_dims].mean(axis=0) * (1.0 - phase_momentum_mean)
                        + mean_delta_pose6_cem[phase_dims] * phase_momentum_mean
                    ).astype(np.float32)
                    std_delta_pose6_cem[phase_dims] = (
                        selected_delta[:, phase_dims].std(axis=0) * (1.0 - phase_momentum_std)
                        + std_delta_pose6_cem[phase_dims] * phase_momentum_std
                    ).astype(np.float32)
                if phase_sample_gripper and phase.get("gripper_search_mode") != "uniform_absolute":
                    mean_gripper_delta_cem = float(
                        selected_gripper.mean() * (1.0 - self.mpc_args["momentum_mean_gripper"])
                        + mean_gripper_delta_cem * self.mpc_args["momentum_mean_gripper"]
                    )
                    std_gripper_delta_cem = float(
                        selected_gripper.std() * (1.0 - self.mpc_args["momentum_std_gripper"])
                        + std_gripper_delta_cem * self.mpc_args["momentum_std_gripper"]
                    )
                phase_search_scale_info = next_phase_search_scale_info
                timing["cem_sort_update_s"] += float(time.perf_counter() - t_sort_update_start)

            if phase_best_payload is not None:
                phase_best_loss_before_noop = float(phase_best_payload.get("best_loss", phase_best_loss))
                phase_current_selection_loss = float(
                    phase_best_payload.get("current_selection_loss", float("nan"))
                )
                phase_noop_due_to_no_improvement = False
                phase_fixed_view_fallback_used = False
                phase_noop_improvement = (
                    phase_current_selection_loss - phase_best_loss_before_noop
                    if np.isfinite(phase_current_selection_loss)
                    else float("nan")
                )
                if (
                    self.fixed_view_noop_fallback_enabled
                    and phase_name == "fixed_pose_coarse"
                    and (not phase_sample_gripper)
                    and phase_dims.size > 0
                    and np.isfinite(phase_noop_improvement)
                    and phase_noop_improvement > 1.0e-6
                ):
                    fixed_view_fallback_payload = dict(phase_best_payload)
                    fixed_view_fallback_payload["target_robot_pose6"] = np.asarray(
                        phase_best_payload["target_robot_pose6"], dtype=np.float32
                    ).copy()
                    fixed_view_fallback_payload["delta_pose6"] = np.asarray(
                        phase_best_payload["delta_pose6"], dtype=np.float32
                    ).copy()
                    fixed_view_fallback_debug = {
                        "available": True,
                        "source_phase": "fixed_pose_coarse",
                        "candidate_index": int(phase_best_payload.get("candidate_index", -1)),
                        "current_fixed_view_objective": float(phase_current_selection_loss),
                        "candidate_fixed_view_objective": float(phase_best_loss_before_noop),
                        "fixed_view_objective_improvement": float(phase_noop_improvement),
                    }
                    if self.a2_no_uot_mode:
                        fixed_view_fallback_debug.update(
                            {
                                "current_fixed_view_l1": float(phase_current_selection_loss),
                                "candidate_fixed_view_l1": float(phase_best_loss_before_noop),
                                "fixed_view_l1_improvement": float(phase_noop_improvement),
                            }
                        )
                if (
                    not self.always_execute_best_candidate
                    and phase_updates_execution_target
                    and (not phase_sample_gripper)
                    and phase_dims.size > 0
                    and np.isfinite(phase_current_selection_loss)
                    and phase_noop_improvement <= 1.0e-6
                ):
                    if (
                        self.fixed_view_noop_fallback_enabled
                        and fixed_view_fallback_payload is not None
                    ):
                        wrist_best_loss = float(phase_best_loss_before_noop)
                        phase_best_payload = dict(fixed_view_fallback_payload)
                        phase_best_payload["best_loss_before_noop"] = wrist_best_loss
                        phase_best_payload["wrist_current_objective"] = float(
                            phase_current_selection_loss
                        )
                        phase_best_payload["wrist_candidate_objective"] = wrist_best_loss
                        phase_best_payload["wrist_objective_improvement"] = float(
                            phase_noop_improvement
                        )
                        if self.a2_no_uot_mode:
                            phase_best_payload["wrist_current_l1"] = float(
                                phase_current_selection_loss
                            )
                            phase_best_payload["wrist_candidate_l1"] = wrist_best_loss
                            phase_best_payload["wrist_l1_improvement"] = float(
                                phase_noop_improvement
                            )
                        phase_best_payload["fixed_view_fallback_used"] = True
                        phase_best_loss = float(phase_best_payload["best_loss"])
                        phase_fixed_view_fallback_used = True
                    else:
                        phase_noop_due_to_no_improvement = True
                        phase_best_payload["target_robot_pose6"] = current_robot_pose6.astype(np.float32).copy()
                        phase_best_payload["delta_pose6"] = np.zeros(6, dtype=np.float32)
                        phase_best_payload["best_loss_before_noop"] = float(phase_best_loss_before_noop)
                        phase_best_payload["best_loss"] = float(phase_current_selection_loss)
                        phase_best_payload["noop_due_to_no_predicted_improvement"] = True
                        phase_best_loss = float(phase_current_selection_loss)
                phase_output_target_robot_pose6 = phase_best_payload["target_robot_pose6"]
                phase_output_delta_pose6 = phase_best_payload["delta_pose6"]
                phase_output_gripper_delta = float(phase_best_payload["gripper_delta"])
                if phase_updates_execution_target:
                    best_pred_by_view = phase_best_payload["pred_by_view"]
                    best_pred_mask_by_view = phase_best_payload["pred_mask_by_view"]
                    best_warp_by_view = phase_best_payload["warp_by_view"]
                    best_warp_mask_by_view = phase_best_payload["warp_mask_by_view"]
                    best_next_state_by_view = phase_best_payload["next_state_by_view"]
                    best_action_by_view = phase_best_payload["action_by_view"]
                    best_target_robot_pose6 = phase_output_target_robot_pose6
                    if phase_dims.size > 0:
                        mean_delta_pose6_cem[phase_dims] = phase_output_delta_pose6[phase_dims]
                    if phase_sample_gripper:
                        mean_gripper_delta_cem = phase_output_gripper_delta
                    if phase.get("next_cursor_on_success") is not None:
                        next_planning_phase_cursor = str(phase["next_cursor_on_success"])
                cem_phase_outputs.append(
                    {
                        "cem_phase": phase_name,
                        "phase_idx": int(phase_idx),
                        "planning_cursor": phase.get("planning_cursor"),
                        "effective_planning_cursor": phase.get("effective_planning_cursor"),
                        "rotation_delta_frame": phase_rotation_delta_frame,
                        "skipped": False,
                        "best_loss": float(phase_best_loss),
                        "best_loss_before_noop": float(
                            phase_best_payload.get("best_loss_before_noop", phase_best_loss)
                        ),
                        "current_selection_loss": float(phase_current_selection_loss),
                        "predicted_loss_improvement": float(phase_noop_improvement),
                        "noop_due_to_no_predicted_improvement": bool(phase_noop_due_to_no_improvement),
                        "fixed_view_fallback_used": bool(phase_fixed_view_fallback_used),
                        "fixed_view_fallback": (
                            dict(fixed_view_fallback_debug)
                            if phase_fixed_view_fallback_used and fixed_view_fallback_debug is not None
                            else None
                        ),
                        "updates_execution_target": bool(phase_updates_execution_target),
                        "top_pose_candidates": phase_top_pose_candidates,
                        "cem_update_momentum_mean": float(phase_momentum_mean),
                        "cem_update_momentum_std": float(phase_momentum_std),
                        "gripper_slot_index_by_view": dict(self.gripper_slot_index_by_view),
                        "gripper_slot_debug_by_view": dict(self.gripper_slot_debug_by_view),
                        "selected_history_by_view": phase_best_payload.get("history_selection_by_view", {}),
                        "view_loss_mode": phase_view_loss_mode,
                        "eval_view_names": list(phase.get("eval_view_names", [])),
                        "pose_view_switch": phase.get("pose_view_switch"),
                        "warmup_anchor_steering_active": bool(
                            phase.get("warmup_anchor_steering_active", False)
                        ),
                        "fixed_view_stagnation_triggered": bool(
                            phase.get("fixed_view_stagnation_triggered", False)
                        ),
                        "selection_mode": (
                            None
                            if phase_best_payload is None
                            else phase_best_payload.get("selection_mode")
                        ),
                        "selected_candidate_index": (
                            None
                            if phase_best_payload is None
                            else phase_best_payload.get("candidate_index")
                        ),
                        "selected_fixed_topk_rank": (
                            None
                            if phase_best_payload is None
                            else phase_best_payload.get("fixed_topk_rank")
                        ),
                        "selected_primary_topk_rank": (
                            None
                            if phase_best_payload is None
                            else phase_best_payload.get("primary_topk_rank")
                        ),
                        "primary_view_loss_mode": primary_view_loss_mode,
                        "rerank_view_names": list(phase.get("rerank_view_names", []))
                        if topk_then_rerank_phase
                        else [],
                        "rerank_view_loss_mode": rerank_view_loss_mode if topk_then_rerank_phase else None,
                        "target_robot_pose6": phase_output_target_robot_pose6.astype(np.float32).tolist(),
                        "delta_pose6_robot": phase_output_delta_pose6.astype(np.float32).tolist(),
                        "gripper_delta": float(phase_output_gripper_delta),
                        "gripper_downpress_max_z": float(phase.get("gripper_downpress_max_z", 0.0)),
                    }
                )
            else:
                cem_phase_outputs.append(
                    {
                        "cem_phase": phase_name,
                        "phase_idx": int(phase_idx),
                        "planning_cursor": phase.get("planning_cursor"),
                        "effective_planning_cursor": phase.get("effective_planning_cursor"),
                        "rotation_delta_frame": phase_rotation_delta_frame,
                        "skipped": True,
                        "skip_reason": "no_phase_best_payload",
                        "view_loss_mode": phase_view_loss_mode,
                        "eval_view_names": list(phase.get("eval_view_names", [])),
                        "pose_view_switch": phase.get("pose_view_switch"),
                        "warmup_anchor_steering_active": bool(
                            phase.get("warmup_anchor_steering_active", False)
                        ),
                        "fixed_view_stagnation_triggered": bool(
                            phase.get("fixed_view_stagnation_triggered", False)
                        ),
                        "primary_view_loss_mode": primary_view_loss_mode,
                        "rerank_view_names": list(phase.get("rerank_view_names", []))
                        if topk_then_rerank_phase
                        else [],
                        "rerank_view_loss_mode": rerank_view_loss_mode if topk_then_rerank_phase else None,
                        "target_robot_pose6": best_target_robot_pose6.astype(np.float32).tolist(),
                        "delta_pose6_robot": mean_delta_pose6_cem.astype(np.float32).tolist(),
                        "gripper_delta": float(mean_gripper_delta_cem),
                    }
                )
        if self.factorized_pose_cem:
            self.slot_planning_phase_cursor = next_planning_phase_cursor
        # CEM phases intentionally evaluate only the views needed for ranking.
        # Run the final selected action once on all views so downstream
        # visualization and history recording remain complete.
        final_delta_pose6_raw = self._pose_delta6(current_robot_pose6, best_target_robot_pose6).astype(np.float32)
        final_delta_pose6 = self._cap_pose_delta_total_norm(
            final_delta_pose6_raw,
            self.execution_translation_cap,
            self.execution_rotation_cap,
        ).astype(np.float32)
        execution_cap_applied = bool(not np.allclose(final_delta_pose6_raw, final_delta_pose6, atol=1.0e-6))
        if execution_cap_applied:
            best_target_robot_pose6 = self._apply_pose_delta6(
                current_robot_pose6,
                final_delta_pose6,
                rotation_delta_frame="parent",
            ).astype(np.float32)
        (
            final_per_view_losses,
            final_gripper_slot_losses,
            final_other_slot_losses,
            final_gripper_mixed_losses,
            final_without_fixed_slot_losses,
            final_pred_by_view,
            final_warp_by_view,
            final_next_state_by_view,
            final_action_by_view,
            final_target_robot_pose6_all,
            final_shared_mass,
        ) = evaluate_pose_delta_batch(
            final_delta_pose6[None, :],
            np.array([mean_gripper_delta_cem], dtype=np.float32),
            eval_view_names=view_names,
        )
        best_pred_by_view = {view_name: final_pred_by_view[view_name] for view_name in view_names}
        best_pred_mask_by_view = {view_name: final_pred_by_view[f"{view_name}__mask"] for view_name in view_names}
        best_warp_by_view = {
            view_name: final_warp_by_view[view_name]
            for view_name in view_names
            if view_name in final_warp_by_view
        }
        best_warp_mask_by_view = {
            view_name: final_warp_by_view[f"{view_name}__mask"]
            for view_name in view_names
            if f"{view_name}__mask" in final_warp_by_view
        }
        best_next_state_by_view = {view_name: final_next_state_by_view[view_name][0] for view_name in view_names}
        best_action_by_view = {view_name: final_action_by_view[view_name][0] for view_name in view_names}
        best_target_robot_pose6 = final_target_robot_pose6_all[0].astype(np.float32).copy()

        candidate_summary = {
            "num_candidates": int(len(candidate_diagnostics)),
            "goal_robot_pose6": None if goal_robot_pose6 is None else goal_robot_pose6.astype(np.float32).tolist(),
            "current_goal_pos_l2": current_goal_pos_l2,
            "current_goal_rot_l2": current_goal_rot_l2,
            "best_by_predicted_loss": None,
            "best_by_goal_pos_l2": None,
            "loss_goal_pos_delta_corr": None,
        }
        if candidate_diagnostics:
            best_loss_diag = min(candidate_diagnostics, key=lambda x: x["predicted_loss"])
            candidate_summary["best_by_predicted_loss"] = {
                key: best_loss_diag[key]
                for key in (
                    "cem_phase",
                    "phase_idx",
                    "cem_iter",
                    "candidate_index",
                    "rank",
                    "predicted_loss",
                    "goal_pos_l2_after",
                    "goal_pos_l2_delta",
                    "target_gripper_model",
                    "delta_pose6_robot",
                    "target_robot_pose6",
                )
            }
            valid_goal_diags = [x for x in candidate_diagnostics if x["goal_pos_l2_after"] is not None]
            if valid_goal_diags:
                best_goal_diag = min(valid_goal_diags, key=lambda x: x["goal_pos_l2_after"])
                candidate_summary["best_by_goal_pos_l2"] = {
                    key: best_goal_diag[key]
                    for key in (
                        "cem_phase",
                        "phase_idx",
                        "cem_iter",
                        "candidate_index",
                        "rank",
                        "predicted_loss",
                        "goal_pos_l2_after",
                        "goal_pos_l2_delta",
                        "target_gripper_model",
                        "delta_pose6_robot",
                        "target_robot_pose6",
                    )
                }
                loss_values = np.array([x["predicted_loss"] for x in valid_goal_diags], dtype=np.float32)
                goal_delta_values = np.array([x["goal_pos_l2_delta"] for x in valid_goal_diags], dtype=np.float32)
                if loss_values.size > 1 and float(loss_values.std()) > 1.0e-8 and float(goal_delta_values.std()) > 1.0e-8:
                    candidate_summary["loss_goal_pos_delta_corr"] = float(
                        np.corrcoef(loss_values, goal_delta_values)[0, 1]
                    )

        ref_view = self._reference_view(view_names)
        best_action_ref = best_action_by_view[ref_view].astype(np.float32)
        best_next_pose_ref = torch.from_numpy(best_next_state_by_view[ref_view]).to(self.device, dtype=torch.float32)
        best_delta_pose6_robot = self._pose_delta6(current_robot_pose6, best_target_robot_pose6).astype(np.float32)
        self.recent_selected_pose_deltas.append(best_delta_pose6_robot.copy())
        self.last_predictor_debug = {
            "optimize_timing": timing,
            "slot_planning_request": planning_request,
            "num_views": int(len(view_names)),
            "samples": int(samples),
            "cem_steps": int(cem_steps),
            "factorized_pose_cem": bool(self.factorized_pose_cem),
            "always_execute_best_candidate": bool(
                self.always_execute_best_candidate
            ),
            "fixed_view_noop_fallback_enabled": bool(
                self.fixed_view_noop_fallback_enabled
            ),
            "fixed_view_fallback": fixed_view_fallback_debug,
            # Retain the A2-named fields for compatibility with existing traces.
            "a2_fixed_view_noop_fallback_enabled": bool(
                self.a2_no_uot_mode and self.fixed_view_noop_fallback_enabled
            ),
            "a2_fixed_view_fallback": (
                fixed_view_fallback_debug if self.a2_no_uot_mode else None
            ),
            "planning_phase_cursor_before": current_planning_phase,
            "effective_planning_phase": effective_planning_phase,
            "planning_phase_redirect_reason": phase_redirect_reason,
            "planning_phase_cursor_after": getattr(self, "slot_planning_phase_cursor", None),
            "fixed_slot_index_by_view": dict(self.fixed_slot_index_by_view),
            "fixed_slot_debug_by_view": dict(self.fixed_slot_debug_by_view),
            "gripper_slot_index_by_view": dict(self.gripper_slot_index_by_view),
            "gripper_slot_debug_by_view": dict(self.gripper_slot_debug_by_view),
            "final_gripper_slot_transport_losses": {
                view_name: float(np.asarray(values, dtype=np.float32).reshape(-1)[0])
                for view_name, values in final_gripper_slot_losses.items()
            },
            "final_other_slot_mean_transport_losses": {
                view_name: float(np.asarray(values, dtype=np.float32).reshape(-1)[0])
                for view_name, values in final_other_slot_losses.items()
            },
            "final_gripper_mixed_transport_losses": {
                view_name: float(np.asarray(values, dtype=np.float32).reshape(-1)[0])
                for view_name, values in final_gripper_mixed_losses.items()
            },
            "factorized_cem_phase_plan": [
                {
                    "name": phase["name"],
                    "dims": phase["dims"].astype(np.int64).tolist(),
                    "planning_cursor": phase.get("planning_cursor"),
                    "effective_planning_cursor": phase.get("effective_planning_cursor"),
                    "view_loss_mode": phase["view_loss_mode"],
                    "primary_view_loss_mode": phase.get("primary_view_loss_mode"),
                    "eval_view_names": list(phase.get("eval_view_names", [])),
                    "rerank_view_names": list(phase.get("rerank_view_names", [])),
                    "rerank_view_loss_mode": phase.get("rerank_view_loss_mode"),
                    "rerank_topk": phase.get("rerank_topk"),
                    "sample_gripper": bool(phase["sample_gripper"]),
                    "num_candidates": int(phase.get("num_candidates", self.mpc_args["samples"])),
                    "num_steps": int(phase["num_steps"]),
                    "sample_scale": float(phase.get("sample_scale", 1.0)),
                    "sample_scale_by_dim": None
                    if phase.get("sample_scale_by_dim") is None
                    else np.asarray(phase["sample_scale_by_dim"], dtype=np.float32).tolist(),
                    "z_offset": float(phase.get("z_offset", 0.0)),
                    "z_offset_stage": phase.get("z_offset_stage"),
                    "rotation_delta_frame": phase.get("rotation_delta_frame", "parent"),
                    "gripper_search_mode": phase.get("gripper_search_mode"),
                    "updates_execution_target": bool(phase.get("updates_execution_target", True)),
                    "momentum_mean": phase.get("momentum_mean"),
                    "momentum_std": phase.get("momentum_std"),
                    "next_cursor_on_success": phase.get("next_cursor_on_success"),
                    "fixed_view_stagnation_triggered": bool(phase.get("fixed_view_stagnation_triggered", False)),
                    "pose_eval_uses_fixed_views": bool(phase.get("pose_eval_uses_fixed_views", False)),
                    "pose_eval_uses_wrist_only": bool(phase.get("pose_eval_uses_wrist_only", False)),
                    "pose_view_switch": phase.get("pose_view_switch"),
                }
                for phase in phase_plan
            ],
            "gripper_cem_steps": int(self.gripper_cem_steps),
            "gripper_samples": int(self.gripper_samples),
            "mpc_eval_batch_size": int(self.mpc_eval_batch_size),
            "adaptive_action_limits": action_limits,
            "action_range_effective": {
                "translation_cap_xyz": np.asarray(action_limits["translation_cap_xyz"], dtype=np.float32).tolist(),
                "translation_std_xyz": np.asarray(action_limits["translation_std_xyz"], dtype=np.float32).tolist(),
                "rotation_cap": float(action_limits["rotation_cap"]),
                "rotation_std": float(action_limits["rotation_std"]),
                "execution_translation_cap_total_norm": float(self.execution_translation_cap),
                "execution_rotation_cap_total_norm": float(self.execution_rotation_cap),
                "execution_cap_applied": bool(execution_cap_applied),
                "final_delta_pose6_raw": final_delta_pose6_raw.astype(np.float32).tolist(),
                "final_delta_pose6_after_execution_cap": final_delta_pose6.astype(np.float32).tolist(),
                "final_translation_norm_raw": float(np.linalg.norm(final_delta_pose6_raw[:3])),
                "final_translation_norm_after_execution_cap": float(np.linalg.norm(final_delta_pose6[:3])),
                "final_rotation_norm_raw": float(np.linalg.norm(final_delta_pose6_raw[3:6])),
                "final_rotation_norm_after_execution_cap": float(np.linalg.norm(final_delta_pose6[3:6])),
                "scale": float(self.slot_action_range_scale),
            },
            "slot_action_range_scale": float(self.slot_action_range_scale),
            "obj_latent_refresh_mode": self.last_obj_latent_refresh_mode,
            "last_obj_latent_refresh_step": int(self.last_obj_latent_refresh_step),
            "last_obj_latent_refresh_reason": self.last_obj_latent_refresh_reason,
            "history_selection": self.last_history_selection_debug,
            "memory_selection": {
                "mode": self.memory_selection_mode,
                "random_seed": int(self.memory_random_seed),
                "sampling_scope": (
                    "current_trajectory_history"
                    if self.memory_selection_mode == "in_trajectory_random"
                    else None
                ),
                "effective_random_seed": int(self.in_trajectory_random_effective_seed),
            },
            "current_pose_source": "latest_observation" if current_obs_views is not None else "history_last_frame",
            "mask_transport_loss": {
                "enabled": bool(self.transport_loss_mix > 0.0),
                "transport_loss_mix": float(self.transport_loss_mix),
                "pointwise_loss_type": "mask_l1",
                "mask_loss_type": "centralized_unbalanced_sinkhorn_ot",
                "mask_centralize_gamma": float(self.mask_centralize_gamma),
                "unbalanced_ot_rho": float(self.unbalanced_ot_rho),
                "unbalanced_ot_eps": float(self.unbalanced_ot_eps),
                "area_normalize_by_slot_max_area": bool(self.mask_transport_area_normalize),
                "area_normalize_eps": float(self.mask_transport_area_normalize_eps),
                "area_normalize_source": (
                    "max_area_over_slot_extraction_frames_and_goals"
                    if self.mask_transport_area_normalize
                    else "raw_uot"
                ),
                "area_normalizer_by_view": {
                    view_name: self._slot_area_normalizer_for_view(view_name, device=self.device)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .tolist()
                    for view_name in view_names
                    if self._slot_area_normalizer_for_view(view_name, device=self.device) is not None
                },
                "scope": "full_image_only",
                "slot_weight_type": "goal_current_abs_delta",
                "slot_weight_mean_by_view": {
                    view_name: float(predictor_cache_by_view[view_name]["slot_change_weight_mean"])
                    for view_name in view_names
                },
                "slot_weight_max_by_view": {
                    view_name: float(predictor_cache_by_view[view_name]["slot_change_weight_max"])
                    for view_name in view_names
                },
                "weight_mean_by_view": {
                    view_name: float(predictor_cache_by_view[view_name]["shared_visible_weight_mean"])
                    for view_name in view_names
                },
                "weight_mass_by_view": {
                    view_name: float(predictor_cache_by_view[view_name]["shared_visible_weight_mass"])
                    for view_name in view_names
                },
            },
            "action_l2_weight": float(self.action_l2_weight),
            "view_loss_gate": {
                "view_loss_mode": self.view_loss_mode,
                "fixed_view_names": list(fixed_view_names),
                "wrist_view_names": list(wrist_view_names),
                "current_loss_by_view": dict(current_loss_by_view),
                "current_fixed_view_loss": float(current_fixed_view_loss),
                "pose_view_switch": dict(pose_view_switch),
                "fixed_view_loss_gate_threshold": float(self.fixed_view_loss_gate_threshold),
                "history": loss_gate_history,
            },
            "warmup_fixed_anchor_steering": warmup_fixed_anchor_steering,
            "cem_phase_outputs": cem_phase_outputs,
            "pose_memory_steering": {
                "enabled": False,
                "disabled_reason": "pose_memory_steering_disabled",
                "topk_per_view": int(self.pose_memory_topk),
                "sample_fraction": float(self.steering_sample_fraction),
                "loss_range_eps": float(self.steering_loss_range_eps),
                "history": steering_history,
                "memory_sizes": {
                    view_name: int(len(self.pose_memory_by_view.get(view_name, [])))
                    for view_name in self.VIEW_ORDER
                },
                "memory_top": self._pose_memory_summary(),
            },
            "uses_physical_goal_pose_prior": bool(self.use_goal_pose_prior),
            "goal_pose_prior_debug": self.last_goal_pose_prior_debug,
            "goal_distance_filter": goal_distance_filter,
            "cem_init_mean_delta_pose6_robot": mean_delta_pose6.astype(np.float32),
            "cem_init_mean_translation_norm": float(np.linalg.norm(mean_delta_pose6[:3])),
            "cem_init_mean_rotation_norm": float(np.linalg.norm(mean_delta_pose6[3:6])),
            "observed_pose_delta6_robot": None
            if observed_pose_delta6 is None
            else observed_pose_delta6.astype(np.float32),
            "observed_translation_norm": None
            if observed_pose_delta6 is None
            else float(np.linalg.norm(observed_pose_delta6[:3])),
            "observed_rotation_norm": None
            if observed_pose_delta6 is None
            else float(np.linalg.norm(observed_pose_delta6[3:6])),
            "observed_motion_gate": {
                "metric": "net_pose_delta_over_window_per_step",
                "window_size": int(observed_window_step_count),
                "required_window_size": int(self.observed_motion_window),
                "window_ready": bool(observed_motion_window_ready),
                "mean_translation_norm": float(observed_translation_norm),
                "mean_rotation_norm": float(observed_rotation_norm),
                "window_delta6_robot": observed_window_delta6.astype(np.float32).tolist(),
                "step_mean_translation_norm": float(observed_step_mean_translation_norm),
                "step_mean_rotation_norm": float(observed_step_mean_rotation_norm),
                "translation_norms": observed_translation_norms.astype(np.float32).tolist(),
                "rotation_norms": observed_rotation_norms.astype(np.float32).tolist(),
            },
            "gripper_position_range": [float(self.gripper_position_min), float(self.gripper_position_max)],
            "optimize_gripper_action": bool(self.optimize_gripper_action),
            "skipped_gripper_cem": bool(skipped_gripper_cem),
            "gripper_cem_skip_reason": gripper_cem_skip_reason,
            "gripper_cem_skip_translation_threshold": float(self.gripper_cem_skip_translation_threshold),
            "gripper_cem_skip_rotation_threshold": float(self.gripper_cem_skip_rotation_threshold),
            "fixed_gripper_target": float(self.fixed_gripper_target),
            "gripper_search_mode": "uniform_absolute_wrist_loss",
            "gripper_min_open_gap": float(self.gripper_min_open_gap),
            "gripper_downpress_max_z": float(self.gripper_downpress_max_z),
            "gripper_uniform_target_range": [
                float(self.gripper_position_min),
                float(
                    min(
                        self.gripper_position_max,
                        max(self.gripper_position_min, self.gripper_position_max - self.gripper_min_open_gap),
                    )
                ),
            ],
            "gripper_delta_std": float(self.gripper_delta_std),
            "gripper_delta_cap": float(self.gripper_delta_cap),
            "best_gripper_delta_model": float(best_action_ref[6]),
            "best_next_gripper_position_model": float(best_next_pose_ref.detach().cpu().numpy().reshape(-1)[6]),
            "best_delta_pose6_robot": best_delta_pose6_robot,
            "recent_selected_pose_delta_norms": [
                {
                    "translation_norm": float(np.linalg.norm(delta[:3])),
                    "rotation_norm": float(np.linalg.norm(delta[3:6])),
                }
                for delta in self.recent_selected_pose_deltas
            ],
            "candidate_diagnostics_summary": candidate_summary,
            "candidate_diagnostics": candidate_diagnostics,
            "warp_modes": {
                view_name: ("copied_context" if "wrist" not in view_name.lower() else "geometric_extrinsics")
                for view_name in view_names
            },
        }
        return (
            best_action_ref,
            best_pred_by_view,
            best_next_pose_ref,
            best_target_robot_pose6.astype(np.float32),
            best_pred_mask_by_view,
            best_warp_by_view,
            best_warp_mask_by_view,
        )

    @torch.no_grad()
    def rollout_latent_for_action(self, context_by_view, states_by_view, extrinsics_by_view, view_names, action7_camera):
        ref_view = self._reference_view(view_names)
        action_tensor = torch.as_tensor(action7_camera, dtype=torch.float32, device=self.device).view(1, 1, 7)
        ref_current_state = states_by_view[ref_view][:, -1:].to(self.device, dtype=torch.float32)
        ref_next_state = compute_new_pose(ref_current_state, action_tensor).squeeze(1)
        ref_last_extrinsics = extrinsics_by_view[ref_view][0, -1].detach().cpu().numpy().astype(np.float32)
        target_robot_pose6 = self._compose_pose6(
            ref_last_extrinsics,
            ref_next_state[0, :6].detach().cpu().numpy().astype(np.float32),
        )
        pred_by_view = {}
        pred_mask_by_view = {}
        warp_by_view = {}
        warp_mask_by_view = {}
        next_state_by_view = {}
        predictor_cache_by_view = self._prepare_predictor_cache(
            context_by_view=context_by_view,
            states_by_view=states_by_view,
            extrinsics_by_view=extrinsics_by_view,
            view_names=view_names,
        )
        for view_name in view_names:
            view_cache = predictor_cache_by_view[view_name]
            obj_latents = view_cache["obj_latents"]
            current_view_pose6 = states_by_view[view_name][0, -1, :6].detach().cpu().numpy().astype(np.float32)
            current_view_gripper = float(states_by_view[view_name][0, -1, 6].item())
            view_extrinsics_np = extrinsics_by_view[view_name][0, -1].detach().cpu().numpy().astype(np.float32)
            is_wrist_view = "wrist" in view_name.lower()
            if is_wrist_view:
                camera_to_ee_pose6 = current_view_pose6
                next_extrinsics_np = self._compose_pose6(
                    target_robot_pose6,
                    self._transform_to_pose6(np.linalg.inv(self._pose6_to_transform(camera_to_ee_pose6))),
                )
                next_pose6 = camera_to_ee_pose6
            else:
                next_extrinsics_np = view_extrinsics_np
                next_pose6 = self._pose_in_parent_frame(target_robot_pose6, next_extrinsics_np)
            target_view_gripper = float(
                np.clip(
                    current_view_gripper + float(action7_camera[6]),
                    self.gripper_position_min,
                    self.gripper_position_max,
                )
            )
            next_state_np = np.concatenate(
                [next_pose6.astype(np.float32), np.array([target_view_gripper], dtype=np.float32)],
                axis=0,
            ).astype(np.float32)
            next_states = torch.from_numpy(next_state_np).to(self.device, dtype=torch.float32).view(1, 1, 7)
            next_extrinsics = torch.from_numpy(next_extrinsics_np).to(self.device, dtype=torch.float32).view(1, 1, 6)
            pred_logits, warped_masks = self._predict_from_cache(
                view_cache,
                next_states=next_states,
                next_extrinsics=next_extrinsics,
                use_copied_warp=not is_wrist_view,
            )
            pred_probs = F.softmax(pred_logits[:, -1], dim=1)
            warped_recon, warped_last_masks = self._warped_recon_from_masks(view_cache, warped_masks)
            pred_mask_by_view[view_name] = pred_probs
            pred_by_view[view_name] = torch.einsum("bkn,bkd->bnd", pred_probs.to(obj_latents.dtype), obj_latents)
            if warped_recon is not None:
                warp_by_view[view_name] = warped_recon
                warp_mask_by_view[view_name] = warped_last_masks
            next_state_by_view[view_name] = next_states.squeeze(1)
        return pred_by_view, pred_mask_by_view, warp_by_view, warp_mask_by_view, next_state_by_view

    @torch.no_grad()
    def rollout_latents_for_actions(self, context_by_view, states_by_view, extrinsics_by_view, view_names, action7_camera_batch):
        action7_camera_batch = np.asarray(action7_camera_batch, dtype=np.float32).reshape(-1, 7)
        batch_size = action7_camera_batch.shape[0]
        ref_view = self._reference_view(view_names)
        action_tensor = torch.as_tensor(action7_camera_batch, dtype=torch.float32, device=self.device).view(batch_size, 1, 7)
        ref_current_state = states_by_view[ref_view][:, -1:].to(self.device, dtype=torch.float32).expand(batch_size, -1, -1)
        ref_next_state = compute_new_pose(ref_current_state, action_tensor).squeeze(1)
        ref_last_extrinsics = extrinsics_by_view[ref_view][0, -1].detach().cpu().numpy().astype(np.float32)
        target_robot_pose6_all = []
        for idx in range(batch_size):
            target_robot_pose6_all.append(
                self._compose_pose6(
                    ref_last_extrinsics,
                    ref_next_state[idx, :6].detach().cpu().numpy().astype(np.float32),
                )
            )
        target_robot_pose6_all = np.stack(target_robot_pose6_all, axis=0).astype(np.float32)

        pred_by_view = {}
        pred_mask_by_view = {}
        warp_by_view = {}
        warp_mask_by_view = {}
        next_state_by_view = {}
        predictor_cache_by_view = self._prepare_predictor_cache(
            context_by_view=context_by_view,
            states_by_view=states_by_view,
            extrinsics_by_view=extrinsics_by_view,
            view_names=view_names,
        )
        for view_name in view_names:
            view_cache = predictor_cache_by_view[view_name]
            obj_latents = view_cache["obj_latents"].expand(batch_size, -1, -1)
            current_view_pose6 = states_by_view[view_name][0, -1, :6].detach().cpu().numpy().astype(np.float32)
            current_view_gripper = float(states_by_view[view_name][0, -1, 6].item())
            view_extrinsics_np = extrinsics_by_view[view_name][0, -1].detach().cpu().numpy().astype(np.float32)
            is_wrist_view = "wrist" in view_name.lower()
            target_state7_batch = []
            target_extrinsics_batch = []
            for idx in range(batch_size):
                target_robot_pose6 = target_robot_pose6_all[idx]
                if is_wrist_view:
                    camera_to_ee_pose6 = current_view_pose6
                    target_view_extrinsics = self._compose_pose6(
                        target_robot_pose6,
                        self._transform_to_pose6(np.linalg.inv(self._pose6_to_transform(camera_to_ee_pose6))),
                    )
                    next_pose6 = camera_to_ee_pose6
                else:
                    target_view_extrinsics = view_extrinsics_np
                    next_pose6 = self._pose_in_parent_frame(target_robot_pose6, target_view_extrinsics)
                target_view_gripper = float(
                    np.clip(
                        current_view_gripper + float(action7_camera_batch[idx, 6]),
                        self.gripper_position_min,
                        self.gripper_position_max,
                    )
                )
                target_state7_batch.append(
                    np.concatenate(
                        [
                            next_pose6.astype(np.float32),
                            np.array([target_view_gripper], dtype=np.float32),
                        ],
                        axis=0,
                    ).astype(np.float32)
                )
                target_extrinsics_batch.append(target_view_extrinsics.astype(np.float32))

            next_states = torch.from_numpy(np.stack(target_state7_batch, axis=0)).to(self.device, dtype=torch.float32).unsqueeze(1)
            next_extrinsics = torch.from_numpy(np.stack(target_extrinsics_batch, axis=0)).to(self.device, dtype=torch.float32).unsqueeze(1)
            pred_logits, warped_masks = self._predict_from_cache(
                view_cache,
                next_states=next_states,
                next_extrinsics=next_extrinsics,
                use_copied_warp=not is_wrist_view,
            )
            pred_probs = F.softmax(pred_logits[:, -1], dim=1)
            warped_recon, warped_last_masks = self._warped_recon_from_masks(view_cache, warped_masks)
            pred_mask_by_view[view_name] = pred_probs
            pred_by_view[view_name] = torch.einsum("bkn,bkd->bnd", pred_probs.to(obj_latents.dtype), obj_latents)
            if warped_recon is not None:
                warp_by_view[view_name] = warped_recon
                warp_mask_by_view[view_name] = warped_last_masks
            next_state_by_view[view_name] = next_states.squeeze(1)
        return pred_by_view, pred_mask_by_view, warp_by_view, warp_mask_by_view, next_state_by_view

    def predict_one_step_candidates(self, data):
        if "pre_obs_sequence" in data:
            pre_obs_sequence = [self._parse_observation_views(item) for item in data["pre_obs_sequence"]]
            if not pre_obs_sequence:
                raise ValueError("pre_obs_sequence is empty")
            pre_obs_views = pre_obs_sequence[-1]
        else:
            pre_obs_views = self._parse_observation_views(data["pre_obs"])
            pre_obs_sequence = [pre_obs_views]
        action7_camera_batch = np.asarray(data["actions"], dtype=np.float32).reshape(-1, 7)
        independent = bool(data.get("independent", True))
        if independent:
            self._clear_temporal_buffers()
        available_views = self._filter_available_views(pre_obs_views)
        self._append_obs_sequence_to_buffers(pre_obs_sequence, available_views)
        self._initialize_fixed_latents_if_needed(pre_obs_views)
        context_by_view, states_by_view, extrinsics_by_view, _ = self._build_context_from_buffers(available_views)
        pred_by_view, pred_mask_by_view, warp_by_view, warp_mask_by_view, pred_next_state_by_view = self.rollout_latents_for_actions(
            context_by_view=context_by_view,
            states_by_view=states_by_view,
            extrinsics_by_view=extrinsics_by_view,
            view_names=available_views,
            action7_camera_batch=action7_camera_batch,
        )

        prediction_ids = []
        ref_view = self._reference_view(available_views)
        for idx in range(action7_camera_batch.shape[0]):
            prediction_id = self.next_pending_prediction_id
            self.next_pending_prediction_id += 1
            prediction_ids.append(prediction_id)
            self.pending_one_step_predictions[prediction_id] = {
                "view_names": list(available_views),
                "reference_view": ref_view,
                "action7_camera": action7_camera_batch[idx].astype(np.float32).copy(),
                "pre_state_by_view": {
                    view_name: pre_obs_views[view_name]["state7"].astype(np.float32).copy()
                    for view_name in available_views
                },
                "pre_extrinsics_by_view": {
                    view_name: pre_obs_views[view_name]["extrinsics6"].astype(np.float32).copy()
                    for view_name in available_views
                },
                "camera_frame_convention_by_view": {
                    view_name: pre_obs_views[view_name].get("camera_frame_convention")
                    for view_name in available_views
                },
                "pred_by_view": {
                    view_name: pred_by_view[view_name][idx].detach().cpu().numpy().astype(np.float32).copy()
                    for view_name in available_views
                },
                "pred_mask_by_view": {
                    view_name: pred_mask_by_view[view_name][idx].detach().cpu().numpy().astype(np.float32).copy()
                    for view_name in available_views
                },
                "pre_mask_by_view": {
                    view_name: context_by_view[view_name]["decoded_mask_probs"][:, -1].squeeze(0).detach().cpu().numpy().astype(np.float32).copy()
                    for view_name in available_views
                },
                "warp_by_view": {
                    view_name: warp_by_view[view_name][idx].detach().cpu().numpy().astype(np.float32).copy()
                    for view_name in available_views
                    if view_name in warp_by_view
                },
                "warp_mask_by_view": {
                    view_name: warp_mask_by_view[view_name][idx].detach().cpu().numpy().astype(np.float32).copy()
                    for view_name in available_views
                    if view_name in warp_mask_by_view
                },
                "pred_next_state_by_view": {
                    view_name: pred_next_state_by_view[view_name][idx].detach().cpu().numpy().astype(np.float32).copy()
                    for view_name in available_views
                },
            }
        return {
            "status": "success",
            "prediction_ids": prediction_ids,
            "num_candidates": int(len(prediction_ids)),
            "view_names": list(available_views),
            "reference_view": ref_view,
        }

    def record_one_step_prediction_post(self, data):
        prediction_id = int(data["prediction_id"])
        if prediction_id not in self.pending_one_step_predictions:
            raise KeyError(f"Unknown pending prediction_id: {prediction_id}")
        pending = self.pending_one_step_predictions[prediction_id]
        post_obs_views = self._parse_observation_views(data["post_obs"])
        available_views = [view_name for view_name in pending["view_names"] if view_name in post_obs_views]
        if not available_views:
            raise ValueError("No overlapping views between pending prediction and post_obs.")

        gt_next_latent_by_view = {}
        gt_next_mask_by_view = {}
        gt_next_state_by_view = {}
        per_view_losses = {}
        per_view_transport_losses = {}
        per_view_warp_gt_losses = {}
        per_view_warp_pred_losses = {}
        for view_name in available_views:
            post_frame = self._preprocess_image_tensor(post_obs_views[view_name]["image"])
            post_decoded = self._decode_single_frame_context(post_frame, self.fixed_obj_latents[view_name])
            gt_next_latent_by_view[view_name] = post_decoded["decoder_recon"].squeeze(0)
            gt_next_mask_by_view[view_name] = post_decoded["decoded_mask_probs"].squeeze(0)
            gt_next_state_by_view[view_name] = post_obs_views[view_name]["state7"].astype(np.float32).copy()

            pred_latent = torch.from_numpy(pending["pred_by_view"][view_name]).to(self.device, dtype=gt_next_latent_by_view[view_name].dtype)
            pred_mask = torch.from_numpy(pending["pred_mask_by_view"][view_name]).to(self.device, dtype=gt_next_mask_by_view[view_name].dtype)
            per_view_losses[view_name] = float(torch.mean(torch.abs(pred_latent - gt_next_latent_by_view[view_name])).item())
            if view_name in pending.get("warp_by_view", {}):
                warp_latent = torch.from_numpy(pending["warp_by_view"][view_name]).to(self.device, dtype=gt_next_latent_by_view[view_name].dtype)
                per_view_warp_gt_losses[view_name] = float(torch.mean(torch.abs(warp_latent - gt_next_latent_by_view[view_name])).item())
                per_view_warp_pred_losses[view_name] = float(torch.mean(torch.abs(warp_latent - pred_latent)).item())
            pre_mask = torch.from_numpy(pending["pre_mask_by_view"][view_name]).to(
                self.device,
                dtype=gt_next_mask_by_view[view_name].dtype,
            )
            slot_weights = self._slot_change_weights(
                pre_mask.unsqueeze(0),
                gt_next_mask_by_view[view_name].unsqueeze(0),
            )
            per_view_transport_losses[view_name] = float(
                self._sinkhorn_mask_transport_loss(
                    pred_mask.unsqueeze(0),
                    gt_next_mask_by_view[view_name].unsqueeze(0),
                    spatial_size=self.crop_size // self.patch_size,
                    slot_weights=slot_weights,
                    area_normalizer=self._slot_area_normalizer_for_view(view_name, device=self.device),
                ).mean().item()
            )

        self.history_z_context.append(
            np.stack([gt_next_latent_by_view[view_name].detach().cpu().numpy() for view_name in available_views], axis=0)
        )
        if pending.get("warp_by_view"):
            self.history_z_warp.append(
                np.stack(
                    [
                        pending["warp_by_view"].get(view_name, gt_next_latent_by_view[view_name].detach().cpu().numpy())
                        for view_name in available_views
                    ],
                    axis=0,
                )
            )
        self.history_z_pred.append(
            np.stack([pending["pred_by_view"][view_name] for view_name in available_views], axis=0)
        )
        self.history_mask_context.append(
            np.stack([gt_next_mask_by_view[view_name].detach().cpu().numpy() for view_name in available_views], axis=0)
        )
        if pending.get("warp_mask_by_view"):
            self.history_mask_warp.append(
                np.stack(
                    [
                        pending["warp_mask_by_view"].get(view_name, gt_next_mask_by_view[view_name].detach().cpu().numpy())
                        for view_name in available_views
                    ],
                    axis=0,
                )
            )
        self.history_mask_pred.append(
            np.stack([pending["pred_mask_by_view"][view_name] for view_name in available_views], axis=0)
        )

        ref_view = pending["reference_view"] if pending["reference_view"] in available_views else available_views[0]
        pred_next_state_ref = pending["pred_next_state_by_view"][ref_view].astype(np.float32).copy()
        gt_next_state_ref = gt_next_state_by_view[ref_view]
        obs_pre_state_ref = pending["pre_state_by_view"][ref_view].astype(np.float32).copy()
        self.history_debug.append(
            {
                "step": len(self.history_debug),
                "mode": "parallel_one_step_action_rollout",
                "prediction_id": int(prediction_id),
                "view_names": list(available_views),
                "reference_view": ref_view,
                "action7_camera": pending["action7_camera"].copy(),
                "per_view_losses": per_view_losses,
                "per_view_iou_losses": per_view_transport_losses,
                "per_view_transport_losses": per_view_transport_losses,
                "per_view_warp_gt_losses": per_view_warp_gt_losses,
                "per_view_warp_pred_losses": per_view_warp_pred_losses,
                "mean_loss": float(np.mean(list(per_view_losses.values()))),
                "mean_iou_loss": float(np.mean(list(per_view_transport_losses.values()))),
                "mean_transport_loss": float(np.mean(list(per_view_transport_losses.values()))),
                "predicted_next_state7_camera": pred_next_state_ref.copy(),
                "gt_next_state7_camera": gt_next_state_ref.copy(),
                "predicted_delta_xyz_camera": (pred_next_state_ref[:3] - obs_pre_state_ref[:3]).astype(np.float32).copy(),
                "gt_delta_xyz_camera": (gt_next_state_ref[:3] - obs_pre_state_ref[:3]).astype(np.float32).copy(),
                "obs_by_view": {
                    view_name: {
                        "pre_state7": pending["pre_state_by_view"][view_name].astype(np.float32).copy(),
                        "post_state7": post_obs_views[view_name]["state7"].astype(np.float32).copy(),
                        "camera_extrinsics": pending["pre_extrinsics_by_view"][view_name].astype(np.float32).copy(),
                    }
                    for view_name in available_views
                },
            }
        )
        self.pending_one_step_predictions.pop(prediction_id, None)
        return {
            "status": "success",
            "prediction_id": int(prediction_id),
            "per_view_losses": per_view_losses,
            "per_view_iou_losses": per_view_transport_losses,
            "per_view_transport_losses": per_view_transport_losses,
            "mean_loss": float(np.mean(list(per_view_losses.values()))),
            "mean_iou_loss": float(np.mean(list(per_view_transport_losses.values()))),
            "mean_transport_loss": float(np.mean(list(per_view_transport_losses.values()))),
        }

    @torch.no_grad()
    def score_observations_to_goal(self, data):
        if self.goal_image_maps is None:
            raise ValueError("Goal images must be set before scoring observations.")
        if "obs_sequence" in data:
            obs_sequence = [self._parse_observation_views(item) for item in data["obs_sequence"]]
            if not obs_sequence:
                raise ValueError("obs_sequence is empty")
        elif "pre_obs_sequence" in data:
            obs_sequence = [self._parse_observation_views(item) for item in data["pre_obs_sequence"]]
            if not obs_sequence:
                raise ValueError("pre_obs_sequence is empty")
        else:
            obs_sequence = [self._parse_observation_views(data["obs"])]

        available_views = self._filter_available_views(obs_sequence[-1])
        initialized_for_score = False
        if not self.fixed_latent_initialized:
            self._clear_temporal_buffers()
            self._append_obs_sequence_to_buffers(obs_sequence, available_views)
            self._initialize_fixed_latents_if_needed(obs_sequence[-1])
            self._clear_temporal_buffers()
            initialized_for_score = True

        if self.goal_features is None or self.goal_mask_probs is None:
            raise ValueError("SLOT goal features are unavailable after latent initialization.")
        goal_idx = int(data.get("goal_idx", self.current_goal_idx))
        goal_idx = int(np.clip(goal_idx, 0, int(self.goal_features.shape[0] - 1)))
        if bool(data.get("store_as_warmup", len(obs_sequence) > 1)):
            self.warmup_score_obs_sequence = self._copy_parsed_obs_sequence(obs_sequence)
        result = self._score_parsed_observations_to_goal(
            obs_sequence=obs_sequence,
            available_views=available_views,
            goal_idx=goal_idx,
            update_memory=True,
            update_warmup_anchor=bool(data.get("update_warmup_anchor", True)),
            memory_source=str(data.get("memory_source", "warmup_observed_goal_score")),
        )
        result["initialized_fixed_latents_for_score"] = bool(initialized_for_score)
        return result

    @torch.no_grad()
    def _score_parsed_observations_to_goal(
        self,
        obs_sequence,
        available_views,
        goal_idx,
        update_memory=True,
        update_warmup_anchor=True,
        memory_source="warmup_observed_goal_score",
    ):
        if self.goal_features is None or self.goal_mask_probs is None:
            raise ValueError("SLOT goal features are unavailable.")
        goal_idx = int(np.clip(int(goal_idx), 0, int(self.goal_features.shape[0] - 1)))
        goal_view_indices = {view_name: idx for idx, view_name in enumerate(self.goal_view_names)}
        records = []
        warmup_masks_by_view = {view_name: [] for view_name in available_views}
        for obs_idx, obs_views in enumerate(obs_sequence):
            step_views = [view_name for view_name in available_views if view_name in obs_views and view_name in goal_view_indices]
            per_view_l1 = {}
            per_view_mask_l1 = {}
            per_view_transport = {}
            per_view_objective = {}
            for view_name in step_views:
                frame = self._preprocess_image_tensor(obs_views[view_name]["image"])
                decoded = self._decode_single_frame_context(frame, self.fixed_obj_latents[view_name])
                recon = decoded["decoder_recon"].squeeze(0).to(self.device, dtype=torch.float32)
                mask_probs = decoded["decoded_mask_probs"].squeeze(0).to(self.device, dtype=torch.float32)
                warmup_masks_by_view.setdefault(view_name, []).append(mask_probs.detach())
                goal_recon = self.goal_features[goal_idx, goal_view_indices[view_name]].to(
                    self.device, dtype=torch.float32
                )
                goal_mask = self.goal_mask_probs[goal_idx, goal_view_indices[view_name]].to(self.device, dtype=torch.float32)
                l1_loss = torch.mean(torch.abs(recon - goal_recon))
                mask_l1_loss = torch.mean(torch.abs(mask_probs - goal_mask))
                slot_weights = self._slot_change_weights(mask_probs.unsqueeze(0), goal_mask.unsqueeze(0))
                transport_loss = self._sinkhorn_mask_transport_loss(
                    mask_probs.unsqueeze(0),
                    goal_mask.unsqueeze(0),
                    spatial_size=self.crop_size // self.patch_size,
                    slot_weights=slot_weights,
                    area_normalizer=self._slot_area_normalizer_for_view(view_name, device=self.device),
                ).mean()
                objective_loss = (
                    (1.0 - self.transport_loss_mix) * mask_l1_loss
                    + self.transport_loss_mix * transport_loss
                )
                per_view_l1[view_name] = float(l1_loss.detach().cpu().item())
                per_view_mask_l1[view_name] = float(mask_l1_loss.detach().cpu().item())
                per_view_transport[view_name] = float(transport_loss.detach().cpu().item())
                per_view_objective[view_name] = float(objective_loss.detach().cpu().item())
            records.append(
                {
                    "obs_idx": int(obs_idx),
                    "goal_idx": int(goal_idx),
                    "view_names": list(step_views),
                    "per_view_l1_losses": per_view_l1,
                    "per_view_mask_l1_losses": per_view_mask_l1,
                    "per_view_transport_losses": per_view_transport,
                    "per_view_objective_losses": per_view_objective,
                    "mean_l1_loss": float(np.mean(list(per_view_l1.values()))) if per_view_l1 else None,
                    "mean_mask_l1_loss": (
                        float(np.mean(list(per_view_mask_l1.values())))
                        if per_view_mask_l1
                        else None
                    ),
                    "mean_transport_loss": float(np.mean(list(per_view_transport.values()))) if per_view_transport else None,
                    "mean_objective_loss": float(np.mean(list(per_view_objective.values()))) if per_view_objective else None,
                }
            )
            if per_view_objective:
                ref_view = self._reference_view(step_views)
                ref_state7 = obs_views[ref_view]["state7"].astype(np.float32)
                ref_extrinsics6 = obs_views[ref_view]["extrinsics6"].astype(np.float32)
                robot_pose6 = self._compose_pose6(ref_extrinsics6, ref_state7[:6]).astype(np.float32)
                current_goal_target = self._current_goal_target()
                if update_memory:
                    self._add_pose_memory_entry(
                        pose6_robot=robot_pose6,
                        gripper_state=float(ref_state7[6]),
                        per_view_losses=per_view_objective,
                        source=memory_source,
                        stage=None if current_goal_target is None else current_goal_target.get("stage"),
                        step=int(obs_idx),
                        extra={
                            "goal_idx": int(goal_idx),
                            "mean_l1_loss": float(np.mean(list(per_view_l1.values()))) if per_view_l1 else None,
                            "mean_mask_l1_loss": (
                                float(np.mean(list(per_view_mask_l1.values())))
                                if per_view_mask_l1
                                else None
                            ),
                            "mean_transport_loss": float(np.mean(list(per_view_transport.values()))) if per_view_transport else None,
                        },
                    )
                    if update_warmup_anchor:
                        self._update_warmup_fixed_anchor(
                            goal_idx=goal_idx,
                            pose6_robot=robot_pose6,
                            gripper_state=float(ref_state7[6]),
                            per_view_losses=per_view_objective,
                            step=int(obs_idx),
                        )
        if update_memory and str(memory_source).startswith("warmup"):
            self._update_fixed_slots_from_warmup_masks(warmup_masks_by_view)
            self._update_gripper_slot_from_warmup_masks(warmup_masks_by_view)
        return {
            "status": "success",
            "goal_idx": int(goal_idx),
            "transport_loss_mix": float(self.transport_loss_mix),
            "records": records,
            "pose_memory_sizes": {
                view_name: int(len(self.pose_memory_by_view.get(view_name, [])))
                for view_name in self.VIEW_ORDER
            },
            "warmup_fixed_anchor": self.warmup_fixed_anchor_by_goal.get(int(goal_idx)),
            "update_warmup_anchor": bool(update_warmup_anchor),
            "fixed_slot_index_by_view": dict(self.fixed_slot_index_by_view),
            "fixed_slot_debug_by_view": dict(self.fixed_slot_debug_by_view),
            "gripper_slot_index_by_view": dict(self.gripper_slot_index_by_view),
            "gripper_slot_debug_by_view": dict(self.gripper_slot_debug_by_view),
        }

    def predict_action(self, data):
        t_predict_start = time.perf_counter()
        t_parse_obs_start = time.perf_counter()
        self.current_completed_stages = set(str(stage) for stage in data.get("completed_stages", []))
        planning_request = str(data.get("slot_planning_request", "joint") or "joint").lower()
        if planning_request not in {"joint", "pose_only", "gripper_only", "score_only"}:
            planning_request = "joint"
        if "pre_obs_sequence" in data:
            obs_sequence = [self._parse_observation_views(item) for item in data["pre_obs_sequence"]]
            if not obs_sequence:
                raise ValueError("pre_obs_sequence is empty")
            # A3 is a strictly causal recent-context baseline.  Its frame 0
            # is the real observation captured before any SLOT warmup motion,
            # supplied separately from the warmup trajectory.  Warmup frames
            # must never become temporal predictor context.  History selectors
            # left-pad frame 0 until real online observations accumulate.
            if self.memory_selection_mode == "recent":
                temporal_initial_obs = data.get("temporal_initial_obs")
                if temporal_initial_obs is None:
                    raise ValueError(
                        "A3 recent context requires temporal_initial_obs on its "
                        "first request; warmup frames cannot initialize temporal history."
                    )
                obs_sequence = [self._parse_observation_views(temporal_initial_obs)]
            obs_views = obs_sequence[-1]
        else:
            obs_sequence = None
            obs_views = self._parse_observation_views(data)
        available_views = self._filter_available_views(obs_views)
        parse_obs_duration_s = time.perf_counter() - t_parse_obs_start

        t_append_start = time.perf_counter()
        if obs_sequence is not None:
            self._append_obs_sequence_to_buffers(obs_sequence, available_views)
        else:
            self._append_obs_views_to_buffers(obs_views, available_views)
        append_duration_s = time.perf_counter() - t_append_start

        t_refresh_start = time.perf_counter()
        initialized_before = self.fixed_latent_initialized
        self._initialize_fixed_latents_if_needed(obs_views)
        refresh_duration_s = time.perf_counter() - t_refresh_start
        initialized_this_step = (not initialized_before) and self.fixed_latent_initialized

        if self.goal_features is None:
            return {
                "action": np.zeros((1, 7), dtype=np.float32),
                "best_target_robot_pose6": None,
                "predicted_next_state7_camera": None,
                "reference_view": None,
                "best_loss": None,
                "per_view_losses": {},
            }

        t_context_start = time.perf_counter()
        context_by_view, states_by_view, extrinsics_by_view, current_latent_by_view = self._build_context_from_buffers(available_views)
        build_context_duration_s = time.perf_counter() - t_context_start

        t_opt_start = time.perf_counter()
        if planning_request == "score_only":
            ref_view_for_score = self._reference_view(available_views)
            ref_state_for_score = obs_views[ref_view_for_score]["state7"].astype(np.float32).reshape(-1)
            current_robot_pose6 = np.asarray(
                data.get("current_robot_pose6", np.zeros(6, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            if current_robot_pose6.shape[0] != 6:
                current_robot_pose6 = np.zeros(6, dtype=np.float32)
            best_action = np.zeros(7, dtype=np.float32)
            best_z_pred_by_view = {
                view_name: current_latent_by_view[view_name]
                for view_name in available_views
            }
            best_next_pose = torch.as_tensor(
                ref_state_for_score,
                dtype=torch.float32,
                device=self.device,
            ).view(1, 7)
            best_target_robot_pose6 = current_robot_pose6.astype(np.float32).copy()
            best_mask_pred_by_view = {
                view_name: context_by_view[view_name]["decoded_mask_probs"][:, -1]
                for view_name in available_views
            }
            best_z_warp_by_view = {}
            best_mask_warp_by_view = {}
            self.last_predictor_debug = {
                "mode": "score_only",
                "cem_phase_outputs": [],
                "candidate_diagnostics_summary": {},
                "candidate_diagnostics": [],
                "mask_transport_loss": None,
                "optimize_timing": {"score_only": True},
            }
        else:
            (
                best_action,
                best_z_pred_by_view,
                best_next_pose,
                best_target_robot_pose6,
                best_mask_pred_by_view,
                best_z_warp_by_view,
                best_mask_warp_by_view,
            ) = self.optimize_action(
                context_by_view=context_by_view,
                states_by_view=states_by_view,
                extrinsics_by_view=extrinsics_by_view,
                view_names=available_views,
                current_obs_views=obs_views,
                planning_request=planning_request,
            )
        optimize_duration_s = time.perf_counter() - t_opt_start

        goal_features = self.goal_features[self.current_goal_idx].to(self.device)
        goal_view_indices = {view_name: idx for idx, view_name in enumerate(self.goal_view_names)}
        per_view_losses = {}
        per_view_mask_l1_losses = {}
        per_view_transport_losses = {}
        per_view_objective_losses = {}
        per_view_gripper_slot_transport_losses = {}
        per_view_other_slot_mean_transport_losses = {}
        per_view_gripper_mixed_transport_losses = {}
        final_gripper_slot_losses = (
            {}
            if self.last_predictor_debug is None
            else self.last_predictor_debug.get("final_gripper_slot_transport_losses", {})
        )
        final_other_slot_losses = (
            {}
            if self.last_predictor_debug is None
            else self.last_predictor_debug.get("final_other_slot_mean_transport_losses", {})
        )
        final_gripper_mixed_losses = (
            {}
            if self.last_predictor_debug is None
            else self.last_predictor_debug.get("final_gripper_mixed_transport_losses", {})
        )
        per_view_warp_goal_losses = {}
        per_view_warp_pred_losses = {}
        pred_stack = []
        goal_stack = []
        for view_name in available_views:
            pred = best_z_pred_by_view[view_name]
            goal = goal_features[goal_view_indices[view_name]]
            l1_loss = torch.mean(torch.abs(pred - goal))
            goal_mask = self.goal_mask_probs[self.current_goal_idx, goal_view_indices[view_name]].to(
                self.device, dtype=torch.float32
            )
            pred_mask = best_mask_pred_by_view[view_name].to(self.device, dtype=torch.float32)
            mask_l1_loss = torch.mean(
                torch.abs(pred_mask - goal_mask.unsqueeze(0).expand_as(pred_mask))
            )
            per_view_losses[view_name] = float(l1_loss.item())
            per_view_mask_l1_losses[view_name] = float(mask_l1_loss.item())
            if self.transport_loss_mix > 0.0:
                current_mask = context_by_view[view_name]["decoded_mask_probs"][:, -1].to(self.device, dtype=torch.float32)
                slot_weights = self._slot_change_weights(current_mask, goal_mask.unsqueeze(0))
                transport_loss = self._sinkhorn_mask_transport_loss(
                    pred_mask,
                    goal_mask.unsqueeze(0).expand_as(pred_mask),
                    spatial_size=self.crop_size // self.patch_size,
                    slot_weights=slot_weights,
                    area_normalizer=self._slot_area_normalizer_for_view(view_name, device=self.device),
                ).mean()
                per_view_transport_losses[view_name] = float(transport_loss.item())
                objective_loss = (
                    (1.0 - self.transport_loss_mix) * mask_l1_loss
                    + self.transport_loss_mix * transport_loss
                )
            else:
                objective_loss = mask_l1_loss
            per_view_objective_losses[view_name] = float(objective_loss.item())
            if view_name in final_gripper_slot_losses:
                per_view_gripper_slot_transport_losses[view_name] = float(
                    np.asarray(final_gripper_slot_losses[view_name], dtype=np.float32).reshape(-1)[0]
                )
            if view_name in final_other_slot_losses:
                per_view_other_slot_mean_transport_losses[view_name] = float(
                    np.asarray(final_other_slot_losses[view_name], dtype=np.float32).reshape(-1)[0]
                )
            if view_name in final_gripper_mixed_losses:
                per_view_gripper_mixed_transport_losses[view_name] = float(
                    np.asarray(final_gripper_mixed_losses[view_name], dtype=np.float32).reshape(-1)[0]
                )
            if view_name in best_z_warp_by_view:
                warp = best_z_warp_by_view[view_name]
                per_view_warp_goal_losses[view_name] = float(torch.mean(torch.abs(warp - goal)).item())
                per_view_warp_pred_losses[view_name] = float(torch.mean(torch.abs(warp - pred)).item())
            pred_stack.append(pred)
            goal_stack.append(goal.unsqueeze(0))
        best_l1_loss = float(torch.mean(torch.abs(torch.cat(pred_stack, dim=1) - torch.cat(goal_stack, dim=1))).item())
        best_mask_l1_loss = float(np.mean(list(per_view_mask_l1_losses.values())))
        best_loss = float(np.mean(list(per_view_objective_losses.values())))

        ref_view = self._reference_view(available_views)
        current_goal_target = self._current_goal_target()
        ref_state7 = obs_views[ref_view]["state7"]
        best_action_np = np.asarray(best_action, dtype=np.float32).reshape(-1)
        best_next_pose_np = best_next_pose.detach().cpu().numpy().reshape(-1).astype(np.float32)
        gripper_execution_downpress_z_robot = 0.0
        if planning_request == "gripper_only":
            gripper_execution_downpress_z_robot = -float(self.gripper_downpress_max_z) * float(
                np.clip(best_next_pose_np[6], self.gripper_position_min, self.gripper_position_max)
            )
        goal_state7_camera_np = None
        goal_delta_xyz_camera = None
        predicted_delta_xyz_camera = best_next_pose_np[:3] - ref_state7[:3]
        predicted_goal_delta_xyz_camera = None
        action_goal_cosine = None
        predicted_step_goal_cosine = None
        if current_goal_target is not None and current_goal_target.get("ee_state7_camera") is not None:
            goal_state7_camera_np = np.asarray(current_goal_target["ee_state7_camera"], dtype=np.float32)
            goal_delta_xyz_camera = goal_state7_camera_np[:3] - ref_state7[:3]
            predicted_goal_delta_xyz_camera = goal_state7_camera_np[:3] - best_next_pose_np[:3]
            action_goal_cosine = self._cosine_similarity(best_action_np[:3], goal_delta_xyz_camera)
            predicted_step_goal_cosine = self._cosine_similarity(predicted_delta_xyz_camera, goal_delta_xyz_camera)

        self.history_z_context.append(
            np.stack([current_latent_by_view[view_name].squeeze(0).detach().cpu().numpy() for view_name in available_views], axis=0)
        )
        if best_z_warp_by_view:
            self.history_z_warp.append(
                np.stack(
                    [
                        best_z_warp_by_view.get(view_name, current_latent_by_view[view_name]).squeeze(0).detach().cpu().numpy()
                        for view_name in available_views
                    ],
                    axis=0,
                )
            )
        self.history_z_pred.append(
            np.stack([best_z_pred_by_view[view_name].squeeze(0).detach().cpu().numpy() for view_name in available_views], axis=0)
        )
        self.history_mask_context.append(
            np.stack(
                [
                    context_by_view[view_name]["decoded_mask_probs"][:, -1].squeeze(0).detach().cpu().numpy()
                    for view_name in available_views
                ],
                axis=0,
            )
        )
        if best_mask_warp_by_view:
            self.history_mask_warp.append(
                np.stack(
                    [
                        best_mask_warp_by_view.get(
                            view_name,
                            context_by_view[view_name]["decoded_mask_probs"][:, -1].squeeze(0),
                        ).squeeze(0).detach().cpu().numpy()
                        for view_name in available_views
                    ],
                    axis=0,
                )
            )
        self.history_mask_pred.append(
            np.stack(
                [best_mask_pred_by_view[view_name].squeeze(0).detach().cpu().numpy() for view_name in available_views],
                axis=0,
            )
        )
        self.history_debug.append(
            {
                "step": len(self.history_debug),
                "view_names": list(available_views),
                "reference_view": ref_view,
                "goal_idx": int(self.current_goal_idx),
                "goal_stage": None if self.goal_state is None else self.goal_state.get("target_stages", [None])[self.current_goal_idx],
                "slot_planning_request": planning_request,
                "best_loss": best_loss,
                "best_l1_loss": best_l1_loss,
                "best_mask_l1_loss": best_mask_l1_loss,
                "per_view_losses": per_view_losses,
                "per_view_mask_l1_losses": per_view_mask_l1_losses,
                "per_view_transport_losses": per_view_transport_losses,
                "per_view_objective_losses": per_view_objective_losses,
                "per_view_gripper_slot_transport_losses": per_view_gripper_slot_transport_losses,
                "per_view_other_slot_mean_transport_losses": per_view_other_slot_mean_transport_losses,
                "per_view_gripper_mixed_transport_losses": per_view_gripper_mixed_transport_losses,
                "fixed_slot_index_by_view": dict(self.fixed_slot_index_by_view),
                "fixed_slot_debug_by_view": dict(self.fixed_slot_debug_by_view),
                "gripper_slot_index_by_view": dict(self.gripper_slot_index_by_view),
                "gripper_slot_debug_by_view": dict(self.gripper_slot_debug_by_view),
                "per_view_warp_goal_losses": per_view_warp_goal_losses,
                "per_view_warp_pred_losses": per_view_warp_pred_losses,
                "best_action": best_action_np,
                "best_target_robot_pose6": best_target_robot_pose6.astype(np.float32).copy(),
                "gripper_execution_downpress_z_robot": float(gripper_execution_downpress_z_robot),
                "predicted_next_state7_camera": best_next_pose_np.copy(),
                "predicted_delta_xyz_camera": predicted_delta_xyz_camera.astype(np.float32).copy(),
                "goal_delta_xyz_camera": None if goal_delta_xyz_camera is None else goal_delta_xyz_camera.astype(np.float32).copy(),
                "predicted_goal_delta_xyz_camera": None if predicted_goal_delta_xyz_camera is None else predicted_goal_delta_xyz_camera.astype(np.float32).copy(),
                "action_goal_cosine": action_goal_cosine,
                "predicted_step_goal_cosine": predicted_step_goal_cosine,
                "goal_name": None if self.goal_state is None else self.goal_state.get("target_names", [None])[self.current_goal_idx],
                "goal_target": current_goal_target,
                "goal_state_summary": None if self.goal_state is None else {
                    "success_case_dir": self.goal_state.get("success_case_dir"),
                    "target_names": self.goal_state.get("target_names"),
                    "target_stages": self.goal_state.get("target_stages"),
                    "available_views": self.goal_state.get("available_views"),
                },
                "obs_by_view": {
                    view_name: {
                        "state7": obs_views[view_name]["state7"].astype(np.float32).copy(),
                        "camera_extrinsics": obs_views[view_name]["extrinsics6"].astype(np.float32).copy(),
                        "camera_frame_convention": obs_views[view_name].get("camera_frame_convention"),
                        "state_summary": {
                            "state_xyz_norm": float(np.linalg.norm(obs_views[view_name]["state7"][:3])),
                            "state_rpy_norm": float(np.linalg.norm(obs_views[view_name]["state7"][3:6])),
                            "gripper_raw": float(obs_views[view_name]["state7"][6]),
                            "extrinsics_xyz_norm": float(np.linalg.norm(obs_views[view_name]["extrinsics6"][:3])),
                            "extrinsics_rpy_norm": float(np.linalg.norm(obs_views[view_name]["extrinsics6"][3:6])),
                        },
                    }
                    for view_name in available_views
                },
                "predictor_debug": self.last_predictor_debug,
                "obj_latent_refresh": {
                    "refreshed_this_step": False,
                    "initialized_this_step": bool(initialized_this_step),
                    "refresh_interval": 0,
                    "last_refresh_step": int(self.last_obj_latent_refresh_step),
                    "last_refresh_reason": self.last_obj_latent_refresh_reason,
                },
            }
        )
        self.predict_step_idx += 1
        total_duration_s = time.perf_counter() - t_predict_start
        return {
            "best_target_robot_pose6": best_target_robot_pose6.astype(np.float32).copy(),
            "predicted_next_state7_camera": best_next_pose_np.copy(),
            "gripper_execution_downpress_z_robot": float(gripper_execution_downpress_z_robot),
            "best_target_gripper_state_model": float(best_next_pose_np[6]),
            "slot_planning_request": planning_request,
            "reference_view": ref_view,
            "best_loss": float(best_loss),
            "best_l1_loss": float(best_l1_loss),
            "best_mask_l1_loss": float(best_mask_l1_loss),
            "per_view_losses": per_view_losses,
            "per_view_mask_l1_losses": per_view_mask_l1_losses,
            "per_view_transport_losses": per_view_transport_losses,
            "per_view_objective_losses": per_view_objective_losses,
            "per_view_gripper_slot_transport_losses": per_view_gripper_slot_transport_losses,
            "per_view_other_slot_mean_transport_losses": per_view_other_slot_mean_transport_losses,
            "per_view_gripper_mixed_transport_losses": per_view_gripper_mixed_transport_losses,
            "fixed_slot_index_by_view": dict(self.fixed_slot_index_by_view),
            "fixed_slot_debug_by_view": dict(self.fixed_slot_debug_by_view),
            "gripper_slot_index_by_view": dict(self.gripper_slot_index_by_view),
            "gripper_slot_debug_by_view": dict(self.gripper_slot_debug_by_view),
            "mask_transport_loss": None
            if self.last_predictor_debug is None
            else self.last_predictor_debug.get("mask_transport_loss"),
            "candidate_diagnostics_summary": None
            if self.last_predictor_debug is None
            else self.last_predictor_debug.get("candidate_diagnostics_summary"),
            "cem_phase_outputs": None
            if self.last_predictor_debug is None
            else self.last_predictor_debug.get("cem_phase_outputs"),
            "candidate_diagnostics": None
            if self.last_predictor_debug is None
            else self.last_predictor_debug.get("candidate_diagnostics"),
            "server_timing": {
                "parse_obs_s": float(parse_obs_duration_s),
                "append_buffers_s": float(append_duration_s),
                "refresh_obj_latents_s": float(refresh_duration_s),
                "build_context_s": float(build_context_duration_s),
                "predict_total_s": float(total_duration_s),
                "optimize_action_s": float(optimize_duration_s),
                "optimize_detail": None if self.last_predictor_debug is None else self.last_predictor_debug.get("optimize_timing"),
            },
        }

    def record_step_with_action(self, data):
        pre_obs_views = self._parse_observation_views(data["pre_obs"])
        post_obs_views = self._parse_observation_views(data["post_obs"])
        independent = bool(data.get("independent", False))
        append_post_obs_to_history = bool(data.get("append_post_obs_to_history", False))
        ref_view_for_action = self._reference_view(self._filter_available_views(pre_obs_views))
        if "action" in data and data["action"] is not None:
            action7_camera = np.asarray(data["action"], dtype=np.float32).reshape(-1)
        else:
            pre_state = pre_obs_views[ref_view_for_action]["state7"].astype(np.float32)
            post_state = post_obs_views[ref_view_for_action]["state7"].astype(np.float32)
            action7_camera = np.concatenate(
                [
                    self._pose_delta6(pre_state[:6], post_state[:6]),
                    np.array([post_state[6] - pre_state[6]], dtype=np.float32),
                ],
                axis=0,
            ).astype(np.float32)
        if action7_camera.shape[0] != 7:
            raise ValueError(f"Expected action with 7 dims, got {action7_camera.shape}")
        available_views = self._filter_available_views(pre_obs_views)
        available_views = [view_name for view_name in available_views if view_name in post_obs_views]
        if not available_views:
            raise ValueError("No overlapping views between pre_obs and post_obs.")

        if bool(data.get("gripper_history_only", False)):
            self._initialize_fixed_latents_if_needed(post_obs_views)
            bank_sizes_before = {
                view_name: int(len(self.gripper_history_bank_by_view.get(view_name, [])))
                for view_name in available_views
            }
            source = str(data.get("history_source", "gripper_calibration") or "gripper_calibration")
            bank_sizes_after = self._append_obs_views_to_gripper_history_bank(
                post_obs_views,
                available_views,
                source=source,
            )
            per_view_losses = {}
            per_view_mask_l1_losses = {}
            per_view_transport_losses = {}
            per_view_objective_losses = {}
            per_view_gripper_slot_transport_losses = {}
            per_view_other_slot_mean_transport_losses = {}
            if self.goal_features is not None and self.goal_mask_probs is not None:
                goal_view_indices = {view_name: idx for idx, view_name in enumerate(self.goal_view_names)}
                goal_features = self.goal_features[self.current_goal_idx].to(self.device)
                spatial_size = self.crop_size // self.patch_size
                for view_name in available_views:
                    if view_name not in self.fixed_obj_latents or view_name not in goal_view_indices:
                        continue
                    view_idx = goal_view_indices[view_name]
                    post_frame = self._preprocess_image_tensor(post_obs_views[view_name]["image"])
                    post_decoded = self._decode_single_frame_context(post_frame, self.fixed_obj_latents[view_name])
                    current_recon = post_decoded["decoder_recon"].squeeze(0)
                    current_mask = post_decoded["decoded_mask_probs"].squeeze(0).to(self.device, dtype=torch.float32)
                    goal_recon = goal_features[view_idx]
                    goal_mask = self.goal_mask_probs[self.current_goal_idx, view_idx].to(
                        self.device,
                        dtype=torch.float32,
                    )
                    l1_loss = torch.mean(torch.abs(current_recon - goal_recon))
                    mask_l1_loss = torch.mean(torch.abs(current_mask - goal_mask))
                    per_view_losses[view_name] = float(l1_loss.item())
                    per_view_mask_l1_losses[view_name] = float(mask_l1_loss.item())
                    if self.transport_loss_mix > 0.0:
                        slot_weights = self._slot_change_weights(current_mask.unsqueeze(0), goal_mask.unsqueeze(0))
                        transport_loss = self._sinkhorn_mask_transport_loss(
                            current_mask.unsqueeze(0),
                            goal_mask.unsqueeze(0),
                            spatial_size=spatial_size,
                            slot_weights=slot_weights,
                            area_normalizer=self._slot_area_normalizer_for_view(view_name, device=self.device),
                        ).mean()
                        objective_loss = (
                            (1.0 - self.transport_loss_mix) * mask_l1_loss
                            + self.transport_loss_mix * transport_loss
                        )
                        per_view_transport_losses[view_name] = float(transport_loss.item())
                        per_slot_transport = self._sinkhorn_mask_transport_loss_per_slot(
                            current_mask.unsqueeze(0),
                            goal_mask.unsqueeze(0),
                            spatial_size=spatial_size,
                            area_normalizer=self._slot_area_normalizer_for_view(view_name, device=self.device),
                        ).squeeze(0)
                        gripper_slot_idx = self.gripper_slot_index_by_view.get(view_name)
                        if gripper_slot_idx is not None and 0 <= int(gripper_slot_idx) < int(per_slot_transport.shape[0]):
                            gripper_slot_loss = per_slot_transport[int(gripper_slot_idx)]
                            per_view_gripper_slot_transport_losses[view_name] = float(gripper_slot_loss.item())
                            if per_slot_transport.shape[0] > 1:
                                other_loss = (
                                    per_slot_transport.sum() - gripper_slot_loss
                                ) / float(per_slot_transport.shape[0] - 1)
                            else:
                                other_loss = gripper_slot_loss
                            per_view_other_slot_mean_transport_losses[view_name] = float(other_loss.item())
                    else:
                        objective_loss = mask_l1_loss
                    per_view_objective_losses[view_name] = float(objective_loss.item())
            self.history_debug.append(
                {
                    "step": len(self.history_debug),
                    "mode": "gripper_history_calibration_record",
                    "view_names": list(available_views),
                    "history_source": source,
                    "bank_sizes_before": bank_sizes_before,
                    "bank_sizes_after": bank_sizes_after,
                    "post_gripper_states": {
                        view_name: float(post_obs_views[view_name]["state7"][6])
                        for view_name in available_views
                    },
                    "pose_history_lengths": {
                        view_name: int(len(self.frame_buffer[view_name]))
                        for view_name in available_views
                    },
                    "per_view_losses": per_view_losses,
                    "per_view_mask_l1_losses": per_view_mask_l1_losses,
                    "per_view_transport_losses": per_view_transport_losses,
                    "per_view_objective_losses": per_view_objective_losses,
                    "per_view_gripper_slot_transport_losses": per_view_gripper_slot_transport_losses,
                    "per_view_other_slot_mean_transport_losses": per_view_other_slot_mean_transport_losses,
                }
            )
            return {
                "status": "success",
                "mode": "gripper_history_only",
                "view_names": list(available_views),
                "history_source": source,
                "gripper_history_bank_sizes_before": bank_sizes_before,
                "gripper_history_bank_sizes_after": bank_sizes_after,
                "pose_history_lengths": {
                    view_name: int(len(self.frame_buffer[view_name]))
                    for view_name in available_views
                },
                "per_view_losses": per_view_losses,
                "per_view_mask_l1_losses": per_view_mask_l1_losses,
                "per_view_transport_losses": per_view_transport_losses,
                "per_view_objective_losses": per_view_objective_losses,
                "per_view_gripper_slot_transport_losses": per_view_gripper_slot_transport_losses,
                "per_view_other_slot_mean_transport_losses": per_view_other_slot_mean_transport_losses,
                "mean_loss": None if not per_view_losses else float(np.mean(list(per_view_losses.values()))),
                "mean_transport_loss": None
                if not per_view_transport_losses
                else float(np.mean(list(per_view_transport_losses.values()))),
                "mean_objective_loss": None
                if not per_view_objective_losses
                else float(np.mean(list(per_view_objective_losses.values()))),
            }

        if independent:
            self._clear_temporal_buffers()
        self._append_obs_views_to_buffers(pre_obs_views, available_views)
        self._initialize_fixed_latents_if_needed(pre_obs_views)
        context_by_view, states_by_view, extrinsics_by_view, _ = self._build_context_from_buffers(available_views)
        pred_by_view, pred_mask_by_view, warp_by_view, warp_mask_by_view, pred_next_state_by_view = self.rollout_latent_for_action(
            context_by_view=context_by_view,
            states_by_view=states_by_view,
            extrinsics_by_view=extrinsics_by_view,
            view_names=available_views,
            action7_camera=action7_camera,
        )

        gt_next_latent_by_view = {}
        gt_next_mask_by_view = {}
        gt_next_state_by_view = {}
        per_view_losses = {}
        per_view_transport_losses = {}
        observed_per_view_losses = {}
        observed_per_view_mask_l1_losses = {}
        observed_per_view_transport_losses = {}
        observed_per_view_objective_losses = {}
        per_view_warp_gt_losses = {}
        per_view_warp_pred_losses = {}
        ref_view = self._reference_view(available_views)
        goal_view_indices = {view_name: idx for idx, view_name in enumerate(self.goal_view_names)}
        goal_features = None if self.goal_features is None else self.goal_features[self.current_goal_idx].to(self.device)
        for view_name in available_views:
            post_frame = self._preprocess_image_tensor(post_obs_views[view_name]["image"])
            post_decoded = self._decode_single_frame_context(post_frame, self.fixed_obj_latents[view_name])
            gt_next_latent_by_view[view_name] = post_decoded["decoder_recon"].squeeze(0)
            gt_next_mask_by_view[view_name] = post_decoded["decoded_mask_probs"].squeeze(0)
            gt_next_state_by_view[view_name] = post_obs_views[view_name]["state7"].astype(np.float32).copy()
            per_view_losses[view_name] = float(torch.mean(torch.abs(pred_by_view[view_name].squeeze(0) - gt_next_latent_by_view[view_name])).item())
            if view_name in warp_by_view:
                per_view_warp_gt_losses[view_name] = float(
                    torch.mean(torch.abs(warp_by_view[view_name].squeeze(0) - gt_next_latent_by_view[view_name])).item()
                )
                per_view_warp_pred_losses[view_name] = float(
                    torch.mean(torch.abs(warp_by_view[view_name].squeeze(0) - pred_by_view[view_name].squeeze(0))).item()
                )
            per_view_transport_losses[view_name] = float(
                self._sinkhorn_mask_transport_loss(
                    pred_mask_by_view[view_name].squeeze(0).unsqueeze(0),
                    gt_next_mask_by_view[view_name].unsqueeze(0),
                    spatial_size=self.crop_size // self.patch_size,
                    area_normalizer=self._slot_area_normalizer_for_view(view_name, device=self.device),
                ).mean().item()
            )
            if goal_features is not None and self.goal_mask_probs is not None and view_name in goal_view_indices:
                view_idx = goal_view_indices[view_name]
                goal_recon = goal_features[view_idx]
                goal_mask = self.goal_mask_probs[self.current_goal_idx, view_idx].to(
                    self.device,
                    dtype=torch.float32,
                )
                observed_l1_loss = torch.mean(
                    torch.abs(gt_next_latent_by_view[view_name] - goal_recon)
                )
                observed_mask_l1_loss = torch.mean(
                    torch.abs(gt_next_mask_by_view[view_name] - goal_mask)
                )
                observed_per_view_losses[view_name] = float(observed_l1_loss.item())
                observed_per_view_mask_l1_losses[view_name] = float(
                    observed_mask_l1_loss.item()
                )
                if self.transport_loss_mix > 0.0:
                    slot_weights = self._slot_change_weights(
                        gt_next_mask_by_view[view_name].unsqueeze(0),
                        goal_mask.unsqueeze(0),
                    )
                    observed_transport_loss = self._sinkhorn_mask_transport_loss(
                        gt_next_mask_by_view[view_name].unsqueeze(0),
                        goal_mask.unsqueeze(0),
                        spatial_size=self.crop_size // self.patch_size,
                        slot_weights=slot_weights,
                        area_normalizer=self._slot_area_normalizer_for_view(view_name, device=self.device),
                    ).mean()
                    observed_objective_loss = (
                        (1.0 - self.transport_loss_mix) * observed_mask_l1_loss
                        + self.transport_loss_mix * observed_transport_loss
                    )
                    observed_per_view_transport_losses[view_name] = float(observed_transport_loss.item())
                else:
                    observed_objective_loss = observed_mask_l1_loss
                observed_per_view_objective_losses[view_name] = float(observed_objective_loss.item())

        self.history_z_context.append(
            np.stack([gt_next_latent_by_view[view_name].detach().cpu().numpy() for view_name in available_views], axis=0)
        )
        if warp_by_view:
            self.history_z_warp.append(
                np.stack(
                    [
                        warp_by_view.get(view_name, gt_next_latent_by_view[view_name].unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                        for view_name in available_views
                    ],
                    axis=0,
                )
            )
        self.history_z_pred.append(
            np.stack([pred_by_view[view_name].squeeze(0).detach().cpu().numpy() for view_name in available_views], axis=0)
        )
        self.history_mask_context.append(
            np.stack([gt_next_mask_by_view[view_name].detach().cpu().numpy() for view_name in available_views], axis=0)
        )
        if warp_mask_by_view:
            self.history_mask_warp.append(
                np.stack(
                    [
                        warp_mask_by_view.get(view_name, gt_next_mask_by_view[view_name].unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                        for view_name in available_views
                    ],
                    axis=0,
                )
            )
        self.history_mask_pred.append(
            np.stack([pred_mask_by_view[view_name].squeeze(0).detach().cpu().numpy() for view_name in available_views], axis=0)
        )
        history_lengths_before_post_append = {
            view_name: int(len(self.frame_buffer[view_name]))
            for view_name in available_views
        }
        if append_post_obs_to_history:
            self._append_obs_views_to_buffers(post_obs_views, available_views)
        history_lengths_after_post_append = {
            view_name: int(len(self.frame_buffer[view_name]))
            for view_name in available_views
        }
        pred_next_state_ref = pred_next_state_by_view[ref_view].detach().cpu().numpy().reshape(-1).astype(np.float32)
        gt_next_state_ref = gt_next_state_by_view[ref_view]
        obs_pre_state_ref = pre_obs_views[ref_view]["state7"].astype(np.float32).copy()
        self.history_debug.append(
            {
                "step": len(self.history_debug),
                "mode": "teacher_forced_action_rollout",
                "independent": bool(independent),
                "append_post_obs_to_history": bool(append_post_obs_to_history),
                "history_lengths_before_post_append": history_lengths_before_post_append,
                "history_lengths_after_post_append": history_lengths_after_post_append,
                "view_names": list(available_views),
                "reference_view": ref_view,
                "action7_camera": action7_camera.copy(),
                "per_view_losses": per_view_losses,
                "per_view_transport_losses": per_view_transport_losses,
                "observed_per_view_losses": observed_per_view_losses,
                "observed_per_view_mask_l1_losses": observed_per_view_mask_l1_losses,
                "observed_per_view_transport_losses": observed_per_view_transport_losses,
                "observed_per_view_objective_losses": observed_per_view_objective_losses,
                "per_view_warp_gt_losses": per_view_warp_gt_losses,
                "per_view_warp_pred_losses": per_view_warp_pred_losses,
                "mean_loss": float(np.mean(list(per_view_losses.values()))),
                "mean_transport_loss": float(np.mean(list(per_view_transport_losses.values()))),
                "predicted_next_state7_camera": pred_next_state_ref.copy(),
                "gt_next_state7_camera": gt_next_state_ref.copy(),
                "predicted_delta_xyz_camera": (pred_next_state_ref[:3] - obs_pre_state_ref[:3]).astype(np.float32).copy(),
                "gt_delta_xyz_camera": (gt_next_state_ref[:3] - obs_pre_state_ref[:3]).astype(np.float32).copy(),
                "obs_by_view": {
                    view_name: {
                        "pre_state7": pre_obs_views[view_name]["state7"].astype(np.float32).copy(),
                        "post_state7": post_obs_views[view_name]["state7"].astype(np.float32).copy(),
                        "camera_extrinsics": pre_obs_views[view_name]["extrinsics6"].astype(np.float32).copy(),
                        "camera_frame_convention": pre_obs_views[view_name].get("camera_frame_convention"),
                    }
                    for view_name in available_views
                },
            }
        )
        return {
            "status": "success",
            "independent": bool(independent),
            "per_view_losses": per_view_losses,
            "per_view_transport_losses": per_view_transport_losses,
            "observed_per_view_losses": observed_per_view_losses,
            "observed_per_view_mask_l1_losses": observed_per_view_mask_l1_losses,
            "observed_per_view_transport_losses": observed_per_view_transport_losses,
            "observed_per_view_objective_losses": observed_per_view_objective_losses,
            "mean_loss": float(np.mean(list(per_view_losses.values()))),
            "mean_transport_loss": float(np.mean(list(per_view_transport_losses.values()))),
            "observed_mean_loss": None
            if not observed_per_view_losses
            else float(np.mean(list(observed_per_view_losses.values()))),
            "observed_mean_mask_l1_loss": None
            if not observed_per_view_mask_l1_losses
            else float(np.mean(list(observed_per_view_mask_l1_losses.values()))),
            "observed_mean_transport_loss": None
            if not observed_per_view_transport_losses
            else float(np.mean(list(observed_per_view_transport_losses.values()))),
            "observed_mean_objective_loss": None
            if not observed_per_view_objective_losses
            else float(np.mean(list(observed_per_view_objective_losses.values()))),
            "append_post_obs_to_history": bool(append_post_obs_to_history),
            "history_lengths_before_post_append": history_lengths_before_post_append,
            "history_lengths_after_post_append": history_lengths_after_post_append,
        }


def main(
    port=8011,
    run_dir=SlotWorldModelAgent.DEFAULT_RUN_DIR,
    ckpt_path=None,
    config_path=None,
    transport_loss_mix=0.0,
    use_goal_pose_prior=False,
    sinkhorn_iters=5,
    sinkhorn_entropy_reg=0.05,
    mask_centralize_gamma=2.0,
    unbalanced_ot_rho=0.2,
    unbalanced_ot_eps=1.0e-8,
    mask_transport_area_normalize=True,
    mask_transport_area_normalize_eps=1.0e-3,
    use_shared_visible_wrist_ot=True,
    shared_visible_sim_threshold=0.55,
    shared_visible_temperature=0.05,
    shared_visible_min_mass=0.02,
    factorized_pose_cem=True,
    factorized_cem_steps_per_phase=None,
    gripper_cem_steps=1,
    gripper_samples=5,
    cem_samples=80,
    cem_elite_topk=10,
    wrist_rerank_topk=10,
    wrist_refine_momentum_mean=0.75,
    wrist_refine_momentum_std=0.90,
    cem_steps=3,
    fixed_loss_search_min_scale=0.35,
    pose_rotation_cap=0.15,
    pose_rotation_std=0.05,
    pose_translation_xy_cap=0.03,
    pose_translation_z_cap=0.3,
    pose_translation_xy_std=0.01,
    pose_translation_z_std=0.1,
    execution_translation_cap=None,
    execution_rotation_cap=None,
    slot_action_range_scale=1.0,
    action_l2_weight=0.0,
    view_loss_mode="gated",
    fixed_view_loss_gate_threshold=0.001,
    optimize_gripper_action=True,
    gripper_position_max=1.0,
    fixed_gripper_target=0.0,
    gripper_min_open_gap=0.05,
    gripper_downpress_max_z=0.05,
    gripper_delta_std=0.03,
    gripper_delta_cap=0.08,
    gripper_cem_skip_translation_threshold=0.03,
    gripper_cem_skip_rotation_threshold=0.04,
    slot_warmup_xy_radius=DEFAULT_SLOT_WARMUP_XY_RADIUS,
    slot_warmup_z_offsets=None,
    slot_warmup_settle_steps=DEFAULT_SLOT_WARMUP_SETTLE_STEPS,
    fixed_slot_min_area_ratio=0.01,
    fixed_slot_max_area_ratio=0.20,
    fixed_slot_entropy_weight=0.7,
    fixed_slot_centroid_weight=0.3,
    slot_steering_memory_topk=DEFAULT_SLOT_STEERING_MEMORY_TOPK,
    slot_steering_sample_fraction=DEFAULT_SLOT_STEERING_SAMPLE_FRACTION,
    slot_steering_loss_range_eps=DEFAULT_SLOT_STEERING_LOSS_RANGE_EPS,
    history_pose_sampling_temperature=0.05,
    history_pose_uniform_mix=0.15,
    memory_selection_mode="relevance",
    memory_random_seed=0,
    always_execute_best_candidate=False,
    reserve_visible_gpu_memory_after_load=True,
    gpu_memory_reserve_fraction=0.85,
    gpu_memory_reserve_margin_mb=4096,
    gpu_memory_reserve_chunk_mb=256,
):
    agent = SlotWorldModelAgent(
        run_dir=run_dir,
        ckpt_path=ckpt_path,
        config_path=config_path,
        transport_loss_mix=transport_loss_mix,
        use_goal_pose_prior=use_goal_pose_prior,
        sinkhorn_iters=sinkhorn_iters,
        sinkhorn_entropy_reg=sinkhorn_entropy_reg,
        mask_centralize_gamma=mask_centralize_gamma,
        unbalanced_ot_rho=unbalanced_ot_rho,
        unbalanced_ot_eps=unbalanced_ot_eps,
        mask_transport_area_normalize=mask_transport_area_normalize,
        mask_transport_area_normalize_eps=mask_transport_area_normalize_eps,
        use_shared_visible_wrist_ot=use_shared_visible_wrist_ot,
        shared_visible_sim_threshold=shared_visible_sim_threshold,
        shared_visible_temperature=shared_visible_temperature,
        shared_visible_min_mass=shared_visible_min_mass,
        factorized_pose_cem=factorized_pose_cem,
        factorized_cem_steps_per_phase=factorized_cem_steps_per_phase,
        gripper_cem_steps=gripper_cem_steps,
        gripper_samples=gripper_samples,
        cem_samples=cem_samples,
        cem_elite_topk=cem_elite_topk,
        wrist_rerank_topk=wrist_rerank_topk,
        wrist_refine_momentum_mean=wrist_refine_momentum_mean,
        wrist_refine_momentum_std=wrist_refine_momentum_std,
        cem_steps=cem_steps,
        fixed_loss_search_min_scale=fixed_loss_search_min_scale,
        pose_rotation_cap=pose_rotation_cap,
        pose_rotation_std=pose_rotation_std,
        pose_translation_xy_cap=pose_translation_xy_cap,
        pose_translation_z_cap=pose_translation_z_cap,
        pose_translation_xy_std=pose_translation_xy_std,
        pose_translation_z_std=pose_translation_z_std,
        execution_translation_cap=execution_translation_cap,
        execution_rotation_cap=execution_rotation_cap,
        slot_action_range_scale=slot_action_range_scale,
        action_l2_weight=action_l2_weight,
        view_loss_mode=view_loss_mode,
        fixed_view_loss_gate_threshold=fixed_view_loss_gate_threshold,
        optimize_gripper_action=optimize_gripper_action,
        gripper_position_max=gripper_position_max,
        fixed_gripper_target=fixed_gripper_target,
        gripper_min_open_gap=gripper_min_open_gap,
        gripper_downpress_max_z=gripper_downpress_max_z,
        gripper_delta_std=gripper_delta_std,
        gripper_delta_cap=gripper_delta_cap,
        gripper_cem_skip_translation_threshold=gripper_cem_skip_translation_threshold,
        gripper_cem_skip_rotation_threshold=gripper_cem_skip_rotation_threshold,
        slot_warmup_xy_radius=slot_warmup_xy_radius,
        slot_warmup_z_offsets=slot_warmup_z_offsets,
        slot_warmup_settle_steps=slot_warmup_settle_steps,
        fixed_slot_min_area_ratio=fixed_slot_min_area_ratio,
        fixed_slot_max_area_ratio=fixed_slot_max_area_ratio,
        fixed_slot_entropy_weight=fixed_slot_entropy_weight,
        fixed_slot_centroid_weight=fixed_slot_centroid_weight,
        slot_steering_memory_topk=slot_steering_memory_topk,
        slot_steering_sample_fraction=slot_steering_sample_fraction,
        slot_steering_loss_range_eps=slot_steering_loss_range_eps,
        history_pose_sampling_temperature=history_pose_sampling_temperature,
        history_pose_uniform_mix=history_pose_uniform_mix,
        memory_selection_mode=memory_selection_mode,
        memory_random_seed=memory_random_seed,
        always_execute_best_candidate=always_execute_best_candidate,
    )
    print(
        "[SLOT-AC] Planning ablation config: "
        f"uot_goal_energy_enabled={agent.transport_loss_mix > 0.0} "
        f"goal_energy_mode={'mask_l1' if agent.transport_loss_mix <= 0.0 else 'mask_l1_uot_mix'} "
        f"goal_pose_prior_enabled={agent.use_goal_pose_prior} "
        f"transport_loss_mix={agent.transport_loss_mix:g} "
        f"memory_selection_mode={agent.memory_selection_mode} "
        f"memory_random_seed={agent.memory_random_seed} "
        f"always_execute_best_candidate={agent.always_execute_best_candidate}",
        flush=True,
    )
    if reserve_visible_gpu_memory_after_load:
        reserve_visible_gpu_memory(
            fraction=gpu_memory_reserve_fraction,
            margin_mb=gpu_memory_reserve_margin_mb,
            chunk_mb=gpu_memory_reserve_chunk_mb,
        )
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print(f"SLOT-AC Action Server listening on port {port}...", flush=True)

    while True:
        try:
            message_bytes = socket.recv()
            message = MsgSerializer.from_bytes(message_bytes)
        except Exception as e:
            print(f"Failed to receive or deserialize message: {e}", flush=True)
            continue

        if message.get("endpoint") == "get_action":
            try:
                socket.send(MsgSerializer.to_bytes(agent.predict_action(message["data"])))
            except Exception as e:
                import traceback
                traceback.print_exc()
                socket.send(MsgSerializer.to_bytes({"error": str(e)}))
        elif message.get("endpoint") == "set_goal":
            payload = message["data"]
            goal_im = payload["goal_image"] if isinstance(payload, dict) else payload
            try:
                self_goal_state = payload.get("goal_state") if isinstance(payload, dict) else None
                goal_images = goal_im if isinstance(goal_im, list) else [goal_im]
                agent.reset()
                agent.set_goal_image(goal_images)
                agent.goal_state = self_goal_state
                agent._prepare_in_trajectory_random_memory()
                socket.send(MsgSerializer.to_bytes({"status": "success", "goal_view_names": agent.goal_view_names}))
            except Exception as e:
                import traceback
                traceback.print_exc()
                socket.send(MsgSerializer.to_bytes({"error": str(e)}))
        elif message.get("endpoint") == "reset":
            try:
                agent.reset()
                socket.send(MsgSerializer.to_bytes({"status": "success"}))
            except Exception as e:
                import traceback
                traceback.print_exc()
                socket.send(MsgSerializer.to_bytes({"error": str(e)}))
        elif message.get("endpoint") == "get_debug_trace":
            socket.send(
                MsgSerializer.to_bytes(
                    {
                        "status": "success",
                        "history_debug": agent.history_debug,
                        "last_predictor_debug": agent.last_predictor_debug,
                        "goal_idx": None if agent.goal_features is None else int(agent.current_goal_idx),
                        "goal_state": agent.goal_state,
                    }
                )
            )
        elif message.get("endpoint") == "get_warmup_config":
            socket.send(MsgSerializer.to_bytes(agent.get_warmup_config()))
        elif message.get("endpoint") == "advance_goal":
            try:
                payload = message.get("data") or {}
                old_goal_idx = int(agent.current_goal_idx)
                max_goal_idx = 0 if agent.goal_features is None else int(agent.goal_features.shape[0] - 1)
                if "goal_idx" in payload:
                    new_goal_idx = int(np.clip(int(payload["goal_idx"]), 0, max_goal_idx))
                else:
                    new_goal_idx = int(min(old_goal_idx + 1, max_goal_idx))
                agent.current_goal_idx = new_goal_idx
                warmup_refresh = agent._refresh_warmup_memory_for_goal(new_goal_idx)
                socket.send(
                    MsgSerializer.to_bytes(
                        {
                            "status": "success",
                            "old_goal_idx": old_goal_idx,
                            "goal_idx": int(agent.current_goal_idx),
                            "max_goal_idx": max_goal_idx,
                            "reset_control_loss_tracking_for_goal": int(new_goal_idx),
                            "warmup_goal_loss_refresh": warmup_refresh,
                        }
                    )
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                socket.send(MsgSerializer.to_bytes({"error": str(e)}))
        elif message.get("endpoint") == "record_step_with_action":
            try:
                socket.send(MsgSerializer.to_bytes(agent.record_step_with_action(message["data"])))
            except Exception as e:
                import traceback
                traceback.print_exc()
                socket.send(MsgSerializer.to_bytes({"error": str(e)}))
        elif message.get("endpoint") == "score_observations_to_goal":
            try:
                socket.send(MsgSerializer.to_bytes(agent.score_observations_to_goal(message["data"])))
            except Exception as e:
                import traceback
                traceback.print_exc()
                socket.send(MsgSerializer.to_bytes({"error": str(e)}))
        elif message.get("endpoint") == "predict_one_step_candidates":
            try:
                socket.send(MsgSerializer.to_bytes(agent.predict_one_step_candidates(message["data"])))
            except Exception as e:
                import traceback
                traceback.print_exc()
                socket.send(MsgSerializer.to_bytes({"error": str(e)}))
        elif message.get("endpoint") == "record_one_step_prediction_post":
            try:
                socket.send(MsgSerializer.to_bytes(agent.record_one_step_prediction_post(message["data"])))
            except Exception as e:
                import traceback
                traceback.print_exc()
                socket.send(MsgSerializer.to_bytes({"error": str(e)}))
        elif message.get("endpoint") == "plot_pca":
            try:
                from torchvision.utils import make_grid
                import matplotlib.pyplot as plt

                if len(agent.history_z_context) > 0:
                    gt_features = torch.from_numpy(np.array(agent.history_z_context))
                    pred_features = torch.from_numpy(np.array(agent.history_z_pred))
                    warp_features = None
                    if len(agent.history_z_warp) == len(agent.history_z_context):
                        warp_features = torch.from_numpy(np.array(agent.history_z_warp))
                    gt_masks = None
                    pred_masks = None
                    warp_masks = None
                    if len(agent.history_mask_context) == len(agent.history_z_context) and len(agent.history_mask_pred) == len(agent.history_z_pred):
                        gt_masks = torch.from_numpy(np.array(agent.history_mask_context))
                        pred_masks = torch.from_numpy(np.array(agent.history_mask_pred))
                    if len(agent.history_mask_warp) == len(agent.history_z_context):
                        warp_masks = torch.from_numpy(np.array(agent.history_mask_warp))
                    goal_features_all = agent.goal_feature_full.cpu()
                    goal_masks_all = None if agent.goal_mask_probs is None else agent.goal_mask_probs.cpu()
                    history_debug = list(agent.history_debug)
                    _, num_views, tokens_per_frame, D = gt_features.shape
                    view_names = history_debug[-1].get("view_names", agent.goal_view_names[:num_views]) if history_debug else agent.goal_view_names[:num_views]
                    goal_view_indices = {view_name: idx for idx, view_name in enumerate(agent.goal_view_names)}
                    num_goals = int(goal_features_all.shape[0])
                    active_goal_idx = int(np.clip(agent.current_goal_idx, 0, max(num_goals - 1, 0)))
                    goal_names = []
                    goal_stages = []
                    if isinstance(agent.goal_state, dict):
                        goal_names = list(agent.goal_state.get("target_names", []))
                        goal_stages = list(agent.goal_state.get("target_stages", []))
                    spatial_size = int(math.isqrt(tokens_per_frame))
                    if spatial_size * spatial_size != tokens_per_frame:
                        raise ValueError(f"tokens_per_frame={tokens_per_frame} is not a square number")
                    num_steps = gt_features.shape[0]
                    total_num_steps = int(num_steps)
                    max_pca_vis_frames = 60
                    if total_num_steps > max_pca_vis_frames:
                        sampled_step_indices_np = np.linspace(
                            0,
                            total_num_steps - 1,
                            max_pca_vis_frames,
                        ).round().astype(np.int64)
                        sampled_step_indices = sampled_step_indices_np.tolist()
                        gt_features = gt_features[sampled_step_indices]
                        pred_features = pred_features[sampled_step_indices]
                        if warp_features is not None:
                            warp_features = warp_features[sampled_step_indices]
                        if gt_masks is not None:
                            gt_masks = gt_masks[sampled_step_indices]
                        if pred_masks is not None:
                            pred_masks = pred_masks[sampled_step_indices]
                        if warp_masks is not None:
                            warp_masks = warp_masks[sampled_step_indices]
                        history_debug = [
                            history_debug[idx] if idx < len(history_debug) else {}
                            for idx in sampled_step_indices
                        ]
                        num_steps = gt_features.shape[0]
                    else:
                        sampled_step_indices = list(range(total_num_steps))
                    rows_per_view = 3 if warp_features is not None else 2
                    fig = plt.figure(figsize=(max(18, num_steps * 2.35 + num_goals * 1.2), max(6, num_views * (7.6 if rows_per_view == 3 else 5.6))))
                    fig.suptitle(
                        f"SLOT latent PCA trajectory: sampled {int(num_steps)}/{total_num_steps} frames",
                        fontsize=12,
                    )
                    gs = fig.add_gridspec(
                        num_views * rows_per_view,
                        2,
                        width_ratios=[max(8, num_steps), max(1.4, num_goals * 1.25)],
                        hspace=0.6,
                        wspace=0.12,
                    )

                    for view_idx, view_name in enumerate(view_names):
                        gt_view = gt_features[:, view_idx]
                        pred_view = pred_features[:, view_idx]
                        warp_view = None if warp_features is None else warp_features[:, view_idx]
                        goal_view_all = goal_features_all[:, goal_view_indices[view_name]]
                        active_goal_view = goal_view_all[active_goal_idx : active_goal_idx + 1]

                        gt_patches = torch.cat([gt_view.reshape(-1, D), goal_view_all.reshape(-1, D)], dim=0)
                        gt_mean = gt_patches.mean(dim=0, keepdim=True)
                        gt_centered = gt_patches - gt_mean
                        _, _, v = torch.pca_lowrank(gt_centered, q=3)
                        gt_proj = gt_centered @ v
                        p_min = gt_proj.min(dim=0, keepdim=True).values
                        p_max = gt_proj.max(dim=0, keepdim=True).values

                        def project_view(feats):
                            proj = (feats.reshape(-1, D) - gt_mean) @ v
                            proj = (proj - p_min) / (p_max - p_min + 1e-6)
                            proj_grid = proj.reshape(-1, spatial_size, spatial_size, 3).permute(0, 3, 1, 2)
                            return F.interpolate(proj_grid, size=(agent.crop_size, agent.crop_size), mode="nearest")

                        gt_imgs = project_view(gt_view)
                        warp_imgs = None if warp_view is None else project_view(warp_view)
                        pred_imgs = project_view(pred_view)
                        goal_imgs = project_view(goal_view_all)

                        gt_goal_l1 = torch.mean(torch.abs(gt_view - active_goal_view[0].unsqueeze(0)), dim=(1, 2)).cpu().numpy()
                        warp_goal_l1 = None if warp_view is None else torch.mean(torch.abs(warp_view - active_goal_view[0].unsqueeze(0)), dim=(1, 2)).cpu().numpy()
                        warp_gt_l1 = None if warp_view is None else torch.mean(torch.abs(warp_view - gt_view), dim=(1, 2)).cpu().numpy()
                        warp_pred_l1 = None if warp_view is None else torch.mean(torch.abs(warp_view - pred_view), dim=(1, 2)).cpu().numpy()
                        pred_goal_l1 = torch.mean(torch.abs(pred_view - active_goal_view[0].unsqueeze(0)), dim=(1, 2)).cpu().numpy()
                        pred_gt_l1 = torch.mean(torch.abs(pred_view - gt_view), dim=(1, 2)).cpu().numpy()
                        gt_goal_transport_loss = None
                        warp_goal_transport_loss = None
                        warp_gt_transport_loss = None
                        pred_goal_transport_loss = None
                        pred_gt_transport_loss = None
                        if gt_masks is not None and pred_masks is not None and goal_masks_all is not None:
                            gt_mask_view = gt_masks[:, view_idx].to(agent.device, dtype=torch.float32)
                            pred_mask_view = pred_masks[:, view_idx].to(agent.device, dtype=torch.float32)
                            goal_mask_view = goal_masks_all[active_goal_idx, goal_view_indices[view_name]].to(agent.device, dtype=torch.float32)
                            goal_mask_seq = goal_mask_view.unsqueeze(0).expand_as(gt_mask_view)
                            slot_weights = agent._slot_change_weights(
                                gt_mask_view[:1],
                                goal_mask_view.unsqueeze(0),
                            )
                            gt_goal_transport_loss = agent._sinkhorn_mask_transport_loss(
                                gt_mask_view,
                                goal_mask_seq,
                                spatial_size=spatial_size,
                                slot_weights=slot_weights,
                                area_normalizer=agent._slot_area_normalizer_for_view(view_name, device=agent.device),
                            ).detach().cpu().numpy()
                            pred_goal_transport_loss = agent._sinkhorn_mask_transport_loss(
                                pred_mask_view,
                                goal_mask_seq,
                                spatial_size=spatial_size,
                                slot_weights=slot_weights,
                                area_normalizer=agent._slot_area_normalizer_for_view(view_name, device=agent.device),
                            ).detach().cpu().numpy()
                            pred_gt_transport_loss = agent._sinkhorn_mask_transport_loss(
                                pred_mask_view,
                                gt_mask_view,
                                spatial_size=spatial_size,
                                slot_weights=slot_weights,
                                area_normalizer=agent._slot_area_normalizer_for_view(view_name, device=agent.device),
                            ).detach().cpu().numpy()
                            if warp_masks is not None:
                                warp_mask_view = warp_masks[:, view_idx].to(agent.device, dtype=torch.float32)
                                warp_goal_transport_loss = agent._sinkhorn_mask_transport_loss(
                                    warp_mask_view,
                                    goal_mask_seq,
                                    spatial_size=spatial_size,
                                    slot_weights=slot_weights,
                                    area_normalizer=agent._slot_area_normalizer_for_view(view_name, device=agent.device),
                                ).detach().cpu().numpy()
                                warp_gt_transport_loss = agent._sinkhorn_mask_transport_loss(
                                    warp_mask_view,
                                    gt_mask_view,
                                    spatial_size=spatial_size,
                                    slot_weights=slot_weights,
                                    area_normalizer=agent._slot_area_normalizer_for_view(view_name, device=agent.device),
                                ).detach().cpu().numpy()

                        gt_row = make_grid(gt_imgs, nrow=num_steps, padding=2, normalize=False)
                        warp_row = None if warp_imgs is None else make_grid(warp_imgs, nrow=num_steps, padding=2, normalize=False)
                        pred_row = make_grid(pred_imgs, nrow=num_steps, padding=2, normalize=False)
                        goal_tile = make_grid(goal_imgs, nrow=num_goals, padding=2, normalize=False)

                        gt_np = gt_row.permute(1, 2, 0).numpy()
                        warp_np = None if warp_row is None else warp_row.permute(1, 2, 0).numpy()
                        pred_np = pred_row.permute(1, 2, 0).numpy()
                        goal_np = goal_tile.permute(1, 2, 0).numpy()

                        row0 = view_idx * rows_per_view
                        ax_gt = fig.add_subplot(gs[row0, 0])
                        ax_warp = None if warp_np is None else fig.add_subplot(gs[row0 + 1, 0])
                        ax_pred = fig.add_subplot(gs[row0 + rows_per_view - 1, 0])
                        ax_goal = fig.add_subplot(gs[row0 : row0 + rows_per_view, 1])
                        ax_gt.imshow(np.clip(gt_np, 0, 1))
                        ax_gt.axis("off")
                        ax_gt.set_title(f"{view_name}: True Latent")
                        if ax_warp is not None:
                            ax_warp.imshow(np.clip(warp_np, 0, 1))
                            ax_warp.axis("off")
                            warp_mode = "copied_context" if "wrist" not in view_name.lower() else "geometric_extrinsics"
                            ax_warp.set_title(f"{view_name}: Warped Input ({warp_mode})")
                        ax_pred.imshow(np.clip(pred_np, 0, 1))
                        ax_pred.axis("off")
                        ax_pred.set_title(f"{view_name}: Predictor Latent")
                        ax_goal.imshow(np.clip(goal_np, 0, 1))
                        ax_goal.axis("off")
                        ax_goal.set_title(f"Goals (active={active_goal_idx})")
                        tile_w = float(agent.crop_size)
                        pad = 2.0
                        goal_total_w = float(goal_np.shape[1])
                        for goal_idx in range(num_goals):
                            goal_center_px = pad + goal_idx * (tile_w + pad) + tile_w / 2.0
                            goal_x_frac = goal_center_px / max(goal_total_w, 1.0)
                            goal_name = goal_names[goal_idx] if goal_idx < len(goal_names) else f"goal{goal_idx}"
                            goal_stage = goal_stages[goal_idx] if goal_idx < len(goal_stages) else ""
                            ax_goal.text(
                                goal_x_frac,
                                -0.035,
                                f"{goal_idx}:{goal_name}\n{goal_stage}",
                                transform=ax_goal.transAxes,
                                ha="center",
                                va="top",
                                fontsize=8,
                                family="monospace",
                            )

                        total_w = float(gt_np.shape[1])
                        for step_idx in range(num_steps):
                            center_px = pad + step_idx * (tile_w + pad) + tile_w / 2.0
                            x_frac = center_px / max(total_w, 1.0)
                            source_step_idx = sampled_step_indices[step_idx] if step_idx < len(sampled_step_indices) else step_idx
                            gt_text = f"t{source_step_idx}\nL_goal={gt_goal_l1[step_idx]:.3f}"
                            warp_text = None
                            if warp_goal_l1 is not None:
                                warp_text = (
                                    f"L_goal={warp_goal_l1[step_idx]:.3f}"
                                    f"\nL_gt={warp_gt_l1[step_idx]:.3f}"
                                    f"\nL_pred={warp_pred_l1[step_idx]:.3f}"
                            )
                            pred_text = f"L_goal={pred_goal_l1[step_idx]:.3f}\nL_gt={pred_gt_l1[step_idx]:.3f}"
                            if gt_goal_transport_loss is not None:
                                gt_text += f"\nOT_goal={gt_goal_transport_loss[step_idx]:.4f}"
                            if warp_text is not None and warp_goal_transport_loss is not None and warp_gt_transport_loss is not None:
                                warp_text += (
                                    f"\nOT_goal={warp_goal_transport_loss[step_idx]:.4f}"
                                    f"\nOT_gt={warp_gt_transport_loss[step_idx]:.4f}"
                                )
                            if pred_goal_transport_loss is not None and pred_gt_transport_loss is not None:
                                pred_text += (
                                    f"\nOT_goal={pred_goal_transport_loss[step_idx]:.4f}"
                                    f"\nOT_gt={pred_gt_transport_loss[step_idx]:.4f}"
                                )
                            if step_idx < len(history_debug):
                                step_l1 = history_debug[step_idx].get("per_view_losses", {}).get(view_name)
                                step_transport = (
                                    history_debug[step_idx].get("per_view_transport_losses", {}).get(view_name)
                                    or history_debug[step_idx].get("per_view_iou_losses", {}).get(view_name)
                                )
                                if step_l1 is not None:
                                    pred_text += f"\nwm_L1={float(step_l1):.3f}"
                                if step_transport is not None:
                                    pred_text += f"\nwm_OT={float(step_transport):.4f}"
                            ax_gt.text(
                                x_frac,
                                -0.045,
                                gt_text,
                                transform=ax_gt.transAxes,
                                ha="center",
                                va="top",
                                fontsize=8,
                                family="monospace",
                            )
                            if ax_warp is not None and warp_text is not None:
                                ax_warp.text(
                                    x_frac,
                                    -0.055,
                                    warp_text,
                                    transform=ax_warp.transAxes,
                                    ha="center",
                                    va="top",
                                    fontsize=8,
                                    family="monospace",
                                )
                            ax_pred.text(
                                x_frac,
                                -0.055,
                                pred_text,
                                transform=ax_pred.transAxes,
                                ha="center",
                                va="top",
                                fontsize=8,
                                family="monospace",
                            )

                    buf = io.BytesIO()
                    fig.subplots_adjust(bottom=0.07, top=0.96)
                    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.35)
                    plt.close(fig)
                    img_bytes = buf.getvalue()
                else:
                    img_bytes = b""
                agent.history_z_context.clear()
                agent.history_z_warp.clear()
                agent.history_z_pred.clear()
                agent.history_mask_context.clear()
                agent.history_mask_warp.clear()
                agent.history_mask_pred.clear()
                socket.send(MsgSerializer.to_bytes({"status": "success", "image_bytes": img_bytes}))
            except Exception as e:
                import traceback
                traceback.print_exc()
                socket.send(MsgSerializer.to_bytes({"error": str(e)}))
        elif message.get("endpoint") == "plot_slot_masks":
            try:
                mask_image = agent._render_initial_goal_mask_figure()
                buf = io.BytesIO()
                mask_image.save(buf, format="PNG")
                socket.send(
                    MsgSerializer.to_bytes(
                        {
                            "status": "success",
                            "visualization": "slot_mask_overlay_v2",
                            "image_bytes": buf.getvalue(),
                        }
                    )
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                socket.send(MsgSerializer.to_bytes({"error": str(e)}))
        elif message.get("endpoint") == "render_current_slot_mask_panels":
            try:
                socket.send(MsgSerializer.to_bytes(agent.render_current_slot_mask_panels(message["data"])))
            except Exception as e:
                import traceback
                traceback.print_exc()
                socket.send(MsgSerializer.to_bytes({"error": str(e)}))
        elif message.get("endpoint") == "render_goal_slot_mask_panels":
            try:
                socket.send(MsgSerializer.to_bytes(agent.render_goal_slot_mask_panels(message.get("data"))))
            except Exception as e:
                import traceback
                traceback.print_exc()
                socket.send(MsgSerializer.to_bytes({"error": str(e)}))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--run-dir", type=str, default=SlotWorldModelAgent.DEFAULT_RUN_DIR)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--transport-loss-mix", type=float, default=1.0)
    parser.add_argument(
        "--disable-uot-goal-energy",
        action="store_true",
        help=(
            "A2: set the inference-time mask-loss/UOT mix to zero (pointwise mask L1 only), "
            "enable the historical V-JEPA-style goal initialization and distance filter above "
            "30 cm, and disable warmup-anchor steering."
        ),
    )
    parser.add_argument(
        "--always-execute-best-candidate",
        action="store_true",
        help=(
            "A1-A4: do not inject a dedicated pose no-op candidate and do not compare "
            "predicted candidate loss against observed current loss; execute the lowest-loss "
            "predicted pose candidate. Leave unset for A0."
        ),
    )
    parser.add_argument("--sinkhorn-iters", type=int, default=5)
    parser.add_argument("--sinkhorn-entropy-reg", type=float, default=0.05)
    parser.add_argument(
        "--mask-centralize-gamma",
        type=float,
        default=2.0,
        help="Power used to concentrate soft masks before OT while preserving each mask's original total mass.",
    )
    parser.add_argument(
        "--unbalanced-ot-rho",
        type=float,
        default=0.5,
        help="KL mass creation/destruction penalty for unbalanced mask OT. Larger values are closer to balanced OT.",
    )
    parser.add_argument(
        "--unbalanced-ot-eps",
        type=float,
        default=1.0e-8,
        help="Numerical epsilon for unbalanced OT mass updates.",
    )
    parser.add_argument(
        "--mask-transport-area-normalize",
        action="store_true",
        default=True,
        help=(
            "Normalize each slot UOT by that slot's max area over the frames used for "
            "fixed obj-latent extraction plus all goals. Enabled by default to prevent "
            "large background slots from dominating visual MPC without over-weighting vanished targets."
        ),
    )
    parser.add_argument(
        "--no-mask-transport-area-normalize",
        dest="mask_transport_area_normalize",
        action="store_false",
        help="Use raw mass-scaled UOT, matching the previous behavior.",
    )
    parser.add_argument(
        "--mask-transport-area-normalize-eps",
        type=float,
        default=1.0e-3,
        help="Minimum max slot area used as denominator for area-normalized UOT.",
    )
    parser.add_argument("--disable-shared-visible-wrist-ot", action="store_true")
    parser.add_argument("--shared-visible-sim-threshold", type=float, default=0.55)
    parser.add_argument("--shared-visible-temperature", type=float, default=0.05)
    parser.add_argument("--shared-visible-min-mass", type=float, default=0.02)
    parser.add_argument("--disable-factorized-pose-cem", action="store_true")
    parser.add_argument(
        "--factorized-cem-steps-per-phase",
        type=int,
        default=0,
        help=(
            "CEM iterations per pose phase. 0 reuses --cem-steps. "
            "In fixed-topk-then-wrist mode, wrist reranking is only applied on the final iteration."
        ),
    )
    parser.add_argument(
        "--gripper-cem-steps",
        type=int,
        default=1,
        help="CEM iterations for the gripper-only phase. Defaults to 1 because gripper has one DoF.",
    )
    parser.add_argument(
        "--gripper-samples",
        type=int,
        default=5,
        help="Number of uniformly sampled gripper candidates for the gripper-only phase.",
    )
    parser.add_argument("--cem-samples", type=int, default=80, help="Number of candidates sampled per CEM iteration.")
    parser.add_argument(
        "--cem-elite-topk",
        type=int,
        default=10,
        help="Number of fixed-view elite candidates used to update the pose CEM distribution.",
    )
    parser.add_argument(
        "--wrist-rerank-topk",
        type=int,
        default=10,
        help=(
            "Legacy rerank top-k kept for compatibility. Pose search now uses fixed-view coarse CEM "
            "followed by wrist-view refinement."
        ),
    )
    parser.add_argument(
        "--wrist-refine-momentum-mean",
        type=float,
        default=0.75,
        help=(
            "CEM mean momentum for wrist-view refinement after fixed-view coarse search. "
            "Larger values update the distribution less aggressively."
        ),
    )
    parser.add_argument(
        "--wrist-refine-momentum-std",
        type=float,
        default=0.90,
        help=(
            "CEM std momentum for wrist-view refinement after fixed-view coarse search. "
            "Larger values shrink / shift the distribution more conservatively."
        ),
    )
    parser.add_argument("--cem-steps", type=int, default=5, help="Default CEM iterations for the pose phase.")
    parser.add_argument(
        "--fixed-loss-search-min-scale",
        type=float,
        default=0.2,
        help=(
            "Minimum pose search scale when fixed-view goal loss is low. "
            "1 disables fixed-loss-based search-range tightening."
        ),
    )
    parser.add_argument("--pose-rotation-cap", type=float, default=0.4)
    parser.add_argument("--pose-rotation-std", type=float, default=0.2)
    parser.add_argument("--pose-translation-xy-cap", type=float, default=0.2)
    parser.add_argument("--pose-translation-z-cap", type=float, default=0.2)
    parser.add_argument("--pose-translation-xy-std", type=float, default=0.1)
    parser.add_argument("--pose-translation-z-std", type=float, default=0.1)
    parser.add_argument(
        "--execution-translation-cap",
        type=float,
        default=0.06,
        help=(
            "Final executed robot-frame xyz delta L2 cap. "
            "0 defaults to max(--pose-translation-xy-cap, --pose-translation-z-cap)."
        ),
    )
    parser.add_argument(
        "--execution-rotation-cap",
        type=float,
        default=0.02,
        help="Final executed robot-frame rotation delta L2 cap. 0 defaults to --pose-rotation-cap.",
    )
    parser.add_argument("--slot-action-range-scale", type=float, default=1.0)
    parser.add_argument("--action-l2-weight", type=float, default=0.0)
    parser.add_argument(
        "--view-loss-mode",
        type=str,
        default="gated",
        choices=("gated", "fixed_only", "wrist_only", "all"),
        help=(
            "Candidate ranking loss across camera views. fixed_only uses only "
            "non-wrist views; gated falls back to wrist when fixed views do not "
            "improve over the current fixed-view loss."
        ),
    )
    parser.add_argument("--fixed-view-loss-gate-threshold", type=float, default=0.003)
    parser.add_argument("--disable-gripper-cem", action="store_true")
    parser.add_argument("--gripper-position-max", type=float, default=1.0)
    parser.add_argument("--fixed-gripper-target", type=float, default=0.0)
    parser.add_argument(
        "--gripper-min-open-gap",
        type=float,
        default=0.05,
        help=(
            "Minimum normalized gap from fully closed for uniform gripper search. "
            "With SLOT convention 0=open, 1=closed, default samples up to 0.95."
        ),
    )
    parser.add_argument(
        "--gripper-downpress-max-z",
        type=float,
        default=0.05,
        help=(
            "Maximum robot-frame downward z offset coupled to gripper search. "
            "With SLOT convention 0=open, 1=closed, target z offset is -value * gripper."
        ),
    )
    parser.add_argument("--gripper-delta-std", type=float, default=0.1)
    parser.add_argument("--gripper-delta-cap", type=float, default=0.3)
    parser.add_argument("--gripper-cem-skip-translation-threshold", type=float, default=0.01)
    parser.add_argument("--gripper-cem-skip-rotation-threshold", type=float, default=0.04)
    parser.add_argument("--slot-warmup-xy-radius", type=float, default=DEFAULT_SLOT_WARMUP_XY_RADIUS)
    parser.add_argument(
        "--slot-warmup-z-offsets",
        type=float,
        nargs="+",
        default=DEFAULT_SLOT_WARMUP_Z_OFFSETS,
        metavar="Z_OFFSET",
        help="Robot-frame z offsets for SLOT warmup. Pass one or more values, e.g. -0.1 -0.2.",
    )
    parser.add_argument("--slot-warmup-settle-steps", type=int, default=DEFAULT_SLOT_WARMUP_SETTLE_STEPS)
    parser.add_argument(
        "--fixed-slot-min-area-ratio",
        type=float,
        default=0.01,
        help=(
            "Minimum average soft-mask area ratio required when selecting the warmup-stable fixed/gripper slot. "
            "This avoids choosing slots that are stable only because they are almost invisible."
        ),
    )
    parser.add_argument(
        "--fixed-slot-max-area-ratio",
        type=float,
        default=0.20,
        help="Maximum average soft-mask area ratio allowed for warmup-stable fixed/gripper slot selection.",
    )
    parser.add_argument(
        "--fixed-slot-entropy-weight",
        type=float,
        default=0.7,
        help="Rank-score weight for spatial mask entropy when selecting the warmup-stable fixed/gripper slot.",
    )
    parser.add_argument(
        "--fixed-slot-centroid-weight",
        type=float,
        default=0.3,
        help="Rank-score weight for warmup centroid jitter when selecting the warmup-stable fixed/gripper slot.",
    )
    parser.add_argument("--slot-steering-memory-topk", type=int, default=DEFAULT_SLOT_STEERING_MEMORY_TOPK)
    parser.add_argument("--slot-steering-sample-fraction", type=float, default=DEFAULT_SLOT_STEERING_SAMPLE_FRACTION)
    parser.add_argument("--slot-steering-loss-range-eps", type=float, default=DEFAULT_SLOT_STEERING_LOSS_RANGE_EPS)
    parser.add_argument(
        "--history-pose-sampling-temperature",
        type=float,
        default=0.05,
        help=(
            "Distance temperature for weighted-random pose history selection. "
            "Smaller values favor nearer history frames more strongly."
        ),
    )
    parser.add_argument(
        "--history-pose-uniform-mix",
        type=float,
        default=0.15,
        help="Uniform probability mixed into pose history sampling to preserve diversity.",
    )
    parser.add_argument(
        "--memory-selection-mode",
        choices=MEMORY_SELECTION_MODES,
        default="relevance",
        help=(
            "Transition-memory policy: relevance is A0, recent is A3, and "
            "in_trajectory_random is A4."
        ),
    )
    parser.add_argument(
        "--memory-random-seed",
        type=int,
        default=0,
        help="Fixed per-case RNG seed used by --memory-selection-mode=in_trajectory_random.",
    )
    parser.add_argument(
        "--reserve-visible-gpu-memory",
        action="store_true",
        default=True,
        help=(
            "After loading SLOT-AC, prefill the PyTorch CUDA cache on all CUDA-visible GPUs "
            "so this server can reuse the VRAM while other processes cannot take it. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-reserve-visible-gpu-memory",
        dest="reserve_visible_gpu_memory",
        action="store_false",
        help=(
            "Disable default GPU memory reservation. Use this when sharing visible GPUs "
            "with another process."
        ),
    )
    parser.add_argument(
        "--gpu-memory-reserve-fraction",
        type=float,
        default=0.85,
        help="Fraction of free VRAM, after margin, to reserve when --reserve-visible-gpu-memory is enabled.",
    )
    parser.add_argument(
        "--gpu-memory-reserve-margin-mb",
        type=float,
        default=4096,
        help="Free VRAM margin in MiB left unreserved for SLOT inference and CUDA workspace.",
    )
    parser.add_argument(
        "--gpu-memory-reserve-chunk-mb",
        type=float,
        default=256,
        help="Chunk size in MiB for GPU memory reservation allocations.",
    )
    parser.add_argument(
        "--gripper-cem-skip-planned-translation-threshold",
        dest="gripper_cem_skip_translation_threshold",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gripper-cem-skip-planned-rotation-threshold",
        dest="gripper_cem_skip_rotation_threshold",
        type=float,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    main(
        port=args.port,
        run_dir=args.run_dir,
        ckpt_path=args.ckpt_path,
        config_path=args.config_path,
        transport_loss_mix=0.0 if args.disable_uot_goal_energy else args.transport_loss_mix,
        use_goal_pose_prior=args.disable_uot_goal_energy,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_entropy_reg=args.sinkhorn_entropy_reg,
        mask_centralize_gamma=args.mask_centralize_gamma,
        unbalanced_ot_rho=args.unbalanced_ot_rho,
        unbalanced_ot_eps=args.unbalanced_ot_eps,
        mask_transport_area_normalize=args.mask_transport_area_normalize,
        mask_transport_area_normalize_eps=args.mask_transport_area_normalize_eps,
        use_shared_visible_wrist_ot=not args.disable_shared_visible_wrist_ot,
        shared_visible_sim_threshold=args.shared_visible_sim_threshold,
        shared_visible_temperature=args.shared_visible_temperature,
        shared_visible_min_mass=args.shared_visible_min_mass,
        factorized_pose_cem=not args.disable_factorized_pose_cem,
        factorized_cem_steps_per_phase=args.factorized_cem_steps_per_phase,
        gripper_cem_steps=args.gripper_cem_steps,
        gripper_samples=args.gripper_samples,
        cem_samples=args.cem_samples,
        cem_elite_topk=args.cem_elite_topk,
        wrist_rerank_topk=args.wrist_rerank_topk,
        wrist_refine_momentum_mean=args.wrist_refine_momentum_mean,
        wrist_refine_momentum_std=args.wrist_refine_momentum_std,
        cem_steps=args.cem_steps,
        fixed_loss_search_min_scale=args.fixed_loss_search_min_scale,
        pose_rotation_cap=args.pose_rotation_cap,
        pose_rotation_std=args.pose_rotation_std,
        pose_translation_xy_cap=args.pose_translation_xy_cap,
        pose_translation_z_cap=args.pose_translation_z_cap,
        pose_translation_xy_std=args.pose_translation_xy_std,
        pose_translation_z_std=args.pose_translation_z_std,
        execution_translation_cap=None
        if args.execution_translation_cap <= 0.0
        else args.execution_translation_cap,
        execution_rotation_cap=None
        if args.execution_rotation_cap <= 0.0
        else args.execution_rotation_cap,
        slot_action_range_scale=args.slot_action_range_scale,
        action_l2_weight=args.action_l2_weight,
        view_loss_mode=args.view_loss_mode,
        fixed_view_loss_gate_threshold=args.fixed_view_loss_gate_threshold,
        optimize_gripper_action=not args.disable_gripper_cem,
        gripper_position_max=args.gripper_position_max,
        fixed_gripper_target=args.fixed_gripper_target,
        gripper_min_open_gap=args.gripper_min_open_gap,
        gripper_downpress_max_z=args.gripper_downpress_max_z,
        gripper_delta_std=args.gripper_delta_std,
        gripper_delta_cap=args.gripper_delta_cap,
        gripper_cem_skip_translation_threshold=args.gripper_cem_skip_translation_threshold,
        gripper_cem_skip_rotation_threshold=args.gripper_cem_skip_rotation_threshold,
        slot_warmup_xy_radius=args.slot_warmup_xy_radius,
        slot_warmup_z_offsets=args.slot_warmup_z_offsets,
        slot_warmup_settle_steps=args.slot_warmup_settle_steps,
        fixed_slot_min_area_ratio=args.fixed_slot_min_area_ratio,
        fixed_slot_max_area_ratio=args.fixed_slot_max_area_ratio,
        fixed_slot_entropy_weight=args.fixed_slot_entropy_weight,
        fixed_slot_centroid_weight=args.fixed_slot_centroid_weight,
        slot_steering_memory_topk=args.slot_steering_memory_topk,
        slot_steering_sample_fraction=args.slot_steering_sample_fraction,
        slot_steering_loss_range_eps=args.slot_steering_loss_range_eps,
        history_pose_sampling_temperature=args.history_pose_sampling_temperature,
        history_pose_uniform_mix=args.history_pose_uniform_mix,
        memory_selection_mode=args.memory_selection_mode,
        memory_random_seed=args.memory_random_seed,
        always_execute_best_candidate=args.always_execute_best_candidate,
        reserve_visible_gpu_memory_after_load=args.reserve_visible_gpu_memory,
        gpu_memory_reserve_fraction=args.gpu_memory_reserve_fraction,
        gpu_memory_reserve_margin_mb=args.gpu_memory_reserve_margin_mb,
        gpu_memory_reserve_chunk_mb=args.gpu_memory_reserve_chunk_mb,
    )
