import os
import sys
import math
import copy
import json
import hashlib
import logging
import fcntl
import torch
import torch.nn.functional as F
import imageio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import ExitStack
from einops import rearrange, repeat
from tqdm import tqdm
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils
from planning.objectives import (
    _slot_area_normalizer,
    _slot_change_weights,
    _uot_mask_loss,
)
from planning.pushobj_context import (
    PUSHOBJ_STRUCTURED_CONTEXT_NAMES,
    PUSHOBJ_STRUCTURED_CONTEXT_TANGENT_RATIOS,
)

log = logging.getLogger(__name__)

_UINT32_MODULUS = 1 << 32


def _uint32_seed(value):
    """Map deterministic integer seed expressions to RandomState's domain."""
    return int(value) % _UINT32_MODULUS


class PlanEvaluator:  # evaluator for planning
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
        slot_pca_config=None,
        slot_uot_config=None,
        retrieval_dataset=None,
        normalized_action_clip=None,
        context_layout_ids=None,
        context_cache_identity=None,
        context_cache_dir=None,
        planning_memory_config=None,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        # MPC replaces obs_0 with the latest environment feedback.  Keep the
        # two frames used for the one-time slot initialization immutable so
        # their visualization cannot drift away from the fixed slot features.
        self.fixed_slot_obs_0 = copy.deepcopy(obs_0)
        self.fixed_slot_obs_g = copy.deepcopy(obs_g)
        self.fixed_slot_state_0 = np.array(state_0, copy=True)
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device
        self.state_0_torch = torch.as_tensor(self.state_0, device=self.device, dtype=torch.float32)
        self.slot_pca_config = slot_pca_config or {}
        self.slot_uot_config = slot_uot_config or {}
        self.retrieval_dataset = retrieval_dataset
        self.context_layout_ids = context_layout_ids
        self.context_cache_identity = str(context_cache_identity or "model")
        self.context_cache_dir = (
            None if context_cache_dir is None else Path(context_cache_dir).resolve()
        )
        planning_memory_config = dict(planning_memory_config or {})
        self.planning_memory_mode = str(
            planning_memory_config.get("mode", "relevance")
        )
        if self.planning_memory_mode not in {
            "relevance",
            "recent",
            "in_trajectory_random",
            "random_bank",
        }:
            raise ValueError(
                "planning_memory.mode must be one of relevance, recent, "
                "in_trajectory_random, random_bank; "
                f"got {self.planning_memory_mode!r}"
            )
        self.planning_memory_num_context = int(
            planning_memory_config.get(
                "num_context", getattr(self.wm, "retrieval_num_context", 5)
            )
        )
        self.planning_memory_history_window = int(
            planning_memory_config.get("history_window", 20)
        )
        self.planning_memory_seed = int(planning_memory_config.get("seed", 0))
        if self.planning_memory_num_context < 1:
            raise ValueError("planning_memory.num_context must be positive")
        if self.planning_memory_history_window < 1:
            raise ValueError("planning_memory.history_window must be positive")
        case_seeds = seed if isinstance(seed, (list, tuple, np.ndarray)) else [seed]
        self._planning_memory_rngs = [
            np.random.default_rng(
                np.random.SeedSequence(
                    [_uint32_seed(self.planning_memory_seed), _uint32_seed(case_seed)]
                )
            )
            for case_seed in case_seeds
        ]
        self._trajectory_memory = None
        self.planning_memory_indices = None
        self._retrieval_origins = None
        self._uot_transport_cache = {}
        self._normalized_action_low = None
        self._normalized_action_high = None
        self.normalized_action_clip = (
            None
            if normalized_action_clip is None
            else float(normalized_action_clip)
        )
        if self.normalized_action_clip is not None and self.normalized_action_clip <= 0:
            raise ValueError(
                "normalized_action_clip must be positive, got "
                f"{self.normalized_action_clip}"
            )
        self._initialize_action_bounds()
        self.fixed_initial_slots = None
        self.fixed_goal_slots = None
        self.fixed_goal_z = None
        self.planning_context_slots = None
        self.planning_context_actions = None
        self.planning_retrieval_context = None
        self.observed_transition_context = None
        self.observed_context_visual = None
        self.observed_context_actions = None
        self.observed_context_pool_indices = None
        self.observed_context_collision_mask = None
        self.observed_context_modes = None
        # Shared online-ICL pools stay on CPU and are reused when MPC receives
        # a new real block endpoint.  This lets every planning block retrieve
        # against its current state instead of freezing the episode's initial
        # nearest neighbours.
        self.observed_context_pools = None
        self.observed_context_num_context = None
        # Populated by eval_actions. MPC uses these to keep the returned plan
        # consistent with the first completed model step that succeeded.
        self.last_effective_actions = None
        self.last_first_success_model_steps = None
        if getattr(self.wm, "is_slot_world_model", False):
            self._initialize_fixed_planning_slots()

        self.plot_full = False  # plot all frames or frames after frameskip

    def _initialize_action_bounds(self):
        action_space = getattr(self.env, "action_space", None)
        # BaseVectorEnv exposes one space per worker, whereas SerialVectorEnv
        # exposes the common Box directly.
        if isinstance(action_space, (list, tuple)):
            action_space = action_space[0] if action_space else None
        if action_space is None or not hasattr(action_space, "low"):
            return
        low = torch.as_tensor(action_space.low, dtype=torch.float32)
        high = torch.as_tensor(action_space.high, dtype=torch.float32)
        mean = torch.as_tensor(self.preprocessor.action_mean, dtype=torch.float32)
        std = torch.as_tensor(self.preprocessor.action_std, dtype=torch.float32)
        if low.numel() != mean.numel() or high.numel() != mean.numel():
            log.warning(
                "Cannot project actions: environment bounds %s do not match action statistics %s",
                tuple(low.shape), tuple(mean.shape),
            )
            return
        std = std.clamp_min(1.0e-8)
        self._normalized_action_low = ((low - mean) / std).to(self.device)
        self._normalized_action_high = ((high - mean) / std).to(self.device)
        if self.normalized_action_clip is not None:
            clip = torch.as_tensor(
                self.normalized_action_clip,
                device=self.device,
                dtype=self._normalized_action_low.dtype,
            )
            self._normalized_action_low = torch.maximum(
                self._normalized_action_low, -clip
            )
            self._normalized_action_high = torch.minimum(
                self._normalized_action_high, clip
            )
        log.info(
            "Normalized primitive-action bounds: low=%s high=%s",
            self._normalized_action_low.detach().cpu().tolist(),
            self._normalized_action_high.detach().cpu().tolist(),
        )

    def project_actions_to_env_bounds(self, actions):
        """Project normalized action chunks to what the environment executes."""
        if self._normalized_action_low is None:
            return actions
        primitive_dim = self._normalized_action_low.numel()
        if actions.shape[-1] % primitive_dim != 0:
            raise ValueError(
                f"Action chunk dim {actions.shape[-1]} is not divisible by primitive dim {primitive_dim}"
            )
        shaped = actions.reshape(*actions.shape[:-1], -1, primitive_dim)
        low = self._normalized_action_low.to(actions.device, actions.dtype)
        high = self._normalized_action_high.to(actions.device, actions.dtype)
        return shaped.clamp(min=low, max=high).reshape_as(actions)

    @staticmethod
    def _detach_slot_dict(slots):
        return {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in slots.items()
        }

    def _initialize_fixed_planning_slots(self):
        self.fixed_slot_obs_0 = copy.deepcopy(self.obs_0)
        self.fixed_slot_obs_g = copy.deepcopy(self.obs_g)
        self.fixed_slot_state_0 = np.array(self.state_0, copy=True)
        trans_obs_0 = move_to_device(self.preprocessor.transform_obs(self.obs_0), self.device)
        trans_obs_g = move_to_device(self.preprocessor.transform_obs(self.obs_g), self.device)
        state_g = torch.as_tensor(self.state_g, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            context_slots, initial_slots, goal_slots, goal_z = self.wm.initialize_planning_slots(
                trans_obs_0,
                trans_obs_g,
                state_g=state_g,
            )
        self.fixed_initial_slots = self._detach_slot_dict(initial_slots)
        self.fixed_goal_slots = self._detach_slot_dict(goal_slots)
        self.fixed_goal_z = self._detach_slot_dict(goal_z)
        self.planning_context_slots = {
            "obj_latents": self.fixed_initial_slots["obj_latents"],
            "mask_logits": context_slots["mask_logits"].clone(),
            "mask_probs": context_slots["mask_probs"].clone(),
        }
        self.planning_context_proprio = self.wm._encode_proprio(
            self.state_0_torch[:, None]
        ).detach()
        self.planning_context_actions = self.wm._normalized_zero_action_chunk(
            self.state_0_torch.shape[0], 1, self.device, self.state_0_torch.dtype
        )
        self._initialize_trajectory_memory()
        self._refresh_planning_retrieval_context()

    def _initialize_trajectory_memory(self):
        """Create the frame-0 self-transition used for cold-start padding."""
        if self.planning_memory_mode not in {"recent", "in_trajectory_random"}:
            self._trajectory_memory = None
            return
        initial_logits = self.planning_context_slots["mask_logits"][:, -1:].detach()
        initial_proprio = self.planning_context_proprio[:, -1:].detach()
        self._trajectory_memory = {
            "source_mask_logits": initial_logits.clone(),
            "target_mask_logits": initial_logits.clone(),
            "source_proprio": initial_proprio.clone(),
            "target_proprio": initial_proprio.clone(),
            "actions": self.planning_context_actions[:, -1:].detach().clone(),
        }

    def _trajectory_memory_selection(self):
        """Select one fixed context set for the next complete CEM request."""
        memory = self._trajectory_memory
        if memory is None:
            return None
        batch_size, memory_size = memory["actions"].shape[:2]
        count = self.planning_memory_num_context
        selections = []
        for batch_index in range(batch_size):
            latest = memory_size - 1
            if self.planning_memory_mode == "recent":
                selected = list(range(max(1, memory_size - count), memory_size))
            else:
                history_count = max(0, count - 1)
                history_pool = np.arange(1, latest, dtype=np.int64)
                chosen_count = min(history_count, int(history_pool.size))
                chosen = (
                    self._planning_memory_rngs[batch_index].choice(
                        history_pool, size=chosen_count, replace=False
                    ).tolist()
                    if chosen_count
                    else []
                )
                selected = sorted(int(index) for index in chosen)
                if latest > 0:
                    selected.append(latest)
            selected = [0] * (count - len(selected)) + selected[-count:]
            selections.append(selected)
        return torch.as_tensor(selections, device=self.device, dtype=torch.long)

    def _materialize_trajectory_memory(self):
        indices = self._trajectory_memory_selection()
        if indices is None:
            return None

        def gather(key):
            value = self._trajectory_memory[key]
            view_shape = (indices.shape[0], indices.shape[1]) + (1,) * (value.ndim - 2)
            return torch.gather(
                value,
                1,
                indices.view(view_shape).expand(-1, -1, *value.shape[2:]),
            )

        source_logits = gather("source_mask_logits")
        target_logits = gather("target_mask_logits")
        source_proprio = gather("source_proprio")
        target_proprio = gather("target_proprio")
        actions = gather("actions")
        context_tokens = self.wm.transition_adapter(
            torch.cat(
                [source_proprio, self.wm.action_encoder(actions.float())], dim=-1
            )
        )
        self.wm.predictor.num_context_frames = max(
            int(self.wm.predictor.num_context_frames), indices.shape[1]
        )
        self.planning_memory_indices = indices.detach().cpu()
        return {
            "obj_latents": self.wm.predictor.fixed_obj_latents.expand(
                indices.shape[0], -1, -1
            ),
            "source_mask_logits": source_logits,
            "source_mask_probs": F.softmax(source_logits, dim=2),
            "target_mask_logits": target_logits,
            "target_mask_probs": F.softmax(target_logits, dim=2),
            "context_tokens": context_tokens,
            "target_proprio": target_proprio,
            # Online trajectory rows are negative and remain distinguishable
            # from the training retrieval bank in diagnostics.
            "indices": -indices - 1,
        }

    def _materialize_random_bank_memory(self):
        """Uniformly sample bank transitions, independent of the query state."""
        bank_size = int(getattr(self.wm, "loaded_retrieval_bank_size", 0))
        count = self.planning_memory_num_context
        if bank_size < count:
            raise RuntimeError(
                f"Retrieval bank has {bank_size} rows, fewer than A4's "
                f"num_context={count}"
            )
        batch_size = int(self.planning_context_proprio.shape[0])
        if len(self._planning_memory_rngs) != batch_size:
            raise RuntimeError(
                "Planning-memory RNG count does not match evaluation batch: "
                f"{len(self._planning_memory_rngs)} != {batch_size}"
            )
        sampled = np.stack(
            [
                rng.choice(bank_size, size=count, replace=False)
                for rng in self._planning_memory_rngs
            ],
            axis=0,
        )
        indices = torch.as_tensor(sampled, device=self.device, dtype=torch.long)
        self.planning_memory_indices = indices.detach().cpu()
        return self.wm.prepare_random_bank_context(indices)

    @torch.no_grad()
    def _refresh_planning_retrieval_context(self):
        """Retrieve once from the current real or predictor-generated state."""
        if self.planning_context_slots is None:
            self.planning_retrieval_context = None
            return
        if self.planning_memory_mode == "random_bank":
            self.planning_retrieval_context = self._detach_slot_dict(
                self._materialize_random_bank_memory()
            )
            return
        if self.observed_transition_context is not None:
            self.planning_retrieval_context = self.observed_transition_context
            return
        if self.planning_memory_mode in {"recent", "in_trajectory_random"}:
            self.planning_retrieval_context = self._detach_slot_dict(
                self._materialize_trajectory_memory()
            )
            return
        retrieved = self.wm.prepare_rollout_retrieval_context(
            self.planning_context_proprio[:, -1:],
            self.planning_context_slots["mask_logits"][:, -1:],
        )
        self.planning_retrieval_context = self._detach_slot_dict(retrieved)

    @torch.no_grad()
    def collect_random_transition_context(
        self,
        num_context,
        random_seed=0,
        pool_size=1000,
        action_std=1.5,
        action_temporal_correlation=0.0,
        min_agent_displacement=0.0,
        min_object_displacement=0.0,
        initial_agent_block_clearance=None,
        require_contact=False,
        contact_tolerance=1.0,
        velocity_std_scale=1.0,
        generation_strategy="random",
    ):
        """Load/build a shared layout pool and retrieve context by proprio."""
        num_context = int(num_context)
        pool_size = int(pool_size)
        action_temporal_correlation = float(action_temporal_correlation)
        min_agent_displacement = float(min_agent_displacement)
        min_object_displacement = float(min_object_displacement)
        initial_agent_block_clearance = (
            None
            if initial_agent_block_clearance is None
            else float(initial_agent_block_clearance)
        )
        require_contact = bool(require_contact)
        contact_tolerance = float(contact_tolerance)
        generation_strategy = str(generation_strategy)
        if num_context < 1:
            raise ValueError("num_context must be positive when collecting test context")
        if pool_size < num_context:
            raise ValueError("context pool must cover the requested context count")
        if not 0.0 <= action_temporal_correlation < 1.0:
            raise ValueError("action_temporal_correlation must lie in [0, 1)")
        if min_agent_displacement < 0.0:
            raise ValueError("min_agent_displacement must be non-negative")
        if min_object_displacement < 0.0:
            raise ValueError("min_object_displacement must be non-negative")
        if (
            initial_agent_block_clearance is not None
            and initial_agent_block_clearance < 0.0
        ):
            raise ValueError("initial_agent_block_clearance must be non-negative")
        if contact_tolerance < 0.0:
            raise ValueError("contact_tolerance must be non-negative")
        valid_generation_strategies = ("random", "pushobj_structured_torque")
        if generation_strategy not in valid_generation_strategies:
            raise ValueError(
                "Unknown test context generation strategy "
                f"{generation_strategy!r}; expected one of "
                f"{valid_generation_strategies}"
            )
        if generation_strategy == "pushobj_structured_torque":
            if getattr(self.wm, "env_name", None) != "pusht":
                raise ValueError(
                    "pushobj_structured_torque context generation supports only PushT/PushObj"
                )
            if num_context < len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES):
                raise ValueError(
                    "Structured PushObj recall needs at least five contexts "
                    "to cover translation and signed moderate/strong torque"
                )
            if not require_contact or initial_agent_block_clearance is None:
                raise ValueError(
                    "Structured PushObj contexts require contact=true and an "
                    "explicit initial agent-block clearance"
                )
        batch_size = int(self.state_0_torch.shape[0])
        initial_state = np.array(self.state_0, copy=True)
        layout_ids = self.context_layout_ids
        if layout_ids is None or len(layout_ids) != batch_size:
            raise RuntimeError(
                "Shared test context pools require one layout_id per task"
            )
        cache_paths = [
            self._context_pool_cache_path(
                layout_id,
                pool_size,
                random_seed,
                action_std,
                action_temporal_correlation,
                min_agent_displacement,
                min_object_displacement,
                initial_agent_block_clearance,
                require_contact,
                contact_tolerance,
                velocity_std_scale,
                generation_strategy,
            )
            for layout_id in layout_ids
        ]
        unique_paths = sorted(set(cache_paths), key=str)
        with ExitStack() as stack:
            for cache_path in unique_paths:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                lock_file = stack.enter_context(
                    cache_path.with_suffix(cache_path.suffix + ".lock").open("a+")
                )
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            missing = [path for path in unique_paths if not path.exists()]
            if missing:
                generated = self._generate_random_context_pools(
                    pool_size,
                    random_seed,
                    action_std,
                    action_temporal_correlation,
                    min_agent_displacement,
                    min_object_displacement,
                    initial_agent_block_clearance,
                    require_contact,
                    contact_tolerance,
                    velocity_std_scale,
                    generation_strategy,
                )
                for batch_index, cache_path in enumerate(cache_paths):
                    if cache_path not in missing or cache_path.exists():
                        continue
                    temporary = cache_path.with_name(
                        f".{cache_path.name}.{os.getpid()}.tmp"
                    )
                    torch.save(generated[batch_index], temporary)
                    os.replace(temporary, cache_path)
                    log.info("Saved shared test context pool to %s", cache_path)
            pools = [torch.load(path, map_location="cpu") for path in cache_paths]

        self.observed_context_pools = pools
        self.observed_context_num_context = num_context
        context = self._reretrieve_observed_transition_context()
        self.observed_context_visual = self._render_selected_context_states(
            context["selected_source_states"],
            context["selected_target_states"],
            initial_state,
        )
        reset_obs, reset_state = self.env.prepare(self.seed, initial_state)
        if not np.allclose(
            np.asarray(reset_state), initial_state, rtol=1.0e-5, atol=1.0e-5
        ):
            raise RuntimeError(
                "Environment changed the requested initial state while resetting "
                "after online context collection"
            )
        self.obs_0 = {
            key: np.expand_dims(values, axis=1)
            for key, values in reset_obs.items()
        }
        self.state_0 = np.array(reset_state, copy=True)
        self.state_0_torch = torch.as_tensor(
            self.state_0, device=self.device, dtype=torch.float32
        )
        self._initialize_fixed_planning_slots()
        interaction_name = (
            "contacts"
            if getattr(self.wm, "env_name", None) == "pusht"
            else "collisions"
        )
        log.info(
            "Retrieved %d contexts (%d %s) from shared %d-transition "
            "layout pools; "
            "environment reset to the original task start",
            num_context,
            int(self.observed_context_collision_mask.sum().item()),
            interaction_name,
            pool_size,
        )
        return context["selected_actions"]

    @torch.no_grad()
    def _reretrieve_observed_transition_context(self):
        """Refresh online ICL neighbours for the current real MPC state."""
        if self.observed_context_pools is None:
            return None
        selected_indices = self._retrieve_from_context_pools(
            self.observed_context_pools,
            self.observed_context_num_context,
        )
        context = self._materialize_context_pool_selection(
            self.observed_context_pools, selected_indices
        )
        predictor_context_keys = (
            "obj_latents",
            "source_mask_logits",
            "source_mask_probs",
            "target_mask_logits",
            "target_mask_probs",
            "context_tokens",
            "target_proprio",
            "indices",
        )
        self.observed_transition_context = self._detach_slot_dict(
            {key: context[key] for key in predictor_context_keys}
        )
        self.observed_context_pool_indices = torch.stack(selected_indices)
        self.observed_context_actions = context["selected_actions"].detach().cpu()
        self.observed_context_collision_mask = context[
            "selected_collision_mask"
        ].detach().cpu()
        self.observed_context_modes = context[
            "selected_context_modes"
        ].detach().cpu()
        self.planning_retrieval_context = self.observed_transition_context
        return context

    def _context_pool_cache_path(
        self,
        layout_id,
        pool_size,
        seed_value,
        action_std,
        action_temporal_correlation,
        min_agent_displacement,
        min_object_displacement,
        initial_agent_block_clearance,
        require_contact,
        contact_tolerance,
        velocity_std_scale,
        generation_strategy,
    ):
        bank_path = Path(self.wm._retrieval_bank_path)
        identity = json.dumps(
            {
                "model": self.context_cache_identity,
                "env": getattr(self.wm, "env_name", "env"),
                "layout": int(torch.as_tensor(layout_id).item()),
                "pool_size": int(pool_size),
                "seed": int(seed_value),
                "action_std": float(action_std),
                "action_temporal_correlation": float(action_temporal_correlation),
                "min_agent_displacement": float(min_agent_displacement),
                "min_object_displacement": float(min_object_displacement),
                "initial_agent_block_clearance": initial_agent_block_clearance,
                "require_contact": bool(require_contact),
                "contact_tolerance": float(contact_tolerance),
                "frameskip": int(self.frameskip),
                # Invalidate pools created when every Wall context source was
                # forced to start with zero velocity.
                "source_state_sampling": (
                    "random_velocity_collision_labels_v2"
                    if getattr(self.wm, "env_name", None) == "wall"
                    else (
                        "pusht_contact_probe_release_v4"
                        if getattr(self.wm, "env_name", None) == "pusht"
                        else "random_velocity_v1"
                    )
                ),
                "velocity_std_scale": float(velocity_std_scale),
                # This also acts as a cache schema/version: structured pools
                # must never silently reuse legacy isotropic random pools.
                "generation_strategy": str(generation_strategy),
                "generation_strategy_schema": 1,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        env_name = str(getattr(self.wm, "env_name", "env")).replace("/", "-")
        layout_value = int(torch.as_tensor(layout_id).item())
        cache_dir = (
            self.context_cache_dir
            if self.context_cache_dir is not None
            else bank_path.parent / "test_context_pool"
        )
        return cache_dir / f"pool_{env_name}_layout{layout_value}_{digest}.pt"

    @torch.no_grad()
    def _generate_random_context_pools(
        self,
        pool_size,
        seed_value,
        action_std,
        action_temporal_correlation,
        min_agent_displacement,
        min_object_displacement,
        initial_agent_block_clearance,
        require_contact,
        contact_tolerance,
        velocity_std_scale,
        generation_strategy,
    ):
        """Generate and encode valid random transitions for each worker layout."""
        batch_size = int(self.state_0_torch.shape[0])
        primitive_dim = self.action_dim_from_bounds
        fields = [
            {key: [] for key in (
                "source_states", "target_states", "actions",
                "source_mask_logits", "target_mask_logits",
                "collision_mask",
                "context_modes",
            )}
            for _ in range(batch_size)
        ]
        chunk_size = 16
        layout_label = ",".join(
            str(int(torch.as_tensor(layout).item()))
            for layout in self.context_layout_ids
        )
        chunk_starts = range(0, pool_size, chunk_size)
        for chunk_start in tqdm(
            chunk_starts,
            total=math.ceil(pool_size / chunk_size),
            desc=f"Online context pool layout={layout_label}",
            unit="batch",
            dynamic_ncols=True,
            # Multi-GPU planning redirects each case's stderr to plan.log.
            # Keep this long-running cache-miss progress visible in the launch
            # terminal while ordinary worker output remains case-local.
            file=sys.__stderr__ or sys.stderr,
        ):
            chunk_count = min(chunk_size, pool_size - chunk_start)
            source_visuals, target_visuals = [], []
            source_states = []
            target_states = []
            action_chunks = []
            collision_masks = []
            context_mode_ids = []
            for offset in range(chunk_count):
                sample_index = chunk_start + offset
                is_pusht = getattr(self.wm, "env_name", None) == "pusht"
                accepted_mask = np.zeros(batch_size, dtype=bool)
                accepted_observed = None
                accepted_states = None
                accepted_actions = None
                accepted_contact_mask = np.zeros(batch_size, dtype=bool)
                for attempt in range(100 if is_pusht else 1):
                    attempt_index = sample_index + attempt * pool_size
                    sample_seeds = [
                        _uint32_seed(
                            int(seed_value)
                            + 1000003 * int(torch.as_tensor(layout).item())
                            + attempt_index
                        )
                        for layout in self.context_layout_ids
                    ]
                    starts, _ = self.env.sample_random_init_goal_states(sample_seeds)
                    if is_pusht and initial_agent_block_clearance is not None:
                        placement_seeds = [
                            _uint32_seed(seed + 700000001)
                            for seed in sample_seeds
                        ]
                        starts = self.env.place_agent_near_block(
                            starts,
                            placement_seeds,
                            np.full(
                                batch_size,
                                float(initial_agent_block_clearance),
                                dtype=np.float64,
                            ),
                        )
                    normalized = []
                    context_mode = (
                        sample_index % len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES)
                        if generation_strategy == "pushobj_structured_torque"
                        else -1
                    )
                    for batch_index, layout in enumerate(self.context_layout_ids):
                        rng = np.random.default_rng(
                            np.random.SeedSequence([
                                _uint32_seed(seed_value),
                                _uint32_seed(torch.as_tensor(layout).item()),
                                _uint32_seed(attempt_index),
                            ])
                        )
                        action_noise = rng.normal(
                            0.0,
                            float(action_std),
                            size=(self.frameskip, primitive_dim),
                        ).astype(np.float32)
                        if action_temporal_correlation > 0.0:
                            innovation_scale = np.sqrt(
                                1.0 - action_temporal_correlation**2
                            )
                            for action_index in range(1, self.frameskip):
                                action_noise[action_index] = (
                                    action_temporal_correlation
                                    * action_noise[action_index - 1]
                                    + innovation_scale * action_noise[action_index]
                                )
                        if is_pusht and require_contact:
                            if primitive_dim != 2:
                                raise ValueError(
                                    "PushObj contact context requires 2-D actions"
                                )
                            inward = (
                                starts[batch_index, 2:4]
                                - starts[batch_index, :2]
                            )
                            inward /= max(np.linalg.norm(inward), 1.0e-8)
                            tangent = np.asarray(
                                [-inward[1], inward[0]], dtype=np.float32
                            )
                            magnitudes = np.linalg.norm(action_noise, axis=-1)
                            magnitudes = np.clip(
                                magnitudes,
                                0.75 * float(action_std),
                                1.25 * float(action_std),
                            )
                            if generation_strategy == "pushobj_structured_torque":
                                tangent_ratio = (
                                    PUSHOBJ_STRUCTURED_CONTEXT_TANGENT_RATIOS[
                                        context_mode
                                    ]
                                )
                                # Small random jitter keeps the pool stochastic
                                # while retaining five clearly separated action
                                # families and the existing random,
                                # shape-conditioned contact placement.
                                ratio_jitter = 0.08 * np.tanh(action_noise[:, 0])
                                ratios = tangent_ratio + ratio_jitter
                                directions = (
                                    inward[None]
                                    + ratios[:, None] * tangent[None]
                                )
                                directions /= np.linalg.norm(
                                    directions, axis=-1, keepdims=True
                                ).clip(1.0e-8)
                            else:
                                heading_jitter = 0.25 * np.tanh(
                                    action_noise[:, 0]
                                )
                                directions = (
                                    np.cos(heading_jitter)[:, None] * inward
                                    + np.sin(heading_jitter)[:, None] * tangent
                                )
                            action_noise = (
                                directions * magnitudes[:, None]
                            ).astype(np.float32)
                            # Break physical contact in the stored target frame.
                            # The preceding actions still provide a sustained
                            # push and measurable object motion.
                            action_noise[-1] = -inward * magnitudes[-1]
                        normalized.append(action_noise.reshape(-1))
                        if (
                            getattr(self.wm, "env_name", None) == "wall"
                            and starts.shape[-1] >= 4
                        ):
                            # Wall reset states used to set vx=vy=0 for every pool
                            # entry. Training/validation transitions contain the
                            # full realized-velocity distribution.
                            velocity_rng = np.random.default_rng(
                                np.random.SeedSequence([
                                    _uint32_seed(seed_value),
                                    _uint32_seed(torch.as_tensor(layout).item()),
                                    _uint32_seed(attempt_index),
                                    1,
                                ])
                            )
                            velocity_mean = (
                                self.wm.state_mean[2:4].detach().cpu().numpy()
                            )
                            velocity_std = (
                                self.wm.state_std[2:4].detach().cpu().numpy()
                            )
                            velocity = velocity_rng.normal(
                                velocity_mean,
                                velocity_std * float(velocity_std_scale),
                            ).astype(np.float32)
                            action_space = getattr(self.env, "action_space", None)
                            if action_space is not None:
                                velocity = np.clip(
                                    velocity,
                                    np.asarray(action_space.low, dtype=np.float32),
                                    np.asarray(action_space.high, dtype=np.float32),
                                )
                            starts[batch_index, 2:4] = velocity
                    actions = self.project_actions_to_env_bounds(
                        torch.as_tensor(
                            np.stack(normalized), device=self.device
                        )[:, None]
                    )
                    primitive_actions = rearrange(
                        actions.cpu(),
                        "b t (f d) -> b (t f) d",
                        f=self.frameskip,
                    )
                    primitive_actions = self.preprocessor.denormalize_actions(
                        primitive_actions
                    ).numpy()
                    observed, states = self.env.rollout(
                        sample_seeds, starts, primitive_actions
                    )
                    if not is_pusht:
                        break
                    agent_displacement = np.linalg.norm(
                        states[:, -1, :2] - states[:, 0, :2], axis=-1
                    )
                    object_displacement = np.linalg.norm(
                        states[:, -1, 2:4] - states[:, 0, 2:4], axis=-1
                    )
                    agent_block_clearances = np.stack(
                        [
                            self.env.agent_block_clearance(states[:, time_index])
                            for time_index in range(states.shape[1])
                        ],
                        axis=1,
                    )
                    pusht_contact_mask = np.min(
                        agent_block_clearances, axis=1
                    ) <= float(contact_tolerance)
                    valid_mask = (
                        (self.env.agent_block_clearance(states[:, -1]) >= 0.0)
                        & (agent_displacement >= min_agent_displacement)
                        & (object_displacement >= min_object_displacement)
                        & ((not require_contact) | pusht_contact_mask)
                    )
                    newly_accepted = valid_mask & ~accepted_mask
                    if np.any(newly_accepted):
                        if accepted_observed is None:
                            accepted_observed = {
                                key: np.empty_like(value)
                                for key, value in observed.items()
                            }
                            accepted_states = np.empty_like(states)
                            accepted_actions = torch.empty_like(actions)
                        for key, value in observed.items():
                            accepted_observed[key][newly_accepted] = value[
                                newly_accepted
                            ]
                        accepted_states[newly_accepted] = states[newly_accepted]
                        accepted_indices = torch.as_tensor(
                            np.flatnonzero(newly_accepted),
                            device=actions.device,
                            dtype=torch.long,
                        )
                        accepted_actions.index_copy_(
                            0,
                            accepted_indices,
                            actions.index_select(0, accepted_indices),
                        )
                        accepted_contact_mask[newly_accepted] = (
                            pusht_contact_mask[newly_accepted]
                        )
                        accepted_mask |= newly_accepted
                    if np.all(accepted_mask):
                        observed = accepted_observed
                        states = accepted_states
                        actions = accepted_actions
                        pusht_contact_mask = accepted_contact_mask
                        break
                else:
                    raise RuntimeError(
                        "Could not generate a valid PushObj context "
                        f"transition after 100 attempts for sample {sample_index}"
                    )
                if getattr(self.wm, "env_name", None) == "wall":
                    realized_velocity = (
                        states[:, 1:, :2] - states[:, :-1, :2]
                    ) / 2.0
                    collision_mask = np.any(
                        np.abs(realized_velocity - primitive_actions) > 1.0e-3,
                        axis=(1, 2),
                    )
                elif is_pusht:
                    collision_mask = pusht_contact_mask
                else:
                    collision_mask = np.zeros(batch_size, dtype=bool)
                source_visuals.append(observed["visual"][:, 0])
                target_visuals.append(observed["visual"][:, -1])
                source_states.append(states[:, 0])
                target_states.append(states[:, -1])
                action_chunks.append(actions[:, 0].cpu())
                collision_masks.append(torch.as_tensor(collision_mask))
                context_mode_ids.append(context_mode)
            source_visual = np.stack(source_visuals, axis=1)
            target_visual = np.stack(target_visuals, axis=1)
            source_state = torch.as_tensor(np.stack(source_states, axis=1)).float()
            target_state = torch.as_tensor(np.stack(target_states, axis=1)).float()
            actions = torch.stack(action_chunks, dim=1).float()
            collision_mask = torch.stack(collision_masks, dim=1).bool()
            context_modes = torch.as_tensor(
                context_mode_ids, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1).contiguous()
            encoded = self.wm.prepare_observed_transition_context(
                self.preprocessor.transform_obs_visual(source_visual).to(self.device),
                self.preprocessor.transform_obs_visual(target_visual).to(self.device),
                source_state.to(self.device),
                target_state.to(self.device),
                actions.to(self.device),
            )
            for batch_index in range(batch_size):
                for key, value in (
                    ("source_states", source_state),
                    ("target_states", target_state),
                    ("actions", actions),
                    ("source_mask_logits", encoded["source_mask_logits"].cpu()),
                    ("target_mask_logits", encoded["target_mask_logits"].cpu()),
                    ("collision_mask", collision_mask),
                    ("context_modes", context_modes),
                ):
                    fields[batch_index][key].append(value[batch_index].cpu())
        return [
            {
                "schema_version": 1,
                "layout_id": int(torch.as_tensor(self.context_layout_ids[index]).item()),
                **{key: torch.cat(parts, dim=0) for key, parts in field.items()},
            }
            for index, field in enumerate(fields)
        ]

    @torch.no_grad()
    def _retrieve_from_context_pools(self, pools, num_context):
        selected = []
        for batch_index, pool in enumerate(pools):
            expected_layout = int(
                torch.as_tensor(self.context_layout_ids[batch_index]).item()
            )
            if int(pool.get("schema_version", -1)) != 1:
                raise RuntimeError("Unsupported test context pool schema")
            if int(pool.get("layout_id", -1)) != expected_layout:
                raise RuntimeError("Test context pool layout does not match task layout")
            source_states = pool["source_states"].to(self.device).float()
            source_proprio = self.wm._encode_proprio(source_states)
            query_proprio = self.planning_context_proprio[batch_index, -1].float()
            mean = source_proprio.mean(dim=0, keepdim=True)
            std = source_proprio.std(dim=0, keepdim=True).clamp_min(1.0e-4)
            normalized_sources = (source_proprio - mean) / std
            normalized_query = (query_proprio - mean) / std
            distances = (normalized_sources - normalized_query).square().sum(dim=-1)
            modes = pool.get("context_modes")
            structured_modes = (
                modes is not None
                and bool((torch.as_tensor(modes) >= 0).all())
                and torch.unique(torch.as_tensor(modes)).numel()
                == len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES)
            )
            if structured_modes:
                modes = torch.as_tensor(modes, device=self.device).long()
                base_quota, remainder = divmod(
                    num_context, len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES)
                )
                chosen_parts = []
                for mode_index in range(len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES)):
                    quota = base_quota + int(mode_index < remainder)
                    if quota == 0:
                        continue
                    eligible = torch.nonzero(
                        modes == mode_index, as_tuple=False
                    ).flatten()
                    if eligible.numel() < quota:
                        raise RuntimeError(
                            f"Structured context mode {mode_index} has only "
                            f"{eligible.numel()} rows for quota {quota}"
                        )
                    local = torch.topk(
                        distances.index_select(0, eligible),
                        k=quota,
                        largest=False,
                        sorted=False,
                    ).indices
                    chosen_parts.append(eligible.index_select(0, local))
                chosen = torch.cat(chosen_parts)
            else:
                chosen = torch.topk(
                    distances,
                    k=num_context,
                    largest=False,
                    sorted=False,
                ).indices
            chosen = chosen[torch.argsort(distances.index_select(0, chosen))]
            selected.append(chosen.cpu())
        return selected

    @torch.no_grad()
    def _materialize_context_pool_selection(self, pools, selected_indices):
        def selected_tensor(key):
            return torch.stack(
                [
                    pool[key].index_select(0, indices)
                    for pool, indices in zip(pools, selected_indices)
                ],
                dim=0,
            ).to(self.device)

        source_states = selected_tensor("source_states").float()
        target_states = selected_tensor("target_states").float()
        actions = selected_tensor("actions").float()
        source_logits = selected_tensor("source_mask_logits").float()
        target_logits = selected_tensor("target_mask_logits").float()
        collision_mask = (
            selected_tensor("collision_mask").bool()
            if all("collision_mask" in pool for pool in pools)
            else torch.zeros(
                actions.shape[:2], device=self.device, dtype=torch.bool
            )
        )
        selected_modes = (
            selected_tensor("context_modes").long()
            if all("context_modes" in pool for pool in pools)
            else torch.full(
                actions.shape[:2], -1, device=self.device, dtype=torch.long
            )
        )
        source_proprio = self.wm._encode_proprio(source_states)
        target_proprio = self.wm._encode_proprio(target_states)
        context_tokens = self.wm.transition_adapter(
            torch.cat(
                [source_proprio, self.wm.action_encoder(actions)], dim=-1
            )
        )
        batch_size = len(pools)
        return {
            "obj_latents": self.wm.predictor.fixed_obj_latents.expand(
                batch_size, -1, -1
            ),
            "source_mask_logits": source_logits,
            "source_mask_probs": F.softmax(source_logits, dim=2),
            "target_mask_logits": target_logits,
            "target_mask_probs": F.softmax(target_logits, dim=2),
            "context_tokens": context_tokens,
            "target_proprio": target_proprio,
            # Negative values distinguish shared test-pool rows from training
            # retrieval-bank rows in rollout traces and visualization.
            "indices": (-torch.stack(selected_indices) - 1).to(self.device),
            "selected_source_states": source_states,
            "selected_target_states": target_states,
            "selected_actions": actions,
            "selected_collision_mask": collision_mask,
            "selected_context_modes": selected_modes,
        }

    def _render_selected_context_states(
        self, source_states, target_states, initial_state
    ):
        source_visuals, target_visuals = [], []
        num_context = int(source_states.shape[1])
        for context_index in range(num_context):
            source_obs, _ = self.env.prepare(
                self.seed, source_states[:, context_index].detach().cpu().numpy()
            )
            target_obs, _ = self.env.prepare(
                self.seed, target_states[:, context_index].detach().cpu().numpy()
            )
            source_visuals.append(source_obs["visual"])
            target_visuals.append(target_obs["visual"])
        self.env.prepare(self.seed, initial_state)
        return {
            "source": np.stack(source_visuals, axis=1),
            "target": np.stack(target_visuals, axis=1),
        }

    @property
    def action_dim_from_bounds(self):
        if self._normalized_action_low is not None:
            return int(self._normalized_action_low.numel())
        action_dim = int(getattr(self.wm, "action_dim", 0))
        if action_dim < 1:
            raise RuntimeError("Cannot infer primitive action dimension")
        return action_dim

    def snapshot_planning_context(self):
        """Capture the query state at the start of one environment MPC block."""
        if self.planning_context_slots is None:
            return None
        return {
            "slots": self._detach_slot_dict(
                {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in self.planning_context_slots.items()
                }
            ),
            "proprio": self.planning_context_proprio.detach().clone(),
            "actions": self.planning_context_actions.detach().clone(),
            "retrieval": self._detach_slot_dict(
                {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in self.planning_retrieval_context.items()
                }
            ),
        }

    def restore_planning_context(self, snapshot):
        """Restore a block-start query after predictor-only action selection."""
        if snapshot is None:
            return
        self.planning_context_slots = snapshot["slots"]
        self.planning_context_proprio = snapshot["proprio"]
        self.planning_context_actions = snapshot["actions"]
        self.planning_retrieval_context = snapshot["retrieval"]

    @torch.no_grad()
    def advance_planning_context_with_prediction(self, action):
        """Predict one model step and retrieve from its predicted state.

        No environment observation is read here.  The predicted mask and
        proprio embedding become the unique query for the next CEM decision.
        """
        if self.planning_context_slots is None:
            raise RuntimeError("Predictive planning context requires a Slot world model")
        action = self.project_actions_to_env_bounds(action.detach())
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )
        output, _ = self.wm.rollout(
            obs_0=trans_obs_0,
            act=action,
            state_0=self.state_0_torch,
            context_slots=self.planning_context_slots,
            context_proprio_embeddings=self.planning_context_proprio,
            context_actions=self.planning_context_actions,
            retrieval_context=self.planning_retrieval_context,
        )
        next_mask = output["mask_probs"][:, -1:].detach()
        self.planning_context_slots = {
            "obj_latents": self.planning_context_slots["obj_latents"],
            "mask_logits": next_mask.clamp_min(1.0e-8).log(),
            "mask_probs": next_mask,
        }
        # This is already state_embedding4/predictor output.  Do not replace it
        # with an embedding computed from an environment proprio observation.
        self.planning_context_proprio = output["proprio"][:, -1:].detach()
        self.planning_context_actions = action[:, -1:]
        self._refresh_planning_retrieval_context()
        log.info(
            "Advanced Slot-AC query from predictor masks=%s proprio_embedding=%s; "
            "retrieved context before next CEM without environment observation",
            tuple(self.planning_context_slots["mask_logits"].shape),
            tuple(self.planning_context_proprio.shape),
        )

    def get_planning_retrieval_context(self, traj_index=None, repeats=None):
        context = self.planning_retrieval_context
        if context is None:
            return None
        if traj_index is not None:
            context = {
                key: value[traj_index : traj_index + 1]
                for key, value in context.items()
            }
        if repeats is not None:
            context = {
                key: value.expand(repeats, *value.shape[1:]).contiguous()
                for key, value in context.items()
            }
        return context

    @torch.no_grad()
    def reset_planning_context_to_initial(self):
        """Restore the real initial query before full-trajectory evaluation."""
        if self.fixed_initial_slots is None:
            return
        self.planning_context_slots = {
            "obj_latents": self.fixed_initial_slots["obj_latents"],
            "mask_logits": self.fixed_initial_slots["mask_logits"][:, -1:].clone(),
            "mask_probs": self.fixed_initial_slots["mask_probs"][:, -1:].clone(),
        }
        initial_state = torch.as_tensor(
            self.fixed_slot_state_0, device=self.device, dtype=torch.float32
        )
        self.planning_context_proprio = self.wm._encode_proprio(
            initial_state[:, None]
        ).detach()
        self.planning_context_actions = self.wm._normalized_zero_action_chunk(
            initial_state.shape[0], 1, self.device, initial_state.dtype
        )
        self._refresh_planning_retrieval_context()

    def reinitialize_planning_context(self):
        """Re-extract the current real start while preserving episode ICL context."""
        if getattr(self.wm, "is_slot_world_model", False):
            self._initialize_fixed_planning_slots()

    def get_fixed_context_slots(self, traj_index=None, repeats=None):
        if self.fixed_initial_slots is None:
            return None
        slots = self.planning_context_slots
        if traj_index is not None:
            slots = {
                key: value[traj_index : traj_index + 1] if isinstance(value, torch.Tensor) else value
                for key, value in slots.items()
            }
        if repeats is not None:
            slots = {
                key: value.expand(repeats, *value.shape[1:]).contiguous()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in slots.items()
            }
        return slots

    def get_planning_context_proprio(self, traj_index=None, repeats=None):
        tokens = self.planning_context_proprio
        if traj_index is not None:
            tokens = tokens[traj_index : traj_index + 1]
        if repeats is not None:
            tokens = tokens.expand(repeats, *tokens.shape[1:]).contiguous()
        return tokens

    def get_planning_context_actions(self, traj_index=None, repeats=None):
        actions = self.planning_context_actions
        if traj_index is not None:
            actions = actions[traj_index : traj_index + 1]
        if repeats is not None:
            actions = actions.expand(repeats, *actions.shape[1:]).contiguous()
        return actions

    @torch.no_grad()
    def update_planning_context(self, action_so_far, executed_obses, executed_states):
        """Observe a completed block and refresh retrieval or trajectory memory."""
        model_steps = action_so_far.shape[1]
        if model_steps < 1:
            return
        final_frame_index = model_steps * self.frameskip
        frame_indices = [step * self.frameskip for step in range(model_steps + 1)]
        real_visual = executed_obses["visual"][:, frame_indices]
        transformed = self.preprocessor.transform_obs_visual(real_visual).to(self.device)
        decoded = self.wm.encode_with_object_latents(
            transformed, self.fixed_initial_slots["obj_latents"]
        )
        real_states = torch.as_tensor(
            executed_states[:, frame_indices], device=self.device
        ).float()
        real_proprio = self.wm._encode_proprio(real_states).detach()
        if self._trajectory_memory is not None:
            appended = {
                "source_mask_logits": decoded["mask_logits"][:, :-1].detach(),
                "target_mask_logits": decoded["mask_logits"][:, 1:].detach(),
                "source_proprio": real_proprio[:, :-1],
                "target_proprio": real_proprio[:, 1:],
                "actions": action_so_far.detach(),
            }
            self._trajectory_memory = {
                key: torch.cat([self._trajectory_memory[key], value], dim=1)
                for key, value in appended.items()
            }
        self.planning_context_slots = {
            "obj_latents": self.fixed_initial_slots["obj_latents"],
            "mask_logits": decoded["mask_logits"][:, -1:],
            "mask_probs": decoded["mask_probs"][:, -1:],
        }
        self.planning_context_proprio = real_proprio[:, -1:]
        self.planning_context_actions = action_so_far.detach()[:, -1:]
        refreshed_context = self._reretrieve_observed_transition_context()
        if refreshed_context is not None:
            self.observed_context_visual = self._render_selected_context_states(
                refreshed_context["selected_source_states"],
                refreshed_context["selected_target_states"],
                real_states[:, -1].detach().cpu().numpy(),
            )
        self._refresh_planning_retrieval_context()
        log.info(
            "Observed Slot-AC block endpoint frame=%d masks=%s proprio=%s; "
            "selected context mode=%s indices=%s before the next planning block",
            final_frame_index,
            tuple(self.planning_context_slots["mask_logits"].shape),
            tuple(self.planning_context_proprio.shape),
            self.planning_memory_mode,
            (
                self.planning_memory_indices.tolist()
                if self.planning_memory_indices is not None
                else self.observed_context_pool_indices.tolist()
                if self.observed_context_pool_indices is not None
                else None
            ),
        )

    def get_fixed_goal_z(self, traj_index=None, repeats=None):
        if self.fixed_goal_z is None:
            return None
        goal_z = self.fixed_goal_z
        if traj_index is not None:
            goal_z = {key: value[traj_index : traj_index + 1] for key, value in goal_z.items()}
        if repeats is not None:
            goal_z = {
                key: value.expand(repeats, *value.shape[1:]).contiguous()
                for key, value in goal_z.items()
            }
        return goal_z

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0
        self.state_0_torch = torch.as_tensor(self.state_0, device=self.device, dtype=torch.float32)

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def _project_states_to_proprio(self, states, proprio):
        """Project raw environment states to the observation proprio schema."""
        states = np.asarray(states)
        if states.shape[-1] == proprio.shape[-1]:
            return states

        indices = getattr(self.wm, "predictor_proprio_indices", None)
        if indices is None:
            raise ValueError(
                "Cannot project frozen state into proprio observation: "
                f"proprio={proprio.shape}, states={states.shape}, and the world "
                "model does not declare predictor_proprio_indices"
            )
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size != proprio.shape[-1]:
            raise ValueError(
                "World-model proprio projection does not match the observation: "
                f"indices={indices.tolist()}, proprio={proprio.shape}, "
                f"states={states.shape}"
            )
        if indices.size == 0 or indices.min() < 0 or indices.max() >= states.shape[-1]:
            raise ValueError(
                "World-model proprio projection is outside the raw state: "
                f"indices={indices.tolist()}, states={states.shape}"
            )
        return np.take(states, indices, axis=-1)

    def freeze_successful_feedback(self, executed_obses, executed_states, successes):
        """Zero terminal velocity before successful cases become MPC context."""
        if not hasattr(self.env, "zero_velocity_states"):
            return executed_obses, executed_states
        successes = np.asarray(successes, dtype=bool)
        frozen_states = self.env.zero_velocity_states(
            executed_states[:, -1], successes
        )
        executed_states[successes, -1] = frozen_states[successes]
        proprio = executed_obses.get("proprio")
        if proprio is not None and np.any(successes):
            frozen_proprio = self._project_states_to_proprio(
                frozen_states, proprio
            )
            proprio[successes, -1] = frozen_proprio[successes]
        return executed_obses, executed_states

    def _freeze_rollouts_after_action_length(
        self, executed_obses, executed_states, action_len
    ):
        """Keep completed trajectories stationary after their success frame."""
        if not hasattr(self.env, "zero_velocity_states"):
            return executed_obses, executed_states
        finite = np.isfinite(action_len)
        if not np.any(finite):
            return executed_obses, executed_states
        stop_indices = np.zeros(len(action_len), dtype=int)
        stop_indices[finite] = (
            np.asarray(action_len)[finite] * self.frameskip
        ).astype(int)
        terminal_states = np.stack(
            [
                executed_states[index, stop_indices[index]]
                for index in range(executed_states.shape[0])
            ]
        )
        frozen_states = self.env.zero_velocity_states(terminal_states, finite)
        frozen_proprio = None
        if "proprio" in executed_obses:
            frozen_proprio = self._project_states_to_proprio(
                frozen_states, executed_obses["proprio"]
            )
        for index in np.flatnonzero(finite):
            stop = stop_indices[index]
            executed_states[index, stop:] = frozen_states[index]
            for key, values in executed_obses.items():
                values[index, stop:] = values[index, stop]
            if frozen_proprio is not None:
                executed_obses["proprio"][index, stop:] = frozen_proprio[index]
        return executed_obses, executed_states

    def _freeze_rollouts_at_first_successful_model_step(
        self, actions, executed_actions, executed_obses, executed_states
    ):
        """Stop each rollout at its first successful model-step boundary.

        A model action contains ``frameskip`` primitive commands. Intermediate
        environment frames are neither observed nor used for success feedback;
        only states after a complete model action are checked. Once such a
        boundary succeeds, preserve it, clear its velocity, and replace every
        later command with the normalized representation of physical zero.
        """
        batch_size, num_states = executed_states.shape[:2]
        num_primitive_steps = num_states - 1
        if num_primitive_steps % self.frameskip != 0:
            raise ValueError(
                "Environment rollout length must contain complete model actions: "
                f"primitive_steps={num_primitive_steps}, frameskip={self.frameskip}"
            )
        num_model_steps = num_primitive_steps // self.frameskip
        first_success_model_steps = np.full(batch_size, np.inf)
        unresolved = np.ones(batch_size, dtype=bool)
        for model_step in range(1, num_model_steps + 1):
            env_step = model_step * self.frameskip
            step_result = self.env.eval_state(
                self.state_g, executed_states[:, env_step]
            )
            step_success = np.asarray(step_result["success"], dtype=bool)
            newly_successful = unresolved & step_success
            first_success_model_steps[newly_successful] = model_step
            unresolved[newly_successful] = False
            if not np.any(unresolved):
                break

        successful = np.isfinite(first_success_model_steps)
        if not np.any(successful):
            return (
                actions,
                executed_actions,
                executed_obses,
                executed_states,
                first_success_model_steps,
            )

        terminal_states = np.stack(
            [
                executed_states[
                    index,
                    int(first_success_model_steps[index]) * self.frameskip,
                ]
                if successful[index]
                else executed_states[index, -1]
                for index in range(batch_size)
            ]
        )
        if hasattr(self.env, "zero_velocity_states"):
            terminal_states = self.env.zero_velocity_states(
                terminal_states, successful
            )
        terminal_proprio = None
        if "proprio" in executed_obses:
            terminal_proprio = self._project_states_to_proprio(
                terminal_states, executed_obses["proprio"]
            )

        executed_actions = np.array(executed_actions, copy=True)
        for index in np.flatnonzero(successful):
            stop_env_step = (
                int(first_success_model_steps[index]) * self.frameskip
            )
            executed_actions[index, stop_env_step:] = 0
            executed_states[index, stop_env_step:] = terminal_states[index]
            for values in executed_obses.values():
                values[index, stop_env_step:] = values[index, stop_env_step]
            if terminal_proprio is not None:
                executed_obses["proprio"][index, stop_env_step:] = terminal_proprio[index]

        # Re-normalize the physical actions after clearing the tail. This is
        # correct even when a dataset's zero action does not normalize to 0.
        normalized = self.preprocessor.normalize_actions(
            torch.as_tensor(executed_actions, dtype=actions.dtype)
        )
        normalized = rearrange(
            normalized, "b (t f) d -> b t (f d)", f=self.frameskip
        ).to(actions.device)
        return (
            normalized,
            executed_actions,
            executed_obses,
            executed_states,
            first_success_model_steps,
        )

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        invalid = (last_index >= traj_data.shape[1]) | (last_index < -traj_data.shape[1])
        if np.any(invalid):
            raise IndexError(
                "Trajectory final-state index is out of bounds: "
                f"time_length={traj_data.shape[1]}, requested_lengths="
                f"{np.asarray(length)[invalid].tolist()}, indices={last_index[invalid].tolist()}"
            )
        if isinstance(traj_data, torch.Tensor):
            batch_index = torch.arange(traj_data.shape[0], device=traj_data.device)
            time_index = torch.as_tensor(last_index, device=traj_data.device)
            traj_data = traj_data[batch_index, time_index].unsqueeze(1)
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """
        Zero out everything after specified indices for each trajectory in the tensor.
        data: tensor
        """
        result = data.clone()  # Clone to preserve the original tensor
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False,
        save_slot_pca=True, save_context_visualization=False,
        predictive_retrieval=False,
    ):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        actions = self.project_actions_to_env_bounds(actions)
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
        # rollout in wm
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(self.obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(self.obs_g), self.device
        )
        self.state_0_torch = torch.as_tensor(self.state_0, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            rollout_kwargs = dict(
                obs_0=trans_obs_0,
                act=actions,
                state_0=self.state_0_torch,
            )
            context_slots = self.get_fixed_context_slots()
            if context_slots is not None:
                rollout_kwargs["context_slots"] = context_slots
                rollout_kwargs["context_proprio_embeddings"] = self.get_planning_context_proprio()
                rollout_kwargs["context_actions"] = self.get_planning_context_actions()
                rollout_kwargs["retrieval_context"] = self.get_planning_retrieval_context()
            if getattr(self.wm, "is_slot_world_model", False):
                rollout_kwargs["return_retrieval_trace"] = True
            if predictive_retrieval and getattr(self.wm, "is_slot_world_model", False):
                # Reproduce action selection semantics: after every model action,
                # the predicted mask/proprio becomes the next retrieval query.
                rollout_kwargs.pop("context_actions", None)
                rollout_kwargs.pop("retrieval_context", None)
                i_z_obses, rollout_trace = self.wm.receding_rollout(**rollout_kwargs)
            else:
                i_z_obses, rollout_trace = self.wm.rollout(**rollout_kwargs)
        # The original dino-wm rollout includes the initial latent (T+1), while
        # Slot-AC returns only the T post-action predictions. Do not apply the
        # dino-wm +1 convention to Slot-AC.
        imagined_length = (
            action_len
            if getattr(self.wm, "is_slot_world_model", False)
            else action_len + 1
        )
        i_final_z_obs = self._get_trajdict_last(i_z_obses, imagined_length)

        # rollout in env
        exec_actions = rearrange(
            actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
        )
        exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
        e_obses, e_states = self.env.rollout(self.seed, self.state_0, exec_actions)
        e_obses, e_states = self._freeze_rollouts_after_action_length(
            e_obses, e_states, action_len
        )
        (
            actions,
            exec_actions,
            e_obses,
            e_states,
            first_success_model_steps,
        ) = self._freeze_rollouts_at_first_successful_model_step(
            actions, exec_actions, e_obses, e_states
        )
        self.last_effective_actions = actions.detach()
        self.last_first_success_model_steps = first_success_model_steps
        e_visuals = e_obses["visual"]
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
            :, 0
        ]  # reduce dim back

        diagnostic_lengths = np.full(n_evals, actions.shape[1], dtype=np.int64)
        finite_action_len = np.isfinite(action_len)
        diagnostic_lengths[finite_action_len] = np.minimum(
            diagnostic_lengths[finite_action_len],
            np.asarray(action_len[finite_action_len], dtype=np.int64),
        )
        finite_first_success = np.isfinite(first_success_model_steps)
        diagnostic_lengths[finite_first_success] = np.minimum(
            diagnostic_lengths[finite_first_success],
            np.asarray(
                first_success_model_steps[finite_first_success], dtype=np.int64
            ),
        )

        # compute eval metrics
        logs, successes = self._compute_rollout_metrics(
            e_state=e_final_state,
            e_obs=e_final_obs,
            i_z_obs=i_final_z_obs,
        )
        if getattr(self.wm, "is_slot_world_model", False):
            logs.update(
                self._compute_proprio_rollout_diagnostics(
                    actions=actions,
                    executed_visuals=e_visuals,
                    executed_states=e_states,
                    autoregressive_proprio=i_z_obses["proprio"],
                    valid_model_steps=diagnostic_lengths,
                )
            )

        if (
            save_slot_pca
            and self.fixed_initial_slots is not None
            and bool(self.slot_pca_config.get("enabled", True))
        ):
            if save_context_visualization:
                self._save_retrieval_context_rgb(
                    filename,
                    retrieval_indices=rollout_trace.get("retrieval_indices"),
                )
            self._save_slot_rollout_pca(
                i_z_obses=i_z_obses,
                e_visuals=e_visuals,
                normalized_actions=actions,
                executed_actions=exec_actions,
                filename=filename,
            )

        # plot trajs (only when video output is requested; the imagined frames
        # are only consumed inside _plot_rollout_compare under `if save_video:`)
        if getattr(self.wm, "decoder", None) is not None and save_video:
            # Chunk decode along batch dim to bound peak GPU memory.
            _dec_chunk = 1
            _vis_parts = []
            _B = i_z_obses["visual"].shape[0]
            for _s in range(0, _B, _dec_chunk):
                _z_part = {_k: _v[_s : _s + _dec_chunk] for _k, _v in i_z_obses.items()}
                _vis_parts.append(self.wm.decode_obs(_z_part)[0]["visual"].cpu())
                torch.cuda.empty_cache()
            i_visuals = torch.cat(_vis_parts, dim=0)
            i_visuals = self._mask_traj(
                i_visuals, action_len + 1
            )  # we have action_len + 1 states
            e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
            e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )

        return logs, successes, e_obses, e_states

    @torch.no_grad()
    def _compute_proprio_rollout_diagnostics(
        self,
        actions,
        executed_visuals,
        executed_states,
        autoregressive_proprio,
        valid_model_steps,
    ):
        """Compare one-step teacher forcing with the actual open-loop rollout.

        Teacher forcing supplies the real mask and encoded real proprio at each
        model-step boundary while retaining the exact retrieval context and
        action used by inference.  Autoregressive predictions are the ones
        already produced by the normal WM rollout.  Reporting both against the
        same encoded target state isolates feedback drift from one-step error.
        """
        batch_size, num_steps = actions.shape[:2]
        if num_steps < 1:
            return {}
        source_indices = np.arange(num_steps, dtype=np.int64) * self.frameskip
        target_indices = source_indices + self.frameskip
        if target_indices[-1] >= executed_states.shape[1]:
            raise ValueError(
                "Executed trajectory is too short for proprio diagnostics: "
                f"states={executed_states.shape}, model_steps={num_steps}, "
                f"frameskip={self.frameskip}"
            )

        source_states = torch.as_tensor(
            executed_states[:, source_indices],
            device=self.device,
            dtype=torch.float32,
        )
        target_states = torch.as_tensor(
            executed_states[:, target_indices],
            device=self.device,
            dtype=torch.float32,
        )
        target_proprio = self.wm._encode_proprio(target_states)
        teacher_forced_parts = []
        retrieval_context = self.get_planning_retrieval_context()

        # Work one environment case at a time. This keeps full-resolution Wall
        # and PushT frames bounded while batching all model steps for that case.
        for batch_index in range(batch_size):
            source_visual = self.preprocessor.transform_obs_visual(
                executed_visuals[
                    batch_index : batch_index + 1, source_indices
                ]
            ).to(self.device)
            object_latents = self.fixed_initial_slots["obj_latents"][
                batch_index : batch_index + 1
            ]
            source_slots = self.wm.encode_with_object_latents(
                source_visual, object_latents
            )
            flat_context_slots = {
                "obj_latents": object_latents.expand(
                    num_steps, *object_latents.shape[1:]
                ),
                "mask_logits": source_slots["mask_logits"].reshape(
                    num_steps, 1, *source_slots["mask_logits"].shape[2:]
                ),
                "mask_probs": source_slots["mask_probs"].reshape(
                    num_steps, 1, *source_slots["mask_probs"].shape[2:]
                ),
            }
            flat_source_proprio = self.wm._encode_proprio(
                source_states[batch_index]
            )[:, None]
            flat_retrieval = None
            if retrieval_context is not None:
                flat_retrieval = {
                    key: value[batch_index : batch_index + 1].expand(
                        num_steps, *value.shape[1:]
                    ).contiguous()
                    for key, value in retrieval_context.items()
                }
            teacher_output, _ = self.wm.rollout(
                obs_0={
                    "visual": source_visual.reshape(
                        num_steps, 1, *source_visual.shape[2:]
                    )
                },
                act=actions[batch_index][:, None],
                state_0=source_states[batch_index],
                context_slots=flat_context_slots,
                context_proprio_embeddings=flat_source_proprio,
                retrieval_context=flat_retrieval,
            )
            teacher_forced_parts.append(teacher_output["proprio"][:, 0])

        teacher_forced_proprio = torch.stack(teacher_forced_parts, dim=0)
        if autoregressive_proprio.shape != target_proprio.shape:
            raise ValueError(
                "Autoregressive proprio and real targets do not align: "
                f"pred={tuple(autoregressive_proprio.shape)}, "
                f"target={tuple(target_proprio.shape)}"
            )
        teacher_error = (teacher_forced_proprio - target_proprio).norm(dim=-1)
        autoregressive_error = (
            autoregressive_proprio - target_proprio
        ).norm(dim=-1)
        target_physical = self.wm._select_predictor_proprio(target_states)
        state_mean = self.wm.state_mean.to(
            device=self.device, dtype=target_physical.dtype
        )
        state_std = self.wm.state_std.to(
            device=self.device, dtype=target_physical.dtype
        )

        def decoded_physical_errors(proprio_embeddings):
            decoded_normalized = self.wm.proprio_decoder(
                proprio_embeddings.float()
            )
            decoded_physical = decoded_normalized * state_std + state_mean
            delta = decoded_physical - target_physical
            return delta[..., :2].norm(dim=-1), delta[..., 2:4].norm(dim=-1)

        teacher_position_error, teacher_velocity_error = decoded_physical_errors(
            teacher_forced_proprio
        )
        autoregressive_position_error, autoregressive_velocity_error = (
            decoded_physical_errors(autoregressive_proprio)
        )
        target_roundtrip_position_error, target_roundtrip_velocity_error = (
            decoded_physical_errors(target_proprio)
        )
        valid_model_steps = torch.as_tensor(
            valid_model_steps, device=self.device, dtype=torch.long
        ).clamp(min=1, max=num_steps)
        valid = torch.arange(num_steps, device=self.device)[None] < (
            valid_model_steps[:, None]
        )
        batch_indices = torch.arange(batch_size, device=self.device)
        final_indices = valid_model_steps - 1

        logs = {
            "mean_teacher_forced_proprio_emb_l2": teacher_error[valid]
            .mean()
            .item(),
            "mean_autoregressive_proprio_emb_l2": autoregressive_error[valid]
            .mean()
            .item(),
            "final_teacher_forced_proprio_emb_l2": teacher_error[
                batch_indices, final_indices
            ]
            .mean()
            .item(),
            "final_autoregressive_proprio_emb_l2": autoregressive_error[
                batch_indices, final_indices
            ]
            .mean()
            .item(),
            "mean_teacher_forced_position_l2_px": teacher_position_error[valid]
            .mean()
            .item(),
            "mean_autoregressive_position_l2_px": autoregressive_position_error[
                valid
            ]
            .mean()
            .item(),
            "final_teacher_forced_position_l2_px": teacher_position_error[
                batch_indices, final_indices
            ]
            .mean()
            .item(),
            "final_autoregressive_position_l2_px": autoregressive_position_error[
                batch_indices, final_indices
            ]
            .mean()
            .item(),
            "mean_teacher_forced_velocity_l2_px_per_s": teacher_velocity_error[
                valid
            ]
            .mean()
            .item(),
            "mean_autoregressive_velocity_l2_px_per_s": (
                autoregressive_velocity_error[valid].mean().item()
            ),
            "final_teacher_forced_velocity_l2_px_per_s": teacher_velocity_error[
                batch_indices, final_indices
            ]
            .mean()
            .item(),
            "final_autoregressive_velocity_l2_px_per_s": (
                autoregressive_velocity_error[batch_indices, final_indices]
                .mean()
                .item()
            ),
            "mean_target_roundtrip_position_l2_px": (
                target_roundtrip_position_error[valid].mean().item()
            ),
            "final_target_roundtrip_position_l2_px": (
                target_roundtrip_position_error[batch_indices, final_indices]
                .mean()
                .item()
            ),
            "mean_target_roundtrip_velocity_l2_px_per_s": (
                target_roundtrip_velocity_error[valid].mean().item()
            ),
            "final_target_roundtrip_velocity_l2_px_per_s": (
                target_roundtrip_velocity_error[batch_indices, final_indices]
                .mean()
                .item()
            ),
        }
        logs["mean_proprio_emb_feedback_gap_l2"] = (
            logs["mean_autoregressive_proprio_emb_l2"]
            - logs["mean_teacher_forced_proprio_emb_l2"]
        )
        for step in range(num_steps):
            step_valid = valid[:, step]
            if not bool(step_valid.any()):
                continue
            logs[f"teacher_forced_proprio_emb_l2_step_{step + 1:02d}"] = (
                teacher_error[step_valid, step].mean().item()
            )
            logs[f"autoregressive_proprio_emb_l2_step_{step + 1:02d}"] = (
                autoregressive_error[step_valid, step].mean().item()
            )
            logs[f"teacher_forced_position_l2_px_step_{step + 1:02d}"] = (
                teacher_position_error[step_valid, step].mean().item()
            )
            logs[f"autoregressive_position_l2_px_step_{step + 1:02d}"] = (
                autoregressive_position_error[step_valid, step].mean().item()
            )
            logs[
                f"teacher_forced_velocity_l2_px_per_s_step_{step + 1:02d}"
            ] = teacher_velocity_error[step_valid, step].mean().item()
            logs[
                f"autoregressive_velocity_l2_px_per_s_step_{step + 1:02d}"
            ] = autoregressive_velocity_error[step_valid, step].mean().item()
            logs[f"target_roundtrip_position_l2_px_step_{step + 1:02d}"] = (
                target_roundtrip_position_error[step_valid, step].mean().item()
            )
            logs[
                f"target_roundtrip_velocity_l2_px_per_s_step_{step + 1:02d}"
            ] = target_roundtrip_velocity_error[step_valid, step].mean().item()
        return logs

    @staticmethod
    def _resolve_root_trajectory(dataset, index):
        """Resolve nested Subset indices without decoding any RGB frames."""
        index = int(index)
        while hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            index = int(dataset.indices[index])
            dataset = dataset.dataset
        return dataset, index

    @staticmethod
    def _retrieval_transition_key(source_state, action, target_state):
        values = torch.cat(
            [
                torch.as_tensor(source_state).detach().cpu().float().flatten(),
                torch.as_tensor(action).detach().cpu().float().flatten(),
                torch.as_tensor(target_state).detach().cpu().float().flatten(),
            ]
        ).contiguous()
        return values.numpy().tobytes()

    def _retrieval_origin_cache_path(self):
        bank_path = getattr(self.wm, "_retrieval_bank_path", None)
        if not bank_path:
            return None
        bank_path = Path(bank_path)
        return bank_path.with_name(f"{bank_path.stem}_rgb_origins.pt")

    def _valid_retrieval_origins(self, payload):
        bank_count = int(self.wm._retrieval_source_states.shape[0])
        bank_path = Path(self.wm._retrieval_bank_path)
        bank_stat = bank_path.stat()
        return (
            isinstance(payload, dict)
            and int(payload.get("bank_count", -1)) == bank_count
            and int(payload.get("frameskip", -1)) == int(self.frameskip)
            and int(payload.get("dataset_length", -1)) == len(self.retrieval_dataset)
            and int(payload.get("bank_size_bytes", -1)) == bank_stat.st_size
            and int(payload.get("bank_mtime_ns", -1)) == bank_stat.st_mtime_ns
            and tuple(payload.get("origins", torch.empty(0)).shape)
            == (bank_count, 2)
            and bool((payload["origins"] >= 0).all())
        )

    def _build_retrieval_origins(self):
        """Match each legacy retrieval-bank row to its real training frames."""
        if self.retrieval_dataset is None:
            raise RuntimeError(
                "Retrieval-context RGB visualization requires the training trajectory dataset"
            )
        bank_source = self.wm._retrieval_source_states.detach().cpu().float()
        bank_target = self.wm._retrieval_target_states.detach().cpu().float()
        bank_actions = self.wm._retrieval_actions.detach().cpu().float()
        bank_count = int(bank_source.shape[0])
        pending = {}
        for bank_index in range(bank_count):
            key = self._retrieval_transition_key(
                bank_source[bank_index], bank_actions[bank_index], bank_target[bank_index]
            )
            pending.setdefault(key, []).append(bank_index)

        origins = torch.full((bank_count, 2), -1, dtype=torch.long)
        remaining = bank_count
        for trajectory_index in range(len(self.retrieval_dataset)):
            root_dataset, root_index = self._resolve_root_trajectory(
                self.retrieval_dataset, trajectory_index
            )
            states = torch.as_tensor(root_dataset.states[root_index]).float()
            actions = torch.as_tensor(root_dataset.actions[root_index]).float()
            selected_states = self.wm._select_predictor_proprio(states).cpu()
            trajectory_length = int(self.retrieval_dataset.get_seq_length(trajectory_index))
            num_starts = min(
                trajectory_length - self.frameskip,
                states.shape[0] - self.frameskip,
                actions.shape[0] - self.frameskip + 1,
            )
            for source_frame in range(max(0, num_starts)):
                action_chunk = actions[
                    source_frame : source_frame + self.frameskip
                ].reshape(-1)
                if action_chunk.numel() != bank_actions.shape[-1]:
                    continue
                key = self._retrieval_transition_key(
                    selected_states[source_frame],
                    action_chunk,
                    selected_states[source_frame + self.frameskip],
                )
                matches = pending.pop(key, None)
                if matches is None:
                    continue
                for bank_index in matches:
                    origins[bank_index] = torch.tensor(
                        [root_index, source_frame], dtype=torch.long
                    )
                remaining -= len(matches)
                if remaining == 0:
                    break
            if remaining == 0:
                break
        if remaining:
            missing = torch.nonzero(origins[:, 0] < 0).flatten()[:20].tolist()
            raise RuntimeError(
                f"Could not map {remaining}/{bank_count} retrieval transitions back "
                f"to training RGB frames; first missing bank indices: {missing}"
            )
        return origins

    def _load_retrieval_origins(self):
        if self._retrieval_origins is not None:
            return self._retrieval_origins
        cache_path = self._retrieval_origin_cache_path()
        if cache_path is None:
            self._retrieval_origins = self._build_retrieval_origins()
            return self._retrieval_origins

        lock_path = cache_path.with_suffix(f"{cache_path.suffix}.lock")
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("a+") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                payload = (
                    torch.load(cache_path, map_location="cpu")
                    if cache_path.exists()
                    else None
                )
                if not self._valid_retrieval_origins(payload):
                    origins = self._build_retrieval_origins()
                    payload = {
                        "bank_count": int(origins.shape[0]),
                        "frameskip": int(self.frameskip),
                        "dataset_length": len(self.retrieval_dataset),
                        "bank_size_bytes": Path(self.wm._retrieval_bank_path).stat().st_size,
                        "bank_mtime_ns": Path(self.wm._retrieval_bank_path).stat().st_mtime_ns,
                        "origins": origins,
                    }
                    temporary = cache_path.with_name(
                        f".{cache_path.name}.{os.getpid()}.tmp"
                    )
                    torch.save(payload, temporary)
                    os.replace(temporary, cache_path)
                self._retrieval_origins = payload["origins"].long()
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError as error:
            log.warning(
                "Could not persist retrieval RGB origins at %s (%s); using memory only",
                cache_path,
                error,
            )
            self._retrieval_origins = self._build_retrieval_origins()
        return self._retrieval_origins

    def _save_retrieval_context_rgb(self, filename, retrieval_indices=None):
        """Draw the real source/target RGB pair for every retrieved context."""
        context = self.planning_retrieval_context
        if context is None:
            return
        if retrieval_indices is None:
            indices = context["indices"][:, None]
        else:
            indices = retrieval_indices
        indices = indices.detach().cpu().long()
        if (indices < 0).any():
            self._save_observed_context_rgb(filename)
            return
        if self.retrieval_dataset is None:
            return
        max_samples = min(
            int(self.slot_pca_config.get("num_samples", self.n_plot_samples)),
            indices.shape[0],
        )
        if max_samples <= 0:
            return
        origins = self._load_retrieval_origins()
        for sample_index in range(max_samples):
            sample_indices = indices[sample_index]
            num_steps, num_context = sample_indices.shape
            fig, axes = plt.subplots(
                num_steps,
                2 * num_context,
                figsize=(max(6.4, 3.1 * 2 * num_context), max(3.0, 3.0 * num_steps)),
                squeeze=False,
            )
            root_dataset, _ = self._resolve_root_trajectory(
                self.retrieval_dataset, 0
            )
            for step_index in range(num_steps):
                for context_index, bank_index_tensor in enumerate(
                    sample_indices[step_index]
                ):
                    bank_index = int(bank_index_tensor)
                    trajectory_index, source_frame = origins[bank_index].tolist()
                    obs, _, _, _ = root_dataset.get_frames(
                        trajectory_index,
                        [source_frame, source_frame + self.frameskip],
                    )
                    visual = obs["visual"]
                    for pair_column, title in enumerate(("source", "target")):
                        column = 2 * context_index + pair_column
                        image = self._display_image(
                            visual[pair_column], tuple(visual[pair_column].shape[-2:])
                        )
                        axes[step_index, column].imshow(
                            image.permute(1, 2, 0).numpy(), interpolation="nearest"
                        )
                        axes[step_index, column].axis("off")
                        axes[step_index, column].set_title(
                            f"step {step_index} · ctx {context_index} {title}\n"
                            f"bank={bank_index}, traj={trajectory_index}, "
                            f"f={source_frame + pair_column * self.frameskip}",
                            fontsize=8,
                        )
            fig.suptitle(
                "Real RGB transitions retrieved for this MPC step",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            output_stem = Path(f"{filename}_context_rgb")
            if max_samples > 1:
                output_stem = output_stem.with_name(
                    f"{output_stem.name}_case{sample_index:03d}"
                )
            output_stem.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
            plt.close(fig)

    def _save_observed_context_rgb(self, filename):
        """Draw online context RGB and every corresponding source/target mask."""
        if self.observed_context_visual is None:
            return
        sources = self.observed_context_visual["source"]
        targets = self.observed_context_visual["target"]
        source_masks = self.observed_transition_context[
            "source_mask_probs"
        ].detach().cpu().float()
        target_masks = self.observed_transition_context[
            "target_mask_probs"
        ].detach().cpu().float()
        num_objects = int(source_masks.shape[2])
        patch_count = int(source_masks.shape[-1])
        spatial_size = int(math.sqrt(patch_count))
        if spatial_size * spatial_size != patch_count:
            raise ValueError(
                f"Online context mask patch count must be square, got {patch_count}"
            )
        max_samples = min(
            int(self.slot_pca_config.get("num_samples", self.n_plot_samples)),
            sources.shape[0],
        )
        for sample_index in range(max_samples):
            num_context = sources.shape[1]
            fig, axes = plt.subplots(
                1 + num_objects,
                2 * num_context,
                figsize=(max(8.0, 2.5 * 2 * num_context), 2.2 * (1 + num_objects)),
                squeeze=False,
            )
            for context_index in range(num_context):
                action_summary = ""
                pool_summary = ""
                if self.observed_context_pool_indices is not None:
                    pool_summary = (
                        f" pool={int(self.observed_context_pool_indices[sample_index, context_index])}"
                    )
                if self.observed_context_modes is not None:
                    mode_index = int(
                        self.observed_context_modes[sample_index, context_index]
                    )
                    if 0 <= mode_index < len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES):
                        pool_summary += (
                            f" mode={PUSHOBJ_STRUCTURED_CONTEXT_NAMES[mode_index]}"
                        )
                if self.observed_context_collision_mask is not None:
                    has_interaction = bool(
                        self.observed_context_collision_mask[
                            sample_index, context_index
                        ]
                    )
                    interaction_name = (
                        "contact"
                        if getattr(self.wm, "env_name", None) == "pusht"
                        else "collision"
                    )
                    pool_summary += (
                        f" {interaction_name}={int(has_interaction)}"
                    )
                if self.observed_context_actions is not None:
                    action = self.observed_context_actions[
                        sample_index, context_index
                    ].reshape(self.frameskip, -1)
                    action_summary = "\nmean(a)=" + np.array2string(
                        action.mean(dim=0).numpy(), precision=2, separator=","
                    )
                for pair_column, title in enumerate(("source", "target")):
                    column = 2 * context_index + pair_column
                    visual = (
                        sources[sample_index, context_index]
                        if pair_column == 0
                        else targets[sample_index, context_index]
                    )
                    visual = np.asarray(visual)
                    if np.issubdtype(visual.dtype, np.floating):
                        visual = visual.astype(np.float32, copy=False)
                        if visual.size and float(np.nanmax(visual)) <= 1.0:
                            visual = visual * 255.0
                        visual = np.clip(visual, 0.0, 255.0).astype(np.uint8)
                    axes[0, column].imshow(visual, interpolation="nearest")
                    axes[0, column].axis("off")
                    axes[0, column].set_title(
                        f"ctx {context_index} {title}{pool_summary}"
                        + (action_summary if pair_column == 0 else ""),
                        fontsize=8,
                    )
                    masks = source_masks if pair_column == 0 else target_masks
                    for object_index in range(num_objects):
                        axis = axes[1 + object_index, column]
                        mask = masks[
                            sample_index, context_index, object_index
                        ].reshape(spatial_size, spatial_size)
                        axis.imshow(
                            mask.numpy(),
                            cmap="viridis",
                            vmin=0.0,
                            vmax=1.0,
                            interpolation="nearest",
                        )
                        axis.axis("off")
                        if column == 0:
                            axis.set_ylabel(f"slot {object_index}", fontsize=8)
            axes[0, 0].set_ylabel("RGB", fontsize=8)
            fig.suptitle(
                "Online ICL context: RGB and source/target slot masks",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            output_stem = Path(f"{filename}_online_context_rgb_masks")
            if max_samples > 1:
                output_stem = output_stem.with_name(
                    f"{output_stem.name}_case{sample_index:03d}"
                )
            output_stem.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight"
            )
            plt.close(fig)

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs):
        """
        Args
            e_state
            e_obs
            i_z_obs
        Return
            logs
            successes
        """
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results['success']

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
            for key, value in eval_results.items()
        }

        print("Success rate: ", logs['success_rate'])
        print(eval_results)

        visual_dists = np.linalg.norm(e_obs["visual"] - self.obs_g["visual"], axis=1)
        mean_visual_dist = np.mean(visual_dists)
        proprio_dists = np.linalg.norm(e_obs["proprio"] - self.obs_g["proprio"], axis=1)
        mean_proprio_dist = np.mean(proprio_dists)

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        if self.fixed_initial_slots is not None:
            e_slots = self.wm.encode_with_object_latents(
                e_obs["visual"],
                self.fixed_initial_slots["obj_latents"],
            )
            e_z_obs = {
                "visual": e_slots["mask_probs"],
                "mask_probs": e_slots["mask_probs"],
                "proprio": self.wm._encode_proprio(
                    torch.as_tensor(
                        e_state[:, None], device=self.device, dtype=torch.float32
                    )
                ),
            }
        else:
            e_z_obs = self.wm.encode_obs(e_obs)
        div_visual_emb = (
            (e_z_obs["visual"] - i_z_obs["visual"]).flatten(1).norm(dim=1).mean().item()
        )
        div_proprio_emb = (
            (e_z_obs["proprio"] - i_z_obs["proprio"]).flatten(1).norm(dim=1).mean().item()
        )

        logs.update({
            "mean_visual_dist": mean_visual_dist,
            "mean_proprio_dist": mean_proprio_dist,
            "mean_div_visual_emb": div_visual_emb,
            "mean_div_proprio_emb": div_proprio_emb,
        })

        if self.fixed_goal_z is not None:
            raw_uot_kwargs = {
                "sinkhorn_iters": int(self.slot_uot_config.get("sinkhorn_iters", 5)),
                "entropy_reg": float(self.slot_uot_config.get("entropy_reg", 0.05)),
                "unbalanced_ot_rho": float(self.slot_uot_config.get("unbalanced_ot_rho", 0.2)),
                "batch_size": int(self.slot_uot_config.get("uot_batch_size", 128)),
                "transport_cache": self._uot_transport_cache,
            }
            goal_masks = self.fixed_goal_z["mask_probs"]
            context_masks = self.planning_context_slots["mask_probs"]
            weighted_uot_kwargs = dict(raw_uot_kwargs)
            area_normalize = bool(
                self.slot_uot_config.get("mask_transport_area_normalize", False)
            )
            weighted_uot_kwargs.update(
                area_normalize=area_normalize,
                area_normalize_eps=float(
                    self.slot_uot_config.get(
                        "mask_transport_area_normalize_eps", 1.0e-3
                    )
                ),
            )
            if area_normalize:
                weighted_uot_kwargs["area_normalizer"] = _slot_area_normalizer(
                    context_masks,
                    goal_masks,
                    eps=weighted_uot_kwargs["area_normalize_eps"],
                )
            if bool(self.slot_uot_config.get("slot_change_weighting", False)):
                weighted_uot_kwargs["slot_weights"] = _slot_change_weights(
                    context_masks[:, -1],
                    goal_masks[:, -1],
                    min_weight=float(
                        self.slot_uot_config.get("slot_change_min_weight", 0.05)
                    ),
                    max_weight=float(
                        self.slot_uot_config.get("slot_change_max_weight", 5.0)
                    ),
                )

            pairs = {
                "actual_goal": (e_z_obs["mask_probs"], goal_masks),
                "imagined_goal": (i_z_obs["mask_probs"], goal_masks),
                "imagined_actual": (i_z_obs["mask_probs"], e_z_obs["mask_probs"]),
            }
            raw_losses = {
                name: _uot_mask_loss(pred, target, **raw_uot_kwargs)
                for name, (pred, target) in pairs.items()
            }
            weighted_losses = {
                name: _uot_mask_loss(pred, target, **weighted_uot_kwargs)
                for name, (pred, target) in pairs.items()
            }
            logs.update(
                {
                    **{
                        f"mean_{name}_uot_raw": value.mean().item()
                        for name, value in raw_losses.items()
                    },
                    **{
                        # Keep the original metric names aligned with the loss
                        # now used by CEM; *_raw preserves historical values.
                        f"mean_{name}_uot": value.mean().item()
                        for name, value in weighted_losses.items()
                    },
                }
            )

        return logs, successes

    def _save_slot_rollout_pca(
        self, i_z_obses, e_visuals, normalized_actions, executed_actions, filename
    ):
        max_samples = min(
            int(self.slot_pca_config.get("num_samples", self.n_plot_samples)),
            i_z_obses["mask_probs"].shape[0],
        )
        if max_samples <= 0:
            return
        actual_visual = self.preprocessor.transform_obs_visual(
            e_visuals[:max_samples, :: self.frameskip]
        ).to(self.device)
        predicted_masks = torch.cat(
            [
                self.planning_context_slots["mask_probs"][:max_samples, -1:],
                i_z_obses["mask_probs"][:max_samples],
            ],
            dim=1,
        )
        for sample_index in range(max_samples):
            obj_latents = self.fixed_initial_slots["obj_latents"][sample_index : sample_index + 1]
            with torch.no_grad():
                actual_slots = self.wm.encode_with_object_latents(
                    actual_visual[sample_index : sample_index + 1],
                    obj_latents,
                )
                pred_recon = self.wm.reconstruct_from_slots(
                    predicted_masks[sample_index : sample_index + 1], obj_latents
                )[0]
                actual_recon = self.wm.reconstruct_from_slots(
                    actual_slots["mask_probs"], obj_latents
                )[0]
                goal_recon = self.wm.reconstruct_from_slots(
                    self.fixed_goal_slots["mask_probs"][sample_index : sample_index + 1],
                    obj_latents,
                )[0]
                initial_recon = self.wm.reconstruct_from_slots(
                    self.planning_context_slots["mask_probs"][sample_index : sample_index + 1, -1:],
                    obj_latents,
                )[0]

            # Match training exactly: fit the shared reconstruction PCA on the
            # teacher-forced context frames t0..t(N-1), then use that same
            # projection/range for predicted and executed reconstructions.
            # This is visualization-only and therefore may use the completed
            # executed rollout without affecting planning.
            pca_context = [feature.detach().cpu() for feature in actual_recon[:-1]]
            if not pca_context:
                pca_context = [actual_recon[0].detach().cpu()]
            recon_pca = self._fit_patch_pca(pca_context)
            pred_pca = torch.stack([
                self._patch_feature_to_pca_image(x.detach().cpu(), recon_pca)
                for x in pred_recon
            ])
            actual_pca = torch.stack([
                self._patch_feature_to_pca_image(x.detach().cpu(), recon_pca)
                for x in actual_recon
            ])
            initial_pca = self._patch_feature_to_pca_image(
                initial_recon[0].detach().cpu(), recon_pca
            )
            goal_pca = self._patch_feature_to_pca_image(
                goal_recon[0].detach().cpu(), recon_pca
            )

            raw_initial = self.preprocessor.transform_obs_visual(
                self.obs_0["visual"][sample_index : sample_index + 1]
            )[0, 0]
            raw_goal = self.preprocessor.transform_obs_visual(
                self.fixed_slot_obs_g["visual"][sample_index : sample_index + 1]
            )[0, 0]
            raw_executed = actual_visual[sample_index]
            output_stem = Path(f"{filename}_slot_pca")
            if max_samples > 1:
                output_stem = output_stem.with_name(f"{output_stem.name}_case{sample_index:03d}")
            output_stem.parent.mkdir(parents=True, exist_ok=True)
            self._save_patch_pca_grid(
                initial_image=raw_initial,
                goal_image=raw_goal,
                initial_pca=initial_pca,
                goal_pca=goal_pca,
                predicted_pca=pred_pca,
                executed_images=raw_executed,
                executed_pca=actual_pca,
                normalized_actions=normalized_actions[sample_index],
                executed_actions=executed_actions[sample_index],
                output_path=output_stem.with_suffix(".png"),
            )
            np.savez_compressed(
                output_stem.with_suffix(".npz"),
                predicted_patch_pca=pred_pca.cpu().numpy(),
                executed_patch_pca=actual_pca.cpu().numpy(),
                initial_patch_pca=initial_pca.cpu().numpy(),
                goal_patch_pca=goal_pca.cpu().numpy(),
                normalized_actions=normalized_actions[sample_index].detach().cpu().numpy(),
                executed_actions=executed_actions[sample_index],
            )
            mask_stem = output_stem.with_name(
                output_stem.name.replace("_slot_pca", "_slot_masks")
            )
            self._save_slot_mask_grid(
                start_masks=self.planning_context_slots["mask_probs"][sample_index, -1],
                goal_masks=self.fixed_goal_slots["mask_probs"][sample_index, -1],
                context_masks=actual_slots["mask_probs"][0, :-1],
                pred_masks=predicted_masks[sample_index, 1:],
                executed_masks=actual_slots["mask_probs"][0, 1:],
                normalized_actions=normalized_actions[sample_index],
                executed_actions=executed_actions[sample_index],
                output_path=mask_stem.with_suffix(".png"),
            )

    @staticmethod
    def _save_slot_mask_grid(
        start_masks,
        goal_masks,
        context_masks,
        pred_masks,
        executed_masks,
        normalized_actions,
        executed_actions,
        output_path,
    ):
        """Save planning masks with the same grouping used during training."""
        num_steps = min(
            context_masks.shape[0],
            pred_masks.shape[0],
            executed_masks.shape[0],
        )
        if num_steps < 1:
            return
        context_masks = context_masks[:num_steps]
        pred_masks = pred_masks[:num_steps]
        executed_masks = executed_masks[:num_steps]
        num_objects = int(context_masks.shape[1])
        patch_count = int(context_masks.shape[-1])
        spatial_size = int(math.sqrt(patch_count))
        if spatial_size * spatial_size != patch_count:
            raise ValueError(f"Mask patch count must be square, got {patch_count}")

        num_cols = num_steps + 2
        fig, axes = plt.subplots(
            3 * num_objects,
            num_cols,
            figsize=(2.0 * num_cols, 1.6 * 3 * num_objects),
            squeeze=False,
        )
        start_masks = start_masks.detach().cpu().float().reshape(
            num_objects, spatial_size, spatial_size
        )
        goal_masks = goal_masks.detach().cpu().float().reshape(
            num_objects, spatial_size, spatial_size
        )
        groups = (
            ("context mask", context_masks),
            ("pred mask", pred_masks),
            ("executed mask", executed_masks),
        )
        for group_index, (group_title, masks) in enumerate(groups):
            masks = masks.detach().cpu().float().reshape(
                num_steps, num_objects, spatial_size, spatial_size
            )
            for object_index in range(num_objects):
                row_index = group_index * num_objects + object_index
                for column_index, (title, reference_masks) in enumerate(
                    (("start", start_masks), ("goal", goal_masks))
                ):
                    axis = axes[row_index, column_index]
                    axis.imshow(
                        reference_masks[object_index].numpy(),
                        cmap="viridis",
                        vmin=0.0,
                        vmax=1.0,
                        interpolation="nearest",
                    )
                    axis.axis("off")
                    if row_index == 0:
                        axis.set_title(title)
                for step_index in range(num_steps):
                    axis = axes[row_index, step_index + 2]
                    axis.imshow(
                        masks[step_index, object_index].numpy(),
                        cmap="viridis",
                        vmin=0.0,
                        vmax=1.0,
                        interpolation="nearest",
                    )
                    axis.axis("off")
                    if row_index == 0:
                        axis.set_title(f"t{step_index} → t{step_index + 1}")
                axes[row_index, 0].set_ylabel(
                    f"{group_title} {object_index}", fontsize=8
                )
        action_text = PlanEvaluator._format_planned_actions(
            normalized_actions, executed_actions, num_steps
        )
        fig.suptitle(action_text, fontsize=7, family="monospace", y=0.998)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _flatten_patch_features(feature):
        if feature.ndim == 2:
            return feature
        if feature.ndim == 3:
            return feature.reshape(-1, feature.shape[-1])
        raise ValueError(f"Unsupported patch feature shape: {tuple(feature.shape)}")

    @classmethod
    def _fit_patch_pca(cls, context_features):
        patches = torch.cat([cls._flatten_patch_features(x) for x in context_features], dim=0).float()
        mean = patches.mean(dim=0, keepdim=True)
        _, _, basis = torch.pca_lowrank(patches - mean, q=3)
        projected = (patches - mean) @ basis
        return mean, basis, projected.amin(0, keepdim=True), projected.amax(0, keepdim=True)

    def _patch_feature_to_pca_image(self, feature, pca):
        mean, basis, pca_min, pca_max = pca
        projected = (self._flatten_patch_features(feature).float() - mean) @ basis
        projected = ((projected - pca_min) / (pca_max - pca_min + 1.0e-8)).clamp(0, 1)
        side = int(math.sqrt(projected.shape[0]))
        if side * side != projected.shape[0]:
            raise ValueError(f"Patch count must be square, got {projected.shape[0]}")
        grid = projected.view(side, side, 3).permute(2, 0, 1).unsqueeze(0)
        return F.interpolate(grid, scale_factor=int(self.wm.patch_size), mode="nearest")[0].cpu()

    @staticmethod
    def _display_image(image, target_hw):
        image = image.detach().cpu().float()
        if image.max() > 1:
            image = image / 255.0
        if image.min() < 0:
            image = (image + 1.0) / 2.0
        if tuple(image.shape[-2:]) != tuple(target_hw):
            image = F.interpolate(image[None], size=target_hw, mode="nearest")[0]
        return image.clamp(0, 1)

    def _save_patch_pca_grid(
        self, initial_image, goal_image, initial_pca, goal_pca,
        predicted_pca, executed_images, executed_pca,
        normalized_actions, executed_actions, output_path,
    ):
        target_hw = tuple(predicted_pca.shape[-2:])
        context_images = [self._display_image(initial_image, target_hw), self._display_image(goal_image, target_hw)]
        executed_images = [self._display_image(x, target_hw) for x in executed_images]
        rows = [
            context_images,
            [initial_pca, goal_pca],
            list(predicted_pca),
            executed_images,
            list(executed_pca),
        ]
        labels = [
            "Input RGB\ninitial / goal",
            "Slot reconstruction PCA\ninitial / goal",
            "Predicted slot\nreconstruction PCA\nt0 = context",
            "Executed environment\nRGB\nt0 = initial",
            "Executed slot\nreconstruction PCA\nt0 = initial",
        ]
        num_cols = max(len(row) for row in rows)
        label_width = 2.6
        content_width = 2.2 * num_cols
        fig_width = label_width + content_width
        fig, axes = plt.subplots(
            len(rows),
            num_cols,
            figsize=(fig_width, 2.2 * len(rows)),
            squeeze=False,
        )
        # Reserve a real label column instead of putting long text inside the
        # first image axis. tight_layout otherwise expands inter-column gaps
        # until every image becomes tiny.
        fig.subplots_adjust(
            left=label_width / fig_width,
            right=0.995,
            top=0.90,
            bottom=0.035,
            wspace=0.12,
            hspace=0.28,
        )
        for row_idx, (row, label) in enumerate(zip(rows, labels)):
            for col_idx, axis in enumerate(axes[row_idx]):
                axis.axis("off")
                if col_idx < len(row):
                    axis.imshow(row[col_idx].permute(1, 2, 0).numpy(), interpolation="nearest")
                    if row_idx >= 2:
                        axis.set_title(f"t{col_idx}")
            first_axis_position = axes[row_idx, 0].get_position()
            fig.text(
                0.02,
                (first_axis_position.y0 + first_axis_position.y1) / 2.0,
                label,
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
            )
        axes[0, 0].set_title("initial")
        if num_cols > 1:
            axes[0, 1].set_title("goal")
        fig.suptitle(
            self._format_planned_actions(
                normalized_actions, executed_actions, len(predicted_pca)
            ),
            fontsize=7,
            family="monospace",
            y=0.995,
        )
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    @staticmethod
    def _format_planned_actions(normalized_actions, executed_actions, num_steps):
        """Compactly annotate each CEM model-step action on planning figures."""
        normalized = torch.as_tensor(normalized_actions).detach().cpu().float()
        executed = np.asarray(executed_actions)
        num_steps = min(int(num_steps), normalized.shape[0])
        if num_steps < 1:
            return "CEM actions: none"
        primitive_per_step = executed.shape[0] // normalized.shape[0]
        lines = ["CEM actions (norm chunk | raw environment actions)"]
        for step in range(num_steps):
            norm_text = np.array2string(
                normalized[step].numpy(), precision=2, suppress_small=True,
                separator=",", max_line_width=1000,
            )
            raw = executed[
                step * primitive_per_step : (step + 1) * primitive_per_step
            ]
            raw_text = np.array2string(
                raw, precision=2, suppress_small=True, separator=",",
                max_line_width=1000,
            ).replace("\n", "")
            lines.append(f"a{step}: {norm_text} | {raw_text}")
        return "\n".join(lines)

    def _plot_rollout_compare(
        self, e_visuals, i_visuals, successes, save_video=False, filename=""
    ):
        """
        i_visuals may have less frames than e_visuals due to frameskip, so pad accordingly
        e_visuals: (b, t, h, w, c)
        i_visuals: (b, t, h, w, c)
        goal: (b, h, w, c)
        """
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        i_visuals = i_visuals.unsqueeze(2)
        i_visuals = torch.cat(
            [i_visuals] + [i_visuals] * (self.frameskip - 1),
            dim=2,
        )  # pad i_visuals (due to frameskip)
        i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
        i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

        correction = 0.3  # to distinguish env visuals and imagined visuals

        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    i_obs = i_visuals[idx, i, ...]
                    e_obs = torch.cat(
                        [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    i_obs = torch.cat(
                        [i_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    frame = torch.cat([e_obs - correction, i_obs], dim=1)
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                    frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                    frame = frame.detach().cpu().numpy()
                    frames.append(frame)
                video_writer = imageio.get_writer(
                    f"{filename}_{idx}_{success_tag}.mp4", fps=12
                )

                for frame in frames:
                    frame = frame * 2 - 1 if frame.min() >= 0 else frame
                    video_writer.append_data(
                        (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                    )
                video_writer.close()

        # pad i_visuals or subsample e_visuals
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]

        n_columns = e_visuals.shape[1]
        assert (
            i_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"

        # add a goal column
        e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
        i_visuals = torch.cat([i_visuals.cpu(), goal_visual - correction], dim=1)
        rollout = torch.cat([e_visuals.cpu() - correction, i_visuals.cpu()], dim=1)
        n_columns += 1

        imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
        imgs_for_plotting = (
            imgs_for_plotting * 2 - 1
            if imgs_for_plotting.min() >= 0
            else imgs_for_plotting
        )
        utils.save_image(
            imgs_for_plotting,
            f"{filename}.png",
            nrow=n_columns,  # nrow is the number of columns
            normalize=True,
            value_range=(-1, 1),
        )
