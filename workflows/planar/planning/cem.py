import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from .base_planner import BasePlanner
from utils import move_to_device


def _tensor_to_numpy(value):
    """Detach a tensor and convert NumPy-unsupported dtypes safely."""
    value = value.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy()


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        var_scale=1.0,
        logging_prefix="plan_0",
        log_filename="logs.json",
        sample_chunk_size=None,
        trajectory_batch_size=1,
        num_action_knots=None,
        action_temporal_correlation=0.0,
        trajectory_prior=None,
        velocity_delta=None,
        proposal_dump_path=None,
        dump_loss_components=False,
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = max(1, int(eval_every))
        self.logging_prefix = logging_prefix
        self.sample_chunk_size = sample_chunk_size or num_samples
        self.trajectory_batch_size = max(1, int(trajectory_batch_size))
        self.proposal_dump_path = (
            None if proposal_dump_path is None else Path(proposal_dump_path).resolve()
        )
        self.dump_loss_components = bool(dump_loss_components)
        if self.dump_loss_components and self.proposal_dump_path is None:
            raise ValueError(
                "dump_loss_components=true requires proposal_dump_path to be set"
            )
        if self.dump_loss_components and not callable(
            getattr(self.objective_fn, "loss_components", None)
        ):
            raise ValueError(
                "dump_loss_components=true requires an objective with a "
                "loss_components callable"
            )
        self.frameskip = int(self.evaluator.frameskip)
        self._obstacle_probe_metadata = None
        if self.action_dim % self.frameskip != 0:
            raise ValueError(
                f"Model action dim {self.action_dim} is not divisible by "
                f"frameskip {self.frameskip}"
            )
        self.primitive_action_dim = self.action_dim // self.frameskip
        self.num_action_knots = (
            None if num_action_knots is None else int(num_action_knots)
        )
        self.action_temporal_correlation = float(action_temporal_correlation)
        if not -0.99 < self.action_temporal_correlation < 0.99:
            raise ValueError(
                "action_temporal_correlation must lie in (-0.99, 0.99)"
            )
        self.velocity_delta = dict(velocity_delta or {})
        self.use_velocity_delta = bool(
            self.velocity_delta.pop("enabled", False)
        )
        self.velocity_delta_max = float(
            self.velocity_delta.pop("max_delta", 3.0)
        )
        self.velocity_delta_initial_std = float(
            self.velocity_delta.pop("initial_std", 2.0)
        )
        self.velocity_delta_min_std = float(
            self.velocity_delta.pop("min_std", 0.25)
        )
        if self.velocity_delta:
            raise ValueError(
                f"Unknown velocity_delta options: {sorted(self.velocity_delta)}"
            )
        self.trajectory_prior = dict(trajectory_prior or {})
        self.use_trajectory_prior = bool(
            self.trajectory_prior.pop("enabled", False)
        )
        self.trajectory_action_scale = float(
            self.trajectory_prior.pop("action_scale", 100.0)
        )
        self.sequential_model_step_search = bool(
            self.trajectory_prior.pop("sequential_model_step_search", False)
        )
        self.sequential_cem_rounds = int(
            self.trajectory_prior.pop("cem_rounds_per_model_step", 1)
        )
        self.scale_sequential_samples = bool(
            self.trajectory_prior.pop(
                "scale_samples_by_inverse_prefix_length", False
            )
        )
        self.particle_cem = dict(
            self.trajectory_prior.pop("particle_cem", {}) or {}
        )
        self.use_particle_cem = bool(self.particle_cem.pop("enabled", False))
        self.particle_min_speed = float(self.particle_cem.pop("min_speed", 3.0))
        self.particle_max_speed = float(self.particle_cem.pop("max_speed", 35.0))
        self.max_heading_change_degrees = float(
            self.particle_cem.pop("max_heading_change_degrees", 30.0)
        )
        self.particle_boundary_min_speed = float(
            self.particle_cem.pop("boundary_min_speed", 20.0)
        )
        self.particle_boundary_max_speed = float(
            self.particle_cem.pop("boundary_max_speed", 80.0)
        )
        self.particle_physical_action_clip_px = float(
            self.particle_cem.pop(
                "physical_action_clip_px", self.particle_boundary_max_speed
            )
        )
        self.particle_boundary_full_direction_probability = float(
            self.particle_cem.pop(
                "boundary_full_direction_probability", 0.2
            )
        )
        self.particle_position_lower_bound = self.particle_cem.pop(
            "position_lower_bound", None
        )
        self.particle_position_upper_bound = self.particle_cem.pop(
            "position_upper_bound", None
        )
        self.particle_max_proposals = int(
            self.particle_cem.pop("max_proposals", 100000)
        )
        self.particle_early_stop_probability = float(
            self.particle_cem.pop("early_stop_probability", 0.0)
        )
        self.particle_post_stop_jitter_max_speed = float(
            self.particle_cem.pop("post_stop_jitter_max_speed", 3.0)
        )
        self.particle_random_init_steps = int(
            self.particle_cem.pop("random_init_steps", 1)
        )
        random_fraction_schedule = self.particle_cem.pop(
            "random_fraction_schedule", None
        )
        self.particle_random_fraction_schedule = (
            None
            if random_fraction_schedule is None
            else tuple(float(value) for value in random_fraction_schedule)
        )
        self.particle_temporal_correlation = float(
            self.particle_cem.pop("temporal_correlation", 0.8)
        )
        self.particle_initial_speed_std = float(
            self.particle_cem.pop("initial_speed_std", 3.0)
        )
        self.particle_final_speed_std = float(
            self.particle_cem.pop("final_speed_std", 0.6)
        )
        self.particle_initial_heading_std = float(
            self.particle_cem.pop("initial_heading_std_degrees", 30.0)
        )
        self.particle_final_heading_std = float(
            self.particle_cem.pop("final_heading_std_degrees", 5.0)
        )
        self.particle_initial_turn_std = float(
            self.particle_cem.pop("initial_turn_std_degrees", 10.0)
        )
        self.particle_final_turn_std = float(
            self.particle_cem.pop("final_turn_std_degrees", 2.0)
        )
        if self.particle_cem:
            raise ValueError(
                "Unknown particle_cem options: "
                f"{sorted(self.particle_cem)}"
            )
        if not 0.0 <= self.particle_min_speed <= self.particle_max_speed:
            raise ValueError("particle_cem requires 0 <= min_speed <= max_speed")
        if (self.particle_position_lower_bound is None) != (
            self.particle_position_upper_bound is None
        ):
            raise ValueError(
                "particle_cem position_lower_bound and position_upper_bound "
                "must be provided together"
            )
        if self.particle_position_lower_bound is not None:
            self.particle_position_lower_bound = float(
                self.particle_position_lower_bound
            )
            self.particle_position_upper_bound = float(
                self.particle_position_upper_bound
            )
            if not (
                self.particle_position_lower_bound
                < self.particle_position_upper_bound
            ):
                raise ValueError(
                    "particle_cem position_lower_bound must be below "
                    "position_upper_bound"
                )
        if self.particle_max_proposals < self.num_samples:
            raise ValueError(
                "particle_cem max_proposals must be at least num_samples"
            )
        if not 0.0 <= self.max_heading_change_degrees <= 180.0:
            raise ValueError("max_heading_change_degrees must lie in [0, 180]")
        if not (
            0.0
            <= self.particle_boundary_min_speed
            <= self.particle_boundary_max_speed
        ):
            raise ValueError(
                "particle_cem requires 0 <= boundary_min_speed <= boundary_max_speed"
            )
        if self.particle_physical_action_clip_px <= 0.0:
            raise ValueError(
                "particle_cem physical_action_clip_px must be positive"
            )
        if not (
            0.0
            <= self.particle_boundary_full_direction_probability
            <= 1.0
        ):
            raise ValueError(
                "boundary_full_direction_probability must lie in [0, 1]"
            )
        if not 0.0 <= self.particle_early_stop_probability <= 1.0:
            raise ValueError("early_stop_probability must lie in [0, 1]")
        if self.particle_post_stop_jitter_max_speed < 0.0:
            raise ValueError("post_stop_jitter_max_speed must be non-negative")
        if self.particle_random_init_steps < 1:
            raise ValueError("particle_cem random_init_steps must be positive")
        if self.use_particle_cem and self.particle_random_init_steps > self.opt_steps:
            raise ValueError(
                "particle_cem random_init_steps must not exceed opt_steps"
            )
        if self.particle_random_fraction_schedule is not None:
            if len(self.particle_random_fraction_schedule) != self.opt_steps:
                raise ValueError(
                    "particle_cem random_fraction_schedule must contain one "
                    "value per opt_step"
                )
            if any(
                value < 0.0 or value > 1.0
                for value in self.particle_random_fraction_schedule
            ):
                raise ValueError(
                    "particle_cem random_fraction_schedule values must lie in [0, 1]"
                )
            expected_random_prefix = (
                (1.0,) * self.particle_random_init_steps
            )
            if (
                self.particle_random_fraction_schedule[
                    : self.particle_random_init_steps
                ] != expected_random_prefix
                or any(
                    value >= 1.0
                    for value in self.particle_random_fraction_schedule[
                        self.particle_random_init_steps :
                    ]
                )
            ):
                raise ValueError(
                    "random_fraction_schedule must start with exactly "
                    "random_init_steps entries equal to 1.0"
                )
        if not 0.0 <= self.particle_temporal_correlation < 1.0:
            raise ValueError("temporal_correlation must lie in [0, 1)")
        if min(
            self.particle_initial_speed_std,
            self.particle_final_speed_std,
            self.particle_initial_heading_std,
            self.particle_final_heading_std,
            self.particle_initial_turn_std,
            self.particle_final_turn_std,
        ) < 0.0:
            raise ValueError("particle_cem mutation standard deviations must be nonnegative")
        if any(
            initial < final
            for initial, final in (
                (self.particle_initial_speed_std, self.particle_final_speed_std),
                (self.particle_initial_heading_std, self.particle_final_heading_std),
                (self.particle_initial_turn_std, self.particle_final_turn_std),
            )
        ):
            raise ValueError(
                "particle_cem initial mutation stds must be at least their final stds"
            )
        if self.sequential_cem_rounds < 1:
            raise ValueError("cem_rounds_per_model_step must be positive")
        if self.use_trajectory_prior:
            if self.trajectory_action_scale <= 0.0:
                raise ValueError("trajectory action_scale must be positive")
            if self.num_action_knots is not None:
                raise ValueError(
                    "trajectory_prior produces all primitive actions directly; "
                    "num_action_knots must be null"
                )
            if self.primitive_action_dim != 2:
                raise ValueError(
                    "PushT trajectory_prior requires 2-D primitive actions, got "
                    f"{self.primitive_action_dim}"
                )
            configured_control_points = self.trajectory_prior.pop(
                "num_control_points", None
            )
            expected_particle_steps = self.horizon * self.frameskip
            if self.use_particle_cem:
                if configured_control_points is not None and (
                    int(configured_control_points) != expected_particle_steps
                ):
                    raise ValueError(
                        "particle_cem uses every primitive action as a control "
                        "parameter: num_control_points, when provided, must equal "
                        f"horizon * frameskip ({expected_particle_steps}), got "
                        f"{configured_control_points}"
                    )
                self.trajectory_num_control_points = expected_particle_steps
            else:
                self.trajectory_num_control_points = int(
                    5
                    if configured_control_points is None
                    else configured_control_points
                )
            if self.trajectory_num_control_points < 2:
                raise ValueError(
                    "trajectory_prior.num_control_points must be at least 2 for PCHIP"
                )
            if self.use_particle_cem and self.sequential_model_step_search:
                raise ValueError(
                    "particle_cem and sequential_model_step_search are mutually exclusive"
                )
            if (
                self.sequential_model_step_search
                and self.trajectory_num_control_points != self.frameskip
            ):
                raise ValueError(
                    "sequential_model_step_search requires one control point per "
                    "primitive action: num_control_points must equal frameskip "
                    f"({self.frameskip}), got {self.trajectory_num_control_points}"
                )
            if not self.use_particle_cem:
                self.trajectory_control_delta_range = float(
                    self.trajectory_prior.get("control_delta_range", 70.0)
                )
                if self.trajectory_control_delta_range <= 0.0:
                    raise ValueError(
                        "trajectory_prior.control_delta_range must be positive"
                    )
                self.trajectory_min_control_delta_std = float(
                    self.trajectory_prior.pop("cem_min_control_delta_std", 2.0)
                )
                if self.trajectory_min_control_delta_std <= 0.0:
                    raise ValueError("cem_min_control_delta_std must be positive")
        if self.num_action_knots is not None:
            if self.num_action_knots < 1:
                raise ValueError(
                    f"num_action_knots must be positive, got {self.num_action_knots}"
                )
        if self.use_velocity_delta:
            if self.use_trajectory_prior or self.num_action_knots is not None:
                raise ValueError(
                    "velocity_delta cannot be combined with trajectory_prior "
                    "or num_action_knots"
                )
            if self.primitive_action_dim != 2:
                raise ValueError(
                    "velocity_delta requires 2-D primitive velocity actions, "
                    f"got {self.primitive_action_dim}"
                )
            if self.evaluator.state_0_torch.shape[-1] < 4:
                raise ValueError(
                    "velocity_delta requires state [x, y, vx, vy]"
                )
            if min(
                self.velocity_delta_max,
                self.velocity_delta_initial_std,
                self.velocity_delta_min_std,
            ) <= 0.0:
                raise ValueError(
                    "velocity_delta max_delta, initial_std, and min_std must be positive"
                )


    @staticmethod
    def _expand_cases(tensor, sample_count):
        """Repeat each case over candidates and flatten case/sample dimensions."""
        num_cases = tensor.shape[0]
        return tensor[:, None].expand(
            num_cases, sample_count, *tensor.shape[1:]
        ).reshape(num_cases * sample_count, *tensor.shape[1:])

    @torch.no_grad()
    def _probe_obstacle_transport_metadata(
        self,
        trans_obs_0,
        context_slots,
        context_proprio,
        context_actions,
        retrieval_context,
    ):
        """Run one bank of random WM actions and identify static mask slots."""
        builder = getattr(self.objective_fn, "build_obstacle_metadata", None)
        num_probes = int(
            getattr(self.objective_fn, "obstacle_probe_num_actions", 0)
        )
        use_fixed_scores = bool(
            getattr(self.objective_fn, "obstacle_use_fixed_slot_scores", False)
        )
        if builder is None or (num_probes <= 0 and not use_fixed_scores):
            return {}
        if self._obstacle_probe_metadata is not None:
            return self._obstacle_probe_metadata
        if context_slots is None:
            raise ValueError("Obstacle probing requires Slot world-model context")

        num_cases = context_slots["mask_probs"].shape[0]
        if use_fixed_scores:
            self._obstacle_probe_metadata = builder(
                context_slots["mask_probs"], None
            )
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(self.objective_fn.obstacle_probe_seed))
            action_std = float(self.objective_fn.obstacle_probe_action_std)
            action_bank = torch.randn(
                1,
                num_probes,
                1,
                self.action_dim,
                device=self.device,
                generator=generator,
            ) * action_std
            action_bank = self.evaluator.project_actions_to_env_bounds(action_bank)
            probe_actions = action_bank.expand(num_cases, -1, -1, -1).reshape(
                num_cases * num_probes, 1, self.action_dim
            )
            rollout_kwargs = {
                "obs_0": {
                    key: self._expand_cases(value, num_probes)
                    for key, value in trans_obs_0.items()
                },
                "act": probe_actions,
                "state_0": self._expand_cases(
                    self.evaluator.state_0_torch[:num_cases], num_probes
                ),
                "context_slots": {
                    key: self._expand_cases(value, num_probes)
                    for key, value in context_slots.items()
                },
                "context_proprio_embeddings": self._expand_cases(
                    context_proprio, num_probes
                ),
                "context_actions": self._expand_cases(context_actions, num_probes),
            }
            if retrieval_context is not None:
                rollout_kwargs["retrieval_context"] = {
                    key: self._expand_cases(value, num_probes)
                    for key, value in retrieval_context.items()
                }
            prediction, _ = self.wm.rollout(**rollout_kwargs)
            predicted_masks = prediction["mask_probs"][:, -1].reshape(
                num_cases,
                num_probes,
                prediction["mask_probs"].shape[-2],
                prediction["mask_probs"].shape[-1],
            )
            self._obstacle_probe_metadata = builder(
                context_slots["mask_probs"], predicted_masks
            )
        diagnostics = getattr(self.objective_fn, "last_obstacle_probe", None)
        if diagnostics is not None:
            changes = diagnostics["obstacle_slot_change"].detach().cpu().numpy()
            mass_changes = diagnostics[
                "obstacle_slot_mass_relative_change"
            ].detach().cpu().numpy()
            local_changes = diagnostics[
                "obstacle_slot_local_change"
            ].detach().cpu().numpy()
            centroid_changes = diagnostics[
                "obstacle_slot_relative_centroid_displacement"
            ].detach().cpu().numpy()
            areas = diagnostics["obstacle_slot_area_fraction"].detach().cpu().numpy()
            obstacles = diagnostics["obstacle_slot_flags"].detach().cpu().numpy()
            moving = diagnostics["moving_slot_flags"].detach().cpu().numpy()
            ignored = diagnostics["ignored_slot_flags"].detach().cpu().numpy()
            occupancy = diagnostics["hardened_obstacle_mask"].detach().cpu().numpy()
            for case_index in range(num_cases):
                print(
                    "WM obstacle probe "
                    f"case={case_index} changes={np.round(changes[case_index], 4).tolist()} "
                    f"mass_changes={np.round(mass_changes[case_index], 4).tolist()} "
                    f"local_changes={np.round(local_changes[case_index], 4).tolist()} "
                    f"relative_centroid_changes="
                    f"{np.round(centroid_changes[case_index], 4).tolist()} "
                    f"areas={np.round(areas[case_index], 4).tolist()} "
                    f"obstacle_slots={np.flatnonzero(obstacles[case_index]).tolist()} "
                    f"moving_slots={np.flatnonzero(moving[case_index]).tolist()} "
                    f"ignored_slots={np.flatnonzero(ignored[case_index]).tolist()} "
                    f"obstacle_mean={occupancy[case_index].mean():.4f} "
                    f"obstacle_hard_fraction="
                    f"{(occupancy[case_index] >= 0.5).mean():.4f}"
                )
        return self._obstacle_probe_metadata

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        if self.use_trajectory_prior:
            parameter_shape = (self.trajectory_num_control_points, 2)
            mu = torch.zeros(n_evals, *parameter_shape)
            if self.use_particle_cem:
                # Particle CEM keeps complete elites instead of fitting a
                # Gaussian; this placeholder is unused by its sampler.
                sigma = torch.zeros_like(mu)
            else:
                # Standard deviation of Uniform(-range, +range).
                sigma = torch.full_like(
                    mu, self.trajectory_control_delta_range / np.sqrt(3.0)
                )
            return mu, sigma
        if self.num_action_knots is not None:
            if actions is None or actions.shape[1] == 0:
                mu = torch.zeros(
                    n_evals,
                    self.num_action_knots,
                    self.primitive_action_dim,
                    device=actions.device if actions is not None else None,
                )
            else:
                mu = self._dense_actions_to_knots(actions)
            sigma = self.var_scale * torch.ones_like(mu)
            return mu, sigma

        if self.use_velocity_delta:
            if actions is None or actions.shape[1] == 0:
                mu = torch.zeros(n_evals, 0, self.action_dim)
            else:
                mu = self._dense_velocity_actions_to_deltas(actions)
            sigma = self.velocity_delta_initial_std * torch.ones(
                n_evals, self.horizon, self.action_dim
            )
            device = mu.device
            remaining_t = self.horizon - mu.shape[1]
            if remaining_t > 0:
                mu = torch.cat(
                    [
                        mu,
                        torch.zeros(
                            n_evals,
                            remaining_t,
                            self.action_dim,
                            device=device,
                        ),
                    ],
                    dim=1,
                )
            return self._project_action_parameters(mu), sigma

        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def _action_stats(self, device, dtype):
        mean = torch.as_tensor(
            self.preprocessor.action_mean, device=device, dtype=dtype
        )
        std = torch.as_tensor(
            self.preprocessor.action_std, device=device, dtype=dtype
        ).clamp_min(1.0e-8)
        return mean, std

    def _raw_velocity_bounds(self, device, dtype):
        normalized_low = self.evaluator._normalized_action_low
        normalized_high = self.evaluator._normalized_action_high
        if normalized_low is None or normalized_high is None:
            raise ValueError(
                "velocity_delta requires finite environment action bounds"
            )
        mean, std = self._action_stats(device, dtype)
        low = normalized_low.to(device=device, dtype=dtype) * std + mean
        high = normalized_high.to(device=device, dtype=dtype) * std + mean
        return low, high

    def _dense_velocity_actions_to_deltas(self, actions):
        """Convert normalized commanded velocities into physical delta-v."""
        mean, std = self._action_stats(actions.device, actions.dtype)
        velocities = actions.reshape(
            actions.shape[0], -1, self.primitive_action_dim
        ) * std + mean
        initial_velocity = self.evaluator.state_0_torch[
            : actions.shape[0], 2:4
        ].to(device=actions.device, dtype=actions.dtype)
        previous = torch.cat(
            [initial_velocity[:, None], velocities[:, :-1]], dim=1
        )
        deltas = velocities - previous
        return deltas.reshape(actions.shape[0], actions.shape[1], self.action_dim)

    def _velocity_deltas_to_dense_actions(self, deltas, case_start=0):
        """Integrate physical delta-v and return normalized velocity chunks."""
        primitive_deltas = deltas.reshape(
            *deltas.shape[:-2],
            deltas.shape[-2] * self.frameskip,
            self.primitive_action_dim,
        )
        num_cases = deltas.shape[0]
        initial_velocity = self.evaluator.state_0_torch[
            case_start : case_start + num_cases, 2:4
        ].to(device=deltas.device, dtype=deltas.dtype)
        leading_shape = primitive_deltas.shape[:-2]
        current = initial_velocity.reshape(
            num_cases,
            *([1] * (len(leading_shape) - 1)),
            self.primitive_action_dim,
        ).expand(*leading_shape, self.primitive_action_dim)
        low, high = self._raw_velocity_bounds(deltas.device, deltas.dtype)
        velocities = []
        for primitive_step in range(primitive_deltas.shape[-2]):
            current = (
                current + primitive_deltas[..., primitive_step, :]
            ).clamp(min=low, max=high)
            velocities.append(current)
        velocities = torch.stack(velocities, dim=-2)
        mean, std = self._action_stats(deltas.device, deltas.dtype)
        normalized = (velocities - mean) / std
        return normalized.reshape(
            *deltas.shape[:-2], deltas.shape[-2], self.action_dim
        )

    def _dense_actions_to_knots(self, actions):
        """Convert model-step action chunks to the CEM knot parameterization."""
        batch_size, model_steps, _ = actions.shape
        dense = actions.reshape(
            batch_size, model_steps * self.frameskip, self.primitive_action_dim
        )
        total_steps = self.horizon * self.frameskip
        if dense.shape[1] < total_steps:
            dense = F.pad(dense, (0, 0, 0, total_steps - dense.shape[1]))
        elif dense.shape[1] > total_steps:
            dense = dense[:, :total_steps]
        return F.interpolate(
            dense.transpose(1, 2),
            size=self.num_action_knots,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)

    def _action_parameters_to_dense(self, action_parameters, case_start=0):
        """Convert CEM parameters into predictor/environment action chunks."""
        if self.use_trajectory_prior and self.use_particle_cem:
            # Each particle parameter is already one relative PushT command;
            # pack five primitive commands per world-model action directly.
            pixel_deltas = action_parameters.reshape(
                *action_parameters.shape[:-2],
                self.horizon,
                self.frameskip,
                self.primitive_action_dim,
            )
            dense = self._pixel_deltas_to_normalized_actions(pixel_deltas)
        elif self.use_trajectory_prior:
            dense = self._spatial_waypoints_to_dense_actions(
                action_parameters, case_start=case_start
            )
        elif self.use_velocity_delta:
            dense = self._velocity_deltas_to_dense_actions(
                action_parameters, case_start=case_start
            )
        elif self.num_action_knots is None:
            dense = action_parameters
        else:
            leading_shape = action_parameters.shape[:-2]
            flat_parameters = action_parameters.reshape(
                -1, self.num_action_knots, self.primitive_action_dim
            )
            dense = F.interpolate(
                flat_parameters.transpose(1, 2),
                size=self.horizon * self.frameskip,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            dense = dense.reshape(*leading_shape, self.horizon, self.action_dim)
        # This is the single boundary between planner parameters and model
        # actions. The predictor must score exactly what eval_actions executes.
        return self.evaluator.project_actions_to_env_bounds(dense)

    def _correlate_action_noise(self, noise):
        """Apply stationary AR(1) correlation over action/knot time."""
        rho = self.action_temporal_correlation
        if rho == 0.0 or noise.shape[-2] <= 1:
            return noise
        innovation_scale = float(np.sqrt(1.0 - rho * rho))
        correlated = [noise[..., 0, :]]
        for time_index in range(1, noise.shape[-2]):
            correlated.append(
                rho * correlated[-1]
                + innovation_scale * noise[..., time_index, :]
            )
        return torch.stack(correlated, dim=-2)

    @staticmethod
    def _pchip_interpolate(waypoints, num_positions):
        """Vectorized, differentiable PCHIP for equally spaced waypoints."""
        delta = waypoints[..., 1:, :] - waypoints[..., :-1, :]
        slopes = torch.zeros_like(waypoints)
        left, right = delta[..., :-1, :], delta[..., 1:, :]
        same_sign = left * right > 0
        denominator = left + right
        safe_denominator = torch.where(
            denominator.abs() > 1.0e-12,
            denominator,
            torch.ones_like(denominator),
        )
        harmonic = 2.0 * left * right / safe_denominator
        slopes[..., 1:-1, :] = torch.where(
            same_sign, harmonic, torch.zeros_like(harmonic)
        )

        first = (3.0 * delta[..., 0, :] - delta[..., 1, :]) / 2.0
        first = torch.where(
            first * delta[..., 0, :] <= 0, torch.zeros_like(first), first
        )
        first = torch.where(
            (delta[..., 0, :] * delta[..., 1, :] <= 0)
            & (first.abs() > 3.0 * delta[..., 0, :].abs()),
            3.0 * delta[..., 0, :],
            first,
        )
        last = (3.0 * delta[..., -1, :] - delta[..., -2, :]) / 2.0
        last = torch.where(
            last * delta[..., -1, :] <= 0, torch.zeros_like(last), last
        )
        last = torch.where(
            (delta[..., -1, :] * delta[..., -2, :] <= 0)
            & (last.abs() > 3.0 * delta[..., -1, :].abs()),
            3.0 * delta[..., -1, :],
            last,
        )
        slopes[..., 0, :], slopes[..., -1, :] = first, last

        query = torch.linspace(
            0.0,
            float(waypoints.shape[-2] - 1),
            num_positions + 1,
            device=waypoints.device,
            dtype=waypoints.dtype,
        )[1:]
        interval = torch.floor(query).long().clamp(max=waypoints.shape[-2] - 2)
        u = query - interval.to(query.dtype)
        u = torch.where(query == waypoints.shape[-2] - 1, torch.ones_like(u), u)
        view_shape = [1] * (waypoints.ndim - 2) + [num_positions, 1]
        u = u.view(*view_shape)
        y0, y1 = waypoints[..., interval, :], waypoints[..., interval + 1, :]
        m0, m1 = slopes[..., interval, :], slopes[..., interval + 1, :]
        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = u**3 - 2 * u**2 + u
        h01 = -2 * u**3 + 3 * u**2
        h11 = u**3 - u**2
        return h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1

    def _case_start_positions(self, action_parameters, case_start=0):
        if action_parameters.ndim < 3 or action_parameters.shape[-1] != 2:
            raise ValueError(
                "Trajectory parameters must have shape [case,...,control_point,2], "
                f"got {tuple(action_parameters.shape)}"
            )
        num_cases = action_parameters.shape[0]
        start = self.evaluator.state_0_torch[
            case_start : case_start + num_cases, :2
        ].to(
            device=action_parameters.device, dtype=action_parameters.dtype
        )
        extra_leading_dims = action_parameters.ndim - 3
        return start.view(num_cases, *([1] * extra_leading_dims), 1, 2)

    def _control_deltas_to_spatial_waypoints(
        self, control_deltas, case_start=0
    ):
        """Accumulate pixel-space control-point displacements from each start."""
        start = self._case_start_positions(control_deltas, case_start=case_start)
        return start + torch.cumsum(control_deltas, dim=-2)

    def _spatial_waypoints_to_dense_actions(self, action_parameters, case_start=0):
        start = self._case_start_positions(action_parameters, case_start=case_start)
        spatial_waypoints = self._control_deltas_to_spatial_waypoints(
            action_parameters, case_start=case_start
        )
        start = start.expand(*action_parameters.shape[:-2], 1, 2)
        waypoints = torch.cat([start, spatial_waypoints], dim=-2)
        primitive_steps = self.horizon * self.frameskip
        positions = self._pchip_interpolate(waypoints, primitive_steps)
        deltas_pixels = torch.diff(
            torch.cat([start, positions], dim=-2), dim=-2
        )
        physical_actions = deltas_pixels / self.trajectory_action_scale
        mean = torch.as_tensor(
            self.preprocessor.action_mean,
            device=physical_actions.device,
            dtype=physical_actions.dtype,
        )
        std = torch.as_tensor(
            self.preprocessor.action_std,
            device=physical_actions.device,
            dtype=physical_actions.dtype,
        ).clamp_min(1.0e-8)
        normalized = (physical_actions - mean) / std
        return normalized.reshape(
            *action_parameters.shape[:-2], self.horizon, self.action_dim
        )

    def _pixel_deltas_to_normalized_actions(self, pixel_deltas):
        """Pack per-model-step pixel deltas into normalized model actions."""
        if self.use_particle_cem:
            norms = torch.linalg.vector_norm(pixel_deltas, dim=-1, keepdim=True)
            scales = torch.clamp(
                self.particle_physical_action_clip_px
                / norms.clamp_min(1.0e-8),
                max=1.0,
            )
            pixel_deltas = pixel_deltas * scales
        physical_actions = pixel_deltas / self.trajectory_action_scale
        mean = torch.as_tensor(
            self.preprocessor.action_mean,
            device=physical_actions.device,
            dtype=physical_actions.dtype,
        )
        std = torch.as_tensor(
            self.preprocessor.action_std,
            device=physical_actions.device,
            dtype=physical_actions.dtype,
        ).clamp_min(1.0e-8)
        normalized = (physical_actions - mean) / std
        actions = normalized.reshape(
            *pixel_deltas.shape[:-3], pixel_deltas.shape[-3], self.action_dim
        )
        return self.evaluator.project_actions_to_env_bounds(actions)

    def _normalized_actions_to_pixel_deltas(self, actions):
        """Invert action preprocessing for sequential beam bookkeeping."""
        normalized = actions.reshape(
            *actions.shape[:-2], actions.shape[-2], self.frameskip, 2
        )
        mean = torch.as_tensor(
            self.preprocessor.action_mean,
            device=normalized.device,
            dtype=normalized.dtype,
        )
        std = torch.as_tensor(
            self.preprocessor.action_std,
            device=normalized.device,
            dtype=normalized.dtype,
        )
        return (normalized * std + mean) * self.trajectory_action_scale

    def _sample_step_extensions(
        self,
        start_positions,
        samples_per_parent,
        control_delta_means=None,
        control_delta_stds=None,
    ):
        """Sample one model step (frameskip spatial points) per beam prefix."""
        from .pusht_trajectory_prior import sample_control_delta_proposals

        starts = start_positions.detach().cpu().numpy()
        parent_count = starts.shape[0]
        if parent_count != len(samples_per_parent):
            raise ValueError("samples_per_parent must match the flattened beam size")
        pieces = []
        means = (
            None
            if control_delta_means is None
            else control_delta_means.detach().cpu().numpy()
        )
        stds = (
            None
            if control_delta_stds is None
            else control_delta_stds.detach().cpu().numpy()
        )
        if (means is None) != (stds is None):
            raise ValueError("control_delta_means and control_delta_stds must be paired")
        expected = (parent_count, self.frameskip, 2)
        if means is not None and (means.shape != expected or stds.shape != expected):
            raise ValueError(
                f"Expected step-extension mean/std {expected}, got "
                f"{means.shape}/{stds.shape}"
            )
        prior_kwargs = dict(self.trajectory_prior)
        prior_kwargs["num_control_points"] = self.frameskip
        for parent_idx, count in enumerate(samples_per_parent):
            if count == 0:
                continue
            rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
            sampled = sample_control_delta_proposals(
                start_positions=starts[parent_idx : parent_idx + 1],
                num_samples=int(count),
                rng=rng,
                control_delta_mean=(
                    None if means is None else means[parent_idx : parent_idx + 1]
                ),
                control_delta_std=(
                    None if stds is None else stds[parent_idx : parent_idx + 1]
                ),
                initial_uniform=means is None,
                **prior_kwargs,
            )[0]
            if means is not None:
                sampled[0] = means[parent_idx]
            pieces.append(
                (
                    parent_idx,
                    torch.as_tensor(sampled, device=self.device, dtype=torch.float32),
                )
            )
        return pieces

    def _project_action_parameters(self, action_parameters):
        if self.use_trajectory_prior:
            if self.use_particle_cem:
                # Speed and heading-constrained samplers already return valid
                # deltas; component clipping would only be redundant.
                return action_parameters
            return action_parameters.clamp(
                -self.trajectory_control_delta_range,
                self.trajectory_control_delta_range,
            )
        if self.use_velocity_delta:
            return action_parameters.clamp(
                -self.velocity_delta_max, self.velocity_delta_max
            )
        return self.evaluator.project_actions_to_env_bounds(action_parameters)

    def _adaptive_particle_mutation_stds(self, elite_parameters):
        """Estimate per-case mutation scales from the current elite spread."""
        elites = elite_parameters.detach().cpu().numpy().astype(np.float64)
        speeds = np.linalg.norm(elites, axis=-1)
        headings = np.arctan2(elites[..., 1], elites[..., 0])
        turns = np.angle(
            np.exp(1j * np.diff(np.unwrap(headings, axis=-1), axis=-1))
        )

        # Measure dispersion across elites at each trajectory step, then take
        # the median over time so model-step boundaries do not dominate it.
        speed_spread = np.median(np.std(speeds, axis=1), axis=-1)

        def circular_std(values, axis):
            resultant = np.abs(np.mean(np.exp(1j * values), axis=axis))
            return np.sqrt(
                np.maximum(0.0, -2.0 * np.log(np.clip(resultant, 1.0e-8, 1.0)))
            )

        heading_spread = np.rad2deg(circular_std(headings[..., 0], axis=1))
        turn_spread = np.rad2deg(
            np.median(circular_std(turns, axis=1), axis=-1)
        )

        def bounded(values, first, final):
            return np.clip(values, min(first, final), max(first, final))

        return (
            bounded(
                speed_spread,
                self.particle_initial_speed_std,
                self.particle_final_speed_std,
            ),
            bounded(
                heading_spread,
                self.particle_initial_heading_std,
                self.particle_final_heading_std,
            ),
            bounded(
                turn_spread,
                self.particle_initial_turn_std,
                self.particle_final_turn_std,
            ),
        )

    def _particle_random_fraction(self, iteration):
        if self.particle_random_fraction_schedule is not None:
            return self.particle_random_fraction_schedule[int(iteration)]
        return float(int(iteration) < self.particle_random_init_steps)

    def _sample_trajectory_proposals(
        self,
        case_start,
        case_end,
        mu,
        sigma,
        initial_uniform,
        iteration=0,
        elite_parameters=None,
        num_samples=None,
    ):
        """Sample valid pixel-space control-point displacements."""
        from .pusht_trajectory_prior import (
            apply_stochastic_early_stop,
            mutate_heading_limited_elites,
            sample_control_delta_proposals,
            sample_heading_limited_proposals,
        )

        start_positions = (
            self.evaluator.state_0_torch[case_start:case_end, :2]
            .detach()
            .cpu()
            .numpy()
        )
        sample_count = self.num_samples if num_samples is None else int(num_samples)
        if sample_count < 1:
            raise ValueError("trajectory proposal count must be positive")
        # NumPy's global RNG is initialized by the planning seed. Draw one
        # child seed per case chunk so repeated MPC calls are deterministic but
        # receive fresh proposal populations.
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        prior_kwargs = dict(self.trajectory_prior)
        if self.use_particle_cem:
            common_kwargs = dict(
                min_speed=self.particle_min_speed,
                max_speed=self.particle_max_speed,
                max_heading_change_degrees=self.max_heading_change_degrees,
                model_step_size=self.frameskip,
                boundary_min_speed=self.particle_boundary_min_speed,
                boundary_max_speed=self.particle_boundary_max_speed,
                boundary_full_direction_probability=(
                    self.particle_boundary_full_direction_probability
                ),
            )
            def draw_population(draw_count):
                if initial_uniform or elite_parameters is None:
                    drawn = sample_heading_limited_proposals(
                        start_positions=start_positions,
                        num_samples=draw_count,
                        rng=rng,
                        num_control_points=self.trajectory_num_control_points,
                        **common_kwargs,
                    )
                    return apply_stochastic_early_stop(
                        drawn,
                        rng=rng,
                        stop_probability=self.particle_early_stop_probability,
                        jitter_max_speed=self.particle_post_stop_jitter_max_speed,
                    )

                speed_std, heading_std, turn_std = (
                    self._adaptive_particle_mutation_stds(elite_parameters)
                )

                return mutate_heading_limited_elites(
                    start_positions=start_positions,
                    elite_control_deltas=elite_parameters.detach().cpu().numpy(),
                    num_samples=draw_count,
                    rng=rng,
                    speed_std=speed_std,
                    initial_heading_std_degrees=heading_std,
                    heading_change_std_degrees=turn_std,
                    temporal_correlation=self.particle_temporal_correlation,
                    **common_kwargs,
                )

            if self.particle_position_lower_bound is None:
                proposals = draw_population(sample_count)
            else:
                accepted = [[] for _ in range(len(start_positions))]
                accepted_counts = np.zeros(len(start_positions), dtype=np.int64)
                proposed_count = 0
                while np.any(accepted_counts < sample_count):
                    remaining_budget = self.particle_max_proposals - proposed_count
                    draw_count = min(
                        max(
                            sample_count,
                            2 * int((sample_count - accepted_counts).max()),
                        ),
                        remaining_budget,
                    )
                    minimum_draw_count = (
                        self.topk
                        if not initial_uniform and elite_parameters is not None
                        else 1
                    )
                    if draw_count < minimum_draw_count:
                        raise RuntimeError(
                            "Could not sample enough in-bounds particle trajectories: "
                            f"accepted={accepted_counts.tolist()}, "
                            f"proposed={proposed_count}"
                        )
                    drawn = draw_population(draw_count)
                    proposed_count += draw_count
                    positions = (
                        start_positions[:, None, None, :]
                        + np.cumsum(drawn, axis=2)
                    )
                    valid = np.all(
                        (positions >= self.particle_position_lower_bound)
                        & (positions <= self.particle_position_upper_bound),
                        axis=(2, 3),
                    )
                    for case_index in range(len(start_positions)):
                        needed = sample_count - accepted_counts[case_index]
                        if needed <= 0:
                            continue
                        keep = drawn[case_index, valid[case_index]][:needed]
                        if len(keep):
                            accepted[case_index].append(keep)
                            accepted_counts[case_index] += len(keep)
                proposals = np.stack(
                    [
                        np.concatenate(case_batches, axis=0)[:sample_count]
                        for case_batches in accepted
                    ],
                    axis=0,
                )
            return torch.as_tensor(
                proposals, device=self.device, dtype=torch.float32
            )
        proposals = sample_control_delta_proposals(
            start_positions=start_positions,
            num_samples=sample_count,
            rng=rng,
            control_delta_mean=mu[case_start:case_end].detach().cpu().numpy(),
            control_delta_std=sigma[case_start:case_end].detach().cpu().numpy(),
            initial_uniform=initial_uniform,
            num_control_points=self.trajectory_num_control_points,
            **prior_kwargs,
        )
        return torch.as_tensor(
            proposals, device=self.device, dtype=torch.float32
        )

    def _plan_sequential_model_steps(
        self,
        trans_obs_0,
        z_obs_g,
        context_slots_all,
        context_proprio_all,
        context_actions_all,
        retrieval_context_all,
        objective_metadata_all,
        evaluate_candidates,
    ):
        """Build a trajectory one model step at a time with a top-k beam.

        A model action contains ``frameskip`` two-dimensional primitive actions.
        In this mode those primitive endpoints are the spatial control points for
        that model step. At each depth, the configured sample budget (optionally
        scaled inversely with prefix length) is spread over the surviving
        prefixes, refined for the configured CEM rounds, and the best ``topk``
        complete prefixes are retained.
        """
        n_evals = trans_obs_0["visual"].shape[0]
        selected_pixel_deltas = []

        for case_idx in range(n_evals):
            # [beam, completed model steps, primitive steps, xy]
            prefixes = torch.empty(
                1, 0, self.frameskip, 2, device=self.device, dtype=torch.float32
            )
            prefix_losses = None
            initial_position = self.evaluator.state_0_torch[
                case_idx, :2
            ].to(device=self.device, dtype=torch.float32)

            for model_step in range(self.horizon):
                stage_samples = self.num_samples
                if self.scale_sequential_samples:
                    stage_samples = int(
                        np.ceil(self.num_samples * self.horizon / (model_step + 1))
                    )
                refinement_parents = prefixes
                refinement_means = None
                refinement_stds = None

                for cem_round in range(self.sequential_cem_rounds):
                    beam_size = refinement_parents.shape[0]
                    base_count, remainder = divmod(stage_samples, beam_size)
                    samples_per_parent = [
                        base_count + int(parent_idx < remainder)
                        for parent_idx in range(beam_size)
                    ]
                    parent_endpoints = initial_position[None] + refinement_parents.sum(
                        dim=(1, 2)
                    )
                    extension_pieces = self._sample_step_extensions(
                        parent_endpoints,
                        samples_per_parent,
                        control_delta_means=refinement_means,
                        control_delta_stds=refinement_stds,
                    )
                    candidates = []
                    for parent_idx, extensions in extension_pieces:
                        parent = refinement_parents[
                            parent_idx : parent_idx + 1
                        ].expand(extensions.shape[0], -1, -1, -1)
                        candidates.append(
                            torch.cat([parent, extensions[:, None]], dim=1)
                        )
                    candidate_deltas = torch.cat(candidates, dim=0)
                    candidate_actions = self._pixel_deltas_to_normalized_actions(
                        candidate_deltas
                    )
                    candidate_deltas = self._normalized_actions_to_pixel_deltas(
                        candidate_actions
                    )
                    candidate_count = candidate_actions.shape[0]
                    loss_parts = []
                    chunk = self.sample_chunk_size
                    for start in range(0, candidate_count, chunk):
                        end = min(start + chunk, candidate_count)
                        chunk_size = end - start
                        obs0_chunk = {
                            key: value[case_idx : case_idx + 1].expand(
                                chunk_size, *value.shape[1:]
                            )
                            for key, value in trans_obs_0.items()
                        }
                        zg_chunk = {
                            key: value[case_idx : case_idx + 1].expand(
                                chunk_size, *value.shape[1:]
                            )
                            for key, value in z_obs_g.items()
                        }
                        rollout_kwargs = dict(
                            obs_0=obs0_chunk,
                            act=candidate_actions[start:end],
                            state_0=self.evaluator.state_0_torch[
                                case_idx : case_idx + 1
                            ].expand(chunk_size, -1),
                        )
                        if context_slots_all is not None:
                            rollout_kwargs["context_slots"] = {
                                key: value[case_idx : case_idx + 1].expand(
                                    chunk_size, *value.shape[1:]
                                )
                                for key, value in context_slots_all.items()
                            }
                            rollout_kwargs["context_proprio_embeddings"] = (
                                context_proprio_all[case_idx : case_idx + 1].expand(
                                    chunk_size, *context_proprio_all.shape[1:]
                                )
                            )
                            rollout_kwargs["context_actions"] = context_actions_all[
                                case_idx : case_idx + 1
                            ].expand(chunk_size, *context_actions_all.shape[1:])
                            if retrieval_context_all is not None:
                                rollout_kwargs["retrieval_context"] = {
                                    key: value[case_idx : case_idx + 1].expand(
                                        chunk_size, *value.shape[1:]
                                    )
                                    for key, value in retrieval_context_all.items()
                                }
                        metadata_chunk = {
                            key: value[case_idx : case_idx + 1].expand(
                                chunk_size, *value.shape[1:]
                            )
                            for key, value in objective_metadata_all.items()
                        }
                        with torch.no_grad():
                            imagined, _ = self.wm.rollout(**rollout_kwargs)
                        loss_parts.append(
                            self.objective_fn(imagined, zg_chunk, **metadata_chunk)
                        )
                    losses = torch.cat(loss_parts, dim=0)
                    keep = min(self.topk, candidate_count)
                    top_indices = torch.argsort(losses)[:keep]
                    prefixes = candidate_deltas[top_indices]
                    prefix_losses = losses[top_indices]

                    # Fix older steps and refine only this step around every
                    # surviving elite during the next CEM round.
                    refinement_parents = prefixes[:, :-1]
                    refinement_means = prefixes[:, -1]
                    elite_std = refinement_means.std(dim=0).clamp_min(
                        self.trajectory_min_control_delta_std
                    )
                    refinement_stds = elite_std[None].expand_as(refinement_means)
                    log_step = (
                        model_step * self.sequential_cem_rounds + cem_round + 1
                    )
                    self.wandb_run.log(
                        {
                            f"{self.logging_prefix}/loss": prefix_losses[0].item(),
                            f"{self.logging_prefix}/beam_step": model_step + 1,
                            f"{self.logging_prefix}/cem_round": cem_round + 1,
                            f"{self.logging_prefix}/num_samples": stage_samples,
                            "step": log_step,
                        }
                    )

            selected_pixel_deltas.append(prefixes[0])

        selected_pixel_deltas = torch.stack(selected_pixel_deltas, dim=0)
        final_actions = self._pixel_deltas_to_normalized_actions(
            selected_pixel_deltas
        )
        if self.evaluator is not None and evaluate_candidates:
            logs, _, _, _ = self.evaluator.eval_actions(
                final_actions,
                filename=f"{self.logging_prefix}_output_sequential",
                save_slot_pca=False,
            )
            logs = {f"{self.logging_prefix}/{key}": value for key, value in logs.items()}
            logs["step"] = self.horizon
            self.wandb_run.log(logs)
            self.dump_logs(logs)
        return final_actions, np.full(n_evals, np.inf)

    def _dump_proposal_batch(
        self,
        action_parameters,
        normalized_actions,
        losses,
        iteration,
        case_start,
        loss_components=None,
        objective_metadata=None,
    ):
        """Persist an opt-in CEM batch for exact, post-run visualization."""
        if self.proposal_dump_path is None:
            return
        output_dir = self.proposal_dump_path
        output_dir.mkdir(parents=True, exist_ok=True)
        primitive_normalized = normalized_actions.detach().cpu().reshape(
            *normalized_actions.shape[:-1], self.frameskip, self.primitive_action_dim
        )
        primitive_actions = self.preprocessor.denormalize_actions(
            primitive_normalized
        )
        prefix = self.logging_prefix.replace("/", "_")
        output_path = output_dir / (
            f"{prefix}_iteration_{iteration + 1:02d}_cases_"
            f"{case_start:04d}_{case_start + normalized_actions.shape[0] - 1:04d}.npz"
        )
        payload = dict(
            action_parameters=_tensor_to_numpy(action_parameters),
            normalized_actions=_tensor_to_numpy(normalized_actions),
            primitive_normalized_actions=_tensor_to_numpy(primitive_normalized),
            primitive_actions=_tensor_to_numpy(primitive_actions),
            losses=_tensor_to_numpy(losses),
            iteration=np.asarray(iteration + 1, dtype=np.int64),
            logging_prefix=np.asarray(self.logging_prefix),
        )
        for name, value in (loss_components or {}).items():
            payload[f"loss_{name}"] = _tensor_to_numpy(value)
        for name, value in (objective_metadata or {}).items():
            if name == "transport_cost_addition" or not torch.is_tensor(value):
                continue
            payload[f"objective_{name}"] = _tensor_to_numpy(
                value[case_start : case_start + normalized_actions.shape[0]]
            )
        obstacle_metadata = getattr(self.objective_fn, "last_obstacle_probe", None)
        for name, value in (obstacle_metadata or {}).items():
            if not torch.is_tensor(value):
                continue
            payload[f"obstacle_{name}"] = _tensor_to_numpy(
                value[case_start : case_start + normalized_actions.shape[0]]
            )
        np.savez_compressed(output_path, **payload)

    def plan(self, obs_0, obs_g, actions=None, evaluate_candidates=True):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.evaluator.get_fixed_goal_z()
        if z_obs_g is None:
            z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]
        best_trajectory_parameters = None
        best_trajectory_losses = None
        elite_trajectory_parameters = None
        elite_trajectory_losses = None
        context_slots_all = self.evaluator.get_fixed_context_slots()
        context_proprio_all = (
            self.evaluator.get_planning_context_proprio()
            if context_slots_all is not None
            else None
        )
        context_actions_all = (
            self.evaluator.get_planning_context_actions()
            if context_slots_all is not None
            else None
        )
        retrieval_context_all = (
            self.evaluator.get_planning_retrieval_context()
            if context_slots_all is not None
            else None
        )
        objective_metadata_all = {}
        metadata_builder = getattr(self.objective_fn, "build_slot_metadata", None)
        if metadata_builder is not None and context_slots_all is not None:
            objective_metadata_all = metadata_builder(
                context_slots_all["mask_probs"], z_obs_g["mask_probs"]
            )
        objective_metadata_all.update(
            self._probe_obstacle_transport_metadata(
                trans_obs_0=trans_obs_0,
                context_slots=context_slots_all,
                context_proprio=context_proprio_all,
                context_actions=context_actions_all,
                retrieval_context=retrieval_context_all,
            )
        )
        predictor_context_cache_all = None
        prepare_predictor_cache = getattr(
            self.wm, "prepare_rollout_predictor_context_cache", None
        )
        expand_predictor_cache = getattr(
            self.wm, "expand_rollout_predictor_context_cache", None
        )
        if (
            retrieval_context_all is not None
            and callable(prepare_predictor_cache)
            and callable(expand_predictor_cache)
        ):
            with torch.no_grad():
                predictor_context_cache_all = prepare_predictor_cache(
                    retrieval_context_all
                )

        if self.use_trajectory_prior and self.sequential_model_step_search:
            return self._plan_sequential_model_steps(
                trans_obs_0=trans_obs_0,
                z_obs_g=z_obs_g,
                context_slots_all=context_slots_all,
                context_proprio_all=context_proprio_all,
                context_actions_all=context_actions_all,
                retrieval_context_all=retrieval_context_all,
                objective_metadata_all=objective_metadata_all,
                evaluate_candidates=evaluate_candidates,
            )

        for i in range(self.opt_steps):
            # Cases retain independent CEM distributions/top-k sets, but small
            # groups share each model call for substantially better GPU use.
            losses = []
            next_elite_trajectory_parameters = (
                torch.empty(
                    n_evals,
                    self.topk,
                    self.trajectory_num_control_points,
                    2,
                    device=self.device,
                )
                if self.use_trajectory_prior and self.use_particle_cem
                else None
            )
            next_elite_trajectory_losses = (
                torch.empty(
                    n_evals,
                    self.topk,
                    device=self.device,
                )
                if self.use_trajectory_prior and self.use_particle_cem
                else None
            )
            for case_start in range(0, n_evals, self.trajectory_batch_size):
                case_end = min(case_start + self.trajectory_batch_size, n_evals)
                num_cases = case_end - case_start
                parameter_shape = (
                    (self.trajectory_num_control_points, 2)
                    if self.use_trajectory_prior
                    else (
                        (self.num_action_knots, self.primitive_action_dim)
                        if self.num_action_knots is not None
                        else (self.horizon, self.action_dim)
                    )
                )
                pure_random_initialization = False
                if self.use_trajectory_prior:
                    random_fraction = (
                        self._particle_random_fraction(i)
                        if self.use_particle_cem
                        else float(i == 0)
                    )
                    pure_random_initialization = (
                        self.use_particle_cem
                        and (
                            elite_trajectory_parameters is None
                            or random_fraction >= 1.0
                        )
                    )
                    if pure_random_initialization:
                        action_parameters = self._sample_trajectory_proposals(
                            case_start,
                            case_end,
                            mu,
                            sigma,
                            initial_uniform=True,
                            iteration=i,
                            elite_parameters=None,
                            num_samples=self.num_samples,
                        )
                    elif self.use_particle_cem:
                        random_count = min(
                            int(round(random_fraction * self.num_samples)),
                            self.num_samples - self.topk,
                        )
                        mutation_count = self.num_samples - random_count
                        action_parameters = self._sample_trajectory_proposals(
                            case_start,
                            case_end,
                            mu,
                            sigma,
                            initial_uniform=False,
                            iteration=i,
                            elite_parameters=elite_trajectory_parameters[
                                case_start:case_end
                            ],
                            num_samples=mutation_count,
                        )
                        if random_count:
                            random_parameters = self._sample_trajectory_proposals(
                                case_start,
                                case_end,
                                mu,
                                sigma,
                                initial_uniform=True,
                                iteration=i,
                                elite_parameters=None,
                                num_samples=random_count,
                            )
                            action_parameters = torch.cat(
                                [action_parameters, random_parameters], dim=1
                            )
                    else:
                        action_parameters = self._sample_trajectory_proposals(
                            case_start,
                            case_end,
                            mu,
                            sigma,
                            initial_uniform=(i == 0),
                            iteration=i,
                            elite_parameters=None,
                        )
                else:
                    action_noise = torch.randn(
                        num_cases,
                        self.num_samples,
                        *parameter_shape,
                        device=self.device,
                    )
                    action_noise = self._correlate_action_noise(action_noise)
                    action_parameters = (
                        action_noise * sigma[case_start:case_end, None]
                        + mu[case_start:case_end, None]
                    )
                action_parameters = self._project_action_parameters(action_parameters)
                if not self.use_particle_cem:
                    action_parameters[:, 0] = mu[case_start:case_end]
                action = self._action_parameters_to_dense(
                    action_parameters, case_start=case_start
                )

                chunk = self.sample_chunk_size
                loss_parts = []
                component_parts = {}
                for start in range(0, self.num_samples, chunk):
                    end = min(start + chunk, self.num_samples)
                    chunk_size = end - start
                    act_chunk = action[:, start:end].reshape(
                        num_cases * chunk_size, self.horizon, self.action_dim
                    )
                    obs0_chunk = {
                        key: self._expand_cases(
                            value[case_start:case_end], chunk_size
                        )
                        for key, value in trans_obs_0.items()
                    }
                    zg_chunk = {
                        key: self._expand_cases(
                            value[case_start:case_end], chunk_size
                        )
                        for key, value in z_obs_g.items()
                    }
                    slots_chunk = None
                    context_proprio_chunk = None
                    context_actions_chunk = None
                    retrieval_context_chunk = None
                    predictor_context_cache_chunk = None
                    if context_slots_all is not None:
                        slots_chunk = {
                            key: self._expand_cases(
                                value[case_start:case_end], chunk_size
                            )
                            for key, value in context_slots_all.items()
                        }
                        context_proprio_chunk = self._expand_cases(
                            context_proprio_all[case_start:case_end], chunk_size
                        )
                        context_actions_chunk = self._expand_cases(
                            context_actions_all[case_start:case_end], chunk_size
                        )
                        if retrieval_context_all is not None:
                            retrieval_context_chunk = {
                                key: self._expand_cases(
                                    value[case_start:case_end], chunk_size
                                )
                                for key, value in retrieval_context_all.items()
                            }
                        if predictor_context_cache_all is not None:
                            predictor_context_cache_chunk = expand_predictor_cache(
                                predictor_context_cache_all,
                                sample_count=chunk_size,
                                case_start=case_start,
                                case_end=case_end,
                            )
                    state_chunk = self._expand_cases(
                        self.evaluator.state_0_torch[case_start:case_end],
                        chunk_size,
                    )
                    objective_metadata_chunk = {
                        key: self._expand_cases(
                            value[case_start:case_end], chunk_size
                        )
                        for key, value in objective_metadata_all.items()
                    }
                    with torch.no_grad():
                        rollout_kwargs = dict(
                            obs_0=obs0_chunk,
                            act=act_chunk,
                            state_0=state_chunk,
                        )
                        if slots_chunk is not None:
                            rollout_kwargs["context_slots"] = slots_chunk
                            rollout_kwargs["context_proprio_embeddings"] = context_proprio_chunk
                            rollout_kwargs["context_actions"] = context_actions_chunk
                            rollout_kwargs["retrieval_context"] = retrieval_context_chunk
                            if predictor_context_cache_chunk is not None:
                                rollout_kwargs["predictor_context_cache"] = (
                                    predictor_context_cache_chunk
                                )
                        # Every candidate in this CEM call starts from the same
                        # observed state and therefore shares one retrieval.
                        # Retrieval is refreshed only after the selected first
                        # action has actually been executed by MPC.
                        i_z_obses_chunk, _ = self.wm.rollout(**rollout_kwargs)
                    if self.dump_loss_components:
                        chunk_components = self.objective_fn.loss_components(
                            i_z_obses_chunk,
                            zg_chunk,
                            **objective_metadata_chunk,
                        )
                        chunk_loss = chunk_components["total"]
                        for name, value in chunk_components.items():
                            component_parts.setdefault(name, []).append(
                                value.reshape(
                                    num_cases, chunk_size, *value.shape[1:]
                                )
                            )
                    else:
                        chunk_loss = self.objective_fn(
                            i_z_obses_chunk,
                            zg_chunk,
                            **objective_metadata_chunk,
                        )
                    loss_parts.append(chunk_loss.view(num_cases, chunk_size))
                loss = torch.cat(loss_parts, dim=1)
                loss_components = {
                    name: torch.cat(parts, dim=1)
                    for name, parts in component_parts.items()
                }
                self._dump_proposal_batch(
                    action_parameters,
                    action,
                    loss,
                    iteration=i,
                    case_start=case_start,
                    loss_components=loss_components,
                    objective_metadata=objective_metadata_all,
                )

                topk_candidate_parameters = action_parameters
                topk_candidate_losses = loss
                if (
                    self.use_particle_cem
                    and pure_random_initialization
                    and elite_trajectory_parameters is not None
                ):
                    # Accumulate the best trajectories across all independent
                    # random-initialization generations instead of discarding
                    # the previous generation's top-k set.
                    topk_candidate_parameters = torch.cat(
                        [
                            elite_trajectory_parameters[case_start:case_end],
                            action_parameters,
                        ],
                        dim=1,
                    )
                    topk_candidate_losses = torch.cat(
                        [
                            elite_trajectory_losses[case_start:case_end],
                            loss,
                        ],
                        dim=1,
                    )
                topk_idx = torch.argsort(topk_candidate_losses, dim=1)[
                    :, : self.topk
                ]
                gather_shape = [
                    -1,
                    -1,
                    *topk_candidate_parameters.shape[2:],
                ]
                topk_action_parameters = torch.gather(
                    topk_candidate_parameters,
                    1,
                    topk_idx.view(num_cases, self.topk, *([1] * len(parameter_shape))).expand(
                        *gather_shape
                    ),
                )
                topk_losses = torch.gather(topk_candidate_losses, 1, topk_idx)
                losses.append(topk_losses[:, 0])
                mu[case_start:case_end] = topk_action_parameters.mean(dim=1)
                mu[case_start:case_end] = self._project_action_parameters(
                    mu[case_start:case_end]
                )
                updated_sigma = topk_action_parameters.std(dim=1)
                if self.use_trajectory_prior:
                    if self.use_particle_cem:
                        next_elite_trajectory_parameters[
                            case_start:case_end
                        ] = topk_action_parameters
                        next_elite_trajectory_losses[
                            case_start:case_end
                        ] = topk_losses
                    else:
                        sigma[case_start:case_end] = updated_sigma.clamp_min(
                            self.trajectory_min_control_delta_std
                        )
                else:
                    if self.use_velocity_delta:
                        updated_sigma = updated_sigma.clamp_min(
                            self.velocity_delta_min_std
                        )
                    sigma[case_start:case_end] = updated_sigma

                # Distribution fitting still uses the top-k mean/std above,
                # but the trajectory eventually executed must be one that was
                # actually scored by the world model.  This also avoids a
                # multimodal elite set averaging into a poor or near-zero
                # action sequence on the final CEM iteration.
                chunk_best_loss, chunk_best_index = loss.min(dim=1)
                chunk_best_parameters = action_parameters[
                    torch.arange(num_cases, device=self.device),
                    chunk_best_index,
                ]
                if best_trajectory_parameters is None:
                    best_trajectory_parameters = torch.zeros(
                        n_evals,
                        *parameter_shape,
                        device=self.device,
                        dtype=action_parameters.dtype,
                    )
                    best_trajectory_losses = torch.full(
                        (n_evals,),
                        float("inf"),
                        device=self.device,
                        dtype=loss.dtype,
                    )
                previous = best_trajectory_losses[case_start:case_end]
                improved = chunk_best_loss < previous
                best_trajectory_losses[case_start:case_end] = torch.where(
                    improved, chunk_best_loss, previous
                )
                parameter_update_mask = improved.view(
                    num_cases, *([1] * len(parameter_shape))
                )
                best_trajectory_parameters[case_start:case_end] = torch.where(
                    parameter_update_mask,
                    chunk_best_parameters,
                    best_trajectory_parameters[case_start:case_end],
                )

            if self.use_particle_cem:
                elite_trajectory_parameters = next_elite_trajectory_parameters
                elite_trajectory_losses = next_elite_trajectory_losses

            iteration_log = {
                f"{self.logging_prefix}/loss": torch.cat(losses).mean().item(),
                "step": i + 1,
            }
            if self.use_particle_cem:
                iteration_log[f"{self.logging_prefix}/random_fraction"] = (
                    self._particle_random_fraction(i)
                )
            if self.use_trajectory_prior and not self.use_particle_cem:
                iteration_log.update(
                    {
                        f"{self.logging_prefix}/trajectory_control_delta_std": float(
                            sigma.mean().item()
                        )
                    }
                )
            self.wandb_run.log(iteration_log)
            should_eval = (
                (i + 1) % self.eval_every == 0
                or i == self.opt_steps - 1
            )
            if self.evaluator is not None and should_eval and evaluate_candidates:
                evaluation_parameters = best_trajectory_parameters
                logs, successes, _, _ = self.evaluator.eval_actions(
                    self._action_parameters_to_dense(evaluation_parameters),
                    filename=f"{self.logging_prefix}_output_{i+1}",
                    # Intermediate CEM diagnostics do not need expensive PCA;
                    # committed MPC executions and final output still save it.
                    save_slot_pca=False,
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        # Return the lowest-loss sampled trajectory, not the final top-k mean.
        # The latter is useful for fitting the next proposal distribution but
        # may never have been evaluated and can be poor for multimodal elites.
        final_parameters = best_trajectory_parameters
        return self._action_parameters_to_dense(final_parameters), np.full(
            n_evals, np.inf
        )  # all actions are valid
