import torch
import hydra
import copy
import numpy as np
from einops import rearrange, repeat
from utils import slice_trajdict_with_t
from .base_planner import BasePlanner


class MPCPlanner(BasePlanner):
    """
    an online planner so feedback from env is allowed
    """

    def __init__(
        self,
        max_iter,
        n_taken_actions,
        sub_planner,
        wm,
        env,  # for online exec
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="mpc",
        log_filename="logs.json",
        save_video=True,
        max_env_steps=None,
        test_context_num=0,
        test_context_seed=0,
        test_context_pool_size=1000,
        test_context_action_std=1.5,
        test_context_action_temporal_correlation=0.0,
        test_context_min_agent_displacement=0.0,
        test_context_min_object_displacement=0.0,
        test_context_initial_agent_block_clearance=None,
        test_context_require_contact=False,
        test_context_contact_tolerance=1.0,
        test_context_velocity_std_scale=1.0,
        test_context_generation_strategy="random",
        predictive_replan_within_block=False,
        replan_every_action=None,
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
        self.env = env
        self.max_iter = np.inf if max_iter is None else max_iter
        self.n_taken_actions = n_taken_actions
        # ``replan_every_action`` was the old name for a short-lived
        # implementation that observed the environment after every model
        # action.  Keep it as a compatibility alias, but the corrected behavior
        # is predictor-only replanning inside an n_taken_actions block.
        if replan_every_action is not None:
            predictive_replan_within_block = bool(replan_every_action)
        self.predictive_replan_within_block = bool(predictive_replan_within_block)
        self.max_env_steps = np.inf if max_env_steps is None else int(max_env_steps)
        self.max_model_steps = (
            np.inf
            if not np.isfinite(self.max_env_steps)
            else self.max_env_steps // int(self.evaluator.frameskip)
        )
        self.test_context_num = int(test_context_num)
        self.test_context_seed = int(test_context_seed)
        self.test_context_pool_size = int(test_context_pool_size)
        self.test_context_action_std = float(test_context_action_std)
        self.test_context_action_temporal_correlation = float(
            test_context_action_temporal_correlation
        )
        self.test_context_min_agent_displacement = float(
            test_context_min_agent_displacement
        )
        self.test_context_min_object_displacement = float(
            test_context_min_object_displacement
        )
        self.test_context_initial_agent_block_clearance = (
            None
            if test_context_initial_agent_block_clearance is None
            else float(test_context_initial_agent_block_clearance)
        )
        self.test_context_require_contact = bool(test_context_require_contact)
        self.test_context_contact_tolerance = float(
            test_context_contact_tolerance
        )
        self.test_context_velocity_std_scale = float(
            test_context_velocity_std_scale
        )
        self.test_context_generation_strategy = str(
            test_context_generation_strategy
        )
        if self.test_context_num < 0:
            raise ValueError("test_context_num must be non-negative")
        if self.test_context_pool_size < self.test_context_num:
            raise ValueError("test_context_pool_size must cover test_context_num")
        if self.test_context_action_std <= 0:
            raise ValueError("test_context_action_std must be positive")
        if not 0.0 <= self.test_context_action_temporal_correlation < 1.0:
            raise ValueError(
                "test_context_action_temporal_correlation must lie in [0, 1)"
            )
        if self.test_context_min_agent_displacement < 0.0:
            raise ValueError(
                "test_context_min_agent_displacement must be non-negative"
            )
        if self.test_context_min_object_displacement < 0.0:
            raise ValueError(
                "test_context_min_object_displacement must be non-negative"
            )
        if (
            self.test_context_initial_agent_block_clearance is not None
            and self.test_context_initial_agent_block_clearance < 0.0
        ):
            raise ValueError(
                "test_context_initial_agent_block_clearance must be non-negative"
            )
        if self.test_context_contact_tolerance < 0.0:
            raise ValueError("test_context_contact_tolerance must be non-negative")
        if self.test_context_velocity_std_scale < 0:
            raise ValueError(
                "test_context_velocity_std_scale must be non-negative"
            )
        if np.isfinite(self.max_env_steps) and self.max_model_steps < 1:
            raise ValueError(
                f"max_env_steps={self.max_env_steps} is shorter than frameskip={self.evaluator.frameskip}"
            )
        self.save_video = save_video
        self.logging_prefix = logging_prefix
        sub_planner["_target_"] = sub_planner["target"]
        self.sub_planner = hydra.utils.instantiate(
            sub_planner,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,  # evaluator is shared for mpc and sub_planner
            wandb_run=self.wandb_run,
            log_filename=None,
        )
        if self.predictive_replan_within_block and getattr(
            self.sub_planner, "use_velocity_delta", False
        ):
            raise ValueError(
                "velocity_delta CEM currently requires "
                "predictive_replan_within_block=false so every delta sequence "
                "starts from the latest observed realized velocity"
            )
        self.is_success = None
        self.action_len = None  # keep track of the step each traj reaches success
        self.iter = 0
        self.executed_model_steps = 0
        self.planned_actions = []

    def _apply_success_mask(self, actions):
        device = actions.device
        mask = torch.tensor(self.is_success).bool()
        actions[mask] = 0
        masked_actions = rearrange(
            actions[mask], "... (f d) -> ... f d", f=self.evaluator.frameskip
        )
        masked_actions = self.preprocessor.normalize_actions(masked_actions.cpu())
        masked_actions = rearrange(masked_actions, "... f d -> ... (f d)")
        actions[mask] = masked_actions.to(device)
        return actions

    def plan(self, obs_0, obs_g, actions=None):
        """
        actions is NOT used
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        cur_obs_0 = obs_0
        if self.test_context_num:
            self.evaluator.collect_random_transition_context(
                self.test_context_num,
                random_seed=self.test_context_seed,
                pool_size=self.test_context_pool_size,
                action_std=self.test_context_action_std,
                action_temporal_correlation=(
                    self.test_context_action_temporal_correlation
                ),
                min_agent_displacement=(
                    self.test_context_min_agent_displacement
                ),
                min_object_displacement=(
                    self.test_context_min_object_displacement
                ),
                initial_agent_block_clearance=(
                    self.test_context_initial_agent_block_clearance
                ),
                require_contact=self.test_context_require_contact,
                contact_tolerance=self.test_context_contact_tolerance,
                velocity_std_scale=self.test_context_velocity_std_scale,
                generation_strategy=self.test_context_generation_strategy,
            )
            cur_obs_0, _ = self.evaluator.get_init_cond()
        memo_actions = None
        while (
            not np.all(self.is_success)
            and self.iter < self.max_iter
            and self.executed_model_steps < self.max_model_steps
        ):
            remaining_model_steps = self.max_model_steps - self.executed_model_steps
            take_count = int(min(self.n_taken_actions, remaining_model_steps))

            if self.predictive_replan_within_block:
                # Select the block one model action at a time.  Between CEM
                # calls, advance only with predictor output and retrieve using
                # the predicted proprio embedding/mask.  The environment is
                # not touched until the complete block has been selected.
                block_start_context = self.evaluator.snapshot_planning_context()
                selected_actions = []
                for block_step in range(take_count):
                    self.sub_planner.logging_prefix = (
                        f"plan_{self.iter}_predstep_{block_step}"
                    )
                    actions, _ = self.sub_planner.plan(
                        obs_0=cur_obs_0,
                        obs_g=obs_g,
                        actions=memo_actions,
                        evaluate_candidates=False,
                    )
                    selected = actions.detach()[:, :1]
                    self._apply_success_mask(selected)
                    selected_actions.append(selected)
                    memo_actions = actions.detach()[:, 1:]
                    if block_step + 1 < take_count:
                        self.evaluator.advance_planning_context_with_prediction(
                            selected
                        )
                taken_actions = torch.cat(selected_actions, dim=1)
                # eval_actions below must replay the whole imagined block from
                # its actual starting state, using the same predictive retrieval
                # boundary after every action.
                self.evaluator.restore_planning_context(block_start_context)
            else:
                self.sub_planner.logging_prefix = f"plan_{self.iter}"
                actions, _ = self.sub_planner.plan(
                    obs_0=cur_obs_0,
                    obs_g=obs_g,
                    actions=memo_actions,
                    # MPC evaluates and executes the selected block below.
                    # A second evaluator rollout inside CEM cannot change the
                    # final-iteration selection and only duplicates inference.
                    evaluate_candidates=False,
                )
                taken_actions = actions.detach()[:, :take_count]
                self._apply_success_mask(taken_actions)
                memo_actions = actions.detach()[:, take_count:]
            print(f"MPC iter {self.iter} Eval ------- ")
            logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                taken_actions,
                None,
                filename=f"plan{self.iter}",
                save_video=self.save_video,
                save_context_visualization=False,
                predictive_retrieval=self.predictive_replan_within_block,
            )
            # eval_actions stops a case at its first successful model-step
            # boundary and clears the rest of this action block.
            if self.evaluator.last_effective_actions is not None:
                taken_actions = self.evaluator.last_effective_actions
            self.planned_actions.append(taken_actions)
            new_successes = successes & ~self.is_success  # Identify new successes
            self.is_success = (
                self.is_success | successes
            )  # Update overall success status
            first_success_model_steps = (
                self.evaluator.last_first_success_model_steps
            )
            for index in np.flatnonzero(new_successes):
                model_step = first_success_model_steps[index]
                if np.isfinite(model_step):
                    model_steps = int(model_step)
                else:
                    model_steps = take_count
                self.action_len[index] = self.executed_model_steps + model_steps

            # Freeze velocity at the successful endpoint before this feedback
            # is used as the next MPC initial state and predictor context. This
            # is required for inertial systems and keeps Wall proprio terminal.
            e_obses, e_states = self.evaluator.freeze_successful_feedback(
                e_obses, e_states, self.is_success
            )
            self.evaluator.update_planning_context(
                action_so_far=taken_actions,
                executed_obses=e_obses,
                executed_states=e_states,
            )

            print("self.is_success: ", self.is_success)
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": self.iter + 1})
            self.wandb_run.log(logs)
            self.dump_logs(logs)

            # update evaluator's init conditions with new env feedback
            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            cur_obs_0 = e_final_obs
            e_final_state = e_states[:, -1]
            self.evaluator.assign_init_cond(
                obs_0=e_final_obs,
                state_0=e_final_state,
            )
            self.iter += 1
            self.executed_model_steps += take_count
            self.sub_planner.logging_prefix = f"plan_{self.iter}"

        if not self.planned_actions:
            raise RuntimeError("MPC produced no actions")
        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )
        self.evaluator.reinitialize_planning_context()

        return planned_actions, self.action_len
