import torch
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from utils import move_to_device


class GDPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        action_noise,
        sample_type,
        lr,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
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
        self.action_noise = action_noise
        self.sample_type = sample_type
        self.lr = lr
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix

    def init_actions(self, obs_0, actions=None):
        """
        Initializes or appends actions for planning, ensuring the output shape is (b, self.horizon, action_dim).
        """
        n_evals = obs_0["visual"].shape[0]
        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            if self.sample_type == "randn":
                new_actions = torch.randn(n_evals, remaining_t, self.action_dim)
            elif self.sample_type == "zero":  # zero action of env
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
                new_actions = rearrange(
                    new_actions, "... (f d) -> ... f d", f=self.evaluator.frameskip
                )
                new_actions = self.preprocessor.normalize_actions(new_actions)
                new_actions = rearrange(new_actions, "... f d -> ... (f d)")
            actions = torch.cat([actions, new_actions.to(device)], dim=1)
        return actions

    def get_action_optimizer(self, actions):
        return torch.optim.SGD([actions], lr=self.lr)

    def plan(self, obs_0, obs_g, actions=None, evaluate_candidates=True):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor
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
        z_obs_g_detached = {key: value.detach() for key, value in z_obs_g.items()}

        context_slots = self.evaluator.get_fixed_context_slots()
        retrieval_context = self.evaluator.get_planning_retrieval_context()
        objective_metadata = {}
        metadata_builder = getattr(self.objective_fn, "build_slot_metadata", None)
        if metadata_builder is not None and context_slots is not None:
            objective_metadata = metadata_builder(
                context_slots["mask_probs"], z_obs_g_detached["mask_probs"]
            )

        actions = self.evaluator.project_actions_to_env_bounds(
            self.init_actions(obs_0, actions).to(self.device)
        )
        actions.requires_grad = True
        optimizer = self.get_action_optimizer(actions)
        n_evals = actions.shape[0]

        for i in range(self.opt_steps):
            optimizer.zero_grad()
            projected_actions = self.evaluator.project_actions_to_env_bounds(actions)
            rollout_kwargs = dict(
                obs_0=trans_obs_0,
                act=projected_actions,
                state_0=self.evaluator.state_0_torch,
            )
            if context_slots is not None:
                rollout_kwargs["context_slots"] = context_slots
                rollout_kwargs["context_proprio_embeddings"] = self.evaluator.get_planning_context_proprio()
                rollout_kwargs["context_actions"] = self.evaluator.get_planning_context_actions()
                rollout_kwargs["retrieval_context"] = retrieval_context
            i_z_obses, i_zs = self.wm.rollout(**rollout_kwargs)
            loss = self.objective_fn(
                i_z_obses, z_obs_g_detached, **objective_metadata
            )  # (n_evals, )
            total_loss = loss.mean() * n_evals  # loss for each eval is independent
            total_loss.backward()
            with torch.no_grad():
                actions_new = actions - optimizer.param_groups[0]["lr"] * actions.grad
                actions_new += (
                    torch.randn_like(actions_new) * self.action_noise
                )  # Add Gaussian noise
                actions_new = self.evaluator.project_actions_to_env_bounds(actions_new)
                actions.copy_(actions_new)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": total_loss.item(), "step": i + 1}
            )
            if (
                self.evaluator is not None
                and evaluate_candidates
                and i % self.eval_every == 0
            ):
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success
        return self.evaluator.project_actions_to_env_bounds(
            actions.detach()
        ), np.full(n_evals, np.inf)  # all actions are valid
