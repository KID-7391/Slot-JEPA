import os
import sys
import warnings

# Legacy Gym/D4RL print optional-dependency notices while they are imported.
# Planning does not use those optional suites, so keep PushT/Wall/Maze worker
# logs focused on actionable runtime diagnostics.
os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
warnings.filterwarnings(
    "ignore",
    message=r"dep_util is Deprecated.*",
    category=DeprecationWarning,
)
try:
    import gym_notices.notices as _gym_notices

    _gym_notices.notices.clear()
except (ImportError, AttributeError):
    pass

import gym
import fcntl
import hashlib
import json
import hydra
import math
import random
import torch
import pickle
import wandb
import logging
import numpy as np
try:
    import submitit
except ImportError:
    submitit = None
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

PLANAR_ROOT = Path(__file__).resolve().parents[1]
if str(PLANAR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLANAR_ROOT))

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


def _context_environment_id(env_info):
    """Return a stable non-negative id for context-pool partitioning."""
    layout_id = env_info.get("layout_id")
    if layout_id is not None:
        return int(layout_id)
    payload = json.dumps(
        env_info, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")

ALL_MODEL_KEYS = [
    "model",
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]


def _replace_hydra_runtime_cwd(value, runtime_cwd):
    """Materialize Hydra cwd interpolations for non-Hydra worker processes."""
    if isinstance(value, dict):
        return {
            key: _replace_hydra_runtime_cwd(item, runtime_cwd)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_hydra_runtime_cwd(item, runtime_cwd) for item in value
        ]
    if isinstance(value, str):
        return value.replace("${hydra:runtime.cwd}", str(runtime_cwd))
    return value


def _load_runtime_model_config(config_path, runtime_cwd):
    """Load a saved training config without requiring an active HydraConfig."""
    cfg = OmegaConf.load(config_path)
    raw = OmegaConf.to_container(cfg, resolve=False)
    raw = _replace_hydra_runtime_cwd(raw, runtime_cwd)
    cfg = OmegaConf.create(raw)
    OmegaConf.resolve(cfg)
    return cfg

def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    if submitit is None:
        raise ImportError("launch_plan_jobs requires the optional `submitit` package")
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        retrieval_dataset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.retrieval_dataset = retrieval_dataset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device
        self.visualization_enabled = bool(cfg_dict.get("visualization", True))

        # have different seeds for each planning instances
        self.eval_case_offset = int(cfg_dict.get("eval_case_offset", 0))
        configured_case_indices = cfg_dict.get("eval_case_indices")
        if configured_case_indices is None:
            self.eval_case_indices = list(
                range(
                    self.eval_case_offset,
                    self.eval_case_offset + int(cfg_dict["n_evals"]),
                )
            )
        else:
            configured_case_indices = [
                int(index) for index in configured_case_indices
            ]
            local_end = self.eval_case_offset + int(cfg_dict["n_evals"])
            if local_end > len(configured_case_indices):
                raise ValueError(
                    "Requested local fixed-target slice "
                    f"[{self.eval_case_offset}:{local_end}], but "
                    f"eval_case_indices contains {len(configured_case_indices)} entries"
                )
            self.eval_case_indices = configured_case_indices[
                self.eval_case_offset : local_end
            ]
        self.eval_seed = [
            cfg_dict["seed"] * case_index + 1
            for case_index in self.eval_case_indices
        ]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]
        self.fixed_env_info = None

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        slot_pca_config = dict(self.cfg_dict.get("slot_pca", {}))
        if not self.visualization_enabled:
            slot_pca_config["enabled"] = False
        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=(
                self.cfg_dict["n_plot_samples"]
                if self.visualization_enabled
                else 0
            ),
            slot_pca_config=slot_pca_config,
            slot_uot_config=self.cfg_dict.get("objective", {}),
            retrieval_dataset=self.retrieval_dataset,
            normalized_action_clip=self.cfg_dict.get("planner", {}).get(
                "normalized_action_clip", None
            ),
            context_layout_ids=(
                None
                if self.fixed_env_info is None
                else [
                    _context_environment_id(info) for info in self.fixed_env_info
                ]
            ),
            context_cache_identity=(
                getattr(self.wm, "_checkpoint_identity", None)
                or f"{self.cfg_dict.get('model_name')}@"
                f"{self.cfg_dict.get('model_epoch')}"
            ),
            context_cache_dir=self.cfg_dict.get("test_context_cache_dir"),
            planning_memory_config=self.cfg_dict.get("planning_memory"),
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        planner_cfg = dict(self.cfg_dict["planner"])
        planner_cfg["test_context_num"] = int(
            self.cfg_dict.get("test_context_num", 0)
        )
        planner_cfg["test_context_seed"] = int(
            self.cfg_dict.get("test_context_seed", self.cfg_dict["seed"])
        )
        planner_cfg["test_context_pool_size"] = int(
            self.cfg_dict.get("test_context_pool_size", 1000)
        )
        planner_cfg["test_context_action_std"] = float(
            self.cfg_dict.get("test_context_action_std", 1.5)
        )
        planner_cfg["test_context_action_temporal_correlation"] = float(
            self.cfg_dict.get("test_context_action_temporal_correlation", 0.0)
        )
        planner_cfg["test_context_min_agent_displacement"] = float(
            self.cfg_dict.get("test_context_min_agent_displacement", 0.0)
        )
        planner_cfg["test_context_min_object_displacement"] = float(
            self.cfg_dict.get("test_context_min_object_displacement", 0.0)
        )
        planner_cfg["test_context_initial_agent_block_clearance"] = (
            None
            if self.cfg_dict.get("test_context_initial_agent_block_clearance") is None
            else float(
                self.cfg_dict["test_context_initial_agent_block_clearance"]
            )
        )
        planner_cfg["test_context_require_contact"] = bool(
            self.cfg_dict.get("test_context_require_contact", False)
        )
        planner_cfg["test_context_contact_tolerance"] = float(
            self.cfg_dict.get("test_context_contact_tolerance", 1.0)
        )
        planner_cfg["test_context_velocity_std_scale"] = float(
            self.cfg_dict.get("test_context_velocity_std_scale", 1.0)
        )
        planner_cfg["test_context_generation_strategy"] = str(
            self.cfg_dict.get("test_context_generation_strategy", "random")
        )
        planner_cfg["save_video"] = (
            self.visualization_enabled
            and self.cfg_dict.get("save_video", True)
        )
        test_budget_multiplier = self.cfg_dict.get(
            "test_trajectory_budget_multiplier", None
        )
        if test_budget_multiplier is not None:
            test_lengths = getattr(self, "test_trajectory_lengths", None)
            if test_lengths is None or len(test_lengths) == 0:
                raise ValueError(
                    "test_trajectory_budget_multiplier requires fixed targets with "
                    "gt_action_lengths metadata"
                )
            budget_multiple = int(
                self.cfg_dict.get("test_trajectory_budget_round_to", 25)
            )
            if budget_multiple < 1:
                raise ValueError("test_trajectory_budget_round_to must be positive")
            max_test_traj_steps = int(np.max(test_lengths))
            raw_budget = float(test_budget_multiplier) * max_test_traj_steps
            planner_cfg["max_env_steps"] = int(
                np.ceil(raw_budget / budget_multiple) * budget_multiple
            )
            log.info(
                "MPC environment-step budget: %d = ceil((%.3g x max test trajectory length %d) / %d) x %d",
                planner_cfg["max_env_steps"],
                float(test_budget_multiplier),
                max_test_traj_steps,
                budget_multiple,
                budget_multiple,
            )
        budget_multiplier = self.cfg_dict.get("train_trajectory_budget_multiplier", None)
        if budget_multiplier is not None and test_budget_multiplier is None:
            source_dset = self.dset
            while hasattr(source_dset, "dataset"):
                source_dset = source_dset.dataset
            max_train_traj_steps = max(
                int(source_dset.get_seq_length(i)) for i in range(len(source_dset))
            )
            planner_cfg["max_env_steps"] = int(
                np.ceil(float(budget_multiplier) * max_train_traj_steps)
            )
            log.info(
                "MPC environment-step budget: %d = %.3g x max training trajectory length %d",
                planner_cfg["max_env_steps"],
                float(budget_multiplier),
                max_train_traj_steps,
            )
        self.planner = hydra.utils.instantiate(
            planner_cfg,
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        # optional: assume planning horizon equals to goal horizon
        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        if bool(cfg_dict.get("dump_plan_targets", True)):
            self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Filter by trajectory-length metadata. Calling self.dset[i] here
        # decodes and normalizes the complete video; with the 6,667-trajectory
        # Maze validation split that made every planning rank decode the whole
        # dataset before it could sample a target.
        valid_traj_ids = [
            i
            for i in range(len(self.dset))
            if int(self.dset.get_seq_length(i)) >= traj_len
        ]
        if len(valid_traj_ids) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            traj_id = random.choice(valid_traj_ids)
            max_offset = int(self.dset.get_seq_length(traj_id)) - traj_len
            obs, act, state, e_info = self.dset[traj_id]
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        case_indices = np.asarray(self.eval_case_indices, dtype=np.int64)

        def slice_cases(value):
            if isinstance(value, dict):
                return {key: slice_cases(item) for key, item in value.items()}
            if isinstance(value, torch.Tensor):
                return value[torch.as_tensor(case_indices, dtype=torch.long)]
            if isinstance(value, np.ndarray):
                return value[case_indices]
            return [value[int(index)] for index in case_indices]

        total_cases = len(data["state_0"])
        if np.any(case_indices < 0) or np.any(case_indices >= total_cases):
            raise ValueError(
                f"Requested fixed test cases {case_indices.tolist()}, but "
                f"{file_path} contains {total_cases} cases"
            )
        stored_seeds = data.get("eval_seeds")
        if stored_seeds is not None:
            expected_seeds = np.asarray(stored_seeds, dtype=np.int64)[case_indices]
            actual_seeds = np.asarray(self.eval_seed, dtype=np.int64)
            if not np.array_equal(expected_seeds, actual_seeds):
                raise ValueError(
                    f"Fixed target seeds do not match requested cases: "
                    f"file={expected_seeds.tolist()}, plan={actual_seeds.tolist()}"
                )
        target_env = data.get("metadata", {}).get("environment")
        if target_env is None:
            target_env = data.get("metadata", {}).get("env")
        paper_environment_families = {
            "wall": {"TwoRoom", "TwoRoomR"},
            "point_maze": {"Maze", "MazeR"},
            "pusht": {"PushT", "PushObj"},
        }
        # Target manifests may use either the internal Gym environment id or
        # the corresponding paper task name. Accept both representations.
        valid_target_envs = {
            str(self.env_name),
            *paper_environment_families.get(str(self.env_name), set()),
        }
        if target_env is not None and str(target_env) not in valid_target_envs:
            raise ValueError(
                f"Fixed targets are for env={target_env!r}, but checkpoint uses "
                f"env={self.env_name!r}"
            )
        self.state_0 = slice_cases(data["state_0"])
        self.state_g = slice_cases(data["state_g"])
        if data.get("env_info") is not None:
            fixed_env_info = slice_cases(data["env_info"])
            # Configure geometry/shape before either endpoint is rendered.
            # This is essential for held-out Wall layouts, U-Maze dimensions,
            # and PushObj shapes stored by the generalization benchmark.
            if isinstance(fixed_env_info, dict):
                fixed_env_info = [
                    {key: value[index] for key, value in fixed_env_info.items()}
                    for index in range(len(case_indices))
                ]
            transform_external_states = getattr(
                self.dset, "transform_external_states", None
            )
            if callable(transform_external_states):
                self.state_0 = transform_external_states(
                    self.state_0, fixed_env_info
                )
                self.state_g = transform_external_states(
                    self.state_g, fixed_env_info
                )
            self.env.update_env(fixed_env_info)
            self.fixed_env_info = fixed_env_info
        rerender = bool(self.cfg_dict.get("rerender_file_targets", False))
        has_saved_observations = "obs_0" in data and "obs_g" in data
        if not has_saved_observations and not rerender:
            raise ValueError(
                f"Fixed target file {file_path} stores physical states only; "
                "set rerender_file_targets=true to render its endpoints"
            )
        if has_saved_observations:
            self.obs_0 = slice_cases(data["obs_0"])
            self.obs_g = slice_cases(data["obs_g"])
        if rerender:
            # Fixed target files may contain RGB frames decoded from an older
            # MP4/rendering pipeline.  Recreate both endpoint observations
            # through the live planning environment so training
            # (render_from_state), goal encoding, and online rollouts share the
            # same renderer.  Keep the recorded states themselves as the
            # evaluation targets.
            obs_0, rendered_state_0 = self.env.prepare(self.eval_seed, self.state_0)
            obs_g, rendered_state_g = self.env.prepare(self.eval_seed, self.state_g)
            # A fixed benchmark is meaningful only if its serialized states are
            # valid states of the live simulator.  Do not silently compare two
            # methods after one renderer clips or otherwise changes an endpoint.
            for name, requested, rendered in (
                ("state_0", self.state_0, rendered_state_0),
                ("state_g", self.state_g, rendered_state_g),
            ):
                requested = np.asarray(requested)
                rendered = np.asarray(rendered)
                if requested.shape != rendered.shape or not np.allclose(
                    requested, rendered, rtol=1.0e-5, atol=1.0e-5
                ):
                    max_error = (
                        float(np.max(np.abs(requested - rendered)))
                        if requested.shape == rendered.shape
                        else float("inf")
                    )
                    raise ValueError(
                        f"Live {self.env_name} environment changed fixed {name}: "
                        f"requested={requested.shape}, rendered={rendered.shape}, "
                        f"max_abs_error={max_error:.6g}"
                    )
            self.obs_0 = {
                key: np.expand_dims(value, axis=1) for key, value in obs_0.items()
            }
            self.obs_g = {
                key: np.expand_dims(value, axis=1) for key, value in obs_g.items()
            }
            log.info(
                "Re-rendered fixed test cases %s through the live %s environment",
                case_indices.tolist(),
                self.env_name,
            )
        self.gt_actions = (
            None if data.get("gt_actions") is None else slice_cases(data["gt_actions"])
        )
        # Keep the complete fixed-test length distribution so multi-GPU ranks
        # all derive the same budget, independent of their local case slice.
        self.test_trajectory_lengths = (
            None
            if data.get("gt_action_lengths") is None
            else np.asarray(data["gt_action_lengths"], dtype=np.int64)
        )
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(),
            action_len,
            save_video=(
                self.visualization_enabled
                and self.cfg_dict.get("save_video", True)
            ),
            save_slot_pca=self.visualization_enabled,
            save_context_visualization=self.visualization_enabled,
            filename="output_final",
            predictive_retrieval=bool(
                getattr(self.planner, "predictive_replan_within_block", False)
            ),
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        logs["final_eval/test_context_num"] = int(
            self.cfg_dict.get("test_context_num", 0)
        )
        logs["final_eval/test_context_pool_size"] = int(
            self.cfg_dict.get("test_context_pool_size", 1000)
        )
        logs["final_eval/test_context_action_std"] = float(
            self.cfg_dict.get("test_context_action_std", 1.5)
        )
        logs["final_eval/test_context_action_temporal_correlation"] = float(
            self.cfg_dict.get("test_context_action_temporal_correlation", 0.0)
        )
        logs["final_eval/test_context_min_agent_displacement"] = float(
            self.cfg_dict.get("test_context_min_agent_displacement", 0.0)
        )
        logs["final_eval/test_context_min_object_displacement"] = float(
            self.cfg_dict.get("test_context_min_object_displacement", 0.0)
        )
        initial_clearance = self.cfg_dict.get(
            "test_context_initial_agent_block_clearance"
        )
        logs["final_eval/test_context_initial_agent_block_clearance"] = (
            -1.0 if initial_clearance is None else float(initial_clearance)
        )
        logs["final_eval/test_context_require_contact"] = int(
            bool(self.cfg_dict.get("test_context_require_contact", False))
        )
        logs["final_eval/test_context_contact_tolerance"] = float(
            self.cfg_dict.get("test_context_contact_tolerance", 1.0)
        )
        logs["final_eval/test_context_velocity_std_scale"] = float(
            self.cfg_dict.get("test_context_velocity_std_scale", 1.0)
        )
        logs["final_eval/test_context_generation_strategy"] = str(
            self.cfg_dict.get("test_context_generation_strategy", "random")
        )
        if np.isfinite(getattr(self.planner, "max_model_steps", np.inf)):
            logs["final_eval/model_step_budget"] = int(
                self.planner.max_model_steps
            )
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(
    model_ckpt,
    train_cfg,
    num_action_repeat,
    device,
    dset=None,
    load_retrieval_bank=True,
    log_checkpoint_load=True,
    checkpoint_payload=None,
):
    if str(train_cfg.model._target_).endswith("SlotMaskWorldModel"):
        if dset is None:
            raise ValueError("Slot model loading requires the trajectory dataset for dimensions and statistics")
        if not model_ckpt.exists():
            raise FileNotFoundError(f"Slot model checkpoint not found: {model_ckpt}")
        # Compose checkpoints on CPU, then move the fully initialized model to
        # its inference device once. This avoids a transient duplicate GPU copy.
        payload = (
            torch.load(model_ckpt, map_location="cpu")
            if checkpoint_payload is None
            else checkpoint_payload
        )
        is_compact_checkpoint = "predictor_state" in payload
        model = hydra.utils.instantiate(
            train_cfg.model,
            state_dim=dset.state_dim,
            action_dim=dset.action_dim * train_cfg.frameskip,
            state_mean=dset.state_mean,
            state_std=dset.state_std,
            action_mean=dset.action_mean,
            action_std=dset.action_std,
            frameskip=train_cfg.frameskip,
            env_name=train_cfg.env.name,
            # Compact inference checkpoints are combined with the shared
            # pretrain state below and loaded strictly in one operation.
            load_pretrained_weights=not is_compact_checkpoint,
        )
        if "predictor_state" in payload or "model_state" in payload:
            is_compact = "predictor_state" in payload
            checkpoint_state = payload[
                "predictor_state" if is_compact else "model_state"
            ]
            model.validate_proprio_encoder_checkpoint(checkpoint_state)
            expected_schema = int(model.transition_pair_schema_version.item())
            schema = checkpoint_state.get("transition_pair_schema_version")
            if schema is None or int(schema.item()) != expected_schema:
                raise RuntimeError(
                    "The selected checkpoint does not use transition-pair schema "
                    f"v{expected_schema}. "
                    "Retrain before planning."
                )
            if train_cfg.model.get("predictor_proprio_indices") is not None:
                proprio_schema = checkpoint_state.get(
                    "predictor_proprio_schema_version"
                )
                if proprio_schema is None or int(proprio_schema.item()) != 2:
                    raise RuntimeError(
                        "The selected PushT checkpoint was trained with privileged "
                        "block state. Retrain with predictor_proprio_indices."
                    )
            if is_compact:
                load_msg = model.load_downstream_predictor_state_dict(
                    checkpoint_state,
                    merge_shared_pretrain=True,
                )
            else:
                load_msg = model.load_state_dict(checkpoint_state, strict=False)
            if "transition_pair_schema_version" in load_msg.missing_keys:
                raise RuntimeError(
                    "The selected checkpoint uses the old non-retrieval context "
                    "predictor. Retrain with the Markov transition-pair model "
                    "before planning."
                )
            state_kind = "compact predictor" if is_compact else "legacy full"
            if log_checkpoint_load:
                print(
                    f"Loaded Slot-AC {state_kind} state from epoch "
                    f"{payload.get('epoch', 'unknown')}: "
                    f"checkpoint_tensors={len(checkpoint_state)}, "
                    f"merged_model_tensors={len(model.state_dict())}, "
                    f"strict_load=True."
                )
        elif "model" in payload:
            model = payload["model"]
            schema = getattr(model, "transition_pair_schema_version", None)
            if schema is None or int(schema.item()) != 6:
                raise RuntimeError(
                    "Serialized Slot model predates transition-pair schema v6"
                )
            if train_cfg.model.get("predictor_proprio_indices") is not None:
                proprio_schema = getattr(
                    model, "predictor_proprio_schema_version", None
                )
                if proprio_schema is None or int(proprio_schema.item()) != 2:
                    raise RuntimeError(
                        "Serialized PushT model uses privileged block state"
                    )
        else:
            raise ValueError(
                "Slot checkpoint has none of `predictor_state`, `model_state`, "
                f"or `model`: {model_ckpt}"
            )
        if not model.has_fixed_slots:
            raise RuntimeError(
                "Slot-AC checkpoint does not contain an initialized fixed predictor "
                "slot bank. Inference is not allowed to reinitialize slots; start a "
                "fresh training run with the current configuration first."
            )
        model = model.to(device)
        model.eval()
        retrieval_bank_path = payload.get("retrieval_bank_path")
        if model.predictor_architecture == "sequential":
            if retrieval_bank_path is not None:
                print(
                    "Sequential predictor ignores checkpoint retrieval bank: "
                    f"{retrieval_bank_path}"
                )
        elif load_retrieval_bank:
            if retrieval_bank_path is None:
                raise RuntimeError("Slot checkpoint does not record a retrieval bank")
            recorded_bank_path = Path(retrieval_bank_path)
            candidates = [recorded_bank_path]
            if not recorded_bank_path.is_absolute():
                # Published layout: <planar checkpoint root>/slot_cache/<bank>.
                candidates = [model_ckpt.parents[3] / recorded_bank_path]
                # Backward compatibility with the former downstream/slot_cache layout.
                candidates.append(model_ckpt.parents[4] / recorded_bank_path)
            retrieval_bank_path = next(
                (candidate for candidate in candidates if candidate.exists()), None
            )
            if retrieval_bank_path is None:
                searched = ", ".join(str(candidate) for candidate in candidates)
                raise FileNotFoundError(
                    f"Retrieval bank {recorded_bank_path} was not found; searched: {searched}"
                )
            model.load_retrieval_bank(retrieval_bank_path)
        elif log_checkpoint_load:
            print("Skipping retrieval-bank loading for encoder-only analysis")
        return model

    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = str(PLANAR_ROOT)
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


class _SingleProcessCacheAccelerator:
    """Minimal Accelerator interface used by inference-time cache building."""

    is_main_process = True
    num_processes = 1
    process_index = 0
    local_process_index = 0

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def wait_for_everyone():
        return None


def _resolve_recorded_retrieval_bank(model_ckpt, recorded_path):
    recorded_path = Path(recorded_path)
    candidates = [recorded_path]
    if not recorded_path.is_absolute():
        candidates = [
            model_ckpt.parents[3] / recorded_path,
            model_ckpt.parents[4] / recorded_path,
        ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _load_or_build_retrieval_bank(
    model,
    model_ckpt,
    model_cfg,
    train_dataset,
    device,
    cache_root,
    auto_build,
):
    """Load a checkpoint bank or derive it from raw training trajectories."""
    if model.predictor_architecture == "sequential":
        return None

    payload = torch.load(model_ckpt, map_location="cpu")
    recorded_path = payload.get("retrieval_bank_path")
    if recorded_path is not None:
        existing_path = _resolve_recorded_retrieval_bank(
            model_ckpt, recorded_path
        )
        if existing_path is not None:
            model.load_retrieval_bank(existing_path)
            return existing_path
    if not auto_build:
        raise FileNotFoundError(
            "Checkpoint retrieval bank is absent and "
            "auto_build_retrieval_bank=false"
        )

    # Reuse the training implementation so transition selection, collision/
    # layout balancing, cache identities, and mask extraction stay identical.
    from train import Trainer

    build_cfg = OmegaConf.create(
        OmegaConf.to_container(model_cfg, resolve=False)
    )
    with open_dict(build_cfg):
        build_cfg.ckpt_base_path = str(Path(cache_root).expanduser().resolve())

    builder = Trainer.__new__(Trainer)
    builder.cfg = build_cfg
    builder.model = model
    builder.datasets = {"train": train_dataset}
    builder.validation_dataset_keys = []
    builder.device = device
    builder.is_slot_model = True
    builder.accelerator = _SingleProcessCacheAccelerator()

    slot_cache_path = builder._slot_cache_file_path("train")
    bank_path = builder._retrieval_bank_file_path(slot_cache_path, train_dataset)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = bank_path.with_suffix(bank_path.suffix + ".build.lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if bank_path.exists():
            model.load_retrieval_bank(bank_path)
        else:
            print(
                "Retrieval bank is missing; rebuilding it from raw training "
                f"trajectories at {bank_path}"
            )
            builder.prepare_slot_cache_if_needed()
            if not bank_path.exists():
                raise FileNotFoundError(
                    f"Inference cache builder did not create {bank_path}"
                )
    return bank_path


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def build_planning_runtime(cfg_dict, load_retrieval_bank=True):
    """Load immutable planning assets once so multiple cases can reuse the GPU model.

    Encoder-only diagnostics may set ``load_retrieval_bank=False`` because slot
    extraction does not use the transition-memory bank. Planning keeps the
    strict default and still requires the recorded bank.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        launch_cwd = Path(hydra.utils.get_original_cwd())
    except ValueError:
        launch_cwd = Path.cwd()

    ckpt_base_path = Path(cfg_dict["ckpt_base_path"])
    if not ckpt_base_path.is_absolute():
        ckpt_base_path = (launch_cwd / ckpt_base_path).resolve()
    model_path = str(ckpt_base_path / "outputs" / cfg_dict["model_name"])
    config_candidates = [
        Path(model_path) / "hydra.yaml",
        Path(model_path) / ".hydra" / "config.yaml",
    ]
    model_config_path = next((path for path in config_candidates if path.exists()), None)
    if model_config_path is None:
        raise FileNotFoundError(f"Could not find training config under {model_path}")
    model_cfg = _load_runtime_model_config(model_config_path, launch_cwd)

    # PushT and PushObj share one Gym environment implementation but use
    # different published success criteria. Keep this evaluation-only choice
    # in the planning config so older training-run configs remain usable.
    geometric_center_eval = cfg_dict.get("use_geometric_center_eval")
    if geometric_center_eval is not None and model_cfg.env.name == "pusht":
        model_cfg.env.kwargs.use_geometric_center_eval = bool(
            geometric_center_eval
        )

    expected_input_size = cfg_dict.get("expected_model_input_size")
    if expected_input_size is not None:
        actual_input_size = int(model_cfg.model.input_size)
        if actual_input_size != int(expected_input_size):
            raise ValueError(
                "Selected checkpoint has model.input_size="
                f"{actual_input_size}, but this planning config requires "
                f"{int(expected_input_size)}. Train/select a matching checkpoint."
            )
    expected_slot_size = cfg_dict.get("expected_slot_spatial_size")
    if expected_slot_size is not None:
        actual_slot_size = int(model_cfg.model.slot_spatial_size)
        if actual_slot_size != int(expected_slot_size):
            raise ValueError(
                "Selected checkpoint has model.slot_spatial_size="
                f"{actual_slot_size}, but this planning config requires "
                f"{int(expected_slot_size)}. Train/select a matching checkpoint."
            )
    expected_predictor_size = cfg_dict.get(
        "expected_predictor_internal_spatial_size"
    )
    if expected_predictor_size is not None:
        actual_predictor_size = int(
            model_cfg.model.get(
                "predictor_internal_spatial_size",
                model_cfg.model.slot_spatial_size,
            )
        )
        if actual_predictor_size != int(expected_predictor_size):
            raise ValueError(
                "Selected checkpoint has model.predictor_internal_spatial_size="
                f"{actual_predictor_size}, but this planning config requires "
                f"{int(expected_predictor_size)}. Train/select a matching checkpoint."
            )
    for expected_key, model_key in (
        ("expected_mask_softmax_temperature", "mask_softmax_temperature"),
        ("expected_mask_sharpen_gamma", "mask_sharpen_gamma"),
    ):
        expected_value = cfg_dict.get(expected_key)
        if expected_value is None:
            continue
        actual_value = float(model_cfg.model.get(model_key, 1.0))
        if not math.isclose(actual_value, float(expected_value), rel_tol=0.0, abs_tol=1.0e-8):
            raise ValueError(
                f"Selected checkpoint has model.{model_key}={actual_value:g}, "
                f"but this planning config requires {float(expected_value):g}. "
                "Train/select a matching checkpoint."
            )
    expected_color_weights_version = cfg_dict.get(
        "expected_color_channel_weights_version"
    )
    if expected_color_weights_version is not None:
        actual_color_weights_version = model_cfg.model.get(
            "color_channel_weights_version"
        )
        if str(actual_color_weights_version) != str(expected_color_weights_version):
            raise ValueError(
                "Selected checkpoint has color_channel_weights_version="
                f"{actual_color_weights_version!r}, but this planning config "
                f"requires {str(expected_color_weights_version)!r}. "
                "Train/select a matching checkpoint."
            )
    expected_outline = cfg_dict.get("expected_draw_block_outline")
    if expected_outline is not None:
        actual_outline = bool(model_cfg.env.dataset.get("draw_block_outline", True))
        if actual_outline != bool(expected_outline):
            raise ValueError(
                "Selected checkpoint has env.dataset.draw_block_outline="
                f"{actual_outline}, but this planning config requires "
                f"{bool(expected_outline)}."
            )
    expected_reference_version = cfg_dict.get("expected_fixed_reference_version")
    if expected_reference_version is not None:
        actual_reference_version = str(
            model_cfg.model.get("fixed_reference_version", "")
        )
        if actual_reference_version != str(expected_reference_version):
            raise ValueError(
                "Selected checkpoint has fixed_reference_version="
                f"{actual_reference_version!r}, but this planning config requires "
                f"{str(expected_reference_version)!r}."
            )
    expected_reference_frames = cfg_dict.get("expected_fixed_reference_num_frames")
    if expected_reference_frames is not None:
        actual_reference_frames = int(
            model_cfg.model.get("fixed_reference_num_frames", 32)
        )
        if actual_reference_frames != int(expected_reference_frames):
            raise ValueError(
                "Selected checkpoint has fixed_reference_num_frames="
                f"{actual_reference_frames}, but this planning config requires "
                f"{int(expected_reference_frames)}."
            )
    expected_object_sizes = cfg_dict.get("expected_object_branch_spatial_sizes")
    if expected_object_sizes is not None:
        configured_object_scales = model_cfg.model.get(
            "predictor_object_branch_scales"
        )
        actual_object_sizes = (
            [16, 12, 6]
            if configured_object_scales is None
            else [int(scale.spatial_size) for scale in configured_object_scales]
        )
        required_object_sizes = [int(size) for size in expected_object_sizes]
        if actual_object_sizes != required_object_sizes:
            raise ValueError(
                "Selected checkpoint has predictor object-branch spatial sizes "
                f"{actual_object_sizes}, but this planning config requires "
                f"{required_object_sizes}. Train/select a matching checkpoint."
            )

    seed(cfg_dict["seed"])
    sample_dsets, trajectory_dsets = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
        random_transition_sampling=False,
        causal_transition_sampling=True,
    )
    dset = trajectory_dsets["valid"]
    retrieval_dataset = trajectory_dsets["train"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model(
        model_ckpt,
        model_cfg,
        num_action_repeat,
        device=device,
        dset=dset,
        # Bank loading is handled below so a missing derived cache can be
        # rebuilt from the raw training trajectories.
        load_retrieval_bank=False,
    )
    if load_retrieval_bank and model.predictor_architecture != "sequential":
        cache_root = Path(cfg_dict.get("inference_cache_dir", launch_cwd / "slot-ac"))
        if not cache_root.is_absolute():
            cache_root = (launch_cwd / cache_root).resolve()
        bank_path = _load_or_build_retrieval_bank(
            model=model,
            model_ckpt=model_ckpt,
            model_cfg=model_cfg,
            train_dataset=sample_dsets["train"],
            device=device,
            cache_root=cache_root,
            auto_build=bool(cfg_dict.get("auto_build_retrieval_bank", True)),
        )
        print(f"Using retrieval bank: {bank_path}")
    checkpoint_stat = model_ckpt.stat()
    model._checkpoint_identity = (
        f"{model_ckpt.resolve()}:{checkpoint_stat.st_size}:"
        f"{checkpoint_stat.st_mtime_ns}"
    )
    inference_retrieval_cfg = cfg_dict.get("inference_retrieval")
    if inference_retrieval_cfg is not None:
        configure_retrieval = getattr(model, "configure_inference_retrieval", None)
        if configure_retrieval is None:
            raise TypeError(
                "inference_retrieval is configured, but the loaded model does not "
                "support retrieval contexts"
            )
        configure_retrieval(
            num_context=inference_retrieval_cfg["num_context"],
            candidate_pool_size=inference_retrieval_cfg["candidate_pool_size"],
        )
    return {
        "device": device,
        "model_cfg": model_cfg,
        "model": model,
        "dset": dset,
        "retrieval_dataset": retrieval_dataset,
    }


def _close_plan_env(env):
    close = getattr(env, "close", None)
    if callable(close):
        close()
        return
    for child in getattr(env, "envs", ()):
        child_close = getattr(child, "close", None)
        if callable(child_close):
            child_close()


def run_planning_with_runtime(cfg_dict, runtime, wandb_run=None):
    """Run one case batch with a resident model and fresh planner/environment state."""
    model_cfg = runtime["model_cfg"]
    seed(cfg_dict["seed"])

    # use serial vector env for environments that are not fork-safe
    if model_cfg.env.name in ("wall", "deformable_env", "point_maze"):
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    try:
        plan_workspace = PlanWorkspace(
            cfg_dict=cfg_dict,
            wm=runtime["model"],
            dset=runtime["dset"],
            retrieval_dataset=runtime["retrieval_dataset"],
            env=env,
            env_name=model_cfg.env.name,
            frameskip=model_cfg.frameskip,
            wandb_run=wandb_run,
        )
        return plan_workspace.perform_planning()
    finally:
        _close_plan_env(env)


def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None
    runtime = build_planning_runtime(cfg_dict)
    return run_planning_with_runtime(cfg_dict, runtime, wandb_run=wandb_run)


@hydra.main(version_base=None, config_path="../conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    # Resolve Hydra/OmegaConf values while the Hydra runtime context is still
    # active. In particular goal_file_path may contain ${hydra:runtime.cwd};
    # passing an unresolved string into planning makes open() use it literally.
    cfg_dict = cfg_to_dict(cfg, resolve=True)
    cfg_dict["wandb_logging"] = bool(cfg.get("wandb_logging", False))
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
