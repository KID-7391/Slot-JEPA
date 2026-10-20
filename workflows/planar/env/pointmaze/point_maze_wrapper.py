import os
import numpy as np
import gym
from env.pointmaze.maze_model import MazeEnv
from maze_geometry import (
    GENERALIZATION_U_LAYOUTS,
    center_u_states,
    centered_u_state_offset,
)
from utils import aggregate_dct

STATE_RANGES = np.array([
    [0.39318362, 3.2198412],  # Range for first dimension
    [0.62660956, 3.2187355],  # Range for second dimension
    [-5.2262554, 5.2262554],  # Range for third dimension
    [-5.2262554, 5.2262554],  # Range for fourth dimension
    # [0.90001136, 3.0999563],  # Range for first dimension of target
    # [0.9000267, 3.0999668]    # Range for second dimension of target
])

# Optional empirical distribution retained for diagnostics.  The planning
# benchmark defaults to ``uniform`` below so that its states exactly match the
# original DINO-WM evaluation protocol.
TRAIN_CONTEXT_VELOCITY_MEAN = np.array(
    [-0.01495813, -0.01917224], dtype=np.float64
)
TRAIN_CONTEXT_VELOCITY_STD = np.array(
    [1.6417354, 1.8969163], dtype=np.float64
)
ENV_VELOCITY_LIMIT = 5.0

class PointMazeWrapper(MazeEnv):
    def __init__(self, random_state_velocity_distribution="uniform", **kwargs):
        super().__init__(**kwargs)
        self.action_dim = self.action_space.shape[0]
        self.random_state_velocity_distribution = str(
            random_state_velocity_distribution
        )
        if self.random_state_velocity_distribution not in ("train_normal", "uniform"):
            raise ValueError(
                "random_state_velocity_distribution must be 'train_normal' or "
                f"'uniform', got {self.random_state_velocity_distribution!r}"
            )
    
    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)

        def sample_velocity():
            if self.random_state_velocity_distribution == "uniform":
                velocity = rs.uniform(
                    low=STATE_RANGES[2:4, 0],
                    high=STATE_RANGES[2:4, 1],
                )
            else:
                velocity = rs.normal(
                    loc=TRAIN_CONTEXT_VELOCITY_MEAN,
                    scale=TRAIN_CONTEXT_VELOCITY_STD,
                )
            if self.random_state_velocity_distribution == "uniform":
                # Do not pre-clip: this is intentionally identical to the
                # original DINO-WM test-state sampler, including its bounds.
                return velocity
            return np.clip(velocity, -ENV_VELOCITY_LIMIT, ENV_VELOCITY_LIMIT)

        def generate_state():
            valid = False
            while not valid:
                x = rs.uniform(0.5, 3.1)
                y = rs.uniform(0.5, 3.1)
                valid = ((0.5 <= x <= 1.1 or 2.5 <= x <= 3.1) and (0.5 <= y <= 3.1))\
                        or ((1.1 < x < 2.5) and (2.5 <= y <= 3.1))
            velocity = sample_velocity()
            state = np.array([x, y, velocity[0], velocity[1]])
            return state

        init_state = generate_state()
        goal_state = generate_state()
        return init_state, goal_state
    
    def update_env(self, env_info):
        if "layout_id" not in env_info:
            return
        split = str(env_info.get("layout_split", "eval"))
        layout_id = int(np.asarray(env_info["layout_id"]).item())
        requested = dict(GENERALIZATION_U_LAYOUTS[split][layout_id])
        if self.u_layout is not None and self.u_layout.get("center_inner_stem", False):
            requested["center_inner_stem"] = True
        if self.u_layout == requested:
            return
        distribution = self.random_state_velocity_distribution
        render_size = self.render_size
        self.close()
        self.__init__(
            random_state_velocity_distribution=distribution,
            render_size=render_size,
            randomize_ball_position=False,
            return_value="state",
            with_target=False,
            u_layout=requested,
        )
    
    def eval_state(self, goal_state, cur_state):
        success = np.linalg.norm(goal_state[:2] - cur_state[:2]) < 0.5
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
            'state_dist': state_dist,
        }

    @staticmethod
    def zero_velocity_state(state):
        """Return a Maze state at the same position with zero linear velocity."""
        state = np.asarray(state).copy()
        if state.shape[-1] < 4:
            raise ValueError(
                f"PointMaze state must contain [x, y, vx, vy], got {state.shape}"
            )
        state[..., 2:4] = 0.0
        return state

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.prepare_for_render()
        self.seed(seed)
        self.set_init_state(init_state)
        obs, state = self.reset()
        return obs, state

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
