from .envs.wall import DotWall
from .envs.wall import WallDatasetConfig
import numpy as np
import torch
import random
from gym import spaces
from torchvision import transforms  


from utils import aggregate_dct
from .data.wall_utils import generate_wall_layouts

ENV_ACTION_DIM = 2
BASE_IMG_SIZE = 65
WALL_IMG_SIZE = 384
WALL_SCALE = WALL_IMG_SIZE / BASE_IMG_SIZE


def _scale_int(value):
    return int(round(value * WALL_SCALE))


STATE_RANGES = np.array([
    [16.6840, 46.9885],
    [4.0083,25.2532]
]) * WALL_SCALE

DEFAULT_CFG = WallDatasetConfig(
    action_angle_noise=0.2,
    action_step_mean=1.0 * WALL_SCALE,
    action_step_std=0.4 * WALL_SCALE,
    action_lower_bd=0.2 * WALL_SCALE,
    action_upper_bd=1.8 * WALL_SCALE,
    batch_size=64,
    device='cuda',
    dot_std=1.7 * WALL_SCALE,
    border_wall_loc=_scale_int(5),
    fix_wall_batch_k=None,
    fix_wall=True,
    fix_door_location=_scale_int(30),
    fix_wall_location=_scale_int(32),
    exclude_wall_train='',
    exclude_door_train='',
    only_wall_val='',
    only_door_val='',
    wall_padding=_scale_int(20),
    door_padding=_scale_int(10),
    wall_width=_scale_int(6),
    door_space=_scale_int(4),
    collision_radius=1.0,
    num_train_layouts=-1,
    img_size=WALL_IMG_SIZE,
    max_step=1 * WALL_SCALE,
    n_steps=17,
    n_steps_reduce_factor=1,
    size=20000,
    val_size=10000,
    train=True,
    repeat_actions=1
)

class WallEnvWrapper(DotWall):
    def __init__(self, rng=42, wall_config=DEFAULT_CFG, fix_wall=True, cross_wall=False, fix_wall_location=None, fix_door_location=None, device='cpu', render_size=WALL_IMG_SIZE, **kwargs):
        super().__init__(rng, wall_config, fix_wall, cross_wall, fix_wall_location=fix_wall_location, fix_door_location=fix_door_location, device=device,**kwargs)
        self.action_dim = ENV_ACTION_DIM
        # Wall actions are signed 2-D velocity commands. DotWall applies each
        # command as ``position += 2 * action`` and reports the realized
        # velocity in state (which can differ after collision). The current
        # trajectory generator samples each Cartesian component independently
        # inside +/- action_upper_bd. Exposing
        # this Gym space keeps CEM's normalized-action projection aligned with
        # the actions used to generate the training set.
        action_limit = float(self.wall_config.action_upper_bd)
        self.action_space = spaces.Box(
            low=-action_limit,
            high=action_limit,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.render_size = int(render_size)
        if self.render_size < 1:
            raise ValueError("render_size must be positive")
        if self.render_size != self.img_size:
            raise ValueError(
                f"Wall now renders natively at {self.img_size}x{self.img_size}; "
                f"render_size must match, got {self.render_size}"
            )
        self.transform = transforms.Resize((self.render_size, self.render_size))

    def eval_state(self, goal_state, cur_state):
        success = np.linalg.norm(goal_state[:2] - cur_state[:2]) < 4.5 * WALL_SCALE
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
            'state_dist': state_dist,
        }

    @staticmethod
    def zero_velocity_state(state):
        """Return the same Wall state with its realized velocity cleared."""
        state = np.asarray(state).copy()
        if state.shape[-1] >= 4:
            state[..., 2:4] = 0
        return state

    def sample_random_init_goal_states(self, seed):
        """
        Return a random state
        """
        return self.generate_random_state(seed)
    
    def update_env(self, env_info): # change door and wall locations
        self.wall_config.fix_door_location = float(
            np.asarray(env_info["fix_door_location"]).item()
        )
        self.wall_config.fix_wall_location = float(
            np.asarray(env_info["fix_wall_location"]).item()
        )
        layouts, other_layouts = generate_wall_layouts(self.wall_config)
        self.layouts = layouts
        self.wall_x, self.hole_y = self._generate_wall()

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        """
        self.seed(seed)
        self.set_init_state(init_state)
        obs, state = self.reset()
        obs['visual'] = self.transform(obs['visual'])
        obs['visual'] = obs['visual'].permute(1, 2, 0)
        return obs, state

    def step_multiple(self, actions):
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            o['visual'] = self.transform(o['visual'])
            o['visual'] = o['visual'].permute(1, 2, 0)
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
