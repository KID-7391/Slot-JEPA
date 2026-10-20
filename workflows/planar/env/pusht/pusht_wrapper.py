import os
import numpy as np
import gym
import pymunk
import shapely.geometry as sg
from env.pusht.pusht_env import PushTEnv, pymunk_to_shapely
from utils import aggregate_dct


# Smallest rotation that leaves each object's geometry unchanged. Shapes not
# listed here are asymmetric and therefore retain the usual 2*pi period.
ROTATION_SYMMETRY_PERIOD = {
    "+": np.pi / 2,
    "square": np.pi / 2,
    "I": np.pi,
    "Z": np.pi,
}


def symmetry_aware_angle_distance(angle_a, angle_b, shape):
    """Return the smallest equivalent angular distance for ``shape``."""
    period = ROTATION_SYMMETRY_PERIOD.get(shape, 2 * np.pi)
    delta = np.asarray(angle_a) - np.asarray(angle_b)
    return np.abs((delta + period / 2) % period - period / 2)


class PushTWrapper(PushTEnv):
    def __init__(
            self, 
            with_velocity=True,
            with_target=True,
            render_size=224,
            draw_block_outline=True,
            shape="T",
            color="LightSlateGray",
            color_override=None,
            use_geometric_center_eval=True,
        ):
        self.color_override = (
            None if color_override is None else str(color_override)
        )
        self.use_geometric_center_eval = bool(use_geometric_center_eval)
        super().__init__(
            with_velocity=with_velocity,
            with_target=with_target,
            render_size=render_size,
            draw_block_outline=draw_block_outline,
            shape=shape,
            color=self.color_override or color,
        )
        self.action_dim = self.action_space.shape[0]

    def agent_block_clearance(self, state):
        """Return signed pusher-to-object clearance for a physical state."""
        if self.space is None or getattr(self, "_sampling_shape", None) != self.shape:
            self._setup()
            self._sampling_shape = self.shape
        block_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        block_body.position = tuple(state[2:4])
        block_body.angle = float(state[4])
        block_geometry = pymunk_to_shapely(block_body, self.block.shapes)
        pusher_center = sg.Point(float(state[0]), float(state[1]))
        agent_radius = max(shape.radius for shape in self.agent.shapes)
        return float(block_geometry.distance(pusher_center) - agent_radius)

    def place_agent_near_block(self, state, seed, clearance):
        """Place the pusher at a seeded point just outside the object geometry."""
        if float(clearance) < 0.0:
            raise ValueError("clearance must be non-negative")
        if self.space is None or getattr(self, "_sampling_shape", None) != self.shape:
            self._setup()
            self._sampling_shape = self.shape
        state = np.asarray(state, dtype=np.float64).copy()
        block_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        block_body.position = tuple(state[2:4])
        block_body.angle = float(state[4])
        block_geometry = pymunk_to_shapely(block_body, self.block.shapes)
        agent_radius = max(shape.radius for shape in self.agent.shapes)
        center_region = block_geometry.buffer(agent_radius + float(clearance))
        boundary = center_region.boundary
        rng = np.random.default_rng(int(seed))
        for _ in range(128):
            candidate = boundary.interpolate(rng.random() * boundary.length)
            position = np.asarray(candidate.coords[0], dtype=np.float64)
            if np.all(position >= agent_radius) and np.all(
                position <= self.window_size - agent_radius
            ):
                state[:2] = position
                achieved = self.agent_block_clearance(state)
                if abs(achieved - float(clearance)) <= 1.0e-3:
                    return state
        raise RuntimeError(
            f"Could not place pusher near shape={self.shape!r} at "
            f"clearance={float(clearance):g}"
        )
    
    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)

        # The pusher and object positions used to be sampled independently,
        # allowing the circular pusher to start deep inside the object. Build
        # the configured shape once and reject candidates using its real
        # collision geometry.
        if self.space is None or getattr(self, "_sampling_shape", None) != self.shape:
            self._setup()
            self._sampling_shape = self.shape
        def has_normal_separation(state, clearance=5.0):
            return self.agent_block_clearance(state) >= clearance
        
        def generate_state():
            for _ in range(1000):
                values = [
                    rs.randint(50, 450),
                    rs.randint(50, 450),
                    rs.randint(100, 400),
                    rs.randint(100, 400),
                    rs.randn() * 2 * np.pi - np.pi,
                ]
                if self.with_velocity:
                    values.extend((0, 0))
                state = np.asarray(values)
                if has_normal_separation(state):
                    return state
            raise RuntimeError(
                f"Could not sample a non-overlapping PushT state for shape={self.shape!r}"
            )
        
        init_state = generate_state()
        goal_state = generate_state()
        
        return init_state, goal_state
    
    def update_env(self, env_info):
        self.shape = env_info['shape']
        self.color = self.color_override or env_info.get('color', 'LightSlateGray')

    def object_geometric_center(self, state):
        """Convert a recorded block-origin pose to its world-space centroid."""
        state = np.asarray(state, dtype=np.float64)
        if self.space is None or not hasattr(self, "block"):
            raise RuntimeError(
                "Object geometry must be initialized before evaluating a state"
            )
        local_centroid = np.asarray(self.block.center_of_gravity, dtype=np.float64)
        angle = state[..., 4]
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotated_offset = np.stack(
            (
                cosine * local_centroid[0] - sine * local_centroid[1],
                sine * local_centroid[0] + cosine * local_centroid[1],
            ),
            axis=-1,
        )
        return state[..., 2:4] + rotated_offset
    
    def eval_state(self, goal_state, cur_state):
        """
        Evaluate task success and controllable-state distance.

        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]

        ``state_dist`` deliberately contains only agent position and velocity.
        The block pose is an outcome of contact rather than a directly
        controllable state, and is already evaluated by the visual objective
        and task-success condition below.
        """
        agent_position_diff = goal_state[:2] - cur_state[:2]
        if self.use_geometric_center_eval:
            # Evaluate every PushT-family shape in the geometry-centered
            # coordinates used by the visual planning objective. Respect each
            # shape's rotational symmetry as well.
            block_position_diff = (
                self.object_geometric_center(goal_state)
                - self.object_geometric_center(cur_state)
            )
            angle_diff = symmetry_aware_angle_distance(
                goal_state[4], cur_state[4], self.shape
            )
        else:
            # Explicit legacy compatibility: evaluate the recorded body origin.
            block_position_diff = goal_state[2:4] - cur_state[2:4]
            raw_angle_diff = np.abs(goal_state[4] - cur_state[4])
            angle_diff = np.minimum(raw_angle_diff, 2 * np.pi - raw_angle_diff)
        pos_diff = np.linalg.norm(
            np.concatenate([agent_position_diff, block_position_diff], axis=-1)
        )
        success = pos_diff < 20 and angle_diff < np.pi / 9

        if goal_state.shape[-1] >= 7 and cur_state.shape[-1] >= 7:
            agent_velocity_diff = goal_state[5:7] - cur_state[5:7]
            controllable_diff = np.concatenate(
                [agent_position_diff, agent_velocity_diff], axis=-1
            )
            agent_velocity_dist = np.linalg.norm(agent_velocity_diff)
        else:
            controllable_diff = agent_position_diff
            agent_velocity_dist = 0.0
        state_dist = np.linalg.norm(controllable_diff)
        return {
            'success': success,
            'state_dist': state_dist,
            'agent_position_dist': np.linalg.norm(agent_position_diff),
            'agent_velocity_dist': agent_velocity_dist,
            'block_angle_dist_rad': angle_diff,
        }

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.seed(seed)
        self.reset_to_state = init_state
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
