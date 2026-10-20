import math
from typing import Optional

import torch
import gym
import numpy as np
import random

from ..data.wall import WallDatasetConfig
from ..data.wall_utils import generate_wall_layouts
from .utils import check_wall_intersect
from ..data.single import DotDataset


class DotWall(gym.Env):
    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        wall_config: Optional[WallDatasetConfig] = None,
        fix_wall: bool = True,
        cross_wall: bool = False,
        level: str = "normal",
        fix_wall_location: Optional[int] = None,
        fix_door_location: Optional[int] = None,
        device="cpu",
    ):
        super().__init__()
        self.wall_config = wall_config
        self.cross_wall = cross_wall
        self.level = level
        self.img_size = wall_config.img_size
        self.device = device
        self.dot_std = wall_config.dot_std
        self.padding = self.dot_std * 2
        self.collision_radius = float(wall_config.collision_radius)
        self.border_padding = wall_config.border_wall_loc - 1 + self.padding
        self.rng = rng or np.random.default_rng()
        if wall_config is not None:
            layouts, other_layouts = generate_wall_layouts(wall_config)
            self.layouts = layouts
        else:
            self.fix_wall = fix_wall
            self.fix_wall_location = fix_wall_location
            self.fix_door_location = fix_door_location
        self.wall_x, self.hole_y = self._generate_wall()
        self.left_wall_x = self.wall_x - self.wall_config.wall_width // 2
        self.right_wall_x = self.wall_x + self.wall_config.wall_width // 2

        self.reset_to_state = None
    
    def channels_to_img(self, wall_img, dot_img):
        h, w = wall_img.shape
        rgb_image = torch.ones((3, h, w), dtype=torch.uint8).to(self.device) * 255
        
        wall_mask = wall_img == 1
        rgb_image[:, wall_mask] = torch.tensor([0, 0, 0], dtype=torch.uint8).to(self.device).unsqueeze(1)

        # Draw the visual marker on top of the wall.  Its Gaussian footprint
        # is larger than the collision radius, so masking it by ``wall_img``
        # incorrectly hides the ball at otherwise valid contact states.
        red_intensity = (dot_img * 255).to(torch.uint8)

        rgb_image[0] = torch.maximum(rgb_image[0], red_intensity)
        rgb_image[1] = torch.minimum(rgb_image[1], 255 - red_intensity)
        rgb_image[2] = torch.minimum(rgb_image[2], 255 - red_intensity)
        return rgb_image


    def reset(self, location=None):
        if location is None:
            location = self.reset_to_state # self.reset_to_state can be None

        self.wall_img = self._render_walls(self.wall_x, self.hole_y)
        if location is None:
            # self._generate_start_and_target()
            # generate_random_state returns a (start, target) pair; reset needs
            # one physical state, not both endpoints stacked as a 2x2 tensor.
            dot_state, _ = self.generate_random_state()
            dot_state = torch.as_tensor(
                dot_state, dtype=torch.float32, device=self.device
            )
            self.dot_position = dot_state[:2]
            self.dot_velocity = dot_state[2:4]
        else:
            location = torch.as_tensor(
                location, dtype=torch.float32, device=self.device
            ).squeeze()
            if location.numel() not in (2, 4):
                raise ValueError(
                    f"Wall state must be [x, y] or [x, y, vx, vy], got {tuple(location.shape)}"
                )
            self.dot_position = location[:2]
            self.dot_velocity = (
                location[2:4] if location.numel() == 4 else torch.zeros_like(self.dot_position)
            )
        self.dot_img = self._render_dot(self.dot_position)
        self.observation = (self.wall_img + self.dot_img) / 2
        self.position_history = [self.dot_position.clone()]
        state = torch.cat([self.dot_position, self.dot_velocity])
        visual = self.channels_to_img(self.wall_img, self.dot_img)
        observation = {'visual': visual.float(), 'proprio': state.float()}
        return observation, state

    def step(self, action: torch.Tensor):
        previous_position = self.dot_position.clone()
        self.dot_position = self._calculate_next_position(action)
        # Actions command velocity, but collisions can clip the displacement.
        # Store the realized velocity so offline states and online rollouts agree.
        self.dot_velocity = (self.dot_position - previous_position) / 2.0
        self.position_history.append(self.dot_position.clone())
        self.dot_img = self._render_dot(self.dot_position)
        self.observation = (self.wall_img + self.dot_img) / 2
        visual = self.channels_to_img(self.wall_img, self.dot_img)
        state = torch.cat([self.dot_position, self.dot_velocity])
        observation = {'visual': visual.float(), 'proprio': state.float()}
        info = {}
        info['state'] = state
        info['pos_agent'] = self.dot_position
        return observation, 0, False, info # observation, reward, done, info

    def _calculate_next_position(self, action):
        next_dot_position = self._generate_transition(self.dot_position, action)
        next_dot_position = next_dot_position.squeeze()

        intersect, intersect_w_noise = check_wall_intersect(
            self.dot_position,
            next_dot_position,
            self.wall_x,
            self.hole_y,
            wall_width=self.wall_config.wall_width,
            door_space=self.wall_config.door_space,
            border_wall_loc=self.wall_config.border_wall_loc,
            img_size=self.wall_config.img_size,
            agent_radius=self.collision_radius,
        )
        if intersect is not None:
            next_dot_position = intersect_w_noise
        return next_dot_position

    def _generate_transition(self, location, action):
        next_location = location + action * 2   # [..., :-1] * action[..., -1]
        return next_location

    def _generate_wall(self):
        layout_codes = list(self.layouts.keys())
        weights = [1] * len(layout_codes)
        code = random.choices(layout_codes, weights=weights, k=1)[0]

        wall_loc = self.layouts[code]["wall_pos"]
        door_loc = self.layouts[code]["door_pos"]

        wall_loc = torch.tensor(wall_loc, device=self.device)
        door_loc = torch.tensor(door_loc, device=self.device)

        return wall_loc, door_loc
    
    def generate_random_state(self, seed=None):
        seed = random.randint(0, 2**32 - 1) if seed is None else seed
        rs = np.random.RandomState(seed)
        start_min_x = self.border_padding
        start_max_x = self.left_wall_x.item() - self.padding
        target_min_x = self.right_wall_x.item() + self.padding
        target_max_x = self.img_size - 1 - self.border_padding
        min_y, max_y = (
            self.border_padding,
            self.img_size - 1 - self.border_padding,
        )
        start_x = rs.uniform(low=start_min_x, high=start_max_x)
        target_x = rs.uniform(low=target_min_x, high=target_max_x)

        start_y = rs.uniform(low=min_y, high=max_y)
        target_y = rs.uniform(low=min_y, high=max_y)

        if rs.uniform(low=0, high=1) < 0.5:  # inverse travel direction 50% of time
            start_x, target_x = target_x, start_x
        return (
            np.array([start_x, start_y, 0.0, 0.0], dtype=np.float32),
            np.array([target_x, target_y, 0.0, 0.0], dtype=np.float32),
        )
        # return np.array([start_x, start_y])

    def _generate_start_and_target(self):
        # We leave 2 * self.dot_std margin when generating state, and don't let the
        # dot approach the border.
        n_steps = self.wall_config.n_steps
        if self.cross_wall:
            if self.level == "easy":
                # we make sure start and goal are within (n_steps/2) steps from door

                avg_dist_n_steps = n_steps * self.wall_config.action_step_mean

                assert (
                    self.wall_config.wall_padding
                    - self.wall_config.wall_width // 2
                    - self.wall_config.border_wall_loc
                    >= math.ceil(avg_dist_n_steps * 3 / 4)
                )

                start_min_x = self.left_wall_x - math.ceil(avg_dist_n_steps * 3 / 4)
                start_max_x = self.left_wall_x - math.ceil(avg_dist_n_steps * 1 / 4)
                target_min_x = self.right_wall_x + math.ceil(avg_dist_n_steps * 1 / 4)
                target_max_x = self.right_wall_x + math.ceil(avg_dist_n_steps * 3 / 4)
                min_y = max(
                    self.hole_y - math.ceil(avg_dist_n_steps * 3 / 4),
                    self.border_padding,
                )
                max_y = min(
                    self.hole_y + math.ceil(avg_dist_n_steps * 3 / 4),
                    self.img_size - 1 - self.border_padding,
                )
            else:
                start_min_x = self.border_padding
                start_max_x = self.left_wall_x - self.padding
                target_min_x = self.right_wall_x + self.padding
                target_max_x = self.img_size - 1 - self.border_padding
                min_y, max_y = (
                    self.border_padding,
                    self.img_size - 1 - self.border_padding,
                )

            start_x = start_min_x + random.random() * (start_max_x - start_min_x)
            target_x = target_min_x + random.random() * (target_max_x - target_min_x)

            start_y = torch.tensor(
                min_y + random.random() * (max_y - min_y), device=self.device
            )
            target_y = torch.tensor(
                min_y + random.random() * (max_y - min_y), device=self.device
            )

            if random.random() < 0.5:  # inverse travel direction 50% of time
                start_x, target_x = target_x, start_x

            self.dot_position = torch.stack([start_x, start_y])
            self.target_position = torch.stack([target_x, target_y])
        else:
            raise NotImplementedError
            effective_range = (self.img_size - 1) - 2 * self.padding
            location = (
                torch.from_numpy(
                    self.rng.random(size=(4,)) * effective_range + self.padding
                )
                .to(self.device)
                .float()
            )
            if self.level == "easy":
                # make the target location to be within a certain distance from start
                min_dist_from_start = math.ceil(n_steps * 2 / 3)
                max_dist_from_start = math.ceil(n_steps * 3 / 2)
                # generate random angle
                angle = (torch.rand(1) * 2 * torch.pi).to(location.device)
                # generate a random distance c within the range
                dist = (
                    torch.rand(1) * (max_dist_from_start - min_dist_from_start)
                    + min_dist_from_start
                ).to(location.device)
                # set new x and y for goal
                location[2] = location[0] + dist * torch.cos(angle)
                location[3] = location[1] + dist * torch.sin(angle)
                location = torch.clamp(
                    location, min=self.padding, max=self.img_size - 1 - self.padding
                )

            self.dot_position = location[:2]
            self.target_position = location[2:]

    def _render_walls(self, wall_loc, hole_loc):
        # Generates an image of the wall with the door and specified wall thickness.
        x = torch.arange(0, self.img_size, device=self.device)
        y = torch.arange(0, self.img_size, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")

        # Calculate the range for the wall based on the wall_width
        half_width = self.wall_config.wall_width // 2

        # Create the wall mask centered at wall_loc with the given wall_width
        wall_mask = (grid_x >= (wall_loc - half_width)) & (
            grid_x <= (wall_loc + half_width)
        )

        # Door logic remains the same
        door_mask = (hole_loc - self.wall_config.door_space <= grid_y) & (
            grid_y <= hole_loc + self.wall_config.door_space
        )

        # Combine the wall and door masks
        res = wall_mask & ~door_mask

        # Convert boolean mask to float
        res = res.float()

        # A one-pixel border in the original 65x65 scene occupies roughly six
        # pixels at 384x384.  Keep the outer margin fixed while scaling the
        # visible wall band itself.
        border_wall_loc = self.wall_config.border_wall_loc
        border_wall_width = max(
            1, int(round(float(border_wall_loc) / 5.0))
        )
        res[:, border_wall_loc - border_wall_width : border_wall_loc] = 1
        res[:, -border_wall_loc : -border_wall_loc + border_wall_width] = 1
        res[border_wall_loc - border_wall_width : border_wall_loc, :] = 1
        res[-border_wall_loc : -border_wall_loc + border_wall_width, :] = 1

        return res

    def _render_dot(self, location):
        x = torch.linspace(
            0, self.img_size - 1, steps=self.img_size, device=self.device
        )
        y = torch.linspace(
            0, self.img_size - 1, steps=self.img_size, device=self.device
        )
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        c = torch.stack([xx, yy], dim=-1)
        squared_distance = (c - location).norm(dim=-1).pow(2)
        img = torch.exp(
            -squared_distance / (2 * self.dot_std * self.dot_std)
        ) / (2 * math.pi * self.dot_std * self.dot_std)
        # The Gaussian tail makes the visible dot substantially larger than its
        # nominal radius. Keep only the disk within one standard deviation.
        img = img.masked_fill(squared_distance > self.dot_std * self.dot_std, 0.0)
        
        max_value = img.max()
        img = img / max_value
        return img

    def _render_dot_and_wall(self, location):
        dot_img = self._render_dot(location)
        return torch.stack([dot_img, self.wall_img * dot_img.max()], dim=0)
    
    def set_init_state(self, init_state):
        self.reset_to_state = (
            torch.as_tensor(init_state, dtype=torch.float32)
            if init_state is not None
            else None
        )
        
    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
