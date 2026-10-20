import sys
import unittest
from pathlib import Path

import torch


PLANAR_ROOT = Path(__file__).resolve().parents[1] / "workflows" / "planar"
sys.path.insert(0, str(PLANAR_ROOT))

from datasets.wall_dset import WallDataset
from env.wall.envs.wall import DotWall


class WallRenderOrderTest(unittest.TestCase):
    def test_dataset_ball_is_composited_over_wall(self):
        dataset = WallDataset.__new__(WallDataset)
        dataset.render_size = 384
        scale = dataset.render_size / 65.0
        wall_x, door_y = 32.0, 30.0
        # Deliberately place the marker center on an opaque part of the wall;
        # this isolates draw order from the collision implementation.
        state = torch.tensor([[wall_x, 20.0, 0.0, 0.0]]) * scale
        image = dataset._render_states(
            state,
            torch.tensor([wall_x * scale]),
            torch.tensor([door_y * scale]),
        )[0]
        x, y = round(wall_x * scale), round(20.0 * scale)
        self.assertEqual(image[:, y, x].tolist(), [255, 0, 0])

    def test_live_env_ball_is_composited_over_wall(self):
        env = DotWall.__new__(DotWall)
        env.device = "cpu"
        wall = torch.ones((3, 3))
        dot = torch.zeros((3, 3))
        dot[1, 1] = 1.0
        image = env.channels_to_img(wall, dot)
        self.assertEqual(image[:, 1, 1].tolist(), [255, 0, 0])


if __name__ == "__main__":
    unittest.main()
