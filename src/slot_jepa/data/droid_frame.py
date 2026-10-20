# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from logging import getLogger
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data

cv2.setNumThreads(0)

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=4,
    crop_size=224,
    rank=0,
    world_size=1,
    camera_views=None,
    stereo_view=False,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    collator=None,
    transform=None,
    camera_frame=True,
    tubelet_size=1,
):
    dataset = DROIDFrameDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        frameskip=tubelet_size,
        camera_frame=camera_frame,
        camera_views=camera_views,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )

    if rank == 0:
        logger.info("DROID frame dataset loader created")
    return data_loader, dist_sampler


class DROIDFrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        frames_per_clip=16,
        fps=4,
        transform=None,
        frameskip=1,
        camera_frame=True,
        camera_views=None,
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.transform = transform
        self.frameskip = frameskip
        self.camera_frame = camera_frame
        self.selected_views = list(
            camera_views or ["exterior_image_1_left", "exterior_image_2_left", "wrist_image_left"]
        )

        samples = list(pd.read_csv(data_path, header=None, delimiter=" ").values[:, 0])
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]

        loaded_video = False
        while not loaded_video:
            try:
                buffer, actions, states, extrinsics, indices = self.load_episode(path)
                loaded_video = True
            except Exception as exc:
                logger.info(f"Encountered exception when loading frame episode {path=} {exc=}")
                loaded_video = False
                index = np.random.randint(self.__len__())
                path = self.samples[index]

        return buffer, actions, states, extrinsics, indices

    def load_episode(self, episode_dir):
        episode_dir = Path(episode_dir)
        label_path = episode_dir / "labels.json"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing labels.json under {episode_dir}")

        with label_path.open("r", encoding="utf-8") as fin:
            meta = json.load(fin)

        frames = meta["frames"]
        total_frames = len(frames)
        if total_frames < self.frames_per_clip:
            raise ValueError(
                f"Episode too short for requested clip length: {episode_dir} "
                f"({total_frames=} < {self.frames_per_clip=})"
            )

        max_start = total_frames - self.frames_per_clip
        start = 0 if max_start == 0 else np.random.randint(0, max_start + 1)
        end = start + self.frames_per_clip
        selected = frames[start:end]

        image_root = episode_dir / "images"
        buffers = []
        states_all = []
        extrinsics_all = []
        actions_all = []

        for camera_key in self.selected_views:
            view_buffer = []
            view_states = []
            view_extrinsics = []
            view_actions = []
            for frame in selected:
                view_meta = frame.get("views", {}).get(camera_key)
                if view_meta is None:
                    raise KeyError(f"Missing view `{camera_key}` metadata in {episode_dir}")
                image_path = image_root / camera_key / view_meta["img_name"]
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Failed to read image {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                view_buffer.append(image)

                if self.camera_frame:
                    view_states.append(view_meta["ee_state7_camera_frame"])
                    view_extrinsics.append(view_meta["camera_extrinsics"])
                else:
                    view_states.append(frame["ee_state7"])
                    view_extrinsics.append(frame["camera_extrinsics"])

            for frame in selected[:-1]:
                if self.camera_frame:
                    action_map = frame.get("action_from_camera_frame_state_diff_by_view")
                    action = None if action_map is None else action_map.get(camera_key)
                else:
                    action = frame["action_from_robot_frame_state_diff"]
                if action is None:
                    raise ValueError(
                        f"Encountered None action for view `{camera_key}` before last frame in {episode_dir}"
                    )
                view_actions.append(action)

            view_buffer = np.stack(view_buffer, axis=0)
            if self.transform is not None:
                view_buffer = self.transform(view_buffer)
            else:
                view_buffer = torch.from_numpy(view_buffer).permute(3, 0, 1, 2).contiguous()
            buffers.append(view_buffer)
            states_all.append(np.asarray(view_states, dtype=np.float32))
            extrinsics_all.append(np.asarray(view_extrinsics, dtype=np.float32))
            actions_all.append(np.asarray(view_actions, dtype=np.float32))

        buffer = torch.stack(buffers, dim=0) if torch.is_tensor(buffers[0]) else np.stack(buffers, axis=0)
        states = np.stack(states_all, axis=0)
        extrinsics = np.stack(extrinsics_all, axis=0)
        actions = np.stack(actions_all, axis=0)

        indices = np.asarray([frame["original_step_idx"] for frame in selected], dtype=np.int64)
        return buffer, actions, states, extrinsics, indices
