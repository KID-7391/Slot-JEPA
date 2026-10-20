import os
import torch
import decord
import numpy as np
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Callable, Optional
from .traj_dset import TrajDataset, TrajSubset, get_train_val_sliced, TrajSlicerDataset
from typing import Optional, Callable, Any
decord.bridge.set_bridge("torch")

class PointMazeDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/point_maze",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale=1.0,
        render_from_state: bool = True,
        render_size: int = 384,
        manifest_file: Optional[str] = None,
        layout_split: str = "train",
        normalization_stats=None,
        render_geometry_version: str = "legacy",
        center_u_stem: bool = False,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.render_from_state = bool(render_from_state)
        self.render_size = int(render_size)
        self._render_env = {}
        self.layout_split = str(layout_split)
        self.render_geometry_version = str(render_geometry_version)
        self.center_u_stem = bool(center_u_stem)
        manifest = None
        if manifest_file is not None:
            manifest = torch.load(
                self.data_path / manifest_file, map_location="cpu", weights_only=False
            )
            states = torch.as_tensor(manifest["states"]).float()
        else:
            states = torch.load(self.data_path / "states.pth").float()
        self.states = states
        self.actions = (
            torch.as_tensor(manifest["actions"]).float()
            if manifest is not None
            else torch.load(self.data_path / "actions.pth").float()
        )
        if manifest is not None and self.actions.shape[1] + 1 == self.states.shape[1]:
            self.actions = torch.cat(
                (self.actions, torch.zeros_like(self.actions[:, :1])), dim=1
            )
        self.actions = self.actions / action_scale  # scaled back up in env
        self.seq_lengths = (
            torch.full((len(states),), states.shape[1], dtype=torch.long)
            if manifest is not None
            else torch.load(self.data_path /'seq_lengths.pth')
        )
        self.layout_ids = (
            torch.as_tensor(manifest["layout_id"]).long()
            if manifest is not None
            else None
        )
        if self.layout_ids is not None and self.layout_ids.shape != (len(self.states),):
            raise ValueError(
                "PointMaze manifest must provide exactly one layout_id per "
                f"trajectory, got {tuple(self.layout_ids.shape)} for "
                f"{len(self.states)} trajectories"
            )
        if self.center_u_stem:
            if self.layout_ids is None:
                raise ValueError("center_u_stem requires a layout manifest")
            from maze_geometry import (
                GENERALIZATION_U_LAYOUTS,
                centered_u_state_offset,
            )

            self.states = self.states.clone()
            for layout_id in torch.unique(self.layout_ids).tolist():
                layout = GENERALIZATION_U_LAYOUTS[self.layout_split][int(layout_id)]
                offset = torch.as_tensor(
                    centered_u_state_offset(layout), dtype=self.states.dtype
                )
                self.states[self.layout_ids == int(layout_id), :, :4] += offset

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        if self.layout_ids is not None:
            self.layout_ids = self.layout_ids[:n]
        self.proprios = self.states.clone()
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalization_stats is not None:
            (
                self.action_mean,
                self.action_std,
                self.state_mean,
                self.state_std,
                self.proprio_mean,
                self.proprio_std,
            ) = [torch.as_tensor(value).clone() for value in normalization_stats]
        elif normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(self.actions, self.seq_lengths)
            self.state_mean, self.state_std = self.get_data_mean_std(self.states, self.seq_lengths)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(self.proprios, self.seq_lengths)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def __getstate__(self):
        """Never serialize live MuJoCo/EGL contexts into DataLoader workers."""
        state = self.__dict__.copy()
        state["_render_env"] = {}
        return state

    def transform_external_states(self, states, env_info):
        """Map raw manifest/benchmark states into this dataset's render frame."""
        states = np.asarray(states).copy()
        if not self.center_u_stem:
            return states
        from maze_geometry import (
            GENERALIZATION_U_LAYOUTS,
            centered_u_state_offset,
        )

        infos = env_info if isinstance(env_info, (list, tuple)) else [env_info]
        if states.shape[0] != len(infos):
            raise ValueError(
                f"Expected one env_info per state, got {len(infos)} for {states.shape}"
            )
        for index, info in enumerate(infos):
            split = str(info.get("layout_split", self.layout_split))
            layout_id = int(np.asarray(info["layout_id"]).item())
            layout = GENERALIZATION_U_LAYOUTS[split][layout_id]
            states[index, :4] += centered_u_state_offset(layout)
        return states

    def _get_render_env(self, layout_id=None):
        """Lazily create one lossless MuJoCo renderer in each loader worker."""
        cache_key = -1 if layout_id is None else int(layout_id)
        if cache_key not in self._render_env:
            os.environ.setdefault("MUJOCO_GL", "egl")
            from env.pointmaze.point_maze_wrapper import (
                GENERALIZATION_U_LAYOUTS,
                PointMazeWrapper,
            )

            u_layout = (
                None
                if layout_id is None
                else dict(GENERALIZATION_U_LAYOUTS[self.layout_split][cache_key])
            )
            if u_layout is not None and self.center_u_stem:
                u_layout["center_inner_stem"] = True
            env = PointMazeWrapper(
                return_value="state",
                with_target=False,
                render_size=self.render_size,
                randomize_ball_position=False,
                u_layout=u_layout,
            )
            env.prepare_for_render()
            self._render_env[cache_key] = env
        return self._render_env[cache_key]

    def _render_states(self, states, layout_id=None):
        if layout_id is not None and torch.as_tensor(layout_id).numel() != 1:
            raise ValueError(
                "A PointMaze trajectory must use one scalar layout_id for all "
                f"frames, got shape {tuple(torch.as_tensor(layout_id).shape)}"
            )
        env = self._get_render_env(layout_id)
        states = torch.as_tensor(states).detach().cpu().numpy()
        if len(states) == 0:
            raise ValueError("Cannot render an empty PointMaze trajectory")

        # mujoco_py keeps a framebuffer per EGL render context.  This dataset
        # caches one simulator for each layout; after another layout/context
        # was active, its first render can return that context's previous frame.
        # Prime the selected context with this trajectory's first state and
        # discard the result, so every returned frame has the same geometry.
        env.set_state(states[0, :2], states[0, 2:4])
        env.set_marker()
        env._render_frame()
        images = []
        for state in states:
            env.set_state(state[:2], state[2:4])
            env.set_marker()
            images.append(torch.from_numpy(np.asarray(env._render_frame()).copy()))
        return torch.stack(images)
    
    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            traj_data = data[traj, :traj_len]
            all_data.append(traj_data)
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        # Random-position Maze data intentionally uses zero velocities and
        # zero actions. Clamp constant dimensions so normalization maps them
        # to zero instead of producing 0 / 0 = NaN.
        data_std = torch.std(all_data, dim=0).clamp_min(1.0e-6)
        return data_mean, data_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        frames = list(frames)
        obs_dir = self.data_path / "obses"
        tensor_path = obs_dir / f"episode_{idx:03d}.pth"
        video_path = obs_dir / f"episode_{idx:03d}.mp4"
        if self.render_from_state:
            image = self._render_states(
                self.states[idx, frames],
                None if self.layout_ids is None else self.layout_ids[idx],
            )
        elif video_path.exists():
            image = VideoReader(str(video_path), num_threads=1).get_batch(list(frames))
        else:
            image = torch.load(tensor_path)
            image = image[frames]
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]

        # THWC for both legacy .pth tensors and native-resolution MP4s.
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {
            "visual": image,
            "proprio": proprio
        }
        env_info = {}
        if self.layout_ids is not None:
            env_info = {
                "layout_id": self.layout_ids[idx],
                "layout_split": self.layout_split,
            }
        return obs, act, state, env_info

    def get_causal_window(self, idx, start, end, frameskip):
        """Read only the RGB frames needed by one causal training window."""
        observation_indices = list(range(start, end, frameskip))
        obs_dir = self.data_path / "obses"
        tensor_path = obs_dir / f"episode_{idx:03d}.pth"
        video_path = obs_dir / f"episode_{idx:03d}.mp4"
        if self.render_from_state:
            image = self._render_states(
                self.states[idx, observation_indices],
                None if self.layout_ids is None else self.layout_ids[idx],
            )
        elif video_path.exists():
            image = VideoReader(str(video_path), num_threads=1).get_batch(
                observation_indices
            )
        else:
            image = torch.load(tensor_path)[observation_indices]
        image = rearrange(image / 255.0, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {
            "visual": image,
            "proprio": self.proprios[idx, observation_indices],
        }
        # Keep all primitive actions between the sampled observations; the
        # slicer concatenates each consecutive `frameskip` block below.
        act = self.actions[idx, start : end - 1]
        state = self.states[idx, observation_indices]
        env_info = {}
        if self.layout_ids is not None:
            env_info = {
                "layout_id": self.layout_ids[idx],
                "layout_split": self.layout_split,
            }
        return obs, act, state, env_info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0
        
def load_point_maze_slice_train_val(
    transform,
    n_rollout=50,
    data_path='data/pusht_dataset',
    normalize_action=False,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    random_transition_sampling: bool = False,
    random_samples_per_trajectory: Optional[int] = None,
    causal_transition_sampling=False,
    max_causal_windows_per_trajectory: Optional[int] = None,
    non_overlapping_causal_windows: bool = False,
    render_from_state: bool = True,
    render_size: int = 384,
    split_mode: str = "random",
    render_geometry_version: str = "legacy",
    center_u_stem: bool = False,
    validation_n_rollout: Optional[int] = None,
    validation_non_overlapping_causal_windows: Optional[bool] = None,
):
    validation_non_overlapping = (
        non_overlapping_causal_windows
        if validation_non_overlapping_causal_windows is None
        else bool(validation_non_overlapping_causal_windows)
    )
    if split_mode == "manifest":
        train_dset = PointMazeDataset(
            n_rollout=n_rollout, transform=transform, data_path=data_path,
            normalize_action=normalize_action, render_from_state=render_from_state,
            render_size=render_size, manifest_file="train_manifest.pt",
            layout_split="train",
            render_geometry_version=render_geometry_version,
            center_u_stem=center_u_stem,
        )
        stats = (
            train_dset.action_mean, train_dset.action_std,
            train_dset.state_mean, train_dset.state_std,
            train_dset.proprio_mean, train_dset.proprio_std,
        )
        val_dset = PointMazeDataset(
            n_rollout=validation_n_rollout, transform=transform, data_path=data_path,
            normalize_action=normalize_action, render_from_state=render_from_state,
            render_size=render_size, manifest_file="eval_manifest.pt",
            layout_split="eval", normalization_stats=stats if normalize_action else None,
            render_geometry_version=render_geometry_version,
            center_u_stem=center_u_stem,
        )
        num_frames = num_hist + num_pred
        kwargs = dict(
            frameskip=frameskip,
            random_transition_sampling=random_transition_sampling,
            random_samples_per_trajectory=random_samples_per_trajectory,
            causal_transition_sampling=causal_transition_sampling,
            max_causal_windows_per_trajectory=max_causal_windows_per_trajectory,
            non_overlapping_causal_windows=non_overlapping_causal_windows,
        )
        validation_kwargs = dict(kwargs)
        validation_kwargs["non_overlapping_causal_windows"] = validation_non_overlapping
        return (
            {
                "train": TrajSlicerDataset(train_dset, num_frames, **kwargs),
                "valid": TrajSlicerDataset(val_dset, num_frames, **validation_kwargs),
            },
            {"train": train_dset, "valid": val_dset},
        )
    if split_mode != "random":
        raise ValueError(f"Unknown PointMaze split_mode: {split_mode!r}")
    dset = PointMazeDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
        render_from_state=render_from_state,
        render_size=render_size,
        render_geometry_version=render_geometry_version,
        center_u_stem=center_u_stem,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset, 
        train_fraction=split_ratio, 
        num_frames=num_hist + num_pred, 
        frameskip=frameskip,
        random_transition_sampling=random_transition_sampling,
        random_samples_per_trajectory=random_samples_per_trajectory,
        causal_transition_sampling=causal_transition_sampling,
        max_causal_windows_per_trajectory=max_causal_windows_per_trajectory,
        non_overlapping_causal_windows=non_overlapping_causal_windows,
    )
    if validation_n_rollout is not None:
        dset_val = TrajSubset(
            dset_val, range(min(int(validation_n_rollout), len(dset_val)))
        )
    if (
        validation_n_rollout is not None
        or validation_non_overlapping != non_overlapping_causal_windows
    ):
        val_slices = TrajSlicerDataset(
            dset_val,
            num_hist + num_pred,
            frameskip,
            random_transition_sampling=random_transition_sampling,
            random_samples_per_trajectory=random_samples_per_trajectory,
            causal_transition_sampling=causal_transition_sampling,
            max_causal_windows_per_trajectory=max_causal_windows_per_trajectory,
            non_overlapping_causal_windows=validation_non_overlapping,
        )

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset
