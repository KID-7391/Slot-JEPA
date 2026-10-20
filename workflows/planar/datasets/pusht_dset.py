import os
import torch
import decord
import pickle
import numpy as np
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Callable, Optional
from .traj_dset import TrajDataset, TrajSlicerDataset
from typing import Optional, Callable, Any
decord.bridge.set_bridge("torch")

# precomputed dataset stats
ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
ACTION_STD = torch.tensor([0.2019, 0.2002])
STATE_MEAN = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 1.9584, -2.93032027,  2.54307914])
STATE_STD = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094])
PROPRIO_MEAN = torch.tensor([236.6155, 264.5674, -2.93032027,  2.54307914])
PROPRIO_STD = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])

# Stable across train/eval manifests and independent of their local ordering.
# Retrieval uses this categorical id to require contexts with the same object
# shape as the query; color is intentionally not part of the id.
PUSHT_SHAPE_IDS = {
    "+": 0,
    "I": 1,
    "L": 2,
    "T": 3,
    "Z": 4,
    "small_tee": 5,
    "square": 6,
}

class PushTDataset(TrajDataset):
    def __init__(
        self,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        data_path: str = "data/pusht_dataset",
        normalize_action: bool = True,
        relative=True,
        action_scale=100.0,
        with_velocity: bool = True, # agent's velocity
        render_from_state: bool = True,
        render_size: int = 384,
        draw_block_outline: bool = True,
        manifest_file: Optional[str] = None,
        normalization_stats=None,
        color_override_palette=None,
        color_override_seed: int = 0,
    ):  
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalize_action = normalize_action
        self.render_from_state = bool(render_from_state)
        self.render_size = int(render_size)
        self.draw_block_outline = bool(draw_block_outline)
        self._render_env = None
        self._render_shape = None
        self._render_color = None
        manifest = None
        if manifest_file is not None:
            manifest = torch.load(
                self.data_path / manifest_file, map_location="cpu", weights_only=False
            )
            self.states = torch.as_tensor(manifest["states"]).float()
            self.actions = torch.as_tensor(manifest["actions"]).float()
        elif relative:
            self.states = torch.load(self.data_path / "states.pth").float()
            self.actions = torch.load(self.data_path / "rel_actions.pth")
        else:
            self.states = torch.load(self.data_path / "states.pth").float()
            self.actions = torch.load(self.data_path / "abs_actions.pth")
        self.actions = self.actions.float()
        # A transition manifest naturally stores T+1 states and T actions.
        # Keep the trajectory dataset's frame-aligned action convention by
        # adding an unused terminal zero action, as the Wall manifest loader
        # does. This also makes full-trajectory inspection safe at frame T.
        if manifest is not None and self.actions.shape[1] + 1 == self.states.shape[1]:
            self.actions = torch.cat(
                (self.actions, torch.zeros_like(self.actions[:, :1])), dim=1
            )
        self.actions = self.actions / action_scale  # scaled back up in env

        if manifest is not None:
            manifest_lengths = manifest.get("seq_lengths")
            self.seq_lengths = (
                [self.states.shape[1]] * len(self.states)
                if manifest_lengths is None
                else torch.as_tensor(manifest_lengths).long().tolist()
            )
            manifest_contacts = manifest.get("contact_steps")
            self.contact_steps = (
                None
                if manifest_contacts is None
                else torch.as_tensor(manifest_contacts, dtype=torch.bool)
            )
        else:
            self.contact_steps = None
            with open(self.data_path / "seq_lengths.pkl", "rb") as f:
                self.seq_lengths = pickle.load(f)
        segments_file = self.data_path / "segments.pkl"
        if segments_file.exists():
            with open(segments_file, "rb") as f:
                self.segments = pickle.load(f)
        else:
            self.segments = None
        
        # load shapes, assume all shapes are 'T' if file not found
        shapes_file = self.data_path / "shapes.pkl"
        if manifest is not None:
            self.shapes = list(manifest["shapes"])
            self.colors = list(manifest.get("colors", ["LightSlateGray"] * len(self.states)))
        elif shapes_file.exists():
            with open(shapes_file, 'rb') as f:
                shapes = pickle.load(f)
                self.shapes = shapes
            self.colors = ["LightSlateGray"] * len(self.states)
        else:
            self.shapes = ['T'] * len(self.states)
            self.colors = ["LightSlateGray"] * len(self.states)

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        if self.contact_steps is not None:
            self.contact_steps = self.contact_steps[:n]
        self.seq_lengths = self.seq_lengths[:n]
        if self.segments is not None:
            self.segments = self.segments[:n]
        self.shapes = self.shapes[:n]
        self.colors = self.colors[:n]
        if color_override_palette is not None:
            palette = [str(color) for color in color_override_palette]
            if not palette or any(not color for color in palette):
                raise ValueError("color_override_palette must contain colors")
            color_rng = np.random.default_rng(int(color_override_seed))
            color_indices = color_rng.integers(0, len(palette), size=n)
            self.colors = [palette[int(index)] for index in color_indices]
        self.color_override_palette = (
            None
            if color_override_palette is None
            else tuple(str(color) for color in color_override_palette)
        )
        self.color_override_seed = int(color_override_seed)
        unknown_shapes = sorted(set(self.shapes).difference(PUSHT_SHAPE_IDS))
        if unknown_shapes:
            raise ValueError(
                f"PushT manifest contains shapes without stable ids: {unknown_shapes}"
            )
        self.layout_ids = torch.tensor(
            [PUSHT_SHAPE_IDS[shape] for shape in self.shapes], dtype=torch.long
        )
        self.proprios = self.states[..., :2].clone()  # For pusht, first 2 dim of states is proprio
        # load velocities and update states and proprios
        self.with_velocity = with_velocity
        if with_velocity and self.states.shape[-1] == 5:
            self.velocities = torch.load(self.data_path / "velocities.pth")
            self.velocities = self.velocities[:n].float()
            self.states = torch.cat([self.states, self.velocities], dim=-1)
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        elif with_velocity:
            self.velocities = self.states[..., 5:7]
            self.proprios = torch.cat([self.proprios, self.velocities], dim=-1)
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        # Folder-based released datasets store statistics at the task root so
        # train and validation use one distribution without hard-coded values.
        stats_path = self.data_path.parent / "normalization_stats.pt"
        if normalization_stats is None and stats_path.exists():
            stats_payload = torch.load(stats_path, map_location="cpu")
            normalization_stats = tuple(
                stats_payload[name]
                for name in (
                    "action_mean",
                    "action_std",
                    "state_mean",
                    "state_std",
                    "proprio_mean",
                    "proprio_std",
                )
            )

        if normalization_stats is not None:
            (
                self.action_mean,
                self.action_std,
                self.state_mean,
                self.state_std,
                self.proprio_mean,
                self.proprio_std,
            ) = [torch.as_tensor(value).clone() for value in normalization_stats]
        elif normalize_action and manifest is not None:
            self.action_mean = self.actions.mean(dim=(0, 1))
            self.action_std = self.actions.std(dim=(0, 1)).clamp_min(1.0e-6)
            self.state_mean = self.states.mean(dim=(0, 1))
            self.state_std = self.states.std(dim=(0, 1)).clamp_min(1.0e-6)
            self.proprio_mean = self.proprios.mean(dim=(0, 1))
            self.proprio_std = self.proprios.std(dim=(0, 1)).clamp_min(1.0e-6)
        elif normalize_action:
            self.action_mean = ACTION_MEAN
            self.action_std = ACTION_STD
            self.state_mean = STATE_MEAN[:self.state_dim]
            self.state_std = STATE_STD[:self.state_dim]
            self.proprio_mean = PROPRIO_MEAN[:self.proprio_dim]
            self.proprio_std = PROPRIO_STD[:self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def _get_render_env(self, shape, color, initial_state):
        """Lazily create a native-resolution renderer, recreating it for a new shape."""
        if self._render_env is None or self._render_shape != shape or self._render_color != color:
            if self._render_env is not None:
                self._render_env.close()
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            from env.pusht.pusht_env import PushTEnv

            self._render_env = PushTEnv(
                render_size=self.render_size,
                reset_to_state=np.asarray(initial_state, dtype=np.float64),
                with_velocity=self.with_velocity,
                with_target=True,
                draw_block_outline=self.draw_block_outline,
                shape=shape,
                color=color,
            )
            self._render_env.reset()
            self._render_shape = shape
            self._render_color = color
        return self._render_env

    def _render_states(self, states, shape, color="LightSlateGray"):
        states = torch.as_tensor(states).detach().cpu().numpy()
        env = self._get_render_env(shape, color, states[0])
        images = []
        for state in states:
            env._set_state(np.asarray(state, dtype=np.float64))
            images.append(torch.from_numpy(np.asarray(env.render(mode="rgb_array")).copy()))
        return torch.stack(images)

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        frames = np.asarray(list(frames), dtype=np.int64)
        if self.segments is None:
            video_path = self.data_path / "obses" / f"episode_{idx:03d}.mp4"
            video_frames = frames
        else:
            segment = self.segments[idx]
            video_path = Path(segment["video_path"])
            if not video_path.is_absolute():
                video_path = (self.data_path / video_path).resolve()
            video_frames = frames + int(segment["start"])
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        shape = self.shapes[idx]
        color = self.colors[idx]

        if self.render_from_state:
            image = self._render_states(state, shape, color)
        else:
            reader = VideoReader(str(video_path), num_threads=1)
            image = reader.get_batch(video_frames)  # THWC
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {
            'shape': shape,
            'color': color,
            # Generic Slot-AC retrieval calls this field layout_id, while for
            # PushObj its semantics are specifically the object shape id.
            'layout_id': self.layout_ids[idx],
        }

    def get_causal_window(self, idx, start, end, frameskip):
        """Render only the observations needed by one causal training window."""
        observation_indices = list(range(start, end, frameskip))
        obs, _, state, env_info = self.get_frames(idx, observation_indices)
        # Preserve every primitive action between consecutive model-step
        # observations. TrajSlicerDataset concatenates this block afterwards.
        act = self.actions[idx, start : end - 1]
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


def load_pusht_slice_train_val(
    transform,
    n_rollout=50,
    data_path="data/pusht_dataset",
    normalize_action=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
    random_transition_sampling=False,
    random_samples_per_trajectory=None,
    causal_transition_sampling=False,
    max_causal_windows_per_trajectory=None,
    non_overlapping_causal_windows=False,
    render_from_state=True,
    render_size=384,
    draw_block_outline=True,
    split_mode="folder",
    train_color_palette=None,
    train_color_seed=0,
    color_override=None,
    validation_data_paths=None,
    validation_n_rollout=None,
    validation_non_overlapping_causal_windows=None,
):
    all_split_color_palette = (
        None if color_override is None else [str(color_override)]
    )
    train_color_override_palette = (
        all_split_color_palette
        if all_split_color_palette is not None
        else train_color_palette
    )
    if split_mode == "manifest":
        train_dset = PushTDataset(
            n_rollout=n_rollout, transform=transform, data_path=data_path,
            normalize_action=normalize_action, with_velocity=with_velocity,
            render_from_state=render_from_state, render_size=render_size,
            draw_block_outline=draw_block_outline,
            manifest_file="train_manifest.pt",
            color_override_palette=train_color_override_palette,
            color_override_seed=train_color_seed,
        )
        stats = (
            train_dset.action_mean, train_dset.action_std,
            train_dset.state_mean, train_dset.state_std,
            train_dset.proprio_mean, train_dset.proprio_std,
        )
        val_dset = PushTDataset(
            n_rollout=validation_n_rollout, transform=transform, data_path=data_path,
            normalize_action=normalize_action, with_velocity=with_velocity,
            render_from_state=render_from_state, render_size=render_size,
            draw_block_outline=draw_block_outline,
            manifest_file="eval_manifest.pt",
            normalization_stats=stats if normalize_action else None,
            color_override_palette=all_split_color_palette,
        )
    elif split_mode == "folder":
        train_dset = PushTDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path + "/train",
            normalize_action=normalize_action,
            with_velocity=with_velocity,
            render_from_state=render_from_state,
            render_size=render_size,
            draw_block_outline=draw_block_outline,
            color_override_palette=train_color_override_palette,
            color_override_seed=train_color_seed,
        )
        val_dset = PushTDataset(
            n_rollout=validation_n_rollout,
            transform=transform,
            data_path=data_path + "/val",
            normalize_action=normalize_action,
            with_velocity=with_velocity,
            render_from_state=render_from_state,
            render_size=render_size,
            draw_block_outline=draw_block_outline,
            color_override_palette=all_split_color_palette,
        )
    else:
        raise ValueError(f"Unknown PushT split_mode: {split_mode!r}")

    # Reuse training statistics for every validation domain.  Recomputing
    # normalization on a small validation set would make its score incomparable.
    stats = (
        train_dset.action_mean,
        train_dset.action_std,
        train_dset.state_mean,
        train_dset.state_std,
        train_dset.proprio_mean,
        train_dset.proprio_std,
    )

    num_frames = num_hist + num_pred
    validation_non_overlapping = (
        non_overlapping_causal_windows
        if validation_non_overlapping_causal_windows is None
        else bool(validation_non_overlapping_causal_windows)
    )
    train_slices = TrajSlicerDataset(
        train_dset, num_frames, frameskip,
        random_transition_sampling=random_transition_sampling,
        random_samples_per_trajectory=random_samples_per_trajectory,
        causal_transition_sampling=causal_transition_sampling,
        max_causal_windows_per_trajectory=max_causal_windows_per_trajectory,
        non_overlapping_causal_windows=non_overlapping_causal_windows,
    )
    val_slices = TrajSlicerDataset(
        val_dset, num_frames, frameskip,
        random_transition_sampling=random_transition_sampling,
        random_samples_per_trajectory=random_samples_per_trajectory,
        causal_transition_sampling=causal_transition_sampling,
        max_causal_windows_per_trajectory=max_causal_windows_per_trajectory,
        non_overlapping_causal_windows=validation_non_overlapping,
    )

    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = train_dset
    traj_dset["valid"] = val_dset

    # Optional named validation sets use the training split's normalization
    # statistics but keep their own geometry, colors, caches, and retrieval
    # contexts.  Keys become ``valid_<name>`` in Trainer.
    for name, validation_path in dict(validation_data_paths or {}).items():
        name = str(name)
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(
                "validation_data_paths keys must contain only letters, digits, "
                f"and underscores; got {name!r}"
            )
        validation_dset = PushTDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=str(validation_path),
            normalize_action=normalize_action,
            with_velocity=with_velocity,
            render_from_state=render_from_state,
            render_size=render_size,
            draw_block_outline=draw_block_outline,
            manifest_file="eval_manifest.pt",
            normalization_stats=stats if normalize_action else None,
            color_override_palette=all_split_color_palette,
        )
        validation_slices = TrajSlicerDataset(
            validation_dset,
            num_frames,
            frameskip,
            random_transition_sampling=random_transition_sampling,
            random_samples_per_trajectory=random_samples_per_trajectory,
            causal_transition_sampling=causal_transition_sampling,
            max_causal_windows_per_trajectory=max_causal_windows_per_trajectory,
            non_overlapping_causal_windows=non_overlapping_causal_windows,
        )
        key = f"valid_{name}"
        if key in datasets:
            raise ValueError(f"Duplicate validation dataset key: {key}")
        datasets[key] = validation_slices
        traj_dset[key] = validation_dset
    return datasets, traj_dset
