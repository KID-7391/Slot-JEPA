import torch
import numpy as np
import decord
from decord import VideoReader
from pathlib import Path
from typing import Callable, Optional
from .traj_dset import TrajDataset, TrajSubset, get_train_val_sliced, TrajSlicerDataset

decord.bridge.set_bridge("torch")

# precomputed dataset stats
ACTION_MEAN = torch.tensor([0.0006, 0.0015])
ACTION_STD = torch.tensor([0.4395, 0.4684])
STATE_MEAN = torch.tensor([0.7518, 0.9239, -3.9702e-05, 3.1550e-04])
STATE_STD = torch.tensor([1.0964, 1.2390, 1.3819, 1.5407])

class WallDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/TwoRoom",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale=1.0,
        render_from_state: bool = True,
        render_size: int = 384,
        render_geometry_version: str = "legacy",
        manifest_file: Optional[str] = None,
        source_coordinate_size: int = 65,
        normalization_stats=None,
    ):  
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.render_from_state = bool(render_from_state)
        self.render_size = int(render_size)
        # This tag is intentionally data-only: Hydra may pass it through the
        # dataset config, while Trainer uses it to invalidate slot/retrieval
        # caches whenever the analytic renderer or collision geometry changes.
        self.render_geometry_version = str(render_geometry_version)
        print("Loading wall dataset...")
        manifest = None
        if manifest_file is not None:
            manifest = torch.load(
                self.data_path / manifest_file, map_location="cpu", weights_only=False
            )
            states = torch.as_tensor(manifest["states"]).float()
            # Manifest coordinates are expressed in the generator's 65x65
            # logical canvas.  The live environment is native render_size.
            spatial_scale = self.render_size / float(source_coordinate_size)
            states = states * spatial_scale
            if states.shape[-1] == 2:
                velocity = torch.zeros_like(states)
                velocity[:, 1:] = (states[:, 1:] - states[:, :-1]) / 2.0
                states = torch.cat((states, velocity), dim=-1)
        else:
            states = torch.load(self.data_path / "states.pth")
        self.states = states
        self.proprios = self.states.clone()
        self.actions = (
            torch.as_tensor(manifest["actions"]).float() * spatial_scale
            if manifest is not None
            else torch.load(self.data_path / "actions.pth")
        )
        if manifest is not None and self.actions.shape[1] + 1 == self.states.shape[1]:
            self.actions = torch.cat(
                (self.actions, torch.zeros_like(self.actions[:, :1])), dim=1
            )
        self.actions = self.actions / action_scale
        if manifest is not None:
            steps = self.states.shape[1]
            self.door_locations = (
                torch.as_tensor(manifest["door_locations"]).float() * spatial_scale
            )[:, None].expand(-1, steps)
            self.wall_locations = (
                torch.as_tensor(manifest["wall_locations"]).float() * spatial_scale
            )[:, None].expand(-1, steps)
            self.layout_ids = torch.as_tensor(manifest["layout_id"]).long()
        else:
            self.door_locations = torch.load(self.data_path / "door_locations.pth")
            self.wall_locations = torch.load(self.data_path / "wall_locations.pth")
            self.layout_ids = None
        collision_types_path = self.data_path / "collision_types.pth"
        self.collision_types = (
            torch.load(collision_types_path).long()
            if collision_types_path.exists()
            else None
        )
        crossing_flags_path = self.data_path / "crossing_flags.pth"
        self.crossing_flags = (
            torch.load(crossing_flags_path).bool()
            if crossing_flags_path.exists()
            else None
        )

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)
            print(f"Loaded {n} rollouts")

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.proprios = self.proprios[:n]
        self.door_locations = self.door_locations[:n]
        self.wall_locations = self.wall_locations[:n]
        if self.layout_ids is not None:
            self.layout_ids = self.layout_ids[:n]
        if self.collision_types is not None:
            self.collision_types = self.collision_types[:n]
        if self.crossing_flags is not None:
            self.crossing_flags = self.crossing_flags[:n]

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]
        self.traj_len = self.states.shape[1]
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
            self.action_mean = self.actions.mean(dim=(0, 1))
            self.action_std = self.actions.std(dim=(0, 1))
            self.state_mean = self.states.mean(dim=(0, 1))
            self.state_std = self.states.std(dim=(0, 1))
            self.proprio_mean = self.proprios.mean(dim=(0, 1))
            self.proprio_std = self.proprios.std(dim=(0, 1))
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std
        self.slot_cache = None

    def _render_states(self, states, wall_locations, door_locations):
        """Render the analytic Wall scene directly, without a video codec."""
        if self.render_size != 384:
            raise ValueError(
                f"State rendering currently matches the native 384 Wall data, got {self.render_size}"
            )
        coords = torch.arange(self.render_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        states = torch.as_tensor(states).detach().cpu().float()
        wall_x = float(torch.as_tensor(wall_locations).flatten()[0])
        door_y = float(torch.as_tensor(door_locations).flatten()[0])
        scale = self.render_size / 65.0
        wall_half_width = float(round(6 * scale) // 2)
        door_half_height = float(round(4 * scale))
        border_wall_loc = round(5 * scale)
        border_wall_width = max(1, round(scale))
        dot_std = 1.7 * scale
        wall = (xx >= wall_x - wall_half_width) & (xx <= wall_x + wall_half_width)
        wall &= ~((yy >= door_y - door_half_height) & (yy <= door_y + door_half_height))
        wall[:, border_wall_loc - border_wall_width : border_wall_loc] = True
        wall[:, -border_wall_loc : -border_wall_loc + border_wall_width] = True
        wall[border_wall_loc - border_wall_width : border_wall_loc, :] = True
        wall[-border_wall_loc : -border_wall_loc + border_wall_width, :] = True
        squared_distance = (
            (xx[None] - states[:, 0, None, None]).square()
            + (yy[None] - states[:, 1, None, None]).square()
        )
        dot = torch.exp(-squared_distance / (2 * dot_std**2))
        dot.masked_fill_(squared_distance > dot_std**2, 0.0)
        dot /= dot.amax(dim=(1, 2), keepdim=True).clamp_min(1.0e-12)
        red = (dot * 255).to(torch.uint8)
        images = torch.full(
            (len(states), 3, self.render_size, self.render_size), 255, dtype=torch.uint8
        )
        images[:, :, wall] = 0
        # The rendered marker is intentionally larger than its point-like
        # collision body.  Composite it after the wall so a valid contact does
        # not make the visible ball (and therefore its slot mask) disappear
        # underneath the obstacle.
        images[:, 0] = torch.maximum(images[:, 0], red)
        images[:, 1] = torch.minimum(images[:, 1], 255 - red)
        images[:, 2] = torch.minimum(images[:, 2], 255 - red)
        return images

    def set_slot_cache(self, slot_cache):
        self.slot_cache = slot_cache

    def get_seq_length(self, idx):
        return self.traj_len

    def get_all_actions(self):
        result = []
        for i in range(len(self.states)):
            T = self.get_seq_length(i)
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
                self.wall_locations[idx, frames],
                self.door_locations[idx, frames],
            )
        elif video_path.exists():
            image = VideoReader(str(video_path), num_threads=1).get_batch(list(frames))
            image = image.permute(0, 3, 1, 2)
        else:
            image = torch.load(tensor_path)[frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        door_location = self.door_locations[idx, frames]
        wall_location = self.wall_locations[idx, frames]

        image = image / 255
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image,"proprio": proprio}
        if self.slot_cache is not None:
            cache_episode = self.slot_cache[idx]
            obs["slot_obj_latents"] = cache_episode["obj_latents"][frames]
            obs["slot_mask_logits"] = cache_episode["mask_logits"][frames]
        env_info = {
            'fix_door_location': door_location[0],
            'fix_wall_location': wall_location[0],
        }
        if self.layout_ids is not None:
            env_info['layout_id'] = self.layout_ids[idx]
        return obs, act, state, env_info

    def get_causal_window(self, idx, start, end, frameskip):
        """Load/render only the frame-aligned observations needed by a cache window."""
        observation_indices = list(range(start, end, frameskip))
        obs, _, state, env_info = self.get_frames(idx, observation_indices)
        # Preserve every primitive action between successive sampled frames;
        # TrajSlicerDataset concatenates each `frameskip` block.
        act = self.actions[idx, start : end - 1]
        return obs, act, state, env_info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return self.states.shape[0] if not self.n_rollout else self.n_rollout
    
    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return imgs

def load_wall_slice_train_val(
    transform,
    n_rollout=50,
    data_path='data/TwoRoom',
    normalize_action=False,
    split_ratio=0.8,
    split_mode="random",
    num_hist=0,
    num_pred=0,
    frameskip=0,
    random_transition_sampling: bool = False,
    random_samples_per_trajectory: int = 16,
    causal_transition_sampling: bool = False,
    max_causal_windows_per_trajectory: Optional[int] = None,
    non_overlapping_causal_windows: bool = False,
    render_from_state: bool = True,
    render_size: int = 384,
    render_geometry_version: str = "legacy",
    source_coordinate_size: int = 65,
    validation_n_rollout: Optional[int] = None,
    validation_non_overlapping_causal_windows: bool = False,
):  
    if split_mode == "random":
        dset = WallDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path,
            normalize_action=normalize_action,
            render_from_state=render_from_state,
            render_size=render_size,
            render_geometry_version=render_geometry_version,
            source_coordinate_size=source_coordinate_size,
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
        if validation_n_rollout is not None or validation_non_overlapping_causal_windows:
            val_slices = TrajSlicerDataset(
                dset_val,
                num_hist + num_pred,
                frameskip,
                random_transition_sampling=random_transition_sampling,
                random_samples_per_trajectory=random_samples_per_trajectory,
                causal_transition_sampling=causal_transition_sampling,
                max_causal_windows_per_trajectory=max_causal_windows_per_trajectory,
                non_overlapping_causal_windows=validation_non_overlapping_causal_windows,
            )
    elif split_mode == "folder":
        dset_train = WallDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path + "/train",
            normalize_action=normalize_action,
            render_from_state=render_from_state,
            render_size=render_size,
            render_geometry_version=render_geometry_version,
            source_coordinate_size=source_coordinate_size,
        )
        dset_val = WallDataset(
            n_rollout=validation_n_rollout,
            transform=transform,
            data_path=data_path + "/val",
            normalize_action=normalize_action,
            render_from_state=render_from_state,
            render_size=render_size,
            render_geometry_version=render_geometry_version,
            source_coordinate_size=source_coordinate_size,
        )
        num_frames = num_hist + num_pred
        train_slices = TrajSlicerDataset(
            dset_train,
            num_frames,
            frameskip,
            random_transition_sampling=random_transition_sampling,
            random_samples_per_trajectory=random_samples_per_trajectory,
            causal_transition_sampling=causal_transition_sampling,
            max_causal_windows_per_trajectory=max_causal_windows_per_trajectory,
            non_overlapping_causal_windows=non_overlapping_causal_windows,
        )
        val_slices = TrajSlicerDataset(
            dset_val,
            num_frames,
            frameskip,
            random_transition_sampling=random_transition_sampling,
            random_samples_per_trajectory=random_samples_per_trajectory,
            causal_transition_sampling=causal_transition_sampling,
            max_causal_windows_per_trajectory=max_causal_windows_per_trajectory,
            non_overlapping_causal_windows=validation_non_overlapping_causal_windows,
        )
    elif split_mode == "manifest":
        dset_train = WallDataset(
            n_rollout=n_rollout,
            transform=transform,
            data_path=data_path,
            normalize_action=normalize_action,
            render_from_state=render_from_state,
            render_size=render_size,
            render_geometry_version=render_geometry_version,
            manifest_file="train_manifest.pt",
            source_coordinate_size=source_coordinate_size,
        )
        stats = (
            dset_train.action_mean, dset_train.action_std,
            dset_train.state_mean, dset_train.state_std,
            dset_train.proprio_mean, dset_train.proprio_std,
        )
        dset_val = WallDataset(
            n_rollout=validation_n_rollout,
            transform=transform,
            data_path=data_path,
            normalize_action=normalize_action,
            render_from_state=render_from_state,
            render_size=render_size,
            render_geometry_version=render_geometry_version,
            manifest_file="eval_manifest.pt",
            source_coordinate_size=source_coordinate_size,
            normalization_stats=stats if normalize_action else None,
        )
        num_frames = num_hist + num_pred
        slicer_kwargs = dict(
            frameskip=frameskip,
            random_transition_sampling=random_transition_sampling,
            random_samples_per_trajectory=random_samples_per_trajectory,
            causal_transition_sampling=causal_transition_sampling,
            max_causal_windows_per_trajectory=max_causal_windows_per_trajectory,
            non_overlapping_causal_windows=non_overlapping_causal_windows,
        )
        train_slices = TrajSlicerDataset(dset_train, num_frames, **slicer_kwargs)
        validation_slicer_kwargs = dict(slicer_kwargs)
        validation_slicer_kwargs["non_overlapping_causal_windows"] = (
            validation_non_overlapping_causal_windows
        )
        val_slices = TrajSlicerDataset(
            dset_val,
            num_frames,
            **validation_slicer_kwargs,
        )
    else:
        raise ValueError(f"Unknown Wall split_mode: {split_mode!r}")

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset
