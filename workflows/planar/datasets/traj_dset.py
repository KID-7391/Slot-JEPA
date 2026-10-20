import abc
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Sequence, List
from torch.utils.data import Dataset, Subset
from torch import default_generator, randperm
from einops import rearrange

# https://github.com/JaidedAI/EasyOCR/issues/1243
def _accumulate(iterable, fn=lambda x, y: x + y):
    "Return running totals"
    # _accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # _accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    try:
        total = next(it)
    except StopIteration:
        return
    yield total
    for element in it:
        total = fn(total, element)
        yield total

class TrajDataset(Dataset, abc.ABC):
    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError


def cached_slot_collate(batch):
    """Stack cached Slot-AC samples without resizing shared storage."""
    observations, actions, states = zip(*batch)
    observation_keys = observations[0].keys()
    if any(item.keys() != observation_keys for item in observations[1:]):
        raise ValueError(
            "Cached slot samples in one batch have different observation keys"
        )
    collated_observations = {
        key: torch.stack([item[key] for item in observations], dim=0)
        for key in observation_keys
    }
    return [
        collated_observations,
        torch.stack(actions, dim=0),
        torch.stack(states, dim=0),
    ]

class TrajSubset(TrajDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset: TrajDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def get_causal_window(self, idx, start, end, frameskip):
        return self.dataset.get_causal_window(
            self.indices[idx], start, end, frameskip
        )

    def __getstate__(self):
        """Serialize the subset wrapper rather than a proxied dataset state.

        Some trajectory datasets define ``__getstate__`` to discard live
        renderer/EGL objects before spawn-based DataLoader workers are created.
        Forwarding that pickle hook through ``__getattr__`` serializes only the
        wrapped dataset state and silently drops this wrapper's ``dataset`` and
        ``indices`` attributes.
        """
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getattr__(self, name):
        # Never proxy Python protocol hooks such as __getstate__. Pickle must
        # operate on the TrajSubset wrapper itself.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        # During spawn-based DataLoader unpickling, `dataset` has not yet been
        # restored when Python probes attributes. Referring to self.dataset in
        # that state recursively invokes this method until RecursionError.
        dataset = self.__dict__.get("dataset")
        if dataset is not None and hasattr(dataset, name):
            return getattr(dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class TrajectoryFrameDataset(Dataset):
    """Flatten trajectories into stable model-observation frame records."""

    def __init__(self, trajectory_dataset: TrajDataset, frame_stride: int = 1):
        self.dataset = trajectory_dataset
        self.frame_stride = int(frame_stride)
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        self.frame_indices_by_trajectory = [
            list(range(0, trajectory_dataset.get_seq_length(index), self.frame_stride))
            for index in range(len(trajectory_dataset))
        ]
        self.frame_keys = [
            (trajectory_index, frame_index)
            for trajectory_index, frame_indices in enumerate(
                self.frame_indices_by_trajectory
            )
            for frame_index in frame_indices
        ]

    def __len__(self):
        return len(self.frame_keys)

    def __getitem__(self, index):
        trajectory_index, frame_index = self.frame_keys[index]
        if hasattr(self.dataset, "get_causal_window"):
            obs, _, state, env_info = self.dataset.get_causal_window(
                trajectory_index, frame_index, frame_index + 1, 1
            )
        else:
            obs, _, state, env_info = self.dataset[trajectory_index]
            obs = {
                key: value if key == "layout_id" else value[frame_index : frame_index + 1]
                for key, value in obs.items()
            }
            state = state[frame_index : frame_index + 1]
        layout_id = env_info.get("layout_id", -1)
        return (
            obs["visual"][0],
            state[0],
            torch.tensor(trajectory_index, dtype=torch.long),
            torch.tensor(frame_index, dtype=torch.long),
            torch.as_tensor(layout_id, dtype=torch.long),
        )


class TrajSlicerDataset(TrajDataset):
    def __init__(
        self,
        dataset: TrajDataset,
        num_frames: int,
        frameskip: int = 1,
        process_actions: str = "concat",
        random_transition_sampling: bool = False,
        random_samples_per_trajectory: Optional[int] = None,
        causal_transition_sampling: bool = False,
        max_causal_windows_per_trajectory: Optional[int] = None,
        non_overlapping_causal_windows: bool = False,
    ):
        self.dataset = dataset
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.process_actions = process_actions
        self.random_transition_sampling = random_transition_sampling
        self.random_samples_per_trajectory = random_samples_per_trajectory
        self.causal_transition_sampling = bool(causal_transition_sampling)
        self.max_causal_windows_per_trajectory = max_causal_windows_per_trajectory
        self.non_overlapping_causal_windows = bool(non_overlapping_causal_windows)
        self.slices = []
        for i in range(len(self.dataset)): 
            T = self.dataset.get_seq_length(i)
            max_transition_start = T - self.frameskip
            if self.random_transition_sampling:
                if max_transition_start < self.num_frames:
                    print(f"Ignored short sequence #{i}: len={T}, num_frames={num_frames}")
                    continue
                num_samples = (
                    int(self.random_samples_per_trajectory)
                    if self.random_samples_per_trajectory is not None
                    else T - num_frames * frameskip + 1
                )
                if num_samples <= 0:
                    raise ValueError("random_samples_per_trajectory must be positive")
                valid_starts = np.arange(max_transition_start)
                for _ in range(num_samples):
                    chosen = np.sort(
                        np.random.choice(valid_starts, size=self.num_frames, replace=False)
                    )
                    self.slices.append((i, chosen.astype(np.int64)))
            elif self.causal_transition_sampling:
                span = (num_frames - 1) * self.frameskip + 1
                if T < span:
                    print(f"Ignored short sequence #{i}: len={T}, required_span={span}")
                else:
                    start_step = (
                        num_frames * self.frameskip
                        if self.non_overlapping_causal_windows
                        else 1
                    )
                    trajectory_slices = [
                        (i, start, start + span)
                        for start in range(0, T - span + 1, start_step)
                    ]
                    if (
                        self.max_causal_windows_per_trajectory is not None
                        and len(trajectory_slices) > self.max_causal_windows_per_trajectory
                    ):
                        keep = np.linspace(
                            0,
                            len(trajectory_slices) - 1,
                            int(self.max_causal_windows_per_trajectory),
                            dtype=np.int64,
                        )
                        trajectory_slices = [trajectory_slices[int(index)] for index in keep]
                    self.slices += trajectory_slices
            else:
                if T - num_frames < 0:
                    print(f"Ignored short sequence #{i}: len={T}, num_frames={num_frames}")
                else:
                    self.slices += [
                        (i, start, start + num_frames * self.frameskip)
                        for start in range(T - num_frames * frameskip + 1)
                    ]  # slice indices follow convention [start, end)
        # randomly permute the slices without forcing ragged/object tuples into a numpy ndarray
        perm = np.random.permutation(len(self.slices))
        self.slices = [self.slices[int(i)] for i in perm]
        
        self.proprio_dim = self.dataset.proprio_dim
        if process_actions == "concat":
            self.action_dim = self.dataset.action_dim * self.frameskip
        else:
            self.action_dim = self.dataset.action_dim

        self.state_dim = self.dataset.state_dim
        self.slot_cache = None
        self.zero_action_context_probability = 0.0
        self.zero_action_full_context_probability = 0.0
        self.zero_action_num_hist = 0
        self.zero_action_validation = False
        self.zero_action_chunk = None


    def get_seq_length(self, idx: int) -> int:
        return self.num_frames

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice_item = self.slices[idx]
        if self.random_transition_sampling:
            i, transition_starts = slice_item
        else:
            i, start, end = slice_item
        direct_causal_window = (
            self.causal_transition_sampling
            and not self.random_transition_sampling
            and hasattr(self.dataset, "get_causal_window")
        )
        env_info = {}
        if self.slot_cache is not None and self.causal_transition_sampling:
            # Cached Slot-AC training does not consume RGB. Resolve any Subset
            # mapping and fetch only resident state/action tensors, avoiding
            # MP4 decode, image transforms, worker IPC, and H2D transfer.
            source_dataset = self.dataset
            source_index = i
            while isinstance(source_dataset, Subset):
                source_index = source_dataset.indices[source_index]
                source_dataset = source_dataset.dataset
            observation_indices = list(range(start, end, self.frameskip))
            obs = {
                "proprio": source_dataset.proprios[source_index, observation_indices],
            }
            layout_ids = getattr(source_dataset, "layout_ids", None)
            if layout_ids is not None:
                env_info["layout_id"] = layout_ids[source_index]
            act = source_dataset.actions[source_index, start : end - 1]
            state = source_dataset.states[source_index, observation_indices]
            # The cache fast path has already materialized the requested
            # causal window. In particular PushTDataset has no
            # get_causal_window method, so without this flag the generic tail
            # below slices these tensors a second time (5 frames -> 1 or 0).
            direct_causal_window = True
        elif direct_causal_window:
            obs, act, state, env_info = self.dataset.get_causal_window(
                i, start, end, self.frameskip
            )
        else:
            obs, act, state, env_info = self.dataset[i]
        if "layout_id" in env_info:
            obs["layout_id"] = torch.as_tensor(env_info["layout_id"]).long()
        # Stable sliced-dataset identity. Retrieval banks persist this id for
        # every stored transition, allowing the model to exclude an exact
        # training transition with a device-side tensor lookup instead of
        # rebuilding byte keys on the CPU in every forward pass.
        obs["sample_index"] = torch.as_tensor(idx, dtype=torch.long)
        if self.random_transition_sampling:
            post_indices = transition_starts + self.frameskip
            pre_state = state[transition_starts]
            post_state = state[post_indices]
            obs_out = {}
            for k, v in obs.items():
                # Per-trajectory metadata is scalar and must not be indexed as
                # though it were a frame sequence.
                obs_out[k] = v if k == "layout_id" else v[post_indices]
            if self.slot_cache is not None:
                cache_item = self.slot_cache["episodes"][i]
                raw_indices = torch.as_tensor(post_indices, dtype=torch.long)
                cache_indices = cache_item["frame_to_cache"].index_select(
                    0, raw_indices
                )
                if (cache_indices < 0).any():
                    raise RuntimeError(
                        "Random transition requested a frame absent from the slot cache"
                    )
                obs_out["slot_mask_logits"] = cache_item["mask_logits"].index_select(
                    0, cache_indices
                )
            obs_out["post_state"] = post_state

            act_chunks = []
            for start_idx in transition_starts:
                act_chunk = act[start_idx : start_idx + self.frameskip]
                if self.process_actions == "concat":
                    act_chunk = rearrange(act_chunk, "f d -> (f d)")
                else:
                    act_chunk = act_chunk.mean(dim=0)
                act_chunks.append(act_chunk)
            act_out = torch.stack(act_chunks, dim=0)
            if (
                (
                    self.zero_action_context_probability > 0.0
                    or self.zero_action_full_context_probability > 0.0
                )
                and self.zero_action_num_hist > 0
            ):
                num_context = min(self.zero_action_num_hist, self.num_frames)
                zero_mask = torch.zeros(self.num_frames, dtype=torch.bool)
                if self.zero_action_validation:
                    # Planning starts with a complete history of zero-action
                    # placeholders, so validation should cover that exact case.
                    zero_mask[:num_context] = True
                else:
                    use_full_zero_context = (
                        torch.rand(())
                        < self.zero_action_full_context_probability
                    )
                    if use_full_zero_context:
                        zero_mask[:num_context] = True
                    else:
                        zero_mask[:num_context] = (
                            torch.rand(num_context)
                            < self.zero_action_context_probability
                        )
                    # Keep at least one identity transition in every training sample.
                    if not zero_mask[:num_context].any():
                        zero_mask[0] = True
                act_out[zero_mask] = self.zero_action_chunk.to(
                    dtype=act_out.dtype, device=act_out.device
                )
                post_state[zero_mask] = pre_state[zero_mask]
                obs_out["zero_action_mask"] = zero_mask
            return tuple([obs_out, act_out, pre_state])

        if not direct_causal_window:
            for k, v in obs.items():
                obs[k] = v[start:end:self.frameskip]
        if self.slot_cache is not None:
            cache_item = self.slot_cache["episodes"][i]
            frame_indices = torch.as_tensor(
                list(range(start, end, self.frameskip)), dtype=torch.long
            )
            cache_indices = cache_item["frame_to_cache"].index_select(
                0, frame_indices
            )
            if (cache_indices < 0).any():
                missing = frame_indices[cache_indices < 0].tolist()
                raise RuntimeError(
                    f"Training window requested uncached raw frames {missing}"
                )
            cached_states = cache_item.get("states")
            if cached_states is not None and not torch.equal(
                cached_states.index_select(0, cache_indices).float().cpu(),
                state.float().cpu(),
            ):
                raise RuntimeError(
                    "Frame slot cache state mismatch. The cache was built for "
                    "different trajectory data and must be rebuilt."
                )
            cached_layout_id = cache_item.get("layout_id")
            if cached_layout_id is not None:
                current_layout_id = obs.get("layout_id")
                if current_layout_id is None or int(cached_layout_id) != int(
                    current_layout_id
                ):
                    raise RuntimeError(
                        "Slot cache sample/layout mismatch. The cache must be rebuilt."
                    )
            obs["slot_mask_logits"] = cache_item["mask_logits"].index_select(
                0, cache_indices
            )
        if not direct_causal_window:
            state = state[start:end:self.frameskip]
        if self.causal_transition_sampling:
            # H+1 observations define H real transitions.  The final action
            # row is an unused placeholder so tensors remain frame-aligned.
            transition_act = act if direct_causal_window else act[start : end - 1]
            transition_act = rearrange(
                transition_act, "(n f) d -> n (f d)", n=self.num_frames - 1
            )
            act = torch.cat([transition_act, torch.zeros_like(transition_act[:1])], dim=0)
        else:
            act = act[start:end]
            act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)  # concat actions
        return tuple([obs, act, state])

    def set_slot_cache(self, slot_cache):
        if not isinstance(slot_cache, dict) or slot_cache.get("storage") != "trajectory_masks":
            raise ValueError(
                "Slot cache uses the obsolete window-level format; rebuild it "
                "with trajectory-mask storage"
            )
        if int(slot_cache.get("schema_version", -1)) != 3:
            raise ValueError(
                "Slot cache predates sparse model-frame indexing; rebuild it"
            )
        episodes = slot_cache.get("episodes", [])
        if len(episodes) != len(self.dataset):
            raise ValueError(
                f"Expected {len(self.dataset)} cached trajectories, got {len(episodes)}"
            )
        for trajectory_index, episode in enumerate(episodes):
            frame_indices = episode.get("frame_indices")
            frame_to_cache = episode.get("frame_to_cache")
            if frame_indices is None or frame_to_cache is None:
                raise ValueError(
                    f"Trajectory {trajectory_index} cache has no frame index mapping"
                )
            if episode["mask_logits"].shape[0] != frame_indices.numel():
                raise ValueError(
                    f"Trajectory {trajectory_index} cache mask/index lengths differ"
                )
            expected_raw_length = self.dataset.get_seq_length(trajectory_index)
            if frame_to_cache.numel() != expected_raw_length:
                raise ValueError(
                    f"Trajectory {trajectory_index} cache maps {frame_to_cache.numel()} "
                    f"raw frames, expected {expected_raw_length}"
                )
        self.slot_cache = slot_cache

    def set_zero_action_augmentation(
        self,
        probability: float,
        full_context_probability: float,
        num_hist: int,
        zero_action_chunk: torch.Tensor,
        validation: bool = False,
    ):
        """Configure identity transitions without changing cached visual slots.

        The selected transition keeps its original visual representation/slots/masks,
        while its action is set to physical zero and its target state to pre-state.
        """
        self.zero_action_context_probability = float(probability)
        self.zero_action_full_context_probability = float(full_context_probability)
        for name, value in (
            ("probability", self.zero_action_context_probability),
            ("full_context_probability", self.zero_action_full_context_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        self.zero_action_num_hist = int(num_hist)
        self.zero_action_validation = bool(validation)
        zero_action_chunk = torch.as_tensor(zero_action_chunk).flatten()
        if zero_action_chunk.numel() != self.action_dim:
            raise ValueError(
                f"Expected zero action chunk with {self.action_dim} values, "
                f"got {zero_action_chunk.numel()}"
            )
        self.zero_action_chunk = zero_action_chunk.clone()


def random_split_traj(
    dataset: TrajDataset,
    lengths: Sequence[int],
    generator: Optional[torch.Generator] = default_generator,
) -> List[TrajSubset]:
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        TrajSubset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


def split_traj_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    crossing_flags = getattr(dataset, "crossing_flags", None)
    if crossing_flags is not None:
        flags = torch.as_tensor(crossing_flags).bool().flatten()
        if flags.numel() != dataset_length:
            raise ValueError(
                f"Expected {dataset_length} crossing flags, got {flags.numel()}"
            )
        generator = torch.Generator().manual_seed(random_seed)
        positives = torch.nonzero(flags, as_tuple=False).flatten()
        negatives = torch.nonzero(~flags, as_tuple=False).flatten()
        positives = positives[torch.randperm(len(positives), generator=generator)]
        negatives = negatives[torch.randperm(len(negatives), generator=generator)]
        train_size = int(train_fraction * dataset_length)
        # Floor the proportional allocation so a small validation split still
        # retains its generated crossing quota instead of rounding all positive
        # examples into training.
        train_positive_count = int(len(positives) * train_size / dataset_length)
        train_positive_count = min(train_positive_count, len(positives), train_size)
        train_negative_count = train_size - train_positive_count
        if train_negative_count > len(negatives):
            overflow = train_negative_count - len(negatives)
            train_negative_count -= overflow
            train_positive_count += overflow
        split_indices = [
            positives[:train_positive_count].tolist()
            + negatives[:train_negative_count].tolist(),
            positives[train_positive_count:].tolist()
            + negatives[train_negative_count:].tolist(),
        ]
        for indices in split_indices:
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices[:] = [indices[index] for index in order]
        return TrajSubset(dataset, split_indices[0]), TrajSubset(dataset, split_indices[1])
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split_traj(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set


def get_train_val_sliced(
    traj_dataset: TrajDataset,
    train_fraction: float = 0.9,
    random_seed: int = 42,
    num_frames: int = 10,
    frameskip: int = 1,
    random_transition_sampling: bool = False,
    random_samples_per_trajectory: Optional[int] = None,
    causal_transition_sampling: bool = False,
    max_causal_windows_per_trajectory: Optional[int] = None,
    non_overlapping_causal_windows: bool = False,
):
    train, val = split_traj_datasets(
        traj_dataset,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    train_slices = TrajSlicerDataset(
        train,
        num_frames,
        frameskip,
        random_transition_sampling=random_transition_sampling,
        random_samples_per_trajectory=random_samples_per_trajectory,
        causal_transition_sampling=causal_transition_sampling,
        max_causal_windows_per_trajectory=max_causal_windows_per_trajectory,
        non_overlapping_causal_windows=non_overlapping_causal_windows,
    )
    val_slices = TrajSlicerDataset(
        val,
        num_frames,
        frameskip,
        random_transition_sampling=random_transition_sampling,
        random_samples_per_trajectory=random_samples_per_trajectory,
        causal_transition_sampling=causal_transition_sampling,
        max_causal_windows_per_trajectory=max_causal_windows_per_trajectory,
        non_overlapping_causal_windows=non_overlapping_causal_windows,
    )
    return train, val, train_slices, val_slices
