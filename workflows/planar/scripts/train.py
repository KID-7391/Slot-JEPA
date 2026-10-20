import os
import sys
import time
import math
import hashlib
import hydra
import torch
import wandb
import logging
import warnings
import threading
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler, Subset
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

PLANAR_ROOT = Path(__file__).resolve().parents[1]
if str(PLANAR_ROOT) not in sys.path:
    sys.path.insert(0, str(PLANAR_ROOT))

from metrics.image_metrics import eval_images
from planning.pushobj_context import (
    PUSHOBJ_STRUCTURED_CONTEXT_NAMES,
)
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors
from datasets.traj_dset import (
    TrajSubset,
    TrajectoryFrameDataset,
    cached_slot_collate,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


def _wall_collision_window_mask(sample_dataset, tolerance=1.0e-3):
    """Return whether each sliced Wall sample contains a physical collision.

    Wall stores realized velocity in state[..., 2:4].  In free motion it is
    exactly the commanded action; a collision truncates one or both velocity
    components.  Inspect every primitive transition covered by the causal
    sample so this label matches the five-action pair consumed by Slot-AC.
    """
    cached = getattr(sample_dataset, "_wall_collision_window_mask", None)
    if cached is not None:
        return cached
    if not hasattr(sample_dataset, "slices") or not hasattr(
        sample_dataset, "frameskip"
    ):
        raise TypeError("Collision balancing requires a sliced Wall dataset")

    source_dataset = sample_dataset.dataset
    subset_mappings = []
    while isinstance(source_dataset, Subset):
        subset_mappings.append(torch.as_tensor(source_dataset.indices).long())
        source_dataset = source_dataset.dataset
    if not hasattr(source_dataset, "states") or not hasattr(source_dataset, "actions"):
        raise TypeError("Collision balancing requires resident Wall states/actions")

    trajectory_indices = []
    starts = []
    ends = []
    for slice_item in sample_dataset.slices:
        if len(slice_item) != 3:
            raise ValueError(
                "Collision balancing requires contiguous causal windows"
            )
        trajectory_index, start, end = slice_item
        trajectory_indices.append(int(trajectory_index))
        starts.append(int(start))
        ends.append(int(end))
    trajectory_indices = torch.as_tensor(trajectory_indices).long()
    for mapping in subset_mappings:
        trajectory_indices = mapping.index_select(0, trajectory_indices)
    starts = torch.as_tensor(starts).long()
    ends = torch.as_tensor(ends).long()
    spans = ends - starts - 1
    if spans.numel() == 0 or torch.any(spans != spans[0]):
        raise ValueError("Collision balancing requires fixed-length windows")
    primitive_steps = int(spans[0])
    offsets = torch.arange(primitive_steps).unsqueeze(0)
    action_indices = starts.unsqueeze(1) + offsets
    state_indices = action_indices + 1

    normalized_actions = source_dataset.actions[
        trajectory_indices.unsqueeze(1), action_indices
    ].float()
    action_mean = torch.as_tensor(source_dataset.action_mean).float()
    action_std = torch.as_tensor(source_dataset.action_std).float()
    commanded_velocity = normalized_actions * action_std + action_mean
    realized_velocity = source_dataset.states[
        trajectory_indices.unsqueeze(1), state_indices, 2:4
    ].float()
    mask = (
        (realized_velocity - commanded_velocity)
        .abs()
        .amax(dim=(-1, -2))
        > float(tolerance)
    ).cpu()
    sample_dataset._wall_collision_window_mask = mask
    return mask


def _wall_window_layout_ids(sample_dataset):
    """Map each sliced Wall sample to its source trajectory's layout."""
    cached = getattr(sample_dataset, "_wall_window_layout_ids", None)
    if cached is not None:
        return cached
    source_dataset = sample_dataset.dataset
    subset_mappings = []
    while isinstance(source_dataset, Subset):
        subset_mappings.append(torch.as_tensor(source_dataset.indices).long())
        source_dataset = source_dataset.dataset
    if getattr(source_dataset, "layout_ids", None) is None:
        return None
    trajectory_indices = torch.as_tensor(
        [int(slice_item[0]) for slice_item in sample_dataset.slices]
    ).long()
    for mapping in subset_mappings:
        trajectory_indices = mapping.index_select(0, trajectory_indices)
    layout_ids = torch.as_tensor(source_dataset.layout_ids).long().index_select(
        0, trajectory_indices
    )
    sample_dataset._wall_window_layout_ids = layout_ids
    return layout_ids


def _pusht_contact_window_mask(sample_dataset, tolerance_px=0.0):
    """Mark sliced PushT/PushObj pairs containing physical pusher contact.

    Every primitive state covered by the causal pair is checked. Block
    polygons are read from PushTEnv so the classifier uses the same geometry
    as the simulator for all supported shapes.
    """
    cache_key = float(tolerance_px)
    cached = getattr(sample_dataset, "_pusht_contact_window_masks", {}).get(
        cache_key
    )
    if cached is not None:
        return cached
    if not hasattr(sample_dataset, "slices"):
        raise TypeError("Contact balancing requires a sliced PushT dataset")

    source_dataset = sample_dataset.dataset
    subset_mappings = []
    while isinstance(source_dataset, Subset):
        subset_mappings.append(torch.as_tensor(source_dataset.indices).long())
        source_dataset = source_dataset.dataset
    if not hasattr(source_dataset, "states") or not hasattr(source_dataset, "shapes"):
        raise TypeError("Contact balancing requires resident PushT states/shapes")

    trajectory_indices = []
    starts = []
    ends = []
    for slice_item in sample_dataset.slices:
        if len(slice_item) != 3:
            raise ValueError("Contact balancing requires contiguous causal windows")
        trajectory_index, start, end = slice_item
        trajectory_indices.append(int(trajectory_index))
        starts.append(int(start))
        ends.append(int(end))
    trajectory_indices = torch.as_tensor(trajectory_indices).long()
    for mapping in subset_mappings:
        trajectory_indices = mapping.index_select(0, trajectory_indices)
    starts = torch.as_tensor(starts).long()
    ends = torch.as_tensor(ends).long()
    spans = ends - starts
    if spans.numel() == 0 or torch.any(spans != spans[0]):
        raise ValueError("Contact balancing requires fixed-length windows")

    # Generated manifests store the simulator's actual collision observation
    # for every primitive transition. Prefer it over endpoint geometry: the
    # collision-safety projection deliberately removes overlap before a state
    # is recorded, so reconstructing contact from those endpoints can miss a
    # collision that occurred inside env.step().
    if getattr(source_dataset, "contact_steps", None) is not None:
        primitive_steps = int(spans[0]) - 1
        if primitive_steps <= 0:
            raise ValueError("Contact windows must cover at least one transition")
        contact_indices = starts.unsqueeze(1) + torch.arange(primitive_steps).unsqueeze(0)
        mask = source_dataset.contact_steps[
            trajectory_indices.unsqueeze(1), contact_indices
        ].any(dim=1).cpu()
        cached_masks = getattr(sample_dataset, "_pusht_contact_window_masks", {})
        cached_masks[cache_key] = mask
        sample_dataset._pusht_contact_window_masks = cached_masks
        return mask
    state_indices = starts.unsqueeze(1) + torch.arange(int(spans[0])).unsqueeze(0)

    # Pymunk vertices are block-local. Reading them from the live environment
    # avoids maintaining a second, silently divergent shape table here.
    from env.pusht.pusht_env import PushTEnv

    polygons_by_shape = {}
    pusher_radius = None
    for shape_name in sorted(set(source_dataset.shapes)):
        env = PushTEnv(shape=shape_name, with_target=False)
        try:
            env.reset()
            polygons_by_shape[shape_name] = [
                torch.tensor(
                    [[vertex.x, vertex.y] for vertex in shape.get_vertices()],
                    dtype=torch.float32,
                )
                for shape in env.block.shapes
            ]
            radius = max(float(shape.radius) for shape in env.agent.shapes)
            pusher_radius = radius if pusher_radius is None else pusher_radius
            if not math.isclose(radius, pusher_radius):
                raise ValueError("PushT shapes use inconsistent pusher radii")
        finally:
            env.close()

    def circle_touches_convex_polygon(points, vertices, radius):
        edge_start = vertices
        edge = torch.roll(vertices, shifts=-1, dims=0) - vertices
        relative = points.unsqueeze(-2) - edge_start
        edge_sq = edge.square().sum(dim=-1).clamp_min(1.0e-12)
        projection = ((relative * edge).sum(dim=-1) / edge_sq).clamp(0.0, 1.0)
        closest = edge_start + projection.unsqueeze(-1) * edge
        distance_sq = (points.unsqueeze(-2) - closest).square().sum(dim=-1)
        cross = edge[..., 0] * relative[..., 1] - edge[..., 1] * relative[..., 0]
        inside = (cross >= -1.0e-5).all(dim=-1) | (cross <= 1.0e-5).all(dim=-1)
        return inside | (distance_sq.amin(dim=-1) <= radius * radius)

    mask = torch.zeros(len(sample_dataset), dtype=torch.bool)
    sample_shapes = np.asarray(source_dataset.shapes, dtype=object)[
        trajectory_indices.numpy()
    ]
    radius = float(pusher_radius) + float(tolerance_px)
    states = source_dataset.states
    chunk_size = 8192
    for shape_name, polygons in polygons_by_shape.items():
        shape_samples = torch.from_numpy(np.flatnonzero(sample_shapes == shape_name)).long()
        for begin in range(0, shape_samples.numel(), chunk_size):
            sample_ids = shape_samples[begin : begin + chunk_size]
            window_states = states[
                trajectory_indices.index_select(0, sample_ids).unsqueeze(1),
                state_indices.index_select(0, sample_ids),
            ].float()
            delta = window_states[..., :2] - window_states[..., 2:4]
            angle = window_states[..., 4]
            cosine = torch.cos(angle)
            sine = torch.sin(angle)
            local_points = torch.stack(
                (
                    cosine * delta[..., 0] + sine * delta[..., 1],
                    -sine * delta[..., 0] + cosine * delta[..., 1],
                ),
                dim=-1,
            )
            touches = torch.zeros(local_points.shape[:2], dtype=torch.bool)
            for polygon in polygons:
                touches |= circle_touches_convex_polygon(local_points, polygon, radius)
            mask[sample_ids] = touches.any(dim=1)

    cached_masks = getattr(sample_dataset, "_pusht_contact_window_masks", {})
    cached_masks[cache_key] = mask
    sample_dataset._pusht_contact_window_masks = cached_masks
    return mask


def _window_layout_ids(sample_dataset):
    """Map sliced samples to source trajectory layout/shape ids."""
    cached = getattr(sample_dataset, "_window_layout_ids", None)
    if cached is not None:
        return cached
    source_dataset = sample_dataset.dataset
    subset_mappings = []
    while isinstance(source_dataset, Subset):
        subset_mappings.append(torch.as_tensor(source_dataset.indices).long())
        source_dataset = source_dataset.dataset
    if getattr(source_dataset, "layout_ids", None) is None:
        return None
    trajectory_indices = torch.as_tensor(
        [int(slice_item[0]) for slice_item in sample_dataset.slices]
    ).long()
    for mapping in subset_mappings:
        trajectory_indices = mapping.index_select(0, trajectory_indices)
    layout_ids = torch.as_tensor(source_dataset.layout_ids).long().index_select(
        0, trajectory_indices
    )
    sample_dataset._window_layout_ids = layout_ids
    return layout_ids


def _pusht_rotation_window_degrees(sample_dataset):
    """Maximum absolute object rotation between model steps in each window."""
    cached = getattr(sample_dataset, "_pusht_rotation_window_degrees", None)
    if cached is not None:
        return cached
    if not hasattr(sample_dataset, "slices"):
        raise TypeError("Rotation balancing requires a sliced PushT dataset")

    source_dataset = sample_dataset.dataset
    subset_mappings = []
    while isinstance(source_dataset, Subset):
        subset_mappings.append(torch.as_tensor(source_dataset.indices).long())
        source_dataset = source_dataset.dataset
    if not hasattr(source_dataset, "states"):
        raise TypeError("Rotation balancing requires resident PushT states")

    trajectory_indices = []
    observation_indices = []
    for slice_item in sample_dataset.slices:
        if len(slice_item) != 3:
            raise ValueError(
                "Rotation balancing requires contiguous causal windows"
            )
        trajectory_index, start, end = slice_item
        indices = list(range(int(start), int(end), int(sample_dataset.frameskip)))
        if len(indices) < 2:
            raise ValueError("Rotation-balanced windows need at least two frames")
        trajectory_indices.append(int(trajectory_index))
        observation_indices.append(indices)
    trajectory_indices = torch.as_tensor(trajectory_indices).long()
    for mapping in subset_mappings:
        trajectory_indices = mapping.index_select(0, trajectory_indices)
    observation_indices = torch.as_tensor(observation_indices).long()
    angles = source_dataset.states[
        trajectory_indices.unsqueeze(1), observation_indices, 4
    ].float()
    delta = angles[:, 1:] - angles[:, :-1]
    wrapped_delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    rotation_degrees = wrapped_delta.abs().amax(dim=1).mul(180.0 / math.pi).cpu()
    sample_dataset._pusht_rotation_window_degrees = rotation_degrees
    return rotation_degrees


class StratifiedBinSampler(Sampler):
    """Sample an equal epoch quota from each numeric bin."""

    def __init__(self, values, lower_edges, seed=0):
        self.values = torch.as_tensor(values).float().flatten()
        self.lower_edges = tuple(float(edge) for edge in lower_edges)
        if not self.lower_edges or self.lower_edges[0] != 0.0:
            raise ValueError("Balanced-bin lower edges must start at zero")
        if any(
            right <= left
            for left, right in zip(self.lower_edges, self.lower_edges[1:])
        ):
            raise ValueError("Balanced-bin lower edges must be strictly increasing")
        boundaries = torch.tensor(self.lower_edges[1:], dtype=torch.float32)
        labels = torch.bucketize(self.values, boundaries, right=True)
        self.bin_indices = [
            torch.nonzero(labels == index, as_tuple=False).flatten().long()
            for index in range(len(self.lower_edges))
        ]
        empty_bins = [
            index for index, indices in enumerate(self.bin_indices)
            if indices.numel() == 0
        ]
        if empty_bins:
            raise ValueError(f"Rotation balancing has empty bins {empty_bins}")
        self.raw_counts = tuple(int(indices.numel()) for indices in self.bin_indices)
        base_quota, remainder = divmod(self.values.numel(), len(self.bin_indices))
        self.bin_quotas = tuple(
            base_quota + int(index < remainder)
            for index in range(len(self.bin_indices))
        )
        self.num_samples = sum(self.bin_quotas)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @staticmethod
    def _sample(indices, count, generator):
        parts = []
        remaining = int(count)
        while remaining:
            permutation = torch.randperm(indices.numel(), generator=generator)
            take = min(remaining, indices.numel())
            parts.append(indices[permutation[:take]])
            remaining -= take
        return torch.cat(parts)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        parts = [
            self._sample(indices, quota, generator)
            for indices, quota in zip(self.bin_indices, self.bin_quotas)
        ]
        indices = torch.cat(parts)
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


def _select_balanced_unique_indices(values, lower_edges, max_count, generator):
    """Select unique rows while making populated numeric bins as even as possible."""
    values = torch.as_tensor(values).float().flatten()
    lower_edges = tuple(float(edge) for edge in lower_edges)
    if not lower_edges or lower_edges[0] != 0.0:
        raise ValueError("Balanced-bin lower edges must start at zero")
    if any(right <= left for left, right in zip(lower_edges, lower_edges[1:])):
        raise ValueError("Balanced-bin lower edges must be strictly increasing")
    labels = torch.bucketize(
        values,
        torch.tensor(lower_edges[1:], dtype=torch.float32),
        right=True,
    )
    bins = [
        torch.nonzero(labels == index, as_tuple=False).flatten().long()
        for index in range(len(lower_edges))
    ]
    capacities = [int(indices.numel()) for indices in bins]
    target = min(int(max_count), sum(capacities))
    quotas = [0] * len(bins)
    assigned = 0
    # Round-robin water filling keeps all populated bins equal until a sparse
    # bin is exhausted, then redistributes its unused quota to the others.
    while assigned < target:
        progressed = False
        for index, capacity in enumerate(capacities):
            if quotas[index] >= capacity:
                continue
            quotas[index] += 1
            assigned += 1
            progressed = True
            if assigned == target:
                break
        if not progressed:
            break
    selected = [
        indices[torch.randperm(indices.numel(), generator=generator)[:quota]]
        for indices, quota in zip(bins, quotas)
        if quota
    ]
    if not selected:
        return torch.empty(0, dtype=torch.long), tuple(capacities), tuple(quotas)
    return torch.cat(selected), tuple(capacities), tuple(quotas)


class CollisionBalancedSampler(Sampler):
    """Balance positive windows, then reshuffle the full list every epoch."""

    def __init__(self, collision_mask, minimum_fraction, seed=0):
        self.collision_indices = torch.nonzero(
            collision_mask, as_tuple=False
        ).flatten().long()
        self.free_indices = torch.nonzero(
            ~collision_mask, as_tuple=False
        ).flatten().long()
        self.minimum_fraction = float(minimum_fraction)
        self.seed = int(seed)
        self.epoch = 0
        if not 0.0 < self.minimum_fraction < 1.0:
            raise ValueError("balanced pair fraction must be between 0 and 1")
        if self.collision_indices.numel() == 0 or self.free_indices.numel() == 0:
            raise ValueError("Pair balancing requires both positive and negative pairs")
        required_collision = math.ceil(
            self.minimum_fraction
            * self.free_indices.numel()
            / (1.0 - self.minimum_fraction)
        )
        self.num_collision = max(self.collision_indices.numel(), required_collision)
        self.num_samples = self.num_collision + self.free_indices.numel()

    def set_epoch(self, epoch):
        """Use an explicit epoch seed shared by every distributed rank."""
        self.epoch = int(epoch)

    @staticmethod
    def _sample(indices, count, generator):
        parts = []
        remaining = int(count)
        while remaining:
            permutation = torch.randperm(indices.numel(), generator=generator)
            take = min(remaining, indices.numel())
            parts.append(indices[permutation[:take]])
            remaining -= take
        return torch.cat(parts)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        collision = self._sample(
            self.collision_indices, self.num_collision, generator
        )
        free = self._sample(self.free_indices, self.free_indices.numel(), generator)
        indices = torch.cat((collision, free))
        indices = indices[
            torch.randperm(indices.numel(), generator=generator)
        ]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


def _synchronize_hydra_run_id():
    """Give every process in one distributed launch the same Hydra run dir."""
    if "DINO_WM_RUN_ID" in os.environ:
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        os.environ["DINO_WM_RUN_ID"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        return

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    launch_id = os.environ.get("SLURM_JOB_ID")
    if not launch_id:
        launch_id = ":".join(
            (
                str(os.getppid()),
                os.environ.get("MASTER_ADDR", "localhost"),
                os.environ.get("MASTER_PORT", "unknown"),
            )
        )
    launch_hash = hashlib.sha1(launch_id.encode("utf-8")).hexdigest()[:16]
    rendezvous_path = Path("/tmp") / f"dino_wm_run_id_{launch_hash}.txt"

    if rank == 0:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        temporary_path = rendezvous_path.with_suffix(f".rank0.{os.getpid()}.tmp")
        temporary_path.write_text(run_id, encoding="utf-8")
        temporary_path.replace(rendezvous_path)
    else:
        deadline = time.monotonic() + 60.0
        while not rendezvous_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for distributed Hydra run id at {rendezvous_path}"
                )
            time.sleep(0.05)
        run_id = rendezvous_path.read_text(encoding="utf-8").strip()

    if not run_id:
        raise RuntimeError(f"Empty distributed Hydra run id in {rendezvous_path}")
    os.environ["DINO_WM_RUN_ID"] = run_id

class Trainer:
    def _silence_non_main_process_logging(self):
        if self.accelerator.is_main_process:
            return
        logging.getLogger().setLevel(logging.ERROR)
        for logger_name in list(logging.root.manager.loggerDict.keys()):
            logging.getLogger(logger_name).setLevel(logging.ERROR)

    def _sync_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _cuda_mem_mb(self):
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.memory_allocated(self.device) / (1024 ** 2)

    def _cuda_peak_mb(self):
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def _reset_cuda_peak(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _tic(self):
        self._sync_device()
        return time.perf_counter()

    def _toc(self, start_time):
        self._sync_device()
        return time.perf_counter() - start_time

    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info(" Multirun setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
            log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
            # ==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                if torch.cuda.is_available():
                    # NCCL may initialize CUDA during process-group creation.
                    # Bind this process first so nonzero local ranks do not
                    # create an incidental context on physical GPU 0.
                    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),  # Set a 5-minute timeout
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()
            # # ==== /init ddp process group ====

        mixed_precision = str(cfg.training.get("mixed_precision", "no")).lower()
        if mixed_precision not in {"no", "bf16", "fp16"}:
            raise ValueError(
                f"Unsupported training.mixed_precision={mixed_precision!r}; "
                "use one of: no, bf16, fp16"
            )
        self.mixed_precision = mixed_precision
        self.accelerator = Accelerator(
            log_with="wandb", mixed_precision=mixed_precision
        )
        self._silence_non_main_process_logging()
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        if (
            mixed_precision == "bf16"
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError(
                f"BF16 was requested but assigned device {self.device} "
                "does not support BF16"
            )
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = str(PLANAR_ROOT)
        self.is_slot_model = bool(str(self.cfg.model._target_).endswith("SlotMaskWorldModel"))

        self.num_reconstruct_samples = int(
            self.cfg.training.get("num_reconstruct_samples", 6)
        )
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        requested_micro_batch = cfg.training.get("micro_batch_size", None)
        self.gradient_accumulation_steps = int(
            cfg.training.get("gradient_accumulation_steps", 1)
        )
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if requested_micro_batch is None:
            cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        else:
            cfg.gpu_batch_size = int(requested_micro_batch)
        expected_batch = (
            cfg.gpu_batch_size
            * self.accelerator.num_processes
            * self.gradient_accumulation_steps
        )
        if expected_batch != cfg.effective_batch_size:
            raise ValueError(
                "micro_batch_size * num_processes * gradient_accumulation_steps "
                f"must equal effective batch size: {expected_batch} != "
                f"{cfg.effective_batch_size}"
            )
        cfg.gradient_accumulation_steps = self.gradient_accumulation_steps
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                self.wandb_run = wandb.init(
                    project="dino_wm_debug",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            else:
                self.wandb_run = wandb.init(
                    project="dino_wm",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed)
        log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
            # Slot-AC now follows dino-wm: every sample is a contiguous temporal
            # window.  Velocity/contact information must come from real history.
            random_transition_sampling=False,
            causal_transition_sampling=bool(self.is_slot_model),
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]
        if len(self.datasets["train"]) == 0:
            source_lengths = [
                self.train_traj_dset.get_seq_length(index)
                for index in range(min(len(self.train_traj_dset), 1024))
            ]
            length_summary = (
                "unknown"
                if not source_lengths
                else f"sampled range [{min(source_lengths)}, {max(source_lengths)}]"
            )
            required_span = (
                (int(self.cfg.num_hist) + int(self.cfg.num_pred) - 1)
                * int(self.cfg.frameskip)
                + 1
            )
            raise ValueError(
                "Training dataset produced zero causal windows: "
                f"num_hist={self.cfg.num_hist}, num_pred={self.cfg.num_pred}, "
                f"frameskip={self.cfg.frameskip} requires at least "
                f"{required_span} frames, but source lengths are {length_summary}."
            )
        self.validation_dataset_keys = [
            key for key in self.datasets if key == "valid" or key.startswith("valid_")
        ]
        self.additional_validation_keys = [
            key for key in self.validation_dataset_keys if key != "valid"
        ]
        self.validation_traj_dsets = {
            key: traj_dsets[key] for key in self.validation_dataset_keys
        }

        global_validation_batch_size = int(
            self.cfg.training.get("validation_batch_size", cfg.effective_batch_size)
        )
        if global_validation_batch_size < self.accelerator.num_processes:
            raise ValueError(
                "validation_batch_size must be at least the number of processes"
            )
        if global_validation_batch_size % self.accelerator.num_processes != 0:
            raise ValueError(
                "validation_batch_size must be divisible by the number of processes"
            )
        # Match training.batch_size semantics: configuration values are global,
        # while each distributed DataLoader receives only its per-rank share.
        self.validation_batch_size = (
            global_validation_batch_size // self.accelerator.num_processes
        )
        dataloader_num_workers = int(
            self.cfg.training.get("num_workers", self.cfg.env.num_workers)
        )
        validation_num_workers = int(
            self.cfg.training.get("validation_num_workers", dataloader_num_workers)
        )
        if dataloader_num_workers < 0 or validation_num_workers < 0:
            raise ValueError("DataLoader worker counts must be non-negative")
        train_sampler = None
        collision_pair_fraction = self.cfg.training.get(
            "collision_pair_fraction", None
        )
        contact_pair_fraction = self.cfg.training.get(
            "contact_pair_fraction", None
        )
        rotation_bin_edges = self.cfg.training.get(
            "rotation_balance_bin_edges_degrees", None
        )
        balancing_modes = sum(
            value is not None
            for value in (
                collision_pair_fraction,
                contact_pair_fraction,
                rotation_bin_edges,
            )
        )
        if balancing_modes > 1:
            raise ValueError(
                "Configure only one training collision/contact/rotation balancing mode"
            )
        if rotation_bin_edges is not None:
            if self.cfg.env.name != "pusht":
                raise ValueError("Rotation balancing supports only PushT/PushObj")
            rotation_degrees = _pusht_rotation_window_degrees(
                self.datasets["train"]
            )
            train_sampler = StratifiedBinSampler(
                rotation_degrees,
                lower_edges=rotation_bin_edges,
                seed=int(self.cfg.training.seed),
            )
            log.info(
                "Rotation-balanced training sampler: edges(deg)=%s; raw=%s; "
                "effective=%s; samples=%d",
                list(train_sampler.lower_edges),
                list(train_sampler.raw_counts),
                list(train_sampler.bin_quotas),
                len(train_sampler),
            )
        elif collision_pair_fraction is not None or contact_pair_fraction is not None:
            balance_label = "collision" if collision_pair_fraction is not None else "contact"
            pair_fraction = (
                collision_pair_fraction
                if collision_pair_fraction is not None
                else contact_pair_fraction
            )
            if self.cfg.env.name == "wall" and collision_pair_fraction is not None:
                collision_mask = _wall_collision_window_mask(self.datasets["train"])
            elif self.cfg.env.name == "pusht" and contact_pair_fraction is not None:
                collision_mask = _pusht_contact_window_mask(
                    self.datasets["train"],
                    tolerance_px=float(
                        self.cfg.training.get("contact_tolerance_px", 0.0)
                    ),
                )
            else:
                raise ValueError(
                    "collision_pair_fraction supports Wall; "
                    "contact_pair_fraction supports PushT/PushObj"
                )
            train_sampler = CollisionBalancedSampler(
                collision_mask,
                minimum_fraction=float(pair_fraction),
                seed=int(self.cfg.training.seed),
            )
            log.info(
                "%s-balanced training sampler: raw=%d/%d (%.2f%%), "
                "effective=%d/%d (%.2f%%)",
                balance_label.capitalize(),
                int(collision_mask.sum()),
                collision_mask.numel(),
                100.0 * collision_mask.float().mean().item(),
                train_sampler.num_collision,
                len(train_sampler),
                100.0 * train_sampler.num_collision / len(train_sampler),
            )
        per_rank_iteration_limit = self.cfg.training.get(
            "max_iterations_per_epoch", None
        )
        if per_rank_iteration_limit is None:
            self.max_local_iterations_per_epoch = None
        else:
            per_rank_iteration_limit = int(per_rank_iteration_limit)
            if per_rank_iteration_limit < 1:
                raise ValueError("max_iterations_per_epoch must be positive or null")
            # Every DDP rank executes the same number of synchronized steps.
            # Since training.batch_size is global, this makes samples/epoch
            # equal to batch_size * iterations, independent of world size.
            self.max_local_iterations_per_epoch = per_rank_iteration_limit
        self.dataloaders = {
            "train": torch.utils.data.DataLoader(
                self.datasets["train"],
                batch_size=self.cfg.gpu_batch_size,
                # Reshuffle before taking the capped prefix every epoch, so
                # the iteration limit samples the full trajectory population.
                shuffle=train_sampler is None,
                sampler=train_sampler,
                num_workers=dataloader_num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=dataloader_num_workers > 0,
                collate_fn=cached_slot_collate if self.is_slot_model else None,
            ),
        }
        for validation_key in self.validation_dataset_keys:
            self.dataloaders[validation_key] = torch.utils.data.DataLoader(
                self.datasets[validation_key],
                batch_size=self.validation_batch_size,
                shuffle=False,
                num_workers=validation_num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=validation_num_workers > 0,
                collate_fn=cached_slot_collate if self.is_slot_model else None,
            )

        log.info(
            "dataloader micro batch size: %s; validation batch size: %s; "
            "train/validation num_workers: %s/%s; gradient accumulation steps: %s; "
            "effective batch size: %s; max train iterations/rank/epoch: %s; "
            "mixed precision: %s",
            self.cfg.gpu_batch_size,
            self.validation_batch_size,
            dataloader_num_workers,
            validation_num_workers,
            self.gradient_accumulation_steps,
            self.cfg.effective_batch_size,
            self.max_local_iterations_per_epoch,
            self.mixed_precision,
        )

        prepared_keys = ["train", *self.validation_dataset_keys]
        prepared_loaders = self.accelerator.prepare(
            *(self.dataloaders[key] for key in prepared_keys)
        )
        for key, loader in zip(prepared_keys, prepared_loaders):
            self.dataloaders[key] = loader

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.model = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        log.info(f"Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}")

        self._keys_to_save = [
            "epoch",
        ]
        if self.is_slot_model:
            self._keys_to_save += ["model", "model_optimizer"]
        else:
            self._keys_to_save += (
                ["encoder", "encoder_optimizer"] if self.train_encoder else []
            )
            self._keys_to_save += (
                ["predictor", "predictor_optimizer"]
                if self.train_predictor and self.cfg.has_predictor
                else []
            )
            self._keys_to_save += (
                ["decoder", "decoder_optimizer"] if self.train_decoder else []
            )
            self._keys_to_save += ["action_encoder", "proprio_encoder"]

        self.init_models()
        self.prepare_slot_cache_if_needed()
        self.init_optimizers()

        self.epoch_log = OrderedDict()
        self.debug_timing = bool(self.cfg.get("debug", False)) and self.is_slot_model
        self.debug_profile_steps = int(self.cfg.training.get("debug_profile_steps", 5))

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            ckpt = {}
            if self.is_slot_model:
                slot_model = self.accelerator.unwrap_model(self.model)
                predictor_state = slot_model.downstream_predictor_state_dict()
                slot_model.validate_proprio_encoder_checkpoint(predictor_state)
                ckpt["epoch"] = self.epoch
                ckpt["checkpoint_format"] = "slot_jepa_planar_predictor_v1"
                ckpt["pretrain_checkpoint"] = "slot_jepa_pretrained.pt"
                ckpt["predictor_state"] = predictor_state
                retrieval_bank_path = self.retrieval_bank_path
                if retrieval_bank_path is not None:
                    retrieval_bank_path = Path(retrieval_bank_path)
                    try:
                        retrieval_bank_path = retrieval_bank_path.relative_to(
                            self._resolve_ckpt_base_dir()
                        )
                    except ValueError:
                        # Keep a usable path for deliberately external cache roots.
                        pass
                    retrieval_bank_path = str(retrieval_bank_path)
                ckpt["retrieval_bank_path"] = retrieval_bank_path
                ckpt["model_optimizer_state"] = self.model_optimizer.state_dict()
                if self.model_lr_scheduler is not None:
                    ckpt["model_lr_scheduler_state"] = (
                        self.model_lr_scheduler.state_dict()
                    )
            else:
                for k in self._keys_to_save:
                    if hasattr(self.__dict__[k], "module"):
                        ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])
                    else:
                        ckpt[k] = self.__dict__[k]
            torch.save(ckpt, "checkpoints/model_latest.pth")
            torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_{self.epoch}.pth")
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        ckpt = torch.load(filename)
        if self.is_slot_model:
            self.epoch = int(ckpt.get("epoch", 0))
            return ckpt
        for k, v in ckpt.items():
            self.__dict__[k] = v
        not_in_ckpt = set(self._keys_to_save) - set(ckpt.keys())
        if len(not_in_ckpt):
            log.warning("Keys not found in ckpt: %s", not_in_ckpt)
        return ckpt

    def init_models(self):
        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        slot_ckpt = None
        if model_ckpt.exists():
            slot_ckpt = self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")

        if self.is_slot_model:
            if self.model is None:
                self.model = hydra.utils.instantiate(
                    self.cfg.model,
                    state_dim=self.datasets["train"].state_dim,
                    action_dim=self.datasets["train"].action_dim,
                    state_mean=self.train_traj_dset.state_mean,
                    state_std=self.train_traj_dset.state_std,
                    action_mean=self.train_traj_dset.action_mean,
                    action_std=self.train_traj_dset.action_std,
                    frameskip=self.cfg.frameskip,
                    env_name=self.cfg.env.name,
                )
            if slot_ckpt is not None and (
                "predictor_state" in slot_ckpt or "model_state" in slot_ckpt
            ):
                is_compact = "predictor_state" in slot_ckpt
                checkpoint_state = slot_ckpt[
                    "predictor_state" if is_compact else "model_state"
                ]
                self.model.validate_proprio_encoder_checkpoint(
                    checkpoint_state
                )
                expected_schema = int(
                    self.model.transition_pair_schema_version.item()
                )
                schema = checkpoint_state.get(
                    "transition_pair_schema_version"
                )
                if schema is None or int(schema.item()) != expected_schema:
                    raise RuntimeError(
                        "The checkpoint does not use transition-pair schema "
                        f"v{expected_schema} "
                        "and cannot be resumed safely. Start a fresh Slot-AC run."
                    )
                if self.cfg.model.get("predictor_proprio_indices") is not None:
                    proprio_schema = checkpoint_state.get(
                        "predictor_proprio_schema_version"
                    )
                    if proprio_schema is None or int(proprio_schema.item()) != 2:
                        raise RuntimeError(
                            "The checkpoint conditions PushT on privileged block state. "
                            "Start a fresh run with predictor_proprio_indices."
                        )
                if is_compact:
                    missing = self.model.load_downstream_predictor_state_dict(
                        checkpoint_state
                    )
                else:
                    missing = self.model.load_state_dict(
                        checkpoint_state, strict=False
                    )
                if bool(self.cfg.model.get("training_retrieval_context", False)):
                    retrieval_training_schema = checkpoint_state.get(
                        "training_retrieval_schema_version"
                    )
                    expected_retrieval_schema = (
                        3
                        if bool(
                            self.cfg.model.get(
                                "training_retrieval_same_layout_only", False
                            )
                        )
                        else (
                            2
                            if bool(
                                self.cfg.model.get(
                                    "training_retrieval_exclude_same_layout", False
                                )
                            )
                            else 1
                        )
                    )
                    if (
                        retrieval_training_schema is None
                        or int(retrieval_training_schema.item())
                        != expected_retrieval_schema
                    ):
                        raise RuntimeError(
                            "The checkpoint retrieval-training schema does not match "
                            "the configured layout-filtering behavior. "
                            "Start a fresh Slot-AC training run."
                        )
                if "transition_pair_schema_version" in missing.missing_keys:
                    raise RuntimeError(
                        "The checkpoint predates the retrieval-context predictor "
                        "and cannot be resumed safely. Start a fresh Slot-AC run."
                    )
                log.info(
                    "Loaded %s slot model state: checkpoint_tensors=%d, "
                    "merged_model_tensors=%d, strict_load=%s.",
                    "compact predictor" if is_compact else "legacy full",
                    len(checkpoint_state),
                    len(self.model.state_dict()),
                    is_compact,
                )
            if not self.model.has_fixed_slots:
                reference_visual = self._build_fixed_slot_reference_frames()
                reference_seed = int(self.cfg.model.get("fixed_reference_seed", 0))
                torch.manual_seed(reference_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(reference_seed)
                self.model.to(self.device)
                self.model.initialize_fixed_slots(
                    reference_visual.unsqueeze(0).to(self.device)
                )
                # Reference initialization must not perturb the subsequent
                # training RNG stream.
                seed(self.cfg.training.seed)
                log.info(
                    "Initialized and stored fixed predictor slot bank from %d "
                    "%s reference frames",
                    len(reference_visual),
                    self.cfg.env.name,
                )
            else:
                log.info("Restored fixed predictor slot bank from checkpoint")
            self.model = self.accelerator.prepare(self.model)
            return

        # initialize encoder
        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
            )
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.proprio_encoder = hydra.utils.instantiate(
            self.cfg.proprio_encoder,
            in_chans=self.datasets["train"].proprio_dim,
            emb_dim=self.cfg.proprio_emb_dim,
        )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        if self.accelerator.is_main_process:
            log.info("Proprio encoder type: %s", type(self.proprio_encoder))
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        self.action_encoder = hydra.utils.instantiate(
            self.cfg.action_encoder,
            in_chans=self.datasets["train"].action_dim,
            emb_dim=self.cfg.action_emb_dim,
        )
        action_emb_dim = self.action_encoder.emb_dim
        if self.accelerator.is_main_process:
            log.info("Action encoder type: %s", type(self.action_encoder))

        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # initialize predictor
        if self.encoder.latent_ndim == 1:  # if feature is 1D
            num_patches = 1
        else:
            decoder_scale = 16  # from vqvae
            num_side_patches = self.cfg.img_size // decoder_scale
            num_patches = num_side_patches**2

        if self.cfg.concat_dim == 0:
            num_patches += 2

        if self.cfg.has_predictor:
            if self.predictor is None:
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=self.encoder.emb_dim
                    + (
                        proprio_emb_dim * self.cfg.num_proprio_repeat
                        + action_emb_dim * self.cfg.num_action_repeat
                    )
                    * (self.cfg.concat_dim),
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        emb_dim=self.encoder.emb_dim,  # 384
                    )
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False
        self.encoder, self.predictor, self.decoder = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder
        )
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
        )

    def _build_fixed_slot_reference_frames(self):
        """Generate the deterministic environment reference clip once per run."""
        if self.cfg.env.name == "point_maze":
            # mujoco_py chooses the first CUDA_VISIBLE_DEVICES entry whenever
            # GPUS is unset, which sends every distributed rank's EGL context
            # to physical GPU 0. Map the Accelerate local rank back to its
            # corresponding visible physical device before creating a renderer.
            local_rank = int(self.accelerator.local_process_index)
            visible_devices = [
                value.strip()
                for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                if value.strip()
            ]
            render_device = (
                visible_devices[local_rank]
                if local_rank < len(visible_devices)
                and visible_devices[local_rank].isdigit()
                else str(local_rank)
            )
            os.environ["GPUS"] = render_device
            log.info(
                "Binding point_maze MuJoCo renderer on local rank %d to EGL GPU %s",
                local_rank,
                render_device,
            )
        import gym
        import env as _registered_envs  # noqa: F401
        from fixed_slot_references import (
            _sample_balanced_layout_dataset_references,
            _sample_configured_live_pusht_references,
            _sample_fixed_slot_references,
            _sample_random_dataset_references,
        )

        reference_seed = int(self.cfg.model.get("fixed_reference_seed", 0))
        num_reference_frames = int(
            self.cfg.model.get("fixed_reference_num_frames", 32)
        )
        if num_reference_frames < 1:
            raise ValueError("model.fixed_reference_num_frames must be positive")
        reference_source = str(
            self.cfg.model.get("fixed_reference_source", "live_env")
        )
        if reference_source == "dataset":
            reference_sampling = str(
                self.cfg.model.get("fixed_reference_sampling", "uniform_frames")
            )
            sampler = {
                "uniform_frames": _sample_random_dataset_references,
                "balanced_layout": _sample_balanced_layout_dataset_references,
            }.get(reference_sampling)
            if sampler is None:
                raise ValueError(
                    "model.fixed_reference_sampling must be 'uniform_frames' or "
                    f"'balanced_layout', got {reference_sampling!r}"
                )
            frames, labels = sampler(
                self.train_traj_dset,
                num_frames=num_reference_frames,
                seed=reference_seed,
            )
            layout_counts = {}
            for label in labels:
                layout_name = str(label[0])
                layout_counts[layout_name] = layout_counts.get(layout_name, 0) + 1
            log.info(
                "Sampled %d fixed slot references from held-in training data "
                "with sampling=%s counts=%s",
                len(frames),
                reference_sampling,
                layout_counts,
            )
            return frames.float()
        if reference_source != "live_env":
            raise ValueError(
                "model.fixed_reference_source must be 'live_env' or 'dataset', "
                f"got {reference_source!r}"
            )
        env_instance = gym.make(
            self.cfg.env.name,
            *self.cfg.env.args,
            **self.cfg.env.kwargs,
        )
        configured_shapes = self.cfg.model.get("fixed_reference_shapes")
        configured_colors = self.cfg.model.get("fixed_reference_colors")
        if configured_shapes is not None or configured_colors is not None:
            if self.cfg.env.name != "pusht":
                env_instance.close()
                raise ValueError(
                    "model.fixed_reference_shapes/colors are supported only for pusht"
                )
            if configured_shapes is None or configured_colors is None:
                env_instance.close()
                raise ValueError(
                    "fixed_reference_shapes and fixed_reference_colors must be set together"
                )
            reference_shapes = [
                str(value)
                for value in OmegaConf.to_container(configured_shapes, resolve=True)
            ]
            reference_colors = [
                str(value)
                for value in OmegaConf.to_container(configured_colors, resolve=True)
            ]
            if not reference_shapes or len(reference_shapes) != len(set(reference_shapes)):
                env_instance.close()
                raise ValueError("fixed_reference_shapes must be nonempty and unique")
            if not reference_colors:
                env_instance.close()
                raise ValueError("fixed_reference_colors must be nonempty")
            training_shapes = set(map(str, getattr(self.train_traj_dset, "shapes", ())))
            if training_shapes and set(reference_shapes) != training_shapes:
                env_instance.close()
                raise ValueError(
                    "PushObj fixed references must cover every training shape exactly; "
                    f"configured={sorted(reference_shapes)}, "
                    f"training={sorted(training_shapes)}"
                )
            frames_per_shape = int(
                self.cfg.model.get(
                    "fixed_reference_frames_per_shape",
                    num_reference_frames // len(reference_shapes),
                )
            )
            try:
                frames, _ = _sample_configured_live_pusht_references(
                    env_instance,
                    seed=reference_seed,
                    num_frames=num_reference_frames,
                    shapes=reference_shapes,
                    colors=reference_colors,
                    frames_per_shape=frames_per_shape,
                )
            finally:
                env_instance.close()
            log.info(
                "Sampled one shared PushObj slot bank from %d live references: "
                "%d frames/shape, shapes=%s, colors=%s",
                len(frames),
                frames_per_shape,
                reference_shapes,
                reference_colors,
            )
            return frames
        configured_layouts = self.cfg.model.get("fixed_reference_layouts")
        pusht_layouts = (
            OmegaConf.to_container(configured_layouts, resolve=True)
            if configured_layouts is not None
            else None
        )
        if pusht_layouts is not None:
            reference_shapes = [str(layout["shape"]) for layout in pusht_layouts]
            if len(reference_shapes) != len(set(reference_shapes)):
                raise ValueError(
                    "model.fixed_reference_layouts contains duplicate PushObj shapes"
                )
            training_shapes = set(map(str, getattr(self.train_traj_dset, "shapes", ())))
            if training_shapes and set(reference_shapes) != training_shapes:
                raise ValueError(
                    "PushObj fixed references must cover every training shape exactly; "
                    f"configured={sorted(reference_shapes)}, "
                    f"training={sorted(training_shapes)}"
                )
        try:
            frames, _ = _sample_fixed_slot_references(
                env_instance,
                env_name=self.cfg.env.name,
                seed=reference_seed,
                num_frames=num_reference_frames,
                pusht_layouts=pusht_layouts,
            )
        finally:
            env_instance.close()
        return frames.float()

    def init_optimizers(self):
        if self.is_slot_model:
            slot_model = self.accelerator.unwrap_model(self.model)
            pretrained_params, new_params = (
                slot_model.split_pretrained_and_new_parameters()
            )
            pretrained_lr = float(self.cfg.training.predictor_lr)
            new_lr_multiplier = float(
                self.cfg.training.get("new_module_lr_multiplier", 10.0)
            )
            if pretrained_lr <= 0.0 or new_lr_multiplier <= 0.0:
                raise ValueError(
                    "predictor_lr and new_module_lr_multiplier must be positive"
                )
            new_module_lr = pretrained_lr * new_lr_multiplier
            weight_decay = float(self.cfg.training.get("weight_decay", 1.0e-2))
            if weight_decay < 0.0:
                raise ValueError("weight_decay must be non-negative")
            self.model_optimizer = torch.optim.AdamW(
                [
                    {"params": pretrained_params, "lr": pretrained_lr},
                    {"params": new_params, "lr": new_module_lr},
                ],
                lr=pretrained_lr,
                weight_decay=weight_decay,
            )
            log.info(
                "Slot AdamW groups: pretrained=%d params lr=%.6g; "
                "new=%d params lr=%.6g (x%.3g); weight_decay=%.6g",
                sum(param.numel() for param in pretrained_params),
                pretrained_lr,
                sum(param.numel() for param in new_params),
                new_module_lr,
                new_lr_multiplier,
                weight_decay,
            )
            model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
            if model_ckpt.exists():
                ckpt = torch.load(model_ckpt, map_location="cpu")
                if "model_optimizer_state" in ckpt:
                    try:
                        self.model_optimizer.load_state_dict(
                            ckpt["model_optimizer_state"]
                        )
                    except ValueError as error:
                        log.warning(
                            "Skipping incompatible optimizer state after model "
                            "parameter changes: %s",
                            error,
                        )
            self.model_optimizer = self.accelerator.prepare(self.model_optimizer)
            self.model_lr_scheduler = None
            scheduler_name = str(
                self.cfg.training.get("lr_scheduler", "constant")
            ).lower()
            if scheduler_name == "cosine":
                min_lr_ratio = float(
                    self.cfg.training.get("min_lr_ratio", 0.1)
                )
                if not 0.0 <= min_lr_ratio <= 1.0:
                    raise ValueError("training.min_lr_ratio must be in [0, 1]")
                epoch_iterations = len(self.dataloaders["train"])
                if self.max_local_iterations_per_epoch is not None:
                    epoch_iterations = min(
                        epoch_iterations, self.max_local_iterations_per_epoch
                    )
                updates_per_epoch = epoch_iterations // self.gradient_accumulation_steps
                if updates_per_epoch < 1:
                    raise ValueError(
                        "Cosine LR scheduling requires at least one complete "
                        "optimizer update per epoch"
                    )
                total_updates = int(self.total_epochs) * updates_per_epoch
                completed_updates = min(
                    int(self.epoch) * updates_per_epoch, total_updates
                )

                def cosine_multiplier(step):
                    progress = min(max(int(step), 0), total_updates) / total_updates
                    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                        1.0 + math.cos(math.pi * progress)
                    )

                for param_group in self.model_optimizer.param_groups:
                    param_group.setdefault("initial_lr", param_group["lr"])
                self.model_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.model_optimizer,
                    lr_lambda=cosine_multiplier,
                    last_epoch=completed_updates - 1,
                )
                if model_ckpt.exists() and "model_lr_scheduler_state" in ckpt:
                    self.model_lr_scheduler.load_state_dict(
                        ckpt["model_lr_scheduler_state"]
                    )
                log.info(
                    "Cosine LR schedule: %d updates (%d/epoch), min ratio=%.3g, "
                    "resumed at update %d; current lr=%s",
                    total_updates,
                    updates_per_epoch,
                    min_lr_ratio,
                    completed_updates,
                    [group["lr"] for group in self.model_optimizer.param_groups],
                )
            elif scheduler_name != "constant":
                raise ValueError(
                    "training.lr_scheduler must be 'constant' or 'cosine', got "
                    f"{scheduler_name!r}"
                )
            return
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.training.encoder_lr,
        )
        self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)
        if self.cfg.has_predictor:
            weight_decay = float(self.cfg.training.get("weight_decay", 1.0e-2))
            self.predictor_optimizer = torch.optim.AdamW(
                self.predictor.parameters(),
                lr=self.cfg.training.predictor_lr,
                weight_decay=weight_decay,
            )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )

            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.action_encoder.parameters(), self.proprio_encoder.parameters()
                ),
                lr=self.cfg.training.action_encoder_lr,
                weight_decay=weight_decay,
            )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )

        if self.cfg.has_decoder:
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=self.cfg.training.decoder_lr
            )
            self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)

    def _resolve_ckpt_base_dir(self):
        ckpt_base = Path(str(self.cfg.ckpt_base_path)).expanduser()
        if not ckpt_base.is_absolute():
            original_cwd = Path(hydra.utils.get_original_cwd())
            ckpt_base = (original_cwd / ckpt_base).resolve()
        return ckpt_base

    def _slot_cache_file_path(self, dataset_name):
        ckpt_base_dir = self._resolve_ckpt_base_dir()
        cache_dir = ckpt_base_dir / "slot_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_cfg = self.cfg.model
        realm_tag = Path(str(model_cfg.realm_ckpt_path)).stem
        sample_dataset = self.datasets.get(
            "train" if dataset_name == "train" else dataset_name.replace("val", "valid", 1)
        )
        trajectory_dataset = getattr(sample_dataset, "dataset", sample_dataset)
        trajectory_indices = list(range(len(trajectory_dataset)))
        source_dataset = trajectory_dataset
        while isinstance(source_dataset, Subset):
            trajectory_indices = [
                int(source_dataset.indices[index]) for index in trajectory_indices
            ]
            source_dataset = source_dataset.dataset
        dataset_tag = Path(
            str(getattr(source_dataset, "data_path", self.cfg.env.dataset.data_path))
        ).name
        trajectory_lengths = [
            int(trajectory_dataset.get_seq_length(index))
            for index in range(len(trajectory_dataset))
        ]
        trajectory_identity = hashlib.sha256(
            repr((trajectory_indices, trajectory_lengths)).encode("utf-8")
        ).hexdigest()[:16]
        frame_stride = (
            int(sample_dataset.frameskip)
            if bool(sample_dataset.non_overlapping_causal_windows)
            and not bool(sample_dataset.random_transition_sampling)
            else 1
        )
        geometry_version = self.cfg.env.dataset.get("render_geometry_version")
        reference_version = model_cfg.get("fixed_reference_version")
        slot_cache_version = model_cfg.get("slot_cache_version", "legacy")
        color_channel_weights_version = model_cfg.get(
            "color_channel_weights_version", "disabled"
        )
        train_color_palette = self.cfg.env.dataset.get(
            "train_color_palette", None
        )
        color_override = self.cfg.env.dataset.get("color_override", None)
        color_palette_tag = (
            "none"
            if dataset_name != "train" or train_color_palette is None
            else ",".join(str(color) for color in train_color_palette)
        )
        color_seed_tag = (
            "none"
            if dataset_name != "train" or train_color_palette is None
            else str(int(self.cfg.env.dataset.get("train_color_seed", 0)))
        )
        # Keep the full cache identity in the digest rather than the filename.
        # Verbose filenames can exceed NAME_MAX once retrieval suffixes are
        # appended, while a digest still invalidates the cache on every field
        # that participated in the former filename.
        cache_identity = "|".join(
            (
                f"env={self.cfg.env.name}",
                f"dataset={dataset_tag}",
                f"split={dataset_name}",
                "storage=trajectory_masks_v2",
                f"frame_stride={frame_stride}",
                f"trajectories={len(trajectory_dataset)}",
                f"trajectory_identity={trajectory_identity}",
                f"img={self.cfg.img_size}",
                f"objects={model_cfg.num_objects}",
                f"grid={model_cfg.get('slot_spatial_size', 16)}",
                f"window={'x'.join(map(str, model_cfg.window_size))}",
                f"slots_per_window={model_cfg.slots_per_window}",
                f"flip_tta={int(bool(model_cfg.get('encoder_flip_tta', False)))}",
                f"temperature={float(model_cfg.get('mask_softmax_temperature', 0.2)):g}",
                f"gamma={float(model_cfg.get('mask_sharpen_gamma', 1.0)):g}",
                f"color_channel_weights={color_channel_weights_version}",
                f"reference_seed={model_cfg.get('fixed_reference_seed', 0)}",
                f"reference_source={model_cfg.get('fixed_reference_source', 'live_env')}",
                f"reference_sampling={model_cfg.get('fixed_reference_sampling', 'uniform_frames')}",
                f"reference_version={reference_version}",
                f"slot_cache_version={slot_cache_version}",
                f"train_color_palette={color_palette_tag}",
                f"train_color_seed={color_seed_tag}",
                f"color_override={color_override}",
                f"render={'state' if bool(self.cfg.env.dataset.get('render_from_state', False)) else 'video'}",
                f"geometry={geometry_version}",
                f"normalization={model_cfg.get('visual_normalization', 'imagenet')}",
                f"realm={realm_tag}",
            )
        )
        cache_digest = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()[:16]

        def short_tag(value, max_length):
            tag = "".join(
                char
                if char.isascii() and (char.isalnum() or char in "-_")
                else "-"
                for char in str(value)
            ).strip("-_")
            return (tag or "cache")[:max_length]

        return cache_dir / (
            f"maskframe_{short_tag(self.cfg.env.name, 16)}_"
            f"{short_tag(dataset_name, 12)}_"
            f"{short_tag(dataset_tag, 40)}_{cache_digest}.pt"
        )

    def _retrieval_bank_file_path(self, train_cache_path, sample_dataset=None):
        bank_size = int(self.cfg.model.get("retrieval_bank_size", 32768))
        collision_fraction = self.cfg.model.get(
            "retrieval_collision_pair_fraction", None
        )
        contact_fraction = self.cfg.model.get(
            "retrieval_contact_pair_fraction", None
        )
        rotation_edges = self.cfg.model.get(
            "retrieval_rotation_balance_bin_edges_degrees", None
        )
        if sum(
            value is not None
            for value in (collision_fraction, contact_fraction, rotation_edges)
        ) > 1:
            raise ValueError(
                "Configure only one retrieval collision/contact/rotation balancing mode"
            )
        collision_tag = ""
        pair_fraction = (
            collision_fraction if collision_fraction is not None else contact_fraction
        )
        if pair_fraction is not None:
            fraction_tag = f"{float(pair_fraction):.4f}".replace(".", "p")
            prefix = "cfrac" if collision_fraction is not None else "contactfrac"
            collision_tag = f"_{prefix}{fraction_tag}"
            if contact_fraction is not None:
                contact_tolerance = float(
                    self.cfg.model.get("retrieval_contact_tolerance_px", 0.0)
                )
                tolerance_tag = f"{contact_tolerance:g}"
                collision_tag += f"_contacttol{tolerance_tag.replace('.', 'p')}"
        if rotation_edges is not None:
            edge_tag = "-".join(f"{float(edge):g}" for edge in rotation_edges)
            collision_tag = f"_rotbins{edge_tag.replace('.', 'p')}"
        proprio_indices = self.cfg.model.get("predictor_proprio_indices")
        proprio_tag = ""
        if proprio_indices is not None:
            proprio_tag = "_p" + "-".join(
                str(int(index)) for index in proprio_indices
            )
        if bool(
            self.cfg.model.get("training_retrieval_same_layout_only", False)
        ):
            layout_tag = "_slayout1"
        elif bool(
            self.cfg.model.get("training_retrieval_exclude_same_layout", False)
        ):
            layout_tag = "_xlayout1"
        else:
            layout_tag = ""
        context_strategy = self.cfg.model.get(
            "training_context_generation_strategy", None
        )
        if context_strategy is None:
            context_tag = ""
        else:
            context_identity = repr(
                (
                    str(context_strategy),
                    int(self.cfg.model.get("training_context_bank_size", 1000)),
                    int(self.cfg.model.get("training_context_seed", 0)),
                    float(self.cfg.model.get("training_context_action_std", 2.0)),
                    float(self.cfg.model.get("training_context_action_temporal_correlation", 0.8)),
                    float(self.cfg.model.get("training_context_min_agent_displacement", 20.0)),
                    float(self.cfg.model.get("training_context_min_object_displacement", 10.0)),
                    float(self.cfg.model.get("training_context_initial_agent_block_clearance", 2.0)),
                    float(self.cfg.model.get("training_context_contact_tolerance", 1.0)),
                )
            )
            context_digest = hashlib.sha256(
                context_identity.encode("utf-8")
            ).hexdigest()[:12]
            context_tag = f"_ctxgenv2-{context_digest}"
            if sample_dataset is None:
                raise ValueError(
                    "Generated context-bank identity requires its shape dataset"
                )
            source_dataset = getattr(sample_dataset, "dataset", sample_dataset)
            while isinstance(source_dataset, Subset):
                source_dataset = source_dataset.dataset
            from datasets.pusht_dset import PUSHT_SHAPE_IDS

            shape_by_id = {value: key for key, value in PUSHT_SHAPE_IDS.items()}
            window_layout_ids = _window_layout_ids(sample_dataset)
            if window_layout_ids is None:
                raise RuntimeError(
                    "Generated PushObj context bank requires dataset layout ids"
                )
            shapes = sorted(
                shape_by_id[int(layout_id)]
                for layout_id in torch.unique(window_layout_ids).tolist()
            )
            if sample_dataset is self.datasets["train"]:
                configured_shapes = self.cfg.model.get(
                    "fixed_reference_shapes", None
                )
                if configured_shapes is not None:
                    shapes = sorted(map(str, configured_shapes))
            generated_identity = repr(
                (
                    context_identity,
                    tuple(shapes),
                    int(self.cfg.img_size),
                    int(self.cfg.model.get("slot_spatial_size", 16)),
                    float(self.cfg.model.get("mask_softmax_temperature", 1.0)),
                    float(self.cfg.model.get("mask_sharpen_gamma", 1.0)),
                    str(
                        self.cfg.model.get(
                            "color_channel_weights_version", "disabled"
                        )
                    ),
                    str(self.cfg.model.get("fixed_reference_version", "legacy")),
                    str(self.cfg.model.get("slot_cache_version", "legacy")),
                    tuple(self.cfg.model.get("predictor_proprio_indices", ())),
                )
            )
            generated_digest = hashlib.sha256(
                generated_identity.encode("utf-8")
            ).hexdigest()[:16]
            shape_tag = "-".join(shapes)
            return train_cache_path.parent / (
                f"context_{self.cfg.env.name}_{shape_tag}_{generated_digest}.pt"
            )
        return train_cache_path.with_name(
            f"{train_cache_path.stem}{proprio_tag}_ret{bank_size}"
            f"_h{self.cfg.num_hist}p{self.cfg.num_pred}f{self.cfg.frameskip}"
            f"{layout_tag}{collision_tag}{context_tag}_idxv1.pt"
        )

    def _build_retrieval_bank(self, sample_dataset, bank_path):
        """Materialize a compact, evenly sampled bank of real transitions."""
        if not self.accelerator.is_main_process:
            return
        if sample_dataset.slot_cache is None:
            raise RuntimeError("Slot cache must be attached before retrieval-bank build")
        bank_size = int(self.cfg.model.get("retrieval_bank_size", 32768))
        transitions_per_window = int(self.cfg.num_hist + self.cfg.num_pred - 1)
        fields = {
            "source_states": [],
            "target_states": [],
            "actions": [],
            "source_mask_logits": [],
            "target_mask_logits": [],
            "layout_ids": [],
            "sample_indices": [],
            "transition_indices": [],
        }
        context_strategy = self.cfg.model.get(
            "training_context_generation_strategy", None
        )
        if context_strategy is not None:
            raise ValueError(
                "Dataset retrieval-bank construction cannot be combined with "
                f"generated context strategy {context_strategy!r}"
            )
        seen_transitions = set()
        stored = 0
        retrieval_model = self.accelerator.unwrap_model(self.model)
        proprio_indices = self.cfg.model.get("predictor_proprio_indices")
        sample_indices = list(range(len(sample_dataset)))
        collision_fraction = self.cfg.model.get(
            "retrieval_collision_pair_fraction", None
        )
        contact_fraction = self.cfg.model.get(
            "retrieval_contact_pair_fraction", None
        )
        rotation_edges = self.cfg.model.get(
            "retrieval_rotation_balance_bin_edges_degrees", None
        )
        if sum(
            value is not None
            for value in (collision_fraction, contact_fraction, rotation_edges)
        ) > 1:
            raise ValueError(
                "Configure only one retrieval collision/contact/rotation balancing mode"
            )
        pair_fraction = (
            collision_fraction if collision_fraction is not None else contact_fraction
        )
        balance_label = "collision" if collision_fraction is not None else "contact"
        selected_collision_count = None
        pair_mask = None
        if rotation_edges is not None:
            if self.cfg.env.name != "pusht":
                raise ValueError(
                    "Retrieval rotation balancing supports only PushT/PushObj"
                )
            rotation_degrees = _pusht_rotation_window_degrees(sample_dataset)
            layout_ids = _window_layout_ids(sample_dataset)
            groups = (
                [None]
                if layout_ids is None
                else [int(value) for value in torch.unique(layout_ids).tolist()]
            )
            generator = torch.Generator().manual_seed(int(self.cfg.training.seed))
            base_quota, quota_remainder = divmod(bank_size, len(groups))
            selected_parts = []
            raw_bin_counts = torch.zeros(len(rotation_edges), dtype=torch.long)
            selected_bin_counts = torch.zeros(len(rotation_edges), dtype=torch.long)
            boundaries = torch.tensor(
                [float(edge) for edge in rotation_edges[1:]], dtype=torch.float32
            )
            all_labels = torch.bucketize(
                rotation_degrees, boundaries, right=True
            )
            for group_index, layout_id in enumerate(groups):
                group_indices = (
                    torch.arange(len(sample_dataset))
                    if layout_id is None
                    else torch.nonzero(
                        layout_ids == layout_id, as_tuple=False
                    ).flatten()
                )
                group_quota = base_quota + int(group_index < quota_remainder)
                local_selected, _, _ = _select_balanced_unique_indices(
                    rotation_degrees[group_indices],
                    rotation_edges,
                    group_quota,
                    generator,
                )
                selected_parts.append(group_indices[local_selected])
                raw_bin_counts += torch.bincount(
                    all_labels[group_indices], minlength=len(rotation_edges)
                )
                selected_bin_counts += torch.bincount(
                    all_labels[group_indices[local_selected]],
                    minlength=len(rotation_edges),
                )
            selected = torch.cat(selected_parts)
            selected = selected[
                torch.randperm(selected.numel(), generator=generator)
            ]
            sample_indices = selected.tolist()
            log.info(
                "Building rotation-balanced retrieval bank: edges(deg)=%s; "
                "raw=%s; selected=%s; total=%d",
                [float(edge) for edge in rotation_edges],
                raw_bin_counts.tolist(),
                selected_bin_counts.tolist(),
                selected.numel(),
            )
        elif pair_fraction is not None:
            pair_fraction = float(pair_fraction)
            if not 0.0 < pair_fraction <= 1.0:
                raise ValueError(
                    "retrieval pair fraction must be in (0, 1]"
                )
            if collision_fraction is not None:
                if self.cfg.env.name != "wall":
                    raise ValueError(
                        "retrieval_collision_pair_fraction supports only Wall"
                    )
                collision_mask = _wall_collision_window_mask(sample_dataset)
            else:
                if self.cfg.env.name != "pusht":
                    raise ValueError(
                        "retrieval_contact_pair_fraction supports PushT/PushObj"
                    )
                collision_mask = _pusht_contact_window_mask(
                    sample_dataset,
                    tolerance_px=float(
                        self.cfg.model.get("retrieval_contact_tolerance_px", 0.0)
                    ),
                )
            pair_mask = collision_mask
            collision_indices = torch.nonzero(
                collision_mask, as_tuple=False
            ).flatten()
            free_indices = torch.nonzero(
                ~collision_mask, as_tuple=False
            ).flatten()
            if collision_indices.numel() == 0:
                raise ValueError(
                    f"No positive {balance_label} pairs are available for retrieval"
                )
            generator = torch.Generator().manual_seed(int(self.cfg.training.seed))
            layout_ids = _window_layout_ids(sample_dataset)
            groups = (
                [None]
                if layout_ids is None
                else [int(value) for value in torch.unique(layout_ids).tolist()]
            )
            base_quota, quota_remainder = divmod(bank_size, len(groups))
            selected_parts = []
            selected_collision_count = 0
            for group_index, layout_id in enumerate(groups):
                group_mask = (
                    torch.ones_like(collision_mask)
                    if layout_id is None
                    else layout_ids == layout_id
                )
                group_collision = torch.nonzero(
                    group_mask & collision_mask, as_tuple=False
                ).flatten()
                group_free = torch.nonzero(
                    group_mask & ~collision_mask, as_tuple=False
                ).flatten()
                quota = base_quota + int(group_index < quota_remainder)
                # Keep entries unique. A small validation layout may produce a
                # smaller group rather than silently falling below the target.
                group_total = min(
                    quota,
                    group_collision.numel() + group_free.numel(),
                    int(group_collision.numel() // pair_fraction),
                )
                group_collision_count = max(
                    math.ceil(pair_fraction * group_total),
                    group_total - group_free.numel(),
                )
                group_free_count = group_total - group_collision_count
                group_collision = group_collision[
                    torch.randperm(group_collision.numel(), generator=generator)[
                        :group_collision_count
                    ]
                ]
                group_free = group_free[
                    torch.randperm(group_free.numel(), generator=generator)[
                        :group_free_count
                    ]
                ]
                selected_parts.extend((group_collision, group_free))
                selected_collision_count += group_collision_count
            selected = torch.cat(selected_parts)
            selected = selected[
                torch.randperm(selected.numel(), generator=generator)
            ]
            sample_indices = selected.tolist()
            selected_total = selected.numel()
            log.info(
                "Building %s-balanced retrieval bank: positive=%d/%d "
                "(%.2f%%), available raw=%d/%d",
                balance_label,
                selected_collision_count,
                selected_total,
                100.0 * selected_collision_count / selected_total,
                int(collision_mask.sum()),
                collision_mask.numel(),
            )
        stored_positive_count = 0
        stored_balance_by_layout = {}
        for sample_index in tqdm(
            sample_indices, desc=f"Retrieval {bank_path.stem}"
        ):
            obs, actions, states = sample_dataset[sample_index]
            layout_id = obs.get("layout_id")
            layout_filtered = bool(
                self.cfg.model.get(
                    "training_retrieval_exclude_same_layout", False
                )
            ) or bool(
                self.cfg.model.get(
                    "training_retrieval_same_layout_only", False
                )
            )
            if layout_id is None and layout_filtered:
                raise RuntimeError(
                    "Training retrieval is configured to filter by query layout, "
                    "but the dataset sample has no layout_id"
                )
            logits = obs["slot_mask_logits"]
            count = min(transitions_per_window, states.shape[0] - 1)
            for transition_index in range(count):
                raw_source_state = states[transition_index].float().cpu()
                raw_target_state = states[transition_index + 1].float().cpu()
                action = actions[transition_index].float().cpu()
                # Overlapping trajectory windows contain exact copies of the
                # same transition. Do not let those copies occupy every top-k
                # neighbour position.
                key = torch.cat(
                    [raw_source_state, action, raw_target_state]
                ).numpy().tobytes()
                if layout_id is not None:
                    key += (
                        torch.as_tensor(layout_id)
                        .cpu()
                        .long()
                        .reshape(1)
                        .numpy()
                        .tobytes()
                    )
                if key in seen_transitions:
                    continue
                seen_transitions.add(key)
                # Persist only the predictor-visible controllable state. For
                # PushT this is [pusher_x, pusher_y, pusher_vx, pusher_vy];
                # block pose remains represented exclusively by masks.
                source_state = retrieval_model._select_predictor_proprio(
                    raw_source_state
                )
                target_state = retrieval_model._select_predictor_proprio(
                    raw_target_state
                )
                fields["source_states"].append(source_state.unsqueeze(0))
                fields["target_states"].append(target_state.unsqueeze(0))
                fields["actions"].append(action.unsqueeze(0))
                fields["source_mask_logits"].append(
                    logits[transition_index].to(torch.float16).unsqueeze(0)
                )
                fields["target_mask_logits"].append(
                    logits[transition_index + 1].to(torch.float16).unsqueeze(0)
                )
                if layout_id is not None:
                    fields["layout_ids"].append(
                        torch.as_tensor(layout_id).long().reshape(1)
                    )
                fields["sample_indices"].append(
                    torch.as_tensor(sample_index).long().reshape(1)
                )
                fields["transition_indices"].append(
                    torch.as_tensor(transition_index).long().reshape(1)
                )
                if pair_mask is not None:
                    is_positive = bool(pair_mask[sample_index].item())
                    stored_positive_count += int(is_positive)
                    layout_key = (
                        None
                        if layout_id is None
                        else int(torch.as_tensor(layout_id).item())
                    )
                    positive_count, total_count = stored_balance_by_layout.get(
                        layout_key, (0, 0)
                    )
                    stored_balance_by_layout[layout_key] = (
                        positive_count + int(is_positive),
                        total_count + 1,
                    )
                stored += 1
                if stored >= bank_size:
                    break
            if stored >= bank_size:
                break
        if stored < int(self.cfg.model.get("retrieval_num_context", 3)):
            raise RuntimeError(
                f"Only {stored} unique transitions are available for retrieval"
            )
        if pair_mask is not None:
            required_positive = math.ceil(pair_fraction * stored)
            if stored_positive_count < required_positive:
                raise RuntimeError(
                    f"Final retrieval bank has only {stored_positive_count}/{stored} "
                    f"{balance_label}-positive pairs; required at least "
                    f"{pair_fraction:.2%} after transition deduplication"
                )
            sparse_balance = {
                layout_id: (positive_count, total_count)
                for layout_id, (positive_count, total_count) in
                stored_balance_by_layout.items()
                if positive_count < math.ceil(pair_fraction * total_count)
            }
            if sparse_balance:
                raise RuntimeError(
                    "Final retrieval bank fell below the requested positive "
                    f"fraction in layouts {sparse_balance}"
                )
        payload = {
            key: torch.cat(values, dim=0)[:bank_size].contiguous()
            for key, values in fields.items()
            if values
        }
        payload["schema_version"] = 4
        payload["dataset_size"] = len(sample_dataset)
        payload["proprio_indices"] = (
            None
            if proprio_indices is None
            else [int(index) for index in proprio_indices]
        )
        if bool(
            self.cfg.model.get("training_retrieval_same_layout_only", False)
        ):
            required_per_layout = int(
                self.cfg.model.get("retrieval_num_context", 3)
            ) + 1
            layout_ids, layout_counts = torch.unique(
                payload["layout_ids"], return_counts=True
            )
            too_small = layout_counts < required_per_layout
            if too_small.any():
                sparse = {
                    int(layout_id): int(count)
                    for layout_id, count in zip(
                        layout_ids[too_small], layout_counts[too_small]
                    )
                }
                raise RuntimeError(
                    "Same-layout retrieval bank has too few transitions for "
                    f"layouts {sparse}; need at least {required_per_layout} each"
                )
        torch.save(payload, bank_path)
        log.info(
            "Saved %d real transitions to retrieval bank %s%s",
            payload["source_states"].shape[0],
            bank_path,
            (
                ""
                if selected_collision_count is None
                else (
                    f" with verified {balance_label} fraction "
                    f"{stored_positive_count / stored:.2%}"
                )
            ),
        )

    @torch.no_grad()
    def _build_generated_pushobj_retrieval_bank(self, sample_dataset, bank_path):
        """Generate the structured PushObj context bank in the simulator.

        This deliberately does not read transitions from ``sample_dataset``;
        the dataset is used only to determine which shape/layout partitions
        the bank must cover and which image transform to apply.
        """
        if not self.accelerator.is_main_process:
            return
        if self.cfg.env.name != "pusht":
            raise ValueError("Generated structured contexts are only valid for PushT/PushObj")
        import gym
        import env as _registered_envs  # noqa: F401
        from env.serial_vector_env import SerialVectorEnv
        from planning.evaluator import PlanEvaluator
        from preprocessor import Preprocessor
        from datasets.pusht_dset import PUSHT_SHAPE_IDS

        source_dataset = getattr(sample_dataset, "dataset", sample_dataset)
        while isinstance(source_dataset, Subset):
            source_dataset = source_dataset.dataset
        shape_by_id = {value: key for key, value in PUSHT_SHAPE_IDS.items()}
        window_layout_ids = _window_layout_ids(sample_dataset)
        if window_layout_ids is None:
            raise RuntimeError(
                "Generated PushObj context bank requires dataset layout ids"
            )
        available_shapes = sorted(
            shape_by_id[int(layout_id)]
            for layout_id in torch.unique(window_layout_ids).tolist()
        )
        configured_shapes = self.cfg.model.get("fixed_reference_shapes", None)
        if configured_shapes is not None and sample_dataset is self.datasets["train"]:
            requested_shapes = list(map(str, configured_shapes))
        else:
            requested_shapes = available_shapes
        missing_shapes = sorted(set(requested_shapes).difference(available_shapes))
        if missing_shapes:
            raise RuntimeError(
                f"Generated context bank requested shapes absent from dataset: {missing_shapes}"
            )
        shape_ids = [PUSHT_SHAPE_IDS[shape] for shape in requested_shapes]

        env_kwargs = OmegaConf.to_container(
            self.cfg.env.kwargs, resolve=True
        )
        envs = []
        for shape in requested_shapes:
            shape_kwargs = dict(env_kwargs)
            shape_kwargs["shape"] = shape
            envs.append(
                gym.make(
                    self.cfg.env.name,
                    *self.cfg.env.args,
                    **shape_kwargs,
                )
            )
        vector_env = SerialVectorEnv(envs)
        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()
        try:
            preprocessor = Preprocessor(
                action_mean=self.train_traj_dset.action_mean,
                action_std=self.train_traj_dset.action_std,
                state_mean=self.train_traj_dset.state_mean,
                state_std=self.train_traj_dset.state_std,
                proprio_mean=self.train_traj_dset.proprio_mean,
                proprio_std=self.train_traj_dset.proprio_std,
                transform=source_dataset.transform,
            )
            # Reuse the exact simulator generator used by planning without
            # constructing planning state or retrieving from an existing bank.
            generator = PlanEvaluator.__new__(PlanEvaluator)
            generator.env = vector_env
            generator.wm = model
            generator.frameskip = int(self.cfg.frameskip)
            generator.preprocessor = preprocessor
            generator.device = self.device
            generator.context_layout_ids = shape_ids
            generator.state_0_torch = torch.zeros(
                len(shape_ids), 1, device=self.device
            )
            generator.normalized_action_clip = None
            generator._normalized_action_low = None
            generator._normalized_action_high = None
            generator._initialize_action_bounds()

            bank_size = int(
                self.cfg.model.get("training_context_bank_size", 1000)
            )
            partition_count = len(shape_ids) * len(
                PUSHOBJ_STRUCTURED_CONTEXT_NAMES
            )
            if bank_size < partition_count or bank_size % partition_count != 0:
                raise ValueError(
                    "training_context_bank_size must divide evenly across every "
                    f"shape and structured mode; got size={bank_size}, "
                    f"shapes={len(shape_ids)}, modes={len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES)}"
                )
            pool_size = bank_size // len(shape_ids)
            pools = generator._generate_random_context_pools(
                pool_size=pool_size,
                seed_value=int(self.cfg.model.get("training_context_seed", 0)),
                action_std=float(
                    self.cfg.model.get("training_context_action_std", 2.0)
                ),
                action_temporal_correlation=float(
                    self.cfg.model.get(
                        "training_context_action_temporal_correlation", 0.8
                    )
                ),
                min_agent_displacement=float(
                    self.cfg.model.get(
                        "training_context_min_agent_displacement", 20.0
                    )
                ),
                min_object_displacement=float(
                    self.cfg.model.get(
                        "training_context_min_object_displacement", 10.0
                    )
                ),
                initial_agent_block_clearance=float(
                    self.cfg.model.get(
                        "training_context_initial_agent_block_clearance", 2.0
                    )
                ),
                require_contact=True,
                contact_tolerance=float(
                    self.cfg.model.get("training_context_contact_tolerance", 1.0)
                ),
                velocity_std_scale=1.0,
                generation_strategy="pushobj_structured_torque",
            )
            payload = {
                "source_states": torch.cat(
                    [model._select_predictor_proprio(pool["source_states"]) for pool in pools]
                ).contiguous(),
                "target_states": torch.cat(
                    [model._select_predictor_proprio(pool["target_states"]) for pool in pools]
                ).contiguous(),
                "actions": torch.cat([pool["actions"] for pool in pools]).contiguous(),
                "source_mask_logits": torch.cat(
                    [pool["source_mask_logits"] for pool in pools]
                ).contiguous(),
                "target_mask_logits": torch.cat(
                    [pool["target_mask_logits"] for pool in pools]
                ).contiguous(),
                "context_modes": torch.cat(
                    [pool["context_modes"] for pool in pools]
                ).contiguous(),
                "layout_ids": torch.cat(
                    [
                        torch.full((pool_size,), layout_id, dtype=torch.long)
                        for layout_id in shape_ids
                    ]
                ),
                "schema_version": 2,
                "proprio_indices": model.predictor_proprio_indices.tolist(),
                "generated_context": True,
                "generation_strategy": "pushobj_structured_torque",
            }
            temporary = bank_path.with_name(
                f".{bank_path.name}.{os.getpid()}.tmp"
            )
            torch.save(payload, temporary)
            os.replace(temporary, bank_path)
            log.info(
                "Saved %d directly generated PushObj contexts (%d per shape, "
                "%d per shape/mode) to %s",
                payload["source_states"].shape[0],
                pool_size,
                pool_size // len(PUSHOBJ_STRUCTURED_CONTEXT_NAMES),
                bank_path,
            )
        finally:
            if was_training:
                model.train()
            for child in vector_env.envs:
                child.close()

    @torch.no_grad()
    def _build_slot_cache(self, sliced_dataset, cache_path):
        if self.accelerator.is_main_process:
            log.info(
                "Building %s slot cache at %s with %d ranks",
                self.cfg.env.name,
                cache_path,
                self.accelerator.num_processes,
            )
        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()

        trajectory_dataset = sliced_dataset.dataset
        # Non-overlapping causal windows always start on the model-step grid,
        # so only frames 0, frameskip, 2*frameskip, ... can ever be observed,
        # independent of num_hist/num_pred. Sliding/random windows retain all
        # raw frames because their start offsets are unrestricted.
        frame_stride = (
            int(sliced_dataset.frameskip)
            if bool(sliced_dataset.non_overlapping_causal_windows)
            and not bool(sliced_dataset.random_transition_sampling)
            else 1
        )
        frame_dataset = TrajectoryFrameDataset(
            trajectory_dataset, frame_stride=frame_stride
        )
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        local_indices = list(range(rank, len(frame_dataset), world_size))
        cache_batch_size = int(self.cfg.training.get("cache_batch_size", 32))
        cache_num_workers = int(self.cfg.training.get("cache_num_workers", 4))
        if cache_batch_size < 1 or cache_num_workers < 0:
            raise ValueError("cache_batch_size must be positive and cache_num_workers non-negative")
        cache_loader_kwargs = {}
        if cache_num_workers > 0:
            # Cache construction starts after the rank process has initialized
            # CUDA and, for rendered datasets, potentially EGL/MuJoCo.  Never
            # fork that initialized process: use clean worker interpreters for
            # every planar environment.  Besides avoiding invalid GL state,
            # this prevents workers from inheriting the rank's CUDA allocator.
            cache_loader_kwargs["multiprocessing_context"] = "spawn"
            # Cache frames are large; one prefetched batch per worker is enough
            # to overlap loading/rendering with encoder execution while keeping
            # host-memory usage bounded across distributed ranks.
            cache_loader_kwargs["prefetch_factor"] = 1
        cache_loader = DataLoader(
            Subset(frame_dataset, local_indices),
            batch_size=cache_batch_size,
            shuffle=False,
            num_workers=cache_num_workers,
            # Cache batches are large (for example, a float32 PushT batch at
            # 512 px is about 96 MiB per rank).  CUDA's pinned-host allocator
            # can retain one large shared mapping after another during this
            # long one-pass workload, causing host memory to grow without a
            # useful steady-state cache.  Pageable memory plus a synchronous
            # copy keeps the batch lifetime bounded.
            pin_memory=False,
            persistent_workers=cache_num_workers > 0,
            **cache_loader_kwargs,
        )
        local_cache = []
        iterator = tqdm(
            cache_loader,
            total=len(cache_loader),
            desc=f"Cache {cache_path.stem} rank{rank}",
            disable=not self.accelerator.is_main_process,
        )
        for visuals, states, trajectory_indices, frame_indices, layout_ids in iterator:
            visual = visuals[:, None].to(
                self.device, dtype=torch.float32, non_blocking=False
            )
            # Cache predictor-ready masks so training does not repeatedly
            # interpolate the extractor's native grid after every load.
            slots = model._extract_slots(visual)
            mask_logits = slots["mask_logits"].detach().cpu().to(torch.float16)
            # DataLoader worker outputs use file-descriptor-backed shared
            # storage.  Keeping a view such as states[batch_index] in the
            # long-lived local_cache also keeps that batch's descriptor open;
            # a long cache build eventually fails with "Too many open files".
            # Clone once per batch so cached state views reference ordinary
            # rank-local storage and the worker batch can be released here.
            cached_states = states.detach().cpu().float().clone()
            local_cache.extend(
                (
                    int(trajectory_indices[batch_index]),
                    int(frame_indices[batch_index]),
                    mask_logits[batch_index, 0],
                    cached_states[batch_index],
                    int(layout_ids[batch_index]),
                )
                for batch_index in range(visual.shape[0])
            )

        # Avoid all_gather_object for multi-GB caches. Each rank writes a local
        # shard and rank zero merges them from disk one at a time.
        shard_path = cache_path.with_name(f"{cache_path.name}.rank{rank}.tmp")
        torch.save(local_cache, shard_path)
        del local_cache
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            episodes = [
                {
                    "frame_indices": torch.as_tensor(
                        frame_dataset.frame_indices_by_trajectory[index],
                        dtype=torch.long,
                    ),
                    "mask_logits": [None]
                    * len(frame_dataset.frame_indices_by_trajectory[index]),
                    "states": [None]
                    * len(frame_dataset.frame_indices_by_trajectory[index]),
                    "layout_id": None,
                }
                for index in range(len(trajectory_dataset))
            ]
            frame_positions = [
                {
                    raw_index: cache_index
                    for cache_index, raw_index in enumerate(frame_indices)
                }
                for frame_indices in frame_dataset.frame_indices_by_trajectory
            ]
            merged_count = 0
            for shard_rank in range(world_size):
                rank_shard_path = cache_path.with_name(
                    f"{cache_path.name}.rank{shard_rank}.tmp"
                )
                rank_cache = torch.load(rank_shard_path, map_location="cpu")
                for (
                    trajectory_index,
                    frame_index,
                    mask_logits,
                    state,
                    layout_id,
                ) in rank_cache:
                    episode = episodes[trajectory_index]
                    cache_index = frame_positions[trajectory_index][frame_index]
                    if episode["mask_logits"][cache_index] is not None:
                        raise RuntimeError(
                            "Duplicate trajectory-frame slot cache entry for "
                            f"({trajectory_index}, {frame_index})"
                        )
                    episode["mask_logits"][cache_index] = mask_logits
                    episode["states"][cache_index] = state
                    if layout_id >= 0:
                        episode["layout_id"] = layout_id
                    merged_count += 1
                del rank_cache
                rank_shard_path.unlink()
            if merged_count != len(frame_dataset):
                missing = [
                    (trajectory_index, frame_index)
                    for trajectory_index, episode in enumerate(episodes)
                    for frame_index, value in enumerate(episode["mask_logits"])
                    if value is None
                ]
                raise RuntimeError(
                    f"Distributed slot cache build incomplete for {cache_path}: "
                    f"expected {len(frame_dataset)} frames, got {merged_count}. "
                    f"Missing first indices: {missing[:20]}"
                )
            for trajectory_index, episode in enumerate(episodes):
                episode["mask_logits"] = torch.stack(episode["mask_logits"])
                episode["states"] = torch.stack(episode["states"])
                raw_length = trajectory_dataset.get_seq_length(trajectory_index)
                frame_to_cache = torch.full((raw_length,), -1, dtype=torch.long)
                frame_to_cache[episode["frame_indices"]] = torch.arange(
                    episode["frame_indices"].numel(), dtype=torch.long
                )
                episode["frame_to_cache"] = frame_to_cache
                if episode["layout_id"] is None:
                    episode.pop("layout_id")
            cache = {
                "schema_version": 3,
                "storage": "trajectory_masks",
                "frame_stride": frame_stride,
                "episodes": episodes,
            }
            temporary = cache_path.with_name(
                f".{cache_path.name}.{os.getpid()}.tmp"
            )
            torch.save(cache, temporary)
            os.replace(temporary, cache_path)
        if was_training:
            model.train()
        self.accelerator.wait_for_everyone()

    def _attach_slot_cache(self, dataset, cache):
        if hasattr(dataset, "set_slot_cache"):
            dataset.set_slot_cache(cache)
        else:
            raise AttributeError(f"Dataset {type(dataset).__name__} does not support slot cache attachment")

    def prepare_slot_cache_if_needed(self, load_retrieval_bank=True):
        if not self.is_slot_model or not bool(self.cfg.training.get("cache_slots", True)):
            return

        cache_targets = [("train", self.datasets["train"])] + [
            (key.replace("valid", "val", 1), self.datasets[key])
            for key in self.validation_dataset_keys
        ]

        loaded_caches = {}
        cache_paths = {}
        for dataset_name, sample_dataset in cache_targets:
            cache_path = self._slot_cache_file_path(dataset_name)
            cache_paths[dataset_name] = cache_path
            if not cache_path.exists():
                self._build_slot_cache(sample_dataset, cache_path)
            if self.accelerator.is_main_process:
                log.info("Loading %s slot cache from %s", self.cfg.env.name, cache_path)
            loaded_caches[dataset_name] = torch.load(cache_path, map_location="cpu")
            dataset_key = (
                "train"
                if dataset_name == "train"
                else dataset_name.replace("val", "valid", 1)
            )
            self._attach_slot_cache(
                self.datasets[dataset_key], loaded_caches[dataset_name]
            )

        if str(self.cfg.model.get("predictor_architecture", "transition_memory")) == "sequential":
            # A1 consumes only contiguous trajectory context. Building and
            # moving a 32k transition bank to every GPU wastes substantial
            # memory (especially for native 32x32 Push-Object masks) and can
            # OOM before the first optimizer step.
            self.retrieval_bank_path = None
            self.validation_retrieval_bank_paths = {}
            self.validation_retrieval_bank_path = None
            if self.accelerator.is_main_process:
                log.info(
                    "Sequential predictor: skipping retrieval-bank build/load"
                )
            return

        def ensure_retrieval_bank(sample_dataset, bank_path):
            if bank_path.exists():
                return
            generated_context = (
                self.cfg.model.get("training_context_generation_strategy")
                == "pushobj_structured_torque"
            )
            if self.accelerator.is_main_process:
                if generated_context:
                    self._build_generated_pushobj_retrieval_bank(
                        sample_dataset, bank_path
                    )
                else:
                    self._build_retrieval_bank(sample_dataset, bank_path)
                if not bank_path.exists():
                    raise FileNotFoundError(
                        f"Retrieval bank builder did not create {bank_path}"
                    )
                return

            # Direct environment generation plus 512px slot encoding can take
            # longer than the process group's five-minute collective timeout.
            # Wait on the atomically published file instead of entering NCCL
            # while rank zero is still generating it.
            wait_timeout = float(
                self.cfg.model.get("training_context_generation_timeout_seconds", 21600)
            )
            deadline = time.monotonic() + wait_timeout
            while not bank_path.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for rank zero to create {bank_path}"
                    )
                time.sleep(5.0)

        retrieval_bank_path = self._retrieval_bank_file_path(
            cache_paths["train"], self.datasets["train"]
        )
        ensure_retrieval_bank(self.datasets["train"], retrieval_bank_path)
        validation_online_context = bool(
            self.cfg.model.get("validation_online_context", False)
        )
        validation_bank_paths = {}
        if validation_online_context:
            for validation_key in self.validation_dataset_keys:
                cache_name = validation_key.replace("valid", "val", 1)
                validation_bank_path = self._retrieval_bank_file_path(
                    cache_paths[cache_name], self.datasets[validation_key]
                )
                ensure_retrieval_bank(
                    self.datasets[validation_key], validation_bank_path
                )
                validation_bank_paths[validation_key] = str(
                    validation_bank_path.resolve()
                )
        self.accelerator.wait_for_everyone()
        self.retrieval_bank_path = str(retrieval_bank_path.resolve())
        self.validation_retrieval_bank_paths = validation_bank_paths
        # Backward-compatible alias used by OOD PCA visualization.
        self.validation_retrieval_bank_path = validation_bank_paths.get("valid")
        if load_retrieval_bank:
            self.accelerator.unwrap_model(self.model).load_retrieval_bank(
                self.retrieval_bank_path
            )

    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    log.info("Logging result for %s at epoch %s: %s", job_name, epoch, result)
                    log_data = {
                        f"{job_name}/{key}": value for key, value in result.items()
                    }
                    log_data["epoch"] = epoch
                    self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        if bool(self.cfg.training.get("cache_only", False)):
            if self.accelerator.is_main_process:
                log.info("Slot caches are ready; exiting because training.cache_only=true")
            self.accelerator.wait_for_everyone()
            return

        # ``training.epochs`` is the absolute target epoch, not the number of
        # additional epochs to append after a resume. The old range ended at
        # ``loaded_epoch + training.epochs`` (for example, resuming epoch 25
        # with epochs=100 incorrectly trained through epoch 125).
        if self.epoch >= self.total_epochs:
            if self.accelerator.is_main_process:
                log.info(
                    "Checkpoint epoch %d already reached target epoch %d; "
                    "skipping training",
                    self.epoch,
                    self.total_epochs,
                )
            self.accelerator.wait_for_everyone()
            return

        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        if self.is_slot_model:
            initial_epoch = self.epoch
            self.epoch = 0
            self.export_slot_pca()
            self.epoch = initial_epoch

        init_epoch = self.epoch + 1  # epoch starts from 1
        for epoch in range(init_epoch, self.total_epochs + 1):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()
            if self.is_slot_model and bool(
                self.cfg.model.get("training_retrieval_context", False)
            ):
                # Retrieval distances use the current learned state encoder.
                # Refresh once per epoch; selected context rows are still
                # re-encoded with gradients inside every forward pass.
                self.accelerator.unwrap_model(
                    self.model
                ).refresh_retrieval_state_embeddings()
            self.train()
            self.accelerator.wait_for_everyone()
            self.val()
            self.logs_flash(step=self.epoch)
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt()
                # main thread only: launch planning jobs on the saved ckpt
                plan_settings = self.cfg.get("plan_settings", None)
                plan_cfg_path = None if plan_settings is None else plan_settings.get("plan_cfg_path", None)
                if (
                    plan_cfg_path is not None
                    and ckpt_path is not None
                ):  # ckpt_path is only not None for main process
                    from plan import build_plan_cfg_dicts, launch_plan_jobs

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, plan_cfg_path
                        ),
                        ckpt_base_path=self.cfg.ckpt_base_path,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=plan_settings.planner,
                        goal_source=plan_settings.goal_source,
                        goal_H=plan_settings.goal_H,
                        alpha=plan_settings.alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(), "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def train(self):
        if self.is_slot_model:
            self.model_optimizer.zero_grad()
        else:
        # Clear stale gradients at the start of every epoch. In accumulation mode,
        # an incomplete trailing micro-batch is intentionally not stepped so every
        # optimizer update has the requested effective batch size.
            self.encoder_optimizer.zero_grad()
            if self.cfg.has_decoder:
                self.decoder_optimizer.zero_grad()
            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

        prev_end_time = time.perf_counter()
        train_loader = self.dataloaders["train"]
        if hasattr(train_loader, "set_epoch"):
            train_loader.set_epoch(self.epoch)
        local_iteration_limit = self.max_local_iterations_per_epoch
        if local_iteration_limit is None:
            epoch_iterator = train_loader
            epoch_iterations = len(train_loader)
        else:
            epoch_iterations = min(local_iteration_limit, len(train_loader))
            epoch_iterator = itertools.islice(train_loader, epoch_iterations)
        for i, data in enumerate(
            tqdm(
                epoch_iterator,
                total=epoch_iterations,
                desc=f"Epoch {self.epoch} Train",
                disable=not self.accelerator.is_main_process,
            )
        ):
            step_timing = {}
            if self.debug_timing:
                step_timing["dataloader_wait_s"] = time.perf_counter() - prev_end_time
                self._reset_cuda_peak()
                step_timing["mem_step_start_mb"] = self._cuda_mem_mb()
                t_transfer = self._tic()
            obs, act, state = data
            plot = i == 0  # only plot from the first batch
            self.model.train()
            if self.debug_timing:
                step_timing["device_transfer_s"] = self._toc(t_transfer)
                step_timing["mem_after_transfer_mb"] = self._cuda_mem_mb()
                step_timing["mem_peak_after_transfer_mb"] = self._cuda_peak_mb()
                t_forward = self._tic()
            with self.accelerator.autocast():
                if self.is_slot_model:
                    z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                        obs, act, state=state
                    )
                else:
                    z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                        obs, act
                    )
            if self.debug_timing:
                step_timing["forward_outer_s"] = self._toc(t_forward)
                step_timing["mem_after_forward_mb"] = self._cuda_mem_mb()
                step_timing["mem_peak_after_forward_mb"] = self._cuda_peak_mb()

            # Divide before backward so accumulated gradients equal the gradient
            # of one full effective batch, independent of micro-batch size.
            if self.debug_timing:
                t_backward = self._tic()
            self.accelerator.backward(
                loss / self.gradient_accumulation_steps
            )
            if self.debug_timing:
                step_timing["backward_only_s"] = self._toc(t_backward)
                step_timing["mem_after_backward_mb"] = self._cuda_mem_mb()
                step_timing["mem_peak_after_backward_mb"] = self._cuda_peak_mb()
            should_step = (i + 1) % self.gradient_accumulation_steps == 0

            if should_step:
                if self.debug_timing:
                    t_step = self._tic()
                if self.is_slot_model:
                    self.model_optimizer.step()
                    if self.model_lr_scheduler is not None:
                        self.model_lr_scheduler.step()
                        self.logs_update(
                            {
                                "train_lr": [
                                    self.model_optimizer.param_groups[0]["lr"]
                                ]
                            }
                        )
                    self.model_optimizer.zero_grad()
                else:
                    if self.model.train_encoder:
                        self.encoder_optimizer.step()
                    if self.cfg.has_decoder and self.model.train_decoder:
                        self.decoder_optimizer.step()
                    if self.cfg.has_predictor and self.model.train_predictor:
                        self.predictor_optimizer.step()
                        self.action_encoder_optimizer.step()

                    self.encoder_optimizer.zero_grad()
                    if self.cfg.has_decoder:
                        self.decoder_optimizer.zero_grad()
                    if self.cfg.has_predictor:
                        self.predictor_optimizer.zero_grad()
                        self.action_encoder_optimizer.zero_grad()
                if self.debug_timing:
                    step_timing["optim_step_zero_s"] = self._toc(t_step)
                    step_timing["mem_after_optim_mb"] = self._cuda_mem_mb()
                    step_timing["mem_peak_after_optim_mb"] = self._cuda_peak_mb()

            if self.debug_timing:
                model_timing = getattr(self.accelerator.unwrap_model(self.model), "latest_timing", {})
                timing_logs = {f"train_time_{k}": [float(v)] for k, v in model_timing.items()}
                timing_logs.update({f"train_time_{k}": [float(v)] for k, v in step_timing.items()})
                self.logs_update(timing_logs)
                if i < self.debug_profile_steps and self.accelerator.is_main_process:
                        log.info(
                            "%s",
                            {
                                "timing_phase": "train",
                                "step": int(i),
                                **{k: round(v[0], 4) for k, v in timing_logs.items()},
                            },
                        )
            if self.cfg.has_decoder and plot and not self.is_slot_model:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = {
                        key: value.mean() for key, value in err_logs.items()
                    }
                    err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = {
                            f"train_img_{k}_pred": [v.mean()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = {
                            f"train_img_{k}_reconstructed": [v.mean()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)
            prev_end_time = time.perf_counter()

    def val(self):
        self.model.eval()
        slot_model = (
            self.accelerator.unwrap_model(self.model)
            if self.is_slot_model
            else None
        )
        validation_bank_path = getattr(
            self, "validation_retrieval_bank_path", None
        )
        if validation_bank_path is not None:
            # Validation contexts come from held-out-layout observations, as
            # a loss-level proxy for the real online contexts used by planning.
            slot_model.load_retrieval_bank(validation_bank_path)
            log.info(
                "Validation uses %d same-layout observed contexts from the "
                "held-out validation split",
                int(self.cfg.model.get("retrieval_num_context", 5)),
            )
        if (not self.is_slot_model) and len(self.train_traj_dset) > 0 and self.cfg.has_predictor:
            with torch.no_grad():
                train_rollout_logs = self.openloop_rollout(
                    self.train_traj_dset, mode="train"
                )
                train_rollout_logs = {
                    f"train_{k}": [v] for k, v in train_rollout_logs.items()
                }
                self.logs_update(train_rollout_logs)
                val_rollout_logs = self.openloop_rollout(self.val_traj_dset, mode="val")
                val_rollout_logs = {
                    f"val_{k}": [v] for k, v in val_rollout_logs.items()
                }
                self.logs_update(val_rollout_logs)

        self.accelerator.wait_for_everyone()
        with torch.no_grad():
            prev_end_time = time.perf_counter()
            for i, data in enumerate(
                tqdm(
                    self.dataloaders["valid"],
                    desc=f"Epoch {self.epoch} Valid",
                    disable=not self.accelerator.is_main_process,
                )
            ):
                step_timing = {}
                if self.debug_timing:
                    step_timing["dataloader_wait_s"] = time.perf_counter() - prev_end_time
                    t_transfer = self._tic()
                obs, act, state = data
                plot = i == 0
                self.model.eval()
                if self.debug_timing:
                    step_timing["device_transfer_s"] = self._toc(t_transfer)
                    t_forward = self._tic()
                with self.accelerator.autocast():
                    if self.is_slot_model:
                        z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                            obs, act, state=state
                        )
                    else:
                        z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                            obs, act
                        )
                if self.debug_timing:
                    step_timing["forward_outer_s"] = self._toc(t_forward)

                if self.debug_timing:
                    model_timing = getattr(self.accelerator.unwrap_model(self.model), "latest_timing", {})
                    timing_logs = {f"val_time_{k}": [float(v)] for k, v in model_timing.items()}
                    timing_logs.update({f"val_time_{k}": [float(v)] for k, v in step_timing.items()})
                    self.logs_update(timing_logs)
                    if i < self.debug_profile_steps and self.accelerator.is_main_process:
                        log.info(
                            "%s",
                            {
                                "timing_phase": "valid",
                                "step": int(i),
                                **{k: round(v[0], 4) for k, v in timing_logs.items()},
                            },
                        )

                if self.cfg.has_decoder and plot and not self.is_slot_model:
                    # only eval images when plotting due to speed
                    if self.cfg.has_predictor:
                        z_obs_out, z_act_out = self.model.separate_emb(z_out)
                        z_gt = self.model.encode_obs(obs)
                        z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                        state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                        err_logs = self.err_eval(z_obs_out, z_tgt)

                        err_logs = {
                            key: value.mean() for key, value in err_logs.items()
                        }
                        err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}

                        self.logs_update(err_logs)

                    if visual_out is not None:
                        for t in range(
                            self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                        ):
                            img_pred_scores = eval_images(
                                visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                            )
                            img_pred_scores = {
                                f"val_img_{k}_pred": [v.mean()]
                                for k, v in img_pred_scores.items()
                            }
                            self.logs_update(img_pred_scores)

                    if visual_reconstructed is not None:
                        for t in range(obs["visual"].shape[1]):
                            img_reconstruction_scores = eval_images(
                                visual_reconstructed[:, t], obs["visual"][:, t]
                            )
                            img_reconstruction_scores = {
                                f"val_img_{k}_reconstructed": [v.mean()]
                                for k, v in img_reconstruction_scores.items()
                            }
                            self.logs_update(img_reconstruction_scores)

                    self.plot_samples(
                        obs["visual"],
                        visual_out,
                        visual_reconstructed,
                        self.epoch,
                        batch=i,
                        num_samples=self.num_reconstruct_samples,
                        phase="valid",
                    )
                val_loss_components = {
                    f"val_{k}": [v] for k, v in loss_components.items()
                }
                self.logs_update(val_loss_components)
                if self.additional_validation_keys:
                    # Keep val_* as a backward-compatible alias while making
                    # the original held-out-shape domain explicit.
                    self.logs_update(
                        {
                            f"val_ood_{k}": [v]
                            for k, v in loss_components.items()
                        }
                    )
                prev_end_time = time.perf_counter()
        if self.is_slot_model:
            self._validate_additional_slot_datasets(slot_model)
        if self.is_slot_model:
            self.export_slot_pca()
        if validation_bank_path is not None:
            # Checkpoints and the next training epoch must always use the
            # immutable training retrieval bank.
            slot_model.load_retrieval_bank(self.retrieval_bank_path)

    def _validate_additional_slot_datasets(self, slot_model):
        """Evaluate named validation domains and log each under its own prefix."""
        validation_bank_paths = getattr(
            self, "validation_retrieval_bank_paths", {}
        )
        for validation_key in self.additional_validation_keys:
            metric_name = validation_key.removeprefix("valid_")
            bank_path = validation_bank_paths.get(validation_key)
            if bank_path is not None:
                slot_model.load_retrieval_bank(bank_path)
            self.accelerator.wait_for_everyone()
            with torch.no_grad():
                for data in tqdm(
                    self.dataloaders[validation_key],
                    desc=f"Epoch {self.epoch} Valid {metric_name.upper()}",
                    disable=not self.accelerator.is_main_process,
                ):
                    obs, act, state = data
                    self.model.eval()
                    with self.accelerator.autocast():
                        _, _, _, _, loss_components = self.model(
                            obs, act, state=state
                        )
                    self.logs_update(
                        {
                            f"val_{metric_name}_{key}": [value]
                            for key, value in loss_components.items()
                        }
                    )
    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.cfg.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.cfg.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                    start : 
                    start + horizon * self.cfg.frameskip + 1 : 
                    self.cfg.frameskip
                ]
            act = act[start : start + horizon * self.cfg.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)

                z_obses, z = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if self.cfg.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png",
                    )
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().reshape(-1)
                length = value.numel()
                increment = value.sum()
            else:
                length = len(value)
                detached_values = [
                    item.detach().float().mean()
                    if isinstance(item, torch.Tensor)
                    else torch.as_tensor(item, device=self.device, dtype=torch.float32)
                    for item in value
                ]
                increment = torch.stack(detached_values).sum()
            count, running_total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                running_total + increment,
            )

    def logs_flash(self, step):
        names = tuple(self.epoch_log.keys())
        local_sums = torch.stack(
            [
                torch.as_tensor(self.epoch_log[key][1], device=self.device).float()
                for key in names
            ]
        )
        local_counts = torch.tensor(
            [self.epoch_log[key][0] for key in names],
            device=self.device,
            dtype=torch.float32,
        )
        # One collective per epoch replaces the former all-gather in every
        # train and validation iteration.
        global_stats = self.accelerator.reduce(
            torch.stack((local_sums, local_counts)), reduction="sum"
        )
        epoch_log = OrderedDict(
            (
                key,
                (global_stats[0, index] / global_stats[1, index].clamp_min(1.0)).item(),
            )
            for index, key in enumerate(names)
        )
        epoch_log["epoch"] = step
        log.info(f"Epoch {self.epoch}  Training loss: {epoch_log['train_loss']:.4f}  \
                Validation loss: {epoch_log['val_loss']:.4f}")
        metric_items = []
        for key, value in epoch_log.items():
            if key == "epoch":
                continue
            if key.startswith("train_") or key.startswith("val_"):
                if isinstance(value, float):
                    metric_items.append(f"{key}={value:.6f}")
                else:
                    metric_items.append(f"{key}={value}")
        if metric_items:
            log.info("Epoch %s metrics: %s", self.epoch, " | ".join(metric_items))

        if self.accelerator.is_main_process:
            self.wandb_run.log(epoch_log)
        self.epoch_log = OrderedDict()

    @staticmethod
    def _slot_reconstruct_from_slots(mask_probs, obj_latents):
        if mask_probs.ndim == 3:
            mask_probs = mask_probs.unsqueeze(1)
        return torch.einsum("btkn,bkd->btnd", mask_probs.to(obj_latents.dtype), obj_latents)

    @staticmethod
    def _slot_flatten_feature_map(feat):
        if feat.ndim == 2:
            return feat
        if feat.ndim == 3:
            if feat.shape[-1] <= 8 and feat.shape[0] > 8 and feat.shape[1] > 8:
                return feat.reshape(-1, feat.shape[-1])
            if feat.shape[0] <= 8 and feat.shape[1] > 8 and feat.shape[2] > 8:
                return feat.permute(1, 2, 0).reshape(-1, feat.shape[0])
            return feat.reshape(-1, feat.shape[-1])
        raise ValueError(f"Unsupported feature shape for PCA: {tuple(feat.shape)}")

    @staticmethod
    def _slot_fit_pca(context_features):
        context_all = torch.cat(
            [Trainer._slot_flatten_feature_map(feat) for feat in context_features],
            dim=0,
        ).float()
        pca_mean = context_all.mean(dim=0, keepdim=True)
        centered = context_all - pca_mean
        _, _, pca_basis = torch.pca_lowrank(centered, q=3)
        proj = centered @ pca_basis
        pca_min = proj.amin(dim=0, keepdim=True)
        pca_max = proj.amax(dim=0, keepdim=True)
        return pca_mean, pca_basis, pca_min, pca_max

    @staticmethod
    def _slot_feature_to_pca_image(feat, pca_mean, pca_basis, pca_min, pca_max, patch_size):
        feat_flat = Trainer._slot_flatten_feature_map(feat).float()
        feat_proj = (feat_flat - pca_mean) @ pca_basis
        feat_proj = ((feat_proj - pca_min) / (pca_max - pca_min + 1.0e-8)).clamp(0.0, 1.0)
        spatial_size = int(math.sqrt(feat_proj.shape[0]))
        if spatial_size * spatial_size != feat_proj.shape[0]:
            raise ValueError(
                f"PCA feature token count is not a square number: {feat_proj.shape[0]} from shape {tuple(feat.shape)}"
            )
        pca_grid = feat_proj.view(spatial_size, spatial_size, 3).permute(2, 0, 1).unsqueeze(0)
        pca_img = F.interpolate(pca_grid, scale_factor=patch_size, mode="nearest").squeeze(0)
        return pca_img

    @staticmethod
    def _slot_unnorm_img(frame):
        frame = frame.detach().cpu().float()
        if frame.max() > 1.0:
            frame = frame / 255.0
        if frame.min() < 0.0:
            frame = (frame + 1.0) / 2.0
        return frame.clamp(0.0, 1.0)

    def _slot_prepare_vis_img(self, frame, target_hw):
        frame = self._slot_unnorm_img(frame)
        if tuple(frame.shape[-2:]) != tuple(target_hw):
            frame = F.interpolate(
                frame.unsqueeze(0),
                size=target_hw,
                mode="nearest",
            ).squeeze(0)
        return frame

    def _save_slot_rollout_pca_grid(
        self,
        context_images,
        context_encoder_features,
        context_features,
        pred_features,
        observed_images,
        observed_features,
        output_path,
    ):
        if not context_images:
            return
        patch_size = int(self.cfg.model.patch_size)
        encoder_pca = self._slot_fit_pca(context_encoder_features)
        shared_pca = self._slot_fit_pca(context_features)
        vis_hw = (patch_size * int(self.cfg.model.input_size // self.cfg.model.patch_size),) * 2

        ctx_img_row = [self._slot_prepare_vis_img(img, vis_hw) for img in context_images]
        ctx_encoder_pca_row = [
            self._slot_feature_to_pca_image(feat, *encoder_pca, patch_size)
            for feat in context_encoder_features
        ]
        ctx_pca_row = [
            self._slot_feature_to_pca_image(feat, *shared_pca, patch_size)
            for feat in context_features
        ]
        pred_pca_row = [
            self._slot_feature_to_pca_image(feat, *shared_pca, patch_size)
            for feat in pred_features
        ]
        obs_img_row = [self._slot_prepare_vis_img(img, vis_hw) for img in observed_images]
        obs_pca_row = [
            self._slot_feature_to_pca_image(feat, *shared_pca, patch_size)
            for feat in observed_features
        ]
        rows = [
            ctx_img_row,
            ctx_encoder_pca_row,
            ctx_pca_row,
            pred_pca_row,
            obs_img_row,
            obs_pca_row,
        ]
        row_titles = [
            "actual prior image (reference)",
            "actual prior encoder pca (reference)",
            "rollout query recon pca",
            "rollout prediction recon pca",
            "actual next image",
            "actual next recon pca",
        ]

        num_cols = len(ctx_img_row)
        fig, axes = plt.subplots(len(rows), num_cols, figsize=(2.4 * num_cols, 2.4 * len(rows)))
        if num_cols == 1:
            axes = np.expand_dims(axes, axis=1)
        for row_idx, row in enumerate(rows):
            for col_idx, img in enumerate(row):
                ax = axes[row_idx, col_idx]
                ax.imshow(
                    np.clip(img.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0),
                    interpolation="nearest",
                )
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(f"t{col_idx} → t{col_idx + 1}")
            axes[row_idx, 0].set_ylabel(row_titles[row_idx], fontsize=9)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def _save_slot_mask_grid(
        self,
        context_masks,
        pred_masks,
        observed_masks,
        output_path,
        retrieved_source_masks=None,
        retrieved_target_masks=None,
    ):
        if not context_masks:
            return
        num_steps = len(context_masks)
        num_objects = int(context_masks[0].shape[0])
        groups = [("rollout query mask", context_masks)]
        if retrieved_source_masks is not None:
            groups.append(("retrieved source mask #1", retrieved_source_masks))
        if retrieved_target_masks is not None:
            groups.append(("retrieved target mask #1", retrieved_target_masks))
        groups.extend(
            [
                ("rollout prediction mask", pred_masks),
                ("actual next mask", observed_masks),
            ]
        )
        fig, axes = plt.subplots(
            len(groups) * num_objects,
            num_steps,
            figsize=(2.0 * num_steps, 1.6 * len(groups) * num_objects),
        )
        if num_steps == 1:
            axes = np.expand_dims(axes, axis=1)
        for group_idx, (group_title, masks_per_step) in enumerate(groups):
            for obj_idx in range(num_objects):
                row_idx = group_idx * num_objects + obj_idx
                for step_idx in range(num_steps):
                    ax = axes[row_idx, step_idx]
                    mask_img = masks_per_step[step_idx][obj_idx].detach().cpu().float().numpy()
                    ax.imshow(mask_img, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
                    ax.axis("off")
                    if row_idx == 0:
                        ax.set_title(f"t{step_idx} → t{step_idx + 1}")
                axes[row_idx, 0].set_ylabel(f"{group_title} {obj_idx}", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    @torch.no_grad()
    def export_slot_pca(self):
        if not self.is_slot_model or not self.accelerator.is_main_process:
            return
        vis_every = int(self.cfg.training.get("pca_visualize_every", 0))
        if vis_every <= 0 or (self.epoch % vis_every != 0):
            return

        vis_cfg = self.cfg.training.get("pca_visualization", {})
        first_episode_idx = int(vis_cfg.get("episode_idx", 0))
        num_episodes = max(1, int(vis_cfg.get("num_episodes", 4)))
        selection = str(vis_cfg.get("selection", "sequential"))
        start = int(vis_cfg.get("start_raw_index", 0))
        num_steps_req = int(vis_cfg.get("num_steps", 12))

        num_available = len(self.val_traj_dset)
        if num_available == 0:
            return
        episode_indices = self._slot_visualization_episode_indices(
            first_episode_idx=first_episode_idx,
            num_episodes=min(num_episodes, num_available),
            selection=selection,
        )
        model = self.accelerator.unwrap_model(self.model)
        original_bank_path = model._retrieval_bank_path
        validation_bank_path = getattr(
            self, "validation_retrieval_bank_path", None
        )
        if (
            validation_bank_path is not None
            and str(original_bank_path) != str(validation_bank_path)
        ):
            model.load_retrieval_bank(validation_bank_path)
        for case_idx, episode_idx in enumerate(episode_indices):
            self._export_slot_pca_episode(
                episode_idx=episode_idx,
                start=start,
                num_steps_req=num_steps_req,
                case_idx=case_idx,
            )
        if (
            original_bank_path is not None
            and str(model._retrieval_bank_path) != str(original_bank_path)
        ):
            model.load_retrieval_bank(original_bank_path)

    def _slot_visualization_episode_indices(
        self, first_episode_idx, num_episodes, selection="sequential"
    ):
        """Select reproducible validation trajectories for slot visualization."""
        dset = self.val_traj_dset
        if selection == "random":
            rng = np.random.default_rng(int(self.cfg.training.seed) + int(self.epoch))
            return rng.choice(len(dset), size=num_episodes, replace=False).tolist()
        if selection not in ("sequential", "collision_balanced"):
            raise ValueError(
                "pca_visualization.selection must be random, sequential, or "
                f"collision_balanced; got {selection!r}"
            )
        if selection == "sequential":
            return [
                (first_episode_idx + offset) % len(dset)
                for offset in range(num_episodes)
            ]

        base_dset = getattr(dset, "dataset", dset)
        collision_types = getattr(base_dset, "collision_types", None)
        subset_indices = getattr(dset, "indices", None)
        if collision_types is None:
            return [
                (first_episode_idx + offset) % len(dset)
                for offset in range(num_episodes)
            ]

        if subset_indices is not None:
            local_types = collision_types[
                torch.as_tensor(subset_indices, dtype=torch.long)
            ]
        else:
            local_types = collision_types[: len(dset)]
        groups = {
            collision_type: torch.nonzero(local_types == collision_type, as_tuple=False)
            .flatten()
            .tolist()
            for collision_type in (0, 1, 2)
        }
        selected = []
        offsets = {collision_type: 0 for collision_type in groups}
        # Round-robin produces direct, central-wall and outer-border examples
        # even when the underlying validation split is randomly ordered.
        while len(selected) < num_episodes:
            added = False
            for collision_type in (0, 1, 2):
                group = groups[collision_type]
                offset = offsets[collision_type]
                if offset >= len(group):
                    continue
                selected.append(group[offset])
                offsets[collision_type] += 1
                added = True
                if len(selected) == num_episodes:
                    break
            if not added:
                break
        return selected

    @torch.no_grad()
    def _export_slot_pca_episode(
        self,
        episode_idx,
        start,
        num_steps_req,
        case_idx,
    ):

        obs, act, state, env_info = self.val_traj_dset[episode_idx]
        layout_id = obs.get("layout_id", env_info.get("layout_id"))
        if layout_id is None and bool(
            self.cfg.model.get("training_retrieval_same_layout_only", False)
        ):
            raise RuntimeError(
                "PCA validation rollout requires the episode layout_id"
            )
        obs_visual, act, state = obs["visual"].float(), act.float(), state.float()
        frameskip = int(self.cfg.frameskip)
        model_indices = list(range(start, len(obs_visual), frameskip))
        num_steps = min(num_steps_req, max(0, len(model_indices) - 1))
        if num_steps < 1:
            return
        model_indices = model_indices[: num_steps + 1]
        action_chunks = torch.stack(
            [act[index : index + frameskip].reshape(-1) for index in model_indices[:-1]]
        ).unsqueeze(0).to(self.device)

        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()
        model.refresh_retrieval_state_embeddings()
        actual_visual = obs_visual[model_indices].unsqueeze(0).to(self.device)
        actual_slots = model._extract_slots(actual_visual)
        initial_slots = {
            "obj_latents": actual_slots["obj_latents"],
            "mask_logits": actual_slots["mask_logits"][:, :1],
            "mask_probs": actual_slots["mask_probs"][:, :1],
        }
        # Emulate repeated MPC decisions. Each true action stands in for the
        # action selected by that decision; after predicting its next state,
        # retrieve again from that prediction before the following action.
        rollout, rollout_trace = model.receding_rollout(
            obs_0={"visual": actual_visual[:, :1]},
            act=action_chunks,
            state_0=state[model_indices[0]].unsqueeze(0).to(self.device),
            context_slots=initial_slots,
            return_retrieval_trace=True,
            retrieval_layout_ids=(
                None
                if layout_id is None
                else torch.as_tensor(layout_id, device=self.device)
                .long()
                .reshape(1)
            ),
        )
        pred_masks = rollout["mask_probs"]
        rollout_context_masks = rollout_trace["query_mask_probs"]
        pred_recon = self._slot_reconstruct_from_slots(
            pred_masks, actual_slots["obj_latents"]
        )
        rollout_context_recon = self._slot_reconstruct_from_slots(
            rollout_context_masks, actual_slots["obj_latents"]
        )
        actual_recon = self._slot_reconstruct_from_slots(
            actual_slots["mask_probs"], actual_slots["obj_latents"]
        )
        context_frames = [obs_visual[index].cpu() for index in model_indices[:-1]]
        observed_frames = [obs_visual[index].cpu() for index in model_indices[1:]]
        context_encoder_feat_maps = [
            actual_slots["features"][0, index].detach().cpu()
            for index in range(num_steps)
        ]
        context_feat_maps = [
            rollout_context_recon[0, index].detach().cpu()
            for index in range(num_steps)
        ]
        pred_feat_maps = [pred_recon[0, index].detach().cpu() for index in range(num_steps)]
        observed_feat_maps = [actual_recon[0, index + 1].detach().cpu() for index in range(num_steps)]
        spatial_size = int(math.sqrt(pred_masks.shape[-1]))
        context_mask_maps = [
            rollout_context_masks[0, index]
            .reshape(model.num_objects, spatial_size, spatial_size)
            .detach()
            .cpu()
            for index in range(num_steps)
        ]
        pred_mask_maps = [
            pred_masks[0, index].reshape(model.num_objects, spatial_size, spatial_size).detach().cpu()
            for index in range(num_steps)
        ]
        observed_mask_maps = [
            actual_slots["mask_probs"][0, index + 1].reshape(model.num_objects, spatial_size, spatial_size).detach().cpu()
            for index in range(num_steps)
        ]
        retrieved_source_mask_maps = [
            rollout_trace["retrieved_source_mask_probs"][0, index, 0]
            .reshape(model.num_objects, spatial_size, spatial_size)
            .detach()
            .cpu()
            for index in range(num_steps)
        ]
        retrieved_target_mask_maps = [
            rollout_trace["retrieved_target_mask_probs"][0, index, 0]
            .reshape(model.num_objects, spatial_size, spatial_size)
            .detach()
            .cpu()
            for index in range(num_steps)
        ]

        vis_dir = Path("pca_vis") / f"epoch_{self.epoch:04d}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = f"case{case_idx:03d}_episode{episode_idx:04d}"
        pca_path = vis_dir / f"{output_prefix}_pca.png"
        self._save_slot_rollout_pca_grid(
            context_images=context_frames,
            context_encoder_features=context_encoder_feat_maps,
            context_features=context_feat_maps,
            pred_features=pred_feat_maps,
            observed_images=observed_frames,
            observed_features=observed_feat_maps,
            output_path=pca_path,
        )
        mask_path = vis_dir / f"{output_prefix}_masks.png"
        self._save_slot_mask_grid(
            context_masks=context_mask_maps,
            pred_masks=pred_mask_maps,
            observed_masks=observed_mask_maps,
            output_path=mask_path,
            retrieved_source_masks=retrieved_source_mask_maps,
            retrieved_target_masks=retrieved_target_mask_maps,
        )
        if was_training:
            model.train()

    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        if self.accelerator.is_main_process:
            os.makedirs(phase, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        if not self.accelerator.is_main_process:
            return
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    _synchronize_hydra_run_id()
    main()
