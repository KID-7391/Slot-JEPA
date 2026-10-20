"""Deterministic reference-frame sampling used to initialize fixed slots.

This module contains only the sampling code needed by planar training and
evaluation.  Keeping it separate avoids depending on the internal diagnostic
and visualization script from which these routines were originally developed.
"""

import numpy as np
import torch
from shapely.geometry import Point
from torch.utils.data import Subset

from env.pusht.pusht_env import pymunk_to_shapely


def _visual_to_chw(visual):
    visual = torch.as_tensor(visual).float()
    if visual.ndim != 3:
        raise ValueError(f"Expected one HWC/CHW image, got {tuple(visual.shape)}")
    if visual.shape[-1] == 3:
        visual = visual.permute(2, 0, 1)
    elif visual.shape[0] != 3:
        raise ValueError(
            f"Cannot identify image channel dimension: {tuple(visual.shape)}"
        )
    if visual.max() > 1.0:
        visual = visual / 255.0
    return visual.clamp(0.0, 1.0)


def _env_reset(env_instance, seed, init_state=None):
    if hasattr(env_instance, "prepare_for_render"):
        env_instance.prepare_for_render()
    if hasattr(env_instance, "seed"):
        env_instance.seed(seed)
    if init_state is not None:
        base_env = env_instance.unwrapped
        if hasattr(base_env, "set_init_state"):
            base_env.set_init_state(init_state)
        elif hasattr(base_env, "reset_to_state"):
            base_env.reset_to_state = init_state
        else:
            raise TypeError(
                f"{type(base_env).__name__} does not support an explicit init_state"
            )
    result = env_instance.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result
    obs = result
    state = np.asarray(obs.get("proprio", [])) if isinstance(obs, dict) else np.array([])
    return obs, state


def _sample_action(env_instance, env_name, rng):
    if env_name == "pusht":
        return rng.normal(0.0, 0.35, size=2).astype(np.float32)
    if env_name == "wall":
        return torch.as_tensor(rng.normal(0.0, 0.8, size=2), dtype=torch.float32)
    action_space = env_instance.action_space
    return rng.uniform(action_space.low, action_space.high).astype(np.float32)


def _sample_live_rollout(
    env_instance, env_name, seed, num_frames, frameskip, init_state=None
):
    rng = np.random.RandomState(seed)
    if init_state is None:
        base_env = env_instance.unwrapped
        if hasattr(base_env, "sample_random_init_goal_states"):
            init_state, _ = base_env.sample_random_init_goal_states(seed)
    obs, state = _env_reset(env_instance, seed, init_state=init_state)
    frames = [_visual_to_chw(obs["visual"])]
    states = [np.asarray(state)]
    total_steps = max(num_frames - 1, 0) * frameskip
    for step_index in range(1, total_steps + 1):
        obs, _, done, info = env_instance.step(
            _sample_action(env_instance, env_name, rng)
        )
        if step_index % frameskip == 0:
            frames.append(_visual_to_chw(obs["visual"]))
            states.append(np.asarray(info.get("state", obs.get("proprio", []))))
        if done and step_index < total_steps:
            obs, state = _env_reset(
                env_instance, seed + step_index + 1, init_state=None
            )
    return torch.stack(frames[:num_frames]), states[:num_frames]


def _resolve_trajectory_index(dataset, trajectory_index):
    source_dataset = dataset
    source_index = int(trajectory_index)
    while isinstance(source_dataset, Subset):
        source_index = int(source_dataset.indices[source_index])
        source_dataset = source_dataset.dataset
    return source_dataset, source_index


def _sample_dataset_trajectory(
    dataset, trajectory_index, start, num_frames, frameskip
):
    trajectory_index = int(trajectory_index)
    if trajectory_index < 0 or trajectory_index >= len(dataset):
        raise IndexError(
            f"Trajectory index {trajectory_index} is outside dataset size {len(dataset)}"
        )
    start = int(start)
    frameskip = int(frameskip)
    if start < 0:
        raise ValueError(f"start must be non-negative, got {start}")
    if frameskip < 1:
        raise ValueError(f"frameskip must be positive, got {frameskip}")
    seq_length = int(dataset.get_seq_length(trajectory_index))
    frame_indices = list(range(start, seq_length, frameskip))[: int(num_frames)]
    if not frame_indices:
        raise ValueError(
            f"Trajectory {trajectory_index} has length {seq_length}, so no frame is "
            f"available from raw index {start}"
        )
    source_dataset, source_index = _resolve_trajectory_index(
        dataset, trajectory_index
    )
    obs, _, state, _ = source_dataset.get_frames(source_index, frame_indices)
    return obs["visual"].float(), state, source_index, frame_indices


def _sample_random_dataset_references(dataset, num_frames, seed):
    num_frames = int(num_frames)
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    lengths = np.asarray(
        [int(dataset.get_seq_length(index)) for index in range(len(dataset))],
        dtype=np.int64,
    )
    cumulative = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths)])
    total_frames = int(cumulative[-1])
    if total_frames < num_frames:
        raise ValueError(
            f"Training dataset has {total_frames} valid frames, fewer than "
            f"the requested {num_frames} references"
        )
    rng = np.random.default_rng(int(seed))
    flat_indices = rng.choice(total_frames, size=num_frames, replace=False)
    frames = []
    labels = []
    for flat_index in flat_indices:
        trajectory_index = int(
            np.searchsorted(cumulative, flat_index, side="right") - 1
        )
        frame_index = int(flat_index - cumulative[trajectory_index])
        sampled, _, source_index, sampled_indices = _sample_dataset_trajectory(
            dataset, trajectory_index, frame_index, 1, 1
        )
        frames.append(sampled[0])
        labels.append(("train_random", source_index, sampled_indices[0]))
    return torch.stack(frames), labels


def _sample_balanced_layout_dataset_references(dataset, num_frames, seed):
    num_frames = int(num_frames)
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    trajectories_by_layout = {}
    for trajectory_index in range(len(dataset)):
        source_dataset, source_index = _resolve_trajectory_index(
            dataset, trajectory_index
        )
        layout_ids = getattr(source_dataset, "layout_ids", None)
        if layout_ids is None:
            raise ValueError(
                "Balanced-layout reference sampling requires dataset layout_ids"
            )
        layout_id = int(torch.as_tensor(layout_ids[source_index]).item())
        trajectories_by_layout.setdefault(layout_id, []).append(trajectory_index)

    layout_ids = sorted(trajectories_by_layout)
    if not layout_ids:
        raise ValueError("Balanced-layout reference sampling found no layouts")
    if num_frames < len(layout_ids):
        raise ValueError(
            f"Cannot cover {len(layout_ids)} layouts with only {num_frames} references"
        )

    rng = np.random.default_rng(int(seed))
    base_count, remainder = divmod(num_frames, len(layout_ids))
    remainder_order = list(rng.permutation(layout_ids))
    counts = {layout_id: base_count for layout_id in layout_ids}
    for layout_id in remainder_order[:remainder]:
        counts[int(layout_id)] += 1

    sampled_by_layout = {}
    for layout_id in layout_ids:
        trajectory_indices = trajectories_by_layout[layout_id]
        lengths = np.asarray(
            [int(dataset.get_seq_length(index)) for index in trajectory_indices],
            dtype=np.int64,
        )
        cumulative = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(lengths)]
        )
        count = counts[layout_id]
        total_frames = int(cumulative[-1])
        if total_frames < count:
            raise ValueError(
                f"Layout {layout_id} has {total_frames} valid frames, fewer than "
                f"the requested {count} references"
            )
        samples = []
        for flat_index in rng.choice(total_frames, size=count, replace=False):
            local_index = int(
                np.searchsorted(cumulative, flat_index, side="right") - 1
            )
            trajectory_index = trajectory_indices[local_index]
            frame_index = int(flat_index - cumulative[local_index])
            sampled, _, source_index, sampled_indices = _sample_dataset_trajectory(
                dataset, trajectory_index, frame_index, 1, 1
            )
            samples.append(
                (
                    sampled[0],
                    (f"train_layout_{layout_id}", source_index, sampled_indices[0]),
                )
            )
        sampled_by_layout[layout_id] = samples

    frames = []
    labels = []
    for sample_index in range(max(counts.values())):
        for layout_id in layout_ids:
            if sample_index < len(sampled_by_layout[layout_id]):
                frame, label = sampled_by_layout[layout_id][sample_index]
                frames.append(frame)
                labels.append(label)
    return torch.stack(frames), labels


def _sample_live_pusht_references(
    env_instance, seed, num_trajectories, frames_per_trajectory
):
    rng = np.random.RandomState(seed + 10000)
    total_frames = num_trajectories * frames_per_trajectory
    base_env = env_instance.unwrapped
    _env_reset(env_instance, seed)

    def install_if_valid(agent, block, block_angle, goal, goal_angle):
        state = np.concatenate([agent, block, [block_angle, 0.0, 0.0]])
        base_env._set_state(state)
        base_env.set_task_goal(np.concatenate([goal, [goal_angle]]))
        agent_xy = np.asarray(base_env.agent.position)
        block_geom = pymunk_to_shapely(base_env.block, base_env.block.shapes)
        goal_body = base_env._get_goal_pose_body(base_env.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, base_env.block.shapes)
        agent_geom = Point(*agent_xy).buffer(15.0)
        clearance = 24.0
        return (
            agent_geom.distance(block_geom) >= clearance
            and agent_geom.distance(goal_geom) >= clearance
            and block_geom.distance(goal_geom) >= clearance
        )

    def random_configuration():
        return np.concatenate(
            [
                rng.uniform(35.0, 477.0, size=2),
                rng.uniform(100.0, 412.0, size=2),
                [rng.uniform(-np.pi, np.pi)],
                rng.uniform(100.0, 412.0, size=2),
                [rng.uniform(-np.pi, np.pi)],
            ]
        )

    def interpolate(start, end, fraction):
        result = start * (1.0 - fraction) + end * fraction
        for angle_index in (4, 7):
            delta = (
                end[angle_index] - start[angle_index] + np.pi
            ) % (2 * np.pi) - np.pi
            result[angle_index] = start[angle_index] + fraction * delta
        return result

    def configuration_is_valid(configuration):
        return install_if_valid(
            configuration[:2],
            configuration[2:4],
            configuration[4],
            configuration[5:7],
            configuration[7],
        )

    keyframe_count = min(8, total_frames)
    for _ in range(4000):
        initial = random_configuration()
        if configuration_is_valid(initial):
            break
    else:
        raise RuntimeError("Could not find an initial PushT reference keyframe")

    keyframes = [initial]
    transition_count = max(total_frames - 1, 0)
    base_steps, extra_steps = divmod(transition_count, max(keyframe_count - 1, 1))
    segment_steps = [
        base_steps + (segment_index < extra_steps)
        for segment_index in range(max(keyframe_count - 1, 0))
    ]
    for steps in segment_steps:
        start = keyframes[-1]
        best = None
        best_score = -float("inf")
        for _ in range(1200):
            candidate = random_configuration()
            angle_changes = [
                abs(
                    (candidate[index] - start[index] + np.pi) % (2 * np.pi)
                    - np.pi
                )
                for index in (4, 7)
            ]
            movements = [
                np.linalg.norm(candidate[:2] - start[:2]),
                np.linalg.norm(candidate[2:4] - start[2:4]),
                np.linalg.norm(candidate[5:7] - start[5:7]),
            ]
            if min(movements[1:]) < 80.0 or min(angle_changes) < 0.35:
                continue
            path = [
                interpolate(start, candidate, step / steps)
                for step in range(1, steps + 1)
            ]
            if not all(configuration_is_valid(value) for value in path):
                continue
            score = sum(movements) / 120.0 + sum(angle_changes)
            if score > best_score:
                best = candidate
                best_score = score
            if score >= 7.0:
                break
        if best is None:
            raise RuntimeError(
                "Could not plan a diverse collision-free PushT reference segment"
            )
        keyframes.append(best)

    configurations = [keyframes[0]]
    for start, end, steps in zip(keyframes[:-1], keyframes[1:], segment_steps):
        configurations.extend(
            interpolate(start, end, step / steps) for step in range(1, steps + 1)
        )
    if len(configurations) != total_frames:
        raise RuntimeError("Internal PushT reference trajectory length mismatch")

    frames = []
    for configuration in configurations:
        if not configuration_is_valid(configuration):
            raise RuntimeError("Planned PushT reference became invalid during rendering")
        frames.append(base_env._render_frame("rgb_array"))
    labels = [("long_ref", 0, index) for index in range(total_frames)]
    return torch.stack([_visual_to_chw(frame) for frame in frames]), labels


def _sample_balanced_live_pusht_references(
    env_instance, seed, num_frames, layouts
):
    layouts = [dict(layout) for layout in layouts]
    if not layouts:
        raise ValueError("model.fixed_reference_layouts must not be empty")
    if any("shape" not in layout for layout in layouts):
        raise ValueError("Each fixed_reference_layouts entry must define shape")
    base_count, remainder = divmod(int(num_frames), len(layouts))
    counts = [base_count + (index < remainder) for index in range(len(layouts))]
    if any(count == 0 for count in counts):
        raise ValueError(
            "fixed_reference_num_frames must be at least the number of layouts"
        )

    base_env = env_instance.unwrapped
    per_layout_frames = []
    per_layout_labels = []
    for layout_index, (layout, count) in enumerate(zip(layouts, counts)):
        if not hasattr(base_env, "update_env"):
            raise TypeError(
                f"{type(base_env).__name__} cannot switch PushT reference layouts"
            )
        base_env.update_env(layout)
        frames, _ = _sample_live_pusht_references(
            env_instance,
            seed=int(seed) + layout_index * 100003,
            num_trajectories=1,
            frames_per_trajectory=count,
        )
        per_layout_frames.append(frames)
        per_layout_labels.append(
            [
                ("balanced_live", str(layout["shape"]), frame_index)
                for frame_index in range(count)
            ]
        )

    frames = []
    labels = []
    for frame_index in range(max(counts)):
        for layout_index, count in enumerate(counts):
            if frame_index < count:
                frames.append(per_layout_frames[layout_index][frame_index])
                labels.append(per_layout_labels[layout_index][frame_index])
    return torch.stack(frames), labels


def _sample_live_maze_references(env_instance, seed, num_frames):
    rng = np.random.RandomState(seed + 30000)
    base_env = env_instance.unwrapped
    base_env.prepare_for_render()
    free_tiles = np.asarray(base_env.empty_and_goal_locations, dtype=np.float64)
    frames = []
    labels = []
    for frame_index in range(num_frames):
        tile = free_tiles[int(rng.randint(len(free_tiles)))]
        position = tile + rng.uniform(-0.35, 0.35, size=base_env.model.nq)
        velocity = np.zeros(base_env.model.nv, dtype=np.float64)
        base_env.set_state(position, velocity)
        base_env.set_marker()
        frames.append(_visual_to_chw(base_env._render_frame()))
        labels.append(("random_maze", 0, frame_index))
    return torch.stack(frames), labels


def _sample_live_wall_references(env_instance, seed, num_frames):
    base_env = env_instance.unwrapped
    if not hasattr(base_env, "sample_random_init_goal_states"):
        raise TypeError(
            f"{type(base_env).__name__} cannot sample fixed Wall references"
        )
    frames = []
    labels = []
    pair_index = 0
    while len(frames) < int(num_frames):
        endpoints = base_env.sample_random_init_goal_states(int(seed) + pair_index)
        for endpoint in endpoints:
            if len(frames) >= int(num_frames):
                break
            frame_index = len(frames)
            obs, _ = _env_reset(
                env_instance, seed=int(seed) + pair_index, init_state=endpoint
            )
            frames.append(_visual_to_chw(obs["visual"]))
            labels.append(("random_wall", pair_index, frame_index))
        pair_index += 1
    return torch.stack(frames), labels


def _sample_fixed_slot_references(
    env_instance, env_name, seed, num_frames, pusht_layouts=None
):
    seed = int(seed)
    num_frames = int(num_frames)
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if env_name == "pusht":
        if pusht_layouts:
            return _sample_balanced_live_pusht_references(
                env_instance, seed, num_frames, pusht_layouts
            )
        return _sample_live_pusht_references(
            env_instance, seed, num_trajectories=1, frames_per_trajectory=num_frames
        )
    if env_name == "point_maze":
        return _sample_live_maze_references(env_instance, seed, num_frames)
    if env_name == "wall":
        return _sample_live_wall_references(env_instance, seed, num_frames)
    frames, _ = _sample_live_rollout(
        env_instance, env_name, seed, num_frames, frameskip=1
    )
    labels = [(f"live_{env_name}", 0, index) for index in range(num_frames)]
    return frames, labels


def _sample_configured_live_pusht_references(
    env_instance, *, seed, num_frames, shapes, colors, frames_per_shape
):
    shapes = [str(value) for value in shapes]
    colors = [str(value) for value in colors]
    frames_per_shape = int(frames_per_shape)
    if not shapes or len(shapes) != len(set(shapes)):
        raise ValueError("fixed_reference_shapes must be nonempty and unique")
    if not colors:
        raise ValueError("fixed_reference_colors must be nonempty")
    if frames_per_shape < len(colors):
        raise ValueError(
            "fixed_reference_frames_per_shape must be at least the number of colors"
        )
    expected_total = frames_per_shape * len(shapes)
    if int(num_frames) != expected_total:
        raise ValueError(
            "fixed_reference_num_frames must equal frames_per_shape * shapes; "
            f"got {num_frames} != {frames_per_shape} * {len(shapes)}"
        )

    per_shape_frames = []
    per_shape_labels = []
    for shape_index, shape in enumerate(shapes):
        frames, labels = _sample_fixed_slot_references(
            env_instance,
            env_name="pusht",
            seed=int(seed) + shape_index * 1000003,
            num_frames=frames_per_shape,
            pusht_layouts=[{"shape": shape, "color": color} for color in colors],
        )
        per_shape_frames.append(frames.float())
        per_shape_labels.append(labels)

    frames = torch.stack(
        [
            per_shape_frames[shape_index][frame_index]
            for frame_index in range(frames_per_shape)
            for shape_index in range(len(shapes))
        ]
    )
    labels = [
        per_shape_labels[shape_index][frame_index]
        for frame_index in range(frames_per_shape)
        for shape_index in range(len(shapes))
    ]
    return frames, labels
