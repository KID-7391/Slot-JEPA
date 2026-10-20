"""Control-point-delta trajectory proposals for PushT."""

import numpy as np
from scipy.interpolate import PchipInterpolator


def smooth_waypoints(waypoints, num_positions=25):
    """Shape-preserving interpolation through start plus control points."""
    waypoints = np.asarray(waypoints, dtype=np.float64)
    parameter = np.arange(len(waypoints), dtype=np.float64)
    query = np.linspace(0.0, float(len(waypoints) - 1), num_positions + 1)[1:]
    return np.stack(
        [
            PchipInterpolator(parameter, waypoints[:, axis])(query)
            for axis in range(2)
        ],
        axis=-1,
    )


def control_deltas_to_waypoints(start_positions, control_deltas):
    """Accumulate K two-dimensional deltas into K spatial control points."""
    starts = np.asarray(start_positions, dtype=np.float64)
    deltas = np.asarray(control_deltas, dtype=np.float64)
    if starts.shape[-1:] != (2,) or deltas.shape[-1:] != (2,):
        raise ValueError("PushT starts and control deltas must have final dimension 2")
    return starts[..., None, :] + np.cumsum(deltas, axis=-2)


def _correlated_noise(rng, length, rho):
    """Draw a unit-variance AR(1) sequence."""
    noise = np.empty(int(length), dtype=np.float64)
    noise[0] = rng.normal()
    innovation_scale = np.sqrt(1.0 - float(rho) ** 2)
    for index in range(1, len(noise)):
        noise[index] = (
            float(rho) * noise[index - 1] + innovation_scale * rng.normal()
        )
    return noise


def sample_heading_limited_proposals(
    start_positions,
    num_samples,
    rng,
    num_control_points=25,
    min_speed=3.0,
    max_speed=35.0,
    max_heading_change_degrees=30.0,
    model_step_size=5,
    boundary_min_speed=20.0,
    boundary_max_speed=80.0,
    boundary_full_direction_probability=0.2,
    max_speed_change=None,
    boundary_max_speed_change=None,
    boundary_speed_resample_probability=0.0,
    boundary_resample_min_speed=0.0,
):
    """Sample paths that are smooth within model steps and flexible at boundaries.

    The first primitive action after each completed model step receives a wider
    speed range.  At the same boundary it may also choose a completely new
    direction.  The initial primitive action is not a boundary action.
    """
    starts = np.asarray(start_positions, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[-1] != 2:
        raise ValueError(f"Expected start_positions [B,2], got {starts.shape}")
    count = int(num_samples)
    steps = int(num_control_points)
    min_speed, max_speed = float(min_speed), float(max_speed)
    max_turn = np.deg2rad(float(max_heading_change_degrees))
    model_step_size = int(model_step_size)
    boundary_min_speed = float(boundary_min_speed)
    boundary_max_speed = float(boundary_max_speed)
    boundary_full_direction_probability = float(
        boundary_full_direction_probability
    )
    max_speed_change = (
        None if max_speed_change is None else float(max_speed_change)
    )
    boundary_max_speed_change = (
        max_speed_change
        if boundary_max_speed_change is None
        else float(boundary_max_speed_change)
    )
    boundary_speed_resample_probability = float(
        boundary_speed_resample_probability
    )
    boundary_resample_min_speed = float(boundary_resample_min_speed)
    if count < 1 or steps < 2:
        raise ValueError("num_samples must be positive and num_control_points >= 2")
    if not 0.0 <= min_speed <= max_speed:
        raise ValueError("Require 0 <= min_speed <= max_speed")
    if model_step_size < 1:
        raise ValueError("model_step_size must be positive")
    if not 0.0 <= boundary_min_speed <= boundary_max_speed:
        raise ValueError("Require 0 <= boundary_min_speed <= boundary_max_speed")
    if not 0.0 <= boundary_full_direction_probability <= 1.0:
        raise ValueError(
            "boundary_full_direction_probability must lie in [0, 1]"
        )
    if not 0.0 <= float(max_heading_change_degrees) <= 180.0:
        raise ValueError("max_heading_change_degrees must lie in [0, 180]")
    if max_speed_change is not None and max_speed_change < 0.0:
        raise ValueError("max_speed_change must be non-negative")
    if (
        boundary_max_speed_change is not None
        and boundary_max_speed_change < 0.0
    ):
        raise ValueError("boundary_max_speed_change must be non-negative")
    if not 0.0 <= boundary_speed_resample_probability <= 1.0:
        raise ValueError(
            "boundary_speed_resample_probability must lie in [0, 1]"
        )
    if not 0.0 <= boundary_resample_min_speed <= boundary_max_speed:
        raise ValueError(
            "boundary_resample_min_speed must lie in [0, boundary_max_speed]"
        )
    output = np.empty((len(starts), count, steps, 2), dtype=np.float32)
    for case_index in range(len(starts)):
        boundary_indices = np.arange(model_step_size, steps, model_step_size)
        if max_speed_change is None:
            speeds = rng.uniform(min_speed, max_speed, size=(count, steps))
            if len(boundary_indices):
                speeds[:, boundary_indices] = rng.uniform(
                    boundary_min_speed,
                    boundary_max_speed,
                    size=(count, len(boundary_indices)),
                )
        else:
            speeds = np.empty((count, steps), dtype=np.float64)
            speeds[:, 0] = rng.uniform(min_speed, max_speed, size=count)
            for step_index in range(1, steps):
                is_boundary = step_index % model_step_size == 0
                delta_limit = (
                    boundary_max_speed_change
                    if is_boundary
                    else max_speed_change
                )
                lower = boundary_min_speed if is_boundary else min_speed
                upper = boundary_max_speed if is_boundary else max_speed
                speeds[:, step_index] = np.clip(
                    speeds[:, step_index - 1]
                    + rng.uniform(-delta_limit, delta_limit, size=count),
                    lower,
                    upper,
                )
                if is_boundary and boundary_speed_resample_probability > 0.0:
                    resample = (
                        rng.random(count)
                        < boundary_speed_resample_probability
                    )
                    speeds[resample, step_index] = rng.uniform(
                        boundary_resample_min_speed,
                        boundary_max_speed,
                        size=int(resample.sum()),
                    )
        headings = np.empty((count, steps), dtype=np.float64)
        headings[:, 0] = rng.uniform(-np.pi, np.pi, size=count)
        for step_index in range(1, steps):
            turn = rng.uniform(-max_turn, max_turn, size=count)
            if step_index % model_step_size == 0:
                unrestricted = (
                    rng.random(count) < boundary_full_direction_probability
                )
                turn[unrestricted] = rng.uniform(
                    -np.pi, np.pi, size=int(unrestricted.sum())
                )
            headings[:, step_index] = headings[:, step_index - 1] + turn
        proposals = speeds[..., None] * np.stack(
            [np.cos(headings), np.sin(headings)], axis=-1
        )
        # Relative PushT actions are PD target offsets from the *current*
        # physical agent position, not per-step spatial displacements.
        # Rejecting their cumulative sum against canvas bounds therefore
        # removes valid commands, especially near an image boundary.
        output[case_index] = proposals
    return output


def apply_stochastic_early_stop(
    proposals,
    rng,
    stop_probability,
    jitter_max_speed,
    temporal_correlation=0.0,
):
    """Replace the suffix after a sampled stop action with local jitter.

    A stop decision is sampled independently after every primitive action.
    The first triggering action remains part of the global-motion prefix;
    only later actions are replaced. Trajectories with no trigger are left
    unchanged. The jitter is isotropic and zero-mean in expectation, unlike
    the heading-limited proposal whose persistent direction tends to travel.
    """
    proposals = np.asarray(proposals)
    if proposals.ndim != 4 or proposals.shape[-1] != 2:
        raise ValueError(
            "Expected proposals [case,sample,step,2], got "
            f"{proposals.shape}"
        )
    probability = float(stop_probability)
    jitter_max_speed = float(jitter_max_speed)
    temporal_correlation = float(temporal_correlation)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("stop_probability must lie in [0, 1]")
    if jitter_max_speed < 0.0:
        raise ValueError("jitter_max_speed must be non-negative")
    if not 0.0 <= temporal_correlation < 1.0:
        raise ValueError("temporal_correlation must lie in [0, 1)")
    if probability == 0.0 or proposals.shape[2] < 2:
        return proposals.copy()

    num_cases, num_samples, num_steps, _ = proposals.shape
    triggers = rng.random((num_cases, num_samples, num_steps)) < probability
    has_stop = triggers.any(axis=-1)
    first_stop = np.argmax(triggers, axis=-1)

    if temporal_correlation == 0.0:
        jitter_speeds = rng.uniform(
            0.0,
            jitter_max_speed,
            size=(num_cases, num_samples, num_steps),
        )
        jitter_headings = rng.uniform(
            -np.pi,
            np.pi,
            size=(num_cases, num_samples, num_steps),
        )
        jitter = jitter_speeds[..., None] * np.stack(
            [np.cos(jitter_headings), np.sin(jitter_headings)], axis=-1
        )
    else:
        innovation_scale = np.sqrt(1.0 - temporal_correlation**2)
        jitter = rng.normal(
            size=(num_cases, num_samples, num_steps, 2)
        )
        for step_index in range(1, num_steps):
            jitter[..., step_index, :] = (
                temporal_correlation * jitter[..., step_index - 1, :]
                + innovation_scale * jitter[..., step_index, :]
            )
        jitter *= jitter_max_speed / 2.0
        jitter_norms = np.linalg.norm(jitter, axis=-1, keepdims=True)
        jitter *= np.minimum(
            1.0,
            jitter_max_speed / np.maximum(jitter_norms, 1.0e-12),
        )

    step_indices = np.arange(num_steps)[None, None, :]
    suffix_mask = has_stop[..., None] & (
        step_indices > first_stop[..., None]
    )
    output = proposals.copy()
    output[suffix_mask] = jitter.astype(output.dtype, copy=False)[suffix_mask]
    return output


def mutate_heading_limited_elites(
    start_positions,
    elite_control_deltas,
    num_samples,
    rng,
    min_speed=3.0,
    max_speed=35.0,
    max_heading_change_degrees=30.0,
    model_step_size=5,
    boundary_min_speed=20.0,
    boundary_max_speed=80.0,
    boundary_full_direction_probability=0.2,
    speed_std=3.0,
    initial_heading_std_degrees=30.0,
    heading_change_std_degrees=10.0,
    temporal_correlation=0.8,
):
    """Resample whole elite paths with temporally correlated mutations.

    Every elite contributes nearly the same number of children. The first
    child is an exact carryover; remaining children inherit that complete
    trajectory and perturb its speed/turn sequences jointly over time.
    """
    starts = np.asarray(start_positions, dtype=np.float64)
    elites = np.asarray(elite_control_deltas, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[-1] != 2:
        raise ValueError(f"Expected start_positions [B,2], got {starts.shape}")
    if elites.ndim != 4 or elites.shape[0] != len(starts) or elites.shape[-1] != 2:
        raise ValueError(
            "Expected elite_control_deltas [case,elite,step,2], got "
            f"{elites.shape}"
        )
    count = int(num_samples)
    num_elites, steps = elites.shape[1:3]
    if count < num_elites:
        raise ValueError("num_samples must be at least the number of elites")
    if not 0.0 <= float(min_speed) <= float(max_speed):
        raise ValueError("Require 0 <= min_speed <= max_speed")
    model_step_size = int(model_step_size)
    boundary_min_speed = float(boundary_min_speed)
    boundary_max_speed = float(boundary_max_speed)
    boundary_full_direction_probability = float(
        boundary_full_direction_probability
    )
    if model_step_size < 1:
        raise ValueError("model_step_size must be positive")
    if not 0.0 <= boundary_min_speed <= boundary_max_speed:
        raise ValueError("Require 0 <= boundary_min_speed <= boundary_max_speed")
    if not 0.0 <= boundary_full_direction_probability <= 1.0:
        raise ValueError(
            "boundary_full_direction_probability must lie in [0, 1]"
        )
    if not 0.0 <= float(max_heading_change_degrees) <= 180.0:
        raise ValueError("max_heading_change_degrees must lie in [0, 180]")
    if not 0.0 <= float(temporal_correlation) < 1.0:
        raise ValueError("temporal_correlation must lie in [0, 1)")

    def per_case_values(value, name):
        values = np.asarray(value, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(len(starts), float(values), dtype=np.float64)
        if values.shape != (len(starts),):
            raise ValueError(
                f"{name} must be scalar or one value per case, got {values.shape}"
            )
        if np.any(values < 0.0):
            raise ValueError(f"{name} must be non-negative")
        return values

    speed_stds = per_case_values(speed_std, "speed_std")
    heading_stds = np.deg2rad(
        per_case_values(initial_heading_std_degrees, "initial_heading_std_degrees")
    )
    turn_stds = np.deg2rad(
        per_case_values(heading_change_std_degrees, "heading_change_std_degrees")
    )
    max_turn = np.deg2rad(float(max_heading_change_degrees))
    base_children, remainder = divmod(count, num_elites)
    output = np.empty((len(starts), count, steps, 2), dtype=np.float32)
    boundary_action_indices = np.arange(model_step_size, steps, model_step_size)
    boundary_turn_indices = boundary_action_indices - 1

    for case_index in range(len(starts)):
        case_speed_std = speed_stds[case_index]
        case_heading_std = heading_stds[case_index]
        case_turn_std = turn_stds[case_index]
        write_index = 0
        for elite_index in range(num_elites):
            parent = elites[case_index, elite_index]
            parent_speeds = np.linalg.norm(parent, axis=-1)
            parent_headings = np.unwrap(np.arctan2(parent[:, 1], parent[:, 0]))
            parent_turns = np.diff(parent_headings)
            children = base_children + int(elite_index < remainder)
            output[case_index, write_index] = parent
            write_index += 1
            for _ in range(children - 1):
                speeds = np.clip(
                    parent_speeds
                    + case_speed_std
                    * _correlated_noise(rng, steps, temporal_correlation),
                    min_speed,
                    max_speed,
                )
                if len(boundary_action_indices):
                    speeds[boundary_action_indices] = np.clip(
                        parent_speeds[boundary_action_indices]
                        + case_speed_std
                        * _correlated_noise(
                            rng,
                            len(boundary_action_indices),
                            temporal_correlation,
                        ),
                        boundary_min_speed,
                        boundary_max_speed,
                    )
                turn_noise = case_turn_std * _correlated_noise(
                    rng, steps - 1, temporal_correlation
                )
                raw_turns = parent_turns + turn_noise
                turns = np.clip(raw_turns, -max_turn, max_turn)
                if len(boundary_turn_indices):
                    # Once CEM selects an unrestricted boundary turn, keep
                    # that structural mode in its descendants. Bounded
                    # parents can still enter the unrestricted mode with
                    # the configured probability.
                    parent_unrestricted = (
                        np.abs(parent_turns[boundary_turn_indices])
                        > max_turn + 1.0e-8
                    )
                    new_unrestricted = (
                        rng.random(len(boundary_turn_indices))
                        < boundary_full_direction_probability
                    )
                    unrestricted = parent_unrestricted | new_unrestricted
                    boundary_turns = np.clip(
                        raw_turns[boundary_turn_indices], -max_turn, max_turn
                    )
                    inherited = unrestricted & parent_unrestricted
                    boundary_turns[inherited] = (
                        raw_turns[boundary_turn_indices][inherited] + np.pi
                    ) % (2.0 * np.pi) - np.pi
                    newly_unrestricted = unrestricted & ~parent_unrestricted
                    boundary_turns[newly_unrestricted] = rng.uniform(
                        -np.pi,
                        np.pi,
                        size=int(newly_unrestricted.sum()),
                    )
                    turns[boundary_turn_indices] = boundary_turns
                headings = np.empty(steps, dtype=np.float64)
                headings[0] = parent_headings[0] + rng.normal(
                    scale=case_heading_std
                )
                headings[1:] = headings[0] + np.cumsum(turns)
                proposal = speeds[:, None] * np.stack(
                    [np.cos(headings), np.sin(headings)], axis=-1
                )
                output[case_index, write_index] = proposal
                write_index += 1
        if write_index != count:
            raise RuntimeError(
                f"Internal particle population mismatch: {write_index} != {count}"
            )
    return output


def sample_control_delta_proposals(
    start_positions,
    num_samples,
    rng,
    control_delta_mean=None,
    control_delta_std=None,
    initial_uniform=False,
    num_control_points=5,
    control_delta_range=70.0,
    margin=15.0,
    canvas_size=512.0,
    max_attempts=100000,
):
    """Sample in-bounds CEM parameters with shape [case, sample, K, 2].

    The initial population is uniform in the configured component-wise range.
    Later populations are Gaussian around the elite statistics and clipped to
    that same range. A proposal is accepted only when all cumulative spatial
    control points remain inside the canvas margin.
    """
    starts = np.asarray(start_positions, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[-1] != 2:
        raise ValueError(f"Expected start_positions [B,2], got {starts.shape}")
    k = int(num_control_points)
    count = int(num_samples)
    search_range = float(control_delta_range)
    if k < 2:
        raise ValueError("num_control_points must be at least 2 for PCHIP")
    if count < 1:
        raise ValueError("num_samples must be positive")
    if search_range <= 0.0:
        raise ValueError("control_delta_range must be positive")
    lower = float(margin)
    upper = float(canvas_size) - lower
    if upper <= lower:
        raise ValueError("canvas_size must be larger than twice margin")
    if not np.all((starts >= lower) & (starts <= upper)):
        raise ValueError("Every PushT start position must lie inside the canvas margin")

    num_cases = len(starts)
    if initial_uniform:
        means = stds = None
    else:
        means = np.asarray(control_delta_mean, dtype=np.float64)
        stds = np.asarray(control_delta_std, dtype=np.float64)
        expected = (num_cases, k, 2)
        if means.shape != expected or stds.shape != expected:
            raise ValueError(
                f"Expected control delta mean/std {expected}, got {means.shape}/{stds.shape}"
            )
        if np.any(stds <= 0.0):
            raise ValueError("control_delta_std must be positive")

    output = np.empty((num_cases, count, k, 2), dtype=np.float32)
    for case_index, start in enumerate(starts):
        accepted = 0
        attempts = 0
        while accepted < count:
            remaining = count - accepted
            draw_count = max(64, remaining * 2)
            attempts += draw_count
            if attempts > int(max_attempts):
                raise RuntimeError(
                    f"Could not sample {count} valid control-delta trajectories "
                    f"from start={start.tolist()} within {max_attempts} draws"
                )
            if initial_uniform:
                proposals = rng.uniform(
                    -search_range, search_range, size=(draw_count, k, 2)
                )
            else:
                proposals = rng.normal(
                    means[case_index], stds[case_index], size=(draw_count, k, 2)
                )
                proposals = np.clip(proposals, -search_range, search_range)
            control_points = control_deltas_to_waypoints(
                np.broadcast_to(start, (draw_count, 2)), proposals
            )
            valid = np.all(
                (control_points >= lower) & (control_points <= upper),
                axis=(-2, -1),
            )
            valid_proposals = proposals[valid][:remaining]
            valid_count = len(valid_proposals)
            output[case_index, accepted : accepted + valid_count] = valid_proposals
            accepted += valid_count
    return output


def sample_control_delta_trajectory(
    start,
    rng,
    num_control_points=5,
    num_positions=25,
    control_delta_range=70.0,
    margin=15.0,
    canvas_size=512.0,
    max_attempts=4096,
):
    """Sample one visualization trajectory using the same prior as CEM."""
    start = np.asarray(start, dtype=np.float64)
    proposals = sample_control_delta_proposals(
        start[None],
        num_samples=1,
        rng=rng,
        initial_uniform=True,
        num_control_points=num_control_points,
        control_delta_range=control_delta_range,
        margin=margin,
        canvas_size=canvas_size,
        max_attempts=max_attempts,
    )
    control_deltas = proposals[0, 0].astype(np.float64)
    control_points = control_deltas_to_waypoints(
        start[None], control_deltas[None]
    )[0]
    waypoints = np.concatenate([start[None], control_points], axis=0)
    positions = smooth_waypoints(waypoints, num_positions=int(num_positions))
    pixel_actions = np.diff(
        np.concatenate([start[None], positions], axis=0), axis=0
    )
    return {
        "positions": positions,
        "waypoints": waypoints,
        "control_deltas": control_deltas,
        "actions": pixel_actions,
    }
