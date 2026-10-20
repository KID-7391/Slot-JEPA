"""Pure geometry helpers shared by Maze data, rendering, and planning."""

import numpy as np


CENTERED_U_STEM_POSITION = np.array([1.8, 1.8], dtype=np.float64)

GENERALIZATION_U_LAYOUTS = {
    "train": (
        {"stem_width": 0.65, "stem_length": 1.40},
        {"stem_width": 0.80, "stem_length": 2.20},
        {"stem_width": 1.00, "stem_length": 1.80},
        {"stem_width": 1.20, "stem_length": 2.60},
        {"stem_width": 1.35, "stem_length": 1.60},
    ),
    "eval": (
        {"stem_width": 0.70, "stem_length": 2.40},
        {"stem_width": 0.90, "stem_length": 1.50},
        {"stem_width": 1.10, "stem_length": 2.30},
        {"stem_width": 1.25, "stem_length": 1.90},
        {"stem_width": 1.40, "stem_length": 2.70},
    ),
}


def continuous_u_stem_center(stem_width, stem_length):
    """Return the inner stem center in the public qpos coordinate frame."""
    return np.array(
        [1.0 + float(stem_width) / 2.0, float(stem_length) / 2.0],
        dtype=np.float64,
    )


def continuous_u_centering_offset(stem_width, stem_length):
    """Translation that places the inner stem at the fixed camera center."""
    return CENTERED_U_STEM_POSITION - continuous_u_stem_center(
        stem_width, stem_length
    )


def centered_u_state_offset(layout):
    """Return [dx, dy, 0, 0] for a centered-stem state transformation."""
    xy = continuous_u_centering_offset(
        layout["stem_width"], layout["stem_length"]
    )
    return np.array([xy[0], xy[1], 0.0, 0.0], dtype=np.float32)


def center_u_states(states, layout):
    """Translate positions into the centered-stem frame; preserve velocities."""
    states = np.asarray(states).copy()
    states[..., :4] += centered_u_state_offset(layout)
    return states
