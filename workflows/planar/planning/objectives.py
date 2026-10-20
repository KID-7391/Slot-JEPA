"""Compatibility imports for the shared Slot-JEPA transport objectives."""

from pathlib import Path
import sys


RELEASE_SRC = Path(__file__).resolve().parents[3] / "src"
if str(RELEASE_SRC) not in sys.path:
    sys.path.insert(0, str(RELEASE_SRC))

from slot_jepa.losses.transport import (  # noqa: E402,F401
    _build_obstacle_transport_metadata,
    _geodesic_obstacle_cost,
    _mask_kl_loss,
    _mask_l1_loss,
    _mask_sobel_loss,
    _slot_area_normalizer,
    _slot_change_weights,
    _straight_line_obstacle_cost,
    _uot_mask_loss,
    create_objective_fn,
    create_uot_objective_fn,
)

__all__ = ["create_objective_fn", "create_uot_objective_fn"]
