"""Visual encoders used by all Slot-JEPA workflows."""

from .factory import (
    DinoV2PatchEncoder,
    init_encoder,
)

__all__ = ["DinoV2PatchEncoder", "init_encoder"]
