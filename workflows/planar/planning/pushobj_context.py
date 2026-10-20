"""Shared PushObj structured-context definitions for training and planning."""


PUSHOBJ_STRUCTURED_CONTEXT_NAMES = (
    "translation",
    "torque_negative_moderate",
    "torque_negative_strong",
    "torque_positive_moderate",
    "torque_positive_strong",
)

PUSHOBJ_STRUCTURED_CONTEXT_TANGENT_RATIOS = (
    0.0,
    0.45,
    1.0,
    -0.45,
    -1.0,
)

