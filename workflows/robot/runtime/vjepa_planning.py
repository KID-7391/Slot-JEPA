from __future__ import annotations

import os
import sys
import datetime
import time
import json
import copy
import random
import types
from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from PIL import Image, ImageDraw
import yaml


def _configure_kit_offline_mode():
    """
    Force Kit/Isaac to resolve extensions from local registries only.

    This must run before any omnigibson / Kit import. Otherwise Kit may start
    syncing the remote extension registry while resolving optional extensions.
    """
    env_defaults = {
        "OMNI_KIT_DISABLE_EXT_REGISTRY": "1",
        "OMNI_KIT_DISABLE_EXTENSIONS_REGISTRY": "1",
        "OV_DISABLE_EXT_REGISTRY": "1",
        "OMNI_KIT_REGISTRY_ENABLED": "0",
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)
    kit_exts_dir = os.environ.setdefault(
        "REALM_KIT_EXTS_DIR",
        os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "realm", "kit_exts"),
    )

    kit_flags = [
        "--ext-folder",
        kit_exts_dir,
        # Keep asset loads synchronous. OmniGibson validates transforms while
        # loading visual meshes; async USD/material loads can leave stale/default
        # transforms and trip xform_prim._set_xform_properties assertions.
        "--/omni.kit.plugin/syncUsdLoads=True",
        "--/rtx/materialDb/syncLoads=True",
        "--/rtx/hydra/materialSyncLoads=True",
        "--/app/extensions/registryEnabled=false",
        "--/app/extensions/registryEnabled=0",
        "--/app/extensions/registryUrl=",
        "--/app/extensions/registryUrls=[]",
        "--/exts/omni.kit.registry.nucleus/enabled=false",
        "--/exts/omni.kit.registry.nucleus/registries=[]",
        "--/exts/omni.kit.registry.nucleus/urls=[]",
    ]
    existing = set(sys.argv)
    added = []
    for flag in kit_flags:
        if flag not in existing:
            sys.argv.append(flag)
            added.append(flag)
    if added:
        print(
            "[vjepa_planning] Kit offline extension registry flags injected: "
            + " ".join(added),
            flush=True,
        )

_OMNIGIBSON_RUNTIME_INITIALIZED = False
og = None
RealmEnvironmentDynamic = None
gm = None

if TYPE_CHECKING:
    import omnigibson as og  # type: ignore[no-redef]
    from realm.environments.env_dynamic import RealmEnvironmentDynamic  # type: ignore[no-redef]


def _install_omni_particle_stub_if_missing():
    try:
        import importlib

        importlib.import_module("omni.particle.system.core.scripts.core")
        importlib.import_module("omni.particle.system.core.scripts.utils")
        return
    except ModuleNotFoundError:
        pass

    omni_mod = sys.modules.get("omni")
    if omni_mod is None:
        return

    class _MissingOmniParticleObject:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None

            return _noop

    module_specs = {
        "omni.particle": "particle",
        "omni.particle.system": "system",
        "omni.particle.system.core": "core",
        "omni.particle.system.core.scripts": "scripts",
        "omni.particle.system.core.scripts.core": "core",
        "omni.particle.system.core.scripts.utils": "utils",
    }
    parent = omni_mod
    for module_name, attr_name in module_specs.items():
        module = sys.modules.get(module_name)
        if module is None:
            module = types.ModuleType(module_name)
            module.__path__ = []
            sys.modules[module_name] = module
        setattr(parent, attr_name, module)
        parent = module
    sys.modules["omni.particle.system.core.scripts.core"].Core = _MissingOmniParticleObject
    sys.modules["omni.particle.system.core.scripts.utils"].Utils = _MissingOmniParticleObject
    print("[vjepa_planning] installed fallback stub for missing omni.particle extension", flush=True)


def _set_omnigibson_macro_if_available(name, value):
    try:
        setattr(gm, name, value)
    except AttributeError:
        # The sim server may set and read these macros before dynamically
        # importing this module. MacroDict then locks the key; in that case the
        # already-active server config should win.
        current_value = getattr(gm, name)
        if current_value != value:
            print(
                f"[vjepa_planning] keeping preconfigured Omnigibson macro "
                f"{name}={current_value!r} (requested {value!r})",
                flush=True,
            )


def _configure_omnigibson_macros_before_import():
    # Omnigibson MacroDict values must be written before full simulator import
    # or environment construction. This keeps direct local runs and sim-server
    # dynamic imports on the same path.
    _install_omni_particle_stub_if_missing()
    _set_omnigibson_macro_if_available("DEFAULT_SIM_STEP_FREQ", 15)
    _set_omnigibson_macro_if_available("DEFAULT_RENDERING_FREQ", 15)
    _set_omnigibson_macro_if_available("DEFAULT_PHYSICS_FREQ", 120)
    _set_omnigibson_macro_if_available("ENABLE_TRANSITION_RULES", False)
    _set_omnigibson_macro_if_available("ENABLE_OBJECT_STATES", True)
    _set_omnigibson_macro_if_available("RENDER_VIEWER_CAMERA", False)
    _set_omnigibson_macro_if_available("ENABLE_HQ_RENDERING", True)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from realm.inference import InferenceClient, extract_from_obs
from realm.eval import SUPPORTED_TASKS, SUPPORTED_PERTURBATIONS

DEFAULT_SUCCESS_CASE_DIR = None
DEFAULT_SUCCESS_CASE_ROOT = os.environ.get(
    "SLOT_JEPA_SUCCESS_CASE_ROOT",
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "artifacts",
            "data",
            "robot",
            "successful_case",
        )
    ),
)
SUPPORTED_SCENE_PROFILES = ["realm_droid10", "impact"]
DEFAULT_TASK_VARIANT = "default"
DEFAULT_VJEPA_ACTION_REPEAT = 1
DEFAULT_VJEPA_MAX_SETTLE_STEPS = 4
DEFAULT_VJEPA_SETTLE_POS_TOL = 0.005
DEFAULT_VJEPA_SETTLE_ROT_TOL = 0.05
DEFAULT_ENABLE_SETTLE_DEBUG = False
DEFAULT_CONTACT_SAFETY_EFFORT_SCALE = 0.25
DEFAULT_ROTATE_DETERMINISTIC_STEP_RAD = 0.2
DEFAULT_ROTATE_DETERMINISTIC_AXIS = "yaw"
DEFAULT_ROTATE_DETERMINISTIC_OVERSHOOT_RAD = 0.25
DEFAULT_ROTATE_DETERMINISTIC_FULL_TURN_RAD = float(2.0 * np.pi)
POST_GRASP_CLOSE_COMMAND = 1.0
POST_GRASP_CLOSE_STEPS = 3
POST_GRASP_DOWNWARD_OFFSET_Z = -0.05
POST_GRASP_LIFT_OFFSET_Z = 0.10
POST_GRASP_LIFT_LARGE_OFFSET_Z = 0.15
POST_GRASP_LIFT_STEPS = 3
POST_GRASP_HOLD_EXTRA_CLOSE = 0.1
ROTATE_POST_GRASP_VALIDATION_STAGE = "GRASP_STABLE"
POST_MOVE_RELEASE_COMMAND = 0.0
POST_MOVE_RELEASE_LIFT_OFFSET_Z = 0.05
POST_MOVE_RELEASE_STEPS = 3
PUT_MOVE_CLOSE_MIN_Z_ROBOT = 0.15
VJEPA_VIEW_ORDER = [
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
]
VJEPA_VIEW_FILE_SUFFIX = {
    "exterior_image_1_left": "base",
    "exterior_image_2_left": "second",
    "wrist_image_left": "wrist",
}
DEFAULT_SLOT_GRIPPER_STABLE_LOSS_WINDOW = 4
DEFAULT_SLOT_GRIPPER_STABLE_FIXED_VIEW_LOSS_RANGE = 0.15
DEFAULT_SLOT_GRIPPER_STABLE_WRIST_VIEW_LOSS_RANGE = 0.50
DEFAULT_SLOT_GRIPPER_MIN_POSE_STEPS_BEFORE_GRIPPER = 5
DEFAULT_SLOT_GRIPPER_CALIBRATION_ROUNDS = 2
DEFAULT_SLOT_GRIPPER_CALIBRATION_SAMPLES = 5
DEFAULT_VJEPA_GRIPPER_GOAL_Z_MARGIN_M = 0.02
LOGGED_PHYSICAL_STAGES = {
    "REACH",
    "GRASP",
    "LIFT_SLIGHT",
    "LIFT_LARGE",
    "MOVE_CLOSE",
    "PLACE_INTO",
    "PLACE_ONTO",
    "ROTATED",
    "SUCCESS",
}
ACTIVE_GRASP_REQUIRED_STAGES = {
    "LIFT_SLIGHT",
    "LIFT_LARGE",
    "MOVE_CLOSE",
    "ROTATED",
    "POUR",
}
POST_CARRY_RELEASE_STAGES = {
    "PLACE_INTO",
    "PLACE_ONTO",
}


def _ensure_omnigibson_runtime_initialized():
    global _OMNIGIBSON_RUNTIME_INITIALIZED, og, RealmEnvironmentDynamic, gm
    if _OMNIGIBSON_RUNTIME_INITIALIZED:
        return
    _configure_kit_offline_mode()
    from omnigibson.macros import gm as _gm

    gm = _gm
    _configure_omnigibson_macros_before_import()

    import omnigibson as _og
    from realm.environments.env_dynamic import RealmEnvironmentDynamic as _RealmEnvironmentDynamic

    og = _og
    RealmEnvironmentDynamic = _RealmEnvironmentDynamic
    _OMNIGIBSON_RUNTIME_INITIALIZED = True


def get_post_grasp_validation_stage(env):
    """Return the task-specific carry validation stage after GRASP.

    Put / stack / pour require LIFT_SLIGHT, pick requires LIFT_LARGE, and
    rotate uses a synthetic stable-grasp validation so a transient GRASP flag
    does not advance into rotation with an unheld object.
    """
    task_progression = getattr(env, "task_progression", None)
    if not isinstance(task_progression, dict):
        return "LIFT_SLIGHT"
    stage_names = list(task_progression.keys())
    if "GRASP" in stage_names:
        post_grasp_stages = stage_names[stage_names.index("GRASP") + 1 :]
        for stage_name in post_grasp_stages:
            if stage_name in {"LIFT_SLIGHT", "LIFT_LARGE"}:
                return stage_name
        if "ROTATED" in post_grasp_stages:
            return ROTATE_POST_GRASP_VALIDATION_STAGE
        return None
    if "LIFT_LARGE" in task_progression:
        return "LIFT_LARGE"
    if "LIFT_SLIGHT" in task_progression:
        return "LIFT_SLIGHT"
    return None


def get_post_grasp_lift_offset_z(validation_stage):
    if validation_stage == ROTATE_POST_GRASP_VALIDATION_STAGE:
        return 0.0
    if validation_stage == "LIFT_LARGE":
        return float(POST_GRASP_LIFT_LARGE_OFFSET_Z)
    return float(POST_GRASP_LIFT_OFFSET_Z)


def describe_post_grasp_validation(validation_stage, offset_z):
    if validation_stage == ROTATE_POST_GRASP_VALIDATION_STAGE:
        return "validated stable grasp by holding the current pose"
    return f"validated {validation_stage} by lifting z {float(offset_z):.3f}"


def task_has_post_carry_release_stage(env) -> bool:
    task_progression = getattr(env, "task_progression", None)
    if not isinstance(task_progression, dict):
        return False
    return any(stage in task_progression for stage in POST_CARRY_RELEASE_STAGES)

EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXAMPLES_DIR, ".."))
REALM_OUTPUT_ROOT = os.path.join(REPO_ROOT, "realm", "outputs")


def to_uint8_rgb_image(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.size == 0:
        raise ValueError("Cannot convert an empty image array to uint8 RGB.")
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if float(np.max(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(arr)


class AsyncImageWriter:
    def __init__(self, max_workers: int = 2, max_pending: int = 32):
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self._max_pending = max(1, int(max_pending))
        self._pending = []

    @staticmethod
    def _save_image(path: str, array: np.ndarray):
        Image.fromarray(to_uint8_rgb_image(array)).save(path)

    def submit(self, path: str, array: np.ndarray):
        array_copy = np.asarray(array).copy()
        future = self._executor.submit(self._save_image, path, array_copy)
        self._pending.append(future)
        if len(self._pending) >= self._max_pending:
            self._pending.pop(0).result()

    def flush(self):
        while self._pending:
            self._pending.pop(0).result()

    def shutdown(self):
        try:
            self.flush()
        finally:
            self._executor.shutdown(wait=True)


def dump_robot_camera_observations(obs, robot_name: str, output_dir: str, prefix: str) -> None:
    """Save all robot-mounted camera RGB observations for sensor-key diagnostics."""
    if not isinstance(obs, dict) or robot_name not in obs or not isinstance(obs[robot_name], dict):
        return
    os.makedirs(output_dir, exist_ok=True)
    records = []
    for key, value in obs[robot_name].items():
        if not isinstance(value, dict) or "rgb" not in value:
            continue
        rgb = value["rgb"]
        rgb_np = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        try:
            rgb_np = to_uint8_rgb_image(rgb_np)
        except ValueError as exc:
            records.append(
                {
                    "key": str(key),
                    "shape": list(np.asarray(rgb_np).shape),
                    "skipped": True,
                    "reason": str(exc),
                }
            )
            continue
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(key))
        image_path = os.path.join(output_dir, f"{prefix}_{safe_key}.png")
        Image.fromarray(rgb_np).save(image_path)
        records.append(
            {
                "key": str(key),
                "shape": list(rgb_np.shape),
                "path": image_path,
            }
        )
    if records:
        with open(os.path.join(output_dir, f"{prefix}_camera_keys.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

IMPACT_TASK_NAME_MAP = {
    "put_green_block_into_bowl": "put_green_block_in_bowl",
}

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return None
    return float(np.dot(a, b) / denom)


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def pose6_to_transform(pose6: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = Rot.from_euler("xyz", pose6[3:6], degrees=False).as_matrix()
    tf[:3, 3] = pose6[:3]
    return tf


def camera_pose_to_transform(position: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = Rot.from_quat(quat_xyzw).as_matrix()
    tf[:3, 3] = position
    return tf


USD_CAMERA_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def convert_camera_transform_convention(tf_world_camera: np.ndarray, convention: str = "usd") -> np.ndarray:
    """Convert Isaac/USD camera frame to the optical frame used by DROID extrinsics."""
    tf = np.asarray(tf_world_camera, dtype=np.float64).copy()
    if convention in {"usd", "identity", None}:
        return tf
    if convention in {"opencv", "usd_to_opencv", "droid"}:
        tf[:3, :3] = tf[:3, :3] @ USD_CAMERA_TO_OPENCV
        return tf
    raise ValueError(f"Unknown camera frame convention: {convention}")


def transform_to_pose6(tf: np.ndarray) -> np.ndarray:
    pos = tf[:3, 3]
    rpy = Rot.from_matrix(tf[:3, :3]).as_euler("xyz", degrees=False)
    return np.concatenate([pos, rpy], axis=0).astype(np.float32)


def load_sim_state(og_module, state):
    if hasattr(og_module.sim, "load_state"):
        try:
            og_module.sim.load_state(state)
        except TypeError:
            og_module.sim.load_state(state=state)
        return
    if hasattr(og_module.sim, "restore_state"):
        try:
            og_module.sim.restore_state(state)
        except TypeError:
            og_module.sim.restore_state(state=state)
        return
    raise AttributeError("Current Omnigibson sim does not expose load_state/restore_state.")


def save_sim_state(og_module):
    if hasattr(og_module.sim, "dump_state"):
        try:
            return copy.deepcopy(og_module.sim.dump_state(serialized=False))
        except TypeError:
            return copy.deepcopy(og_module.sim.dump_state())
    if hasattr(og_module.sim, "save_state"):
        try:
            return copy.deepcopy(og_module.sim.save_state(serialized=False))
        except TypeError:
            return copy.deepcopy(og_module.sim.save_state())
    raise AttributeError("Current Omnigibson sim does not expose dump_state/save_state.")


def save_python_task_state(env):
    return {
        "was_lifted": copy.deepcopy(getattr(env, "was_lifted", None)) if hasattr(env, "was_lifted") else None,
        "task_progression": copy.deepcopy(getattr(env, "task_progression", None)),
    }


def load_python_task_state(env, state):
    if not isinstance(state, dict):
        return
    if hasattr(env, "was_lifted"):
        env.was_lifted = bool(state.get("was_lifted", False))
    saved_progression = state.get("task_progression")
    task_progression = getattr(env, "task_progression", None)
    if isinstance(task_progression, dict) and isinstance(saved_progression, dict):
        task_progression.clear()
        task_progression.update(copy.deepcopy(saved_progression))


def reset_python_task_flags(env):
    if hasattr(env, "was_lifted"):
        env.was_lifted = False
    task_progression = getattr(env, "task_progression", None)
    if isinstance(task_progression, dict):
        for key in task_progression:
            task_progression[key] = False


def clear_failed_grasp_candidate_flags(env):
    """Clear task flags caused by a candidate grasp that failed lift validation."""
    if hasattr(env, "was_lifted"):
        env.was_lifted = False
    task_progression = getattr(env, "task_progression", None)
    if not isinstance(task_progression, dict):
        return []
    stages_to_clear = (
        {"GRASP"}
        | set(ACTIVE_GRASP_REQUIRED_STAGES)
        | set(POST_CARRY_RELEASE_STAGES)
        | {"SUCCESS"}
    )
    cleared = []
    for stage in stages_to_clear:
        if stage in task_progression and bool(task_progression.get(stage, False)):
            task_progression[stage] = False
            cleared.append(stage)
    return sorted(cleared)


def gripper_to_env_command(gripper_state):
    value = float(np.asarray(gripper_state).reshape(-1)[0])
    if value < 0.0:
        return -1.0
    # DROID/model convention: 0.0 = open, 1.0 = closed. Only this boundary
    # function converts the normalized state to the simulator command range.
    value = float(np.clip(value, 0.0, 1.0))
    return float(2.0 * value - 1.0)


def get_raw_gripper_position_from_obs(obs, robot_name="DROID"):
    try:
        proprio = obs[robot_name]["proprio"]
        proprio_np = proprio.cpu().numpy() if hasattr(proprio, "cpu") else np.asarray(proprio)
        return float(np.asarray(proprio_np).reshape(-1)[7])
    except Exception:
        return None


def normalize_raw_gripper_position(raw_value):
    # Omnigibson proprio[7] follows the same DROID/model convention observed
    # here after normalization: roughly 0=open and 1=closed.
    return float(np.clip(float(raw_value) / 0.05, 0.0, 1.0))


def gripper_state_for_model(raw_value, units="raw_width"):
    raw_value = float(raw_value)
    if units == "raw_width":
        return float(np.clip(raw_value, 0.0, 0.05))
    if units == "normalized":
        return normalize_raw_gripper_position(raw_value)
    raise ValueError(f"Unsupported gripper model units: {units}")


def model_gripper_to_normalized(value, units="raw_width"):
    value = float(np.asarray(value).reshape(-1)[0])
    if units == "raw_width":
        return float(np.clip(value / 0.05, 0.0, 1.0))
    if units == "normalized":
        return float(np.clip(value, 0.0, 1.0))
    raise ValueError(f"Unsupported gripper model units: {units}")


def hold_current_robot_pose(env, robot_pose6, gripper_state):
    hold_action = np.concatenate(
        [
            np.asarray(robot_pose6, dtype=np.float32).reshape(6),
            np.array([gripper_to_env_command(gripper_state)], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    obs, rew, terminated, truncated, info = env.step(hold_action)
    return obs


def restore_sim_start_state(env, initial_sim_state, initial_robot_pose6, initial_gripper_state):
    load_sim_state(og, copy.deepcopy(initial_sim_state))
    reset_python_task_flags(env)
    obs = hold_current_robot_pose(env, initial_robot_pose6, initial_gripper_state)
    og.sim.render()
    return obs


def angle_wrap(angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles, dtype=np.float32)
    return ((angles + np.pi) % (2 * np.pi) - np.pi).astype(np.float32)


def pose_error_components(target_pose6: np.ndarray, current_pose6: np.ndarray) -> tuple[np.ndarray, float, float]:
    target_pose6 = np.asarray(target_pose6, dtype=np.float32).reshape(6)
    current_pose6 = np.asarray(current_pose6, dtype=np.float32).reshape(6)
    pos_err = target_pose6[:3] - current_pose6[:3]
    rot_err = angle_wrap(target_pose6[3:6] - current_pose6[3:6])
    pos_l2 = float(np.linalg.norm(pos_err))
    rot_l2 = float(np.linalg.norm(rot_err))
    return np.concatenate([pos_err, rot_err], axis=0).astype(np.float32), pos_l2, rot_l2


def pose_delta6(start_pose6: np.ndarray, end_pose6: np.ndarray) -> np.ndarray:
    start_pose6 = np.asarray(start_pose6, dtype=np.float32).reshape(6)
    end_pose6 = np.asarray(end_pose6, dtype=np.float32).reshape(6)
    pos_delta = end_pose6[:3] - start_pose6[:3]
    rot_delta = angle_wrap(end_pose6[3:6] - start_pose6[3:6])
    return np.concatenate([pos_delta, rot_delta], axis=0).astype(np.float32)


def compute_target_distance_change(current_pose6: np.ndarray, commanded_pose6: np.ndarray, target_pose6: np.ndarray) -> dict:
    current_pose6 = np.asarray(current_pose6, dtype=np.float32).reshape(6)
    commanded_pose6 = np.asarray(commanded_pose6, dtype=np.float32).reshape(6)
    target_pose6 = np.asarray(target_pose6, dtype=np.float32).reshape(6)
    before_err6, before_pos_l2, before_rot_l2 = pose_error_components(target_pose6, current_pose6)
    after_err6, after_pos_l2, after_rot_l2 = pose_error_components(target_pose6, commanded_pose6)
    return {
        "before_error6_robot": before_err6.astype(np.float32).copy(),
        "after_error6_robot": after_err6.astype(np.float32).copy(),
        "before_pos_l2_robot": float(before_pos_l2),
        "after_pos_l2_robot": float(after_pos_l2),
        "delta_pos_l2_robot": float(after_pos_l2 - before_pos_l2),
        "before_rot_l2_robot": float(before_rot_l2),
        "after_rot_l2_robot": float(after_rot_l2),
        "delta_rot_l2_robot": float(after_rot_l2 - before_rot_l2),
    }


def _load_vis_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def annotate_image_with_text_lines(image_path: str, lines, fill=(255, 255, 255), bg=(0, 0, 0)) -> None:
    if not os.path.exists(image_path):
        return
    img = Image.open(image_path).convert("RGB")
    text_lines = [str(line) for line in lines if line is not None and str(line)]
    if not text_lines:
        img.save(image_path)
        return
    line_h = 14
    pad = 4
    overlay_h = pad * 2 + line_h * len(text_lines)
    annotated = Image.new("RGB", (img.width, img.height + overlay_h), bg)
    draw = ImageDraw.Draw(annotated)
    for idx, line in enumerate(text_lines):
        draw.text((pad, pad + idx * line_h), line, fill=fill)
    annotated.paste(img, (0, overlay_h))
    annotated.save(image_path)


def save_wrist_image_with_cem_phase_outputs(image_path: str, wrist_image, model_response) -> None:
    img = Image.fromarray(to_uint8_rgb_image(wrist_image)).convert("RGB")
    phase_outputs = (model_response or {}).get("cem_phase_outputs") or []
    synthesized_from_generic_fields = False
    if not phase_outputs and isinstance(model_response, dict):
        best_target_robot_pose6 = model_response.get("best_target_robot_pose6")
        if best_target_robot_pose6 is not None:
            phase_outputs = [
                {
                    "cem_phase": "single_pass",
                    "target_robot_pose6": np.asarray(best_target_robot_pose6, dtype=np.float32).reshape(-1).tolist(),
                    "loss": model_response.get("best_loss"),
                    "per_view_losses": model_response.get("per_view_losses"),
                    "reference_view": model_response.get("reference_view"),
                    "skipped": False,
                }
            ]
            synthesized_from_generic_fields = True

    def _format_skip_reason(item):
        reason = item.get("skip_reason")
        detail = item.get("skip_reason_detail") or {}
        parts = []
        if detail.get("skip_by_insufficient_window"):
            parts.append("need_hist")
        if detail.get("skip_by_xyz"):
            value = detail.get("observed_translation_norm")
            threshold = detail.get("translation_threshold")
            if value is not None and threshold is not None:
                parts.append(f"xyz:{float(value):.3f}>{float(threshold):.3f}")
            else:
                parts.append("xyz")
        if detail.get("skip_by_rotation"):
            value = detail.get("observed_rotation_norm")
            threshold = detail.get("rotation_threshold")
            if value is not None and threshold is not None:
                parts.append(f"rot:{float(value):.3f}>{float(threshold):.3f}")
            else:
                parts.append("rot")
        if detail.get("skip_by_wrist_refine_skipped"):
            parts.append("wrist_skip")
        if parts:
            return ",".join(parts)
        return str(reason) if reason else ""

    group_order = ["fixed", "wrist", "gripper"]
    grouped = {}
    for item in phase_outputs:
        phase_name = str(item.get("cem_phase", ""))
        if "gripper" in phase_name:
            group = "gripper"
        elif "wrist" in phase_name:
            group = "wrist"
        else:
            group = "fixed"
        grouped[group] = item

    lines = [f"CEM phase outputs: n={len(phase_outputs)}"]
    if not phase_outputs:
        lines.append("phase diagnostics unavailable in model_response")
    elif synthesized_from_generic_fields:
        lines.append("showing synthesized single-pass diagnostics from generic model_response fields")
    for group in group_order:
        item = grouped.get(group)
        if item is None:
            if synthesized_from_generic_fields:
                continue
            lines.append(f"{group}: missing")
            continue
        phase_name = str(item.get("cem_phase", group))
        pose = np.asarray(item.get("target_robot_pose6", []), dtype=np.float32).reshape(-1)
        if pose.size >= 6:
            xyz = np.round(pose[:3], 3).tolist()
            rpy = np.round(pose[3:6], 3).tolist()
            pose_text = f"xyz={xyz} rpy={rpy}"
        else:
            pose_text = "pose=None"
        skipped = bool(item.get("skipped", False))
        suffix = " SKIP" if skipped else ""
        if skipped:
            skip_text = _format_skip_reason(item)
            if skip_text:
                suffix += f"({skip_text})"
        gripper_delta = item.get("gripper_delta")
        grip_text = "" if gripper_delta is None else f" dg={float(gripper_delta):+.3f}"
        loss = item.get("loss")
        loss_text = "" if loss is None else f" loss={float(loss):.4f}"
        ref_view = item.get("reference_view")
        ref_text = "" if ref_view is None else f" ref={ref_view}"
        lines.append(f"{group}/{phase_name}:{suffix} {pose_text}{grip_text}{loss_text}{ref_text}")
    per_view_losses = (model_response or {}).get("per_view_losses") or {}
    for view_name in VJEPA_VIEW_ORDER:
        if view_name not in per_view_losses:
            continue
        lines.append(f"{VJEPA_VIEW_FILE_SUFFIX.get(view_name, view_name)} loss={float(per_view_losses[view_name]):.4f}")

    line_h = 16
    pad = 6
    overlay_h = pad * 2 + line_h * len(lines)
    annotated = Image.new("RGB", (img.width, img.height + overlay_h), (0, 0, 0))
    draw = ImageDraw.Draw(annotated)
    for idx, line in enumerate(lines):
        draw.text((pad, pad + idx * line_h), line, fill=(255, 255, 255))
    annotated.paste(img, (0, overlay_h))
    annotated.save(image_path)


def append_slot_mask_panel(image, slot_panel):
    if slot_panel is None:
        return to_uint8_rgb_image(image)
    img = Image.fromarray(to_uint8_rgb_image(image)).convert("RGB")
    panel = Image.fromarray(to_uint8_rgb_image(slot_panel)).convert("RGB")
    if panel.height != img.height:
        scale = img.height / max(panel.height, 1)
        new_w = max(1, int(round(panel.width * scale)))
        nearest_resample = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
        panel = panel.resize((new_w, img.height), resample=nearest_resample)
    pad = 8
    canvas = Image.new("RGB", (img.width + pad + panel.width, img.height), color=(255, 255, 255))
    canvas.paste(img, (0, 0))
    canvas.paste(panel, (img.width + pad, 0))
    return np.asarray(canvas, dtype=np.uint8)


def get_slot_mask_panels_for_payload(client, payload):
    if not hasattr(client, "render_current_slot_mask_panels"):
        return {}
    try:
        response = client.render_current_slot_mask_panels(payload)
    except Exception as exc:
        if "Unknown endpoint: render_current_slot_mask_panels" in str(exc):
            return {}
        print(f"SLOT mask panel rendering issue: {exc}", flush=True)
        return {}
    if not isinstance(response, dict):
        return {}
    panels = response.get("mask_panels_by_view") or {}
    if not isinstance(panels, dict):
        return {}
    return panels


def get_goal_slot_mask_panels(client):
    try:
        response = client.render_goal_slot_mask_panels()
    except Exception as exc:
        print(f"SLOT goal mask panel rendering issue: {exc}", flush=True)
        return []
    if not isinstance(response, dict):
        return []
    panels = response.get("mask_panels_by_goal") or []
    if not isinstance(panels, list):
        print(
            f"SLOT goal mask panel rendering returned unexpected type: {type(panels).__name__}",
            flush=True,
        )
        return []
    print(
        f"SLOT goal mask panel rendering returned {len(panels)} goal panel groups.",
        flush=True,
    )
    return panels


def save_goal_images_with_slot_masks(task_debug_dir: str, goal_images, goal_state: dict, goal_panels_by_goal) -> int:
    if not goal_panels_by_goal:
        return 0
    view_dirs = {
        "exterior_image_1_left": "base_cam",
        "exterior_image_2_left": "second_cam",
        "wrist_image_left": "wrist_cam",
    }
    target_names = (goal_state or {}).get("target_names", [])
    saved_count = 0
    for goal_idx, img_map in enumerate(goal_images):
        if goal_idx >= len(goal_panels_by_goal):
            continue
        panels_by_view = goal_panels_by_goal[goal_idx] or {}
        target_name = target_names[goal_idx] if goal_idx < len(target_names) else f"goal_{goal_idx}"
        safe_target_name = str(target_name).replace(os.sep, "_")
        for view_name, img in img_map.items():
            out_dir_name = view_dirs.get(view_name)
            if out_dir_name is None:
                continue
            panel = panels_by_view.get(view_name)
            if panel is None:
                continue
            out_path = os.path.join(task_debug_dir, out_dir_name, f"goal_{goal_idx:02d}_{safe_target_name}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            Image.fromarray(append_slot_mask_panel(img, panel)).save(out_path)
            saved_count += 1
    return saved_count


def create_settle_visual_summary(
    task_debug_dir: str,
    settle_vis_dir: str,
    settle_history,
    max_columns: int = 8,
    output_path: str | None = None,
):
    step_count = len(settle_history)
    if step_count <= 0:
        return

    labels = [("start", "start"), *[(f"{idx:03d}", str(idx)) for idx in range(1, step_count + 1)], ("end", "end")]
    view_suffixes = ["base", "second", "wrist"]
    sample_path = os.path.join(task_debug_dir, "start_base.png")
    if not os.path.exists(sample_path):
        return

    sample_img = _load_vis_image(sample_path)
    img_w, img_h = sample_img.size
    text_h = 42
    pad = 8
    cols = min(max_columns, len(labels))
    rows = int(np.ceil(len(labels) / cols))
    canvas_w = cols * (img_w + pad) + pad
    canvas_h_per_view = rows * (img_h + text_h + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h_per_view * len(view_suffixes)), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for view_idx, suffix in enumerate(view_suffixes):
        y_offset = view_idx * canvas_h_per_view
        for item_idx, (file_prefix, display_name) in enumerate(labels):
            col = item_idx % cols
            row = item_idx // cols
            x0 = pad + col * (img_w + pad)
            y0 = y_offset + pad + row * (img_h + text_h + pad)
            if file_prefix == "start":
                path = os.path.join(task_debug_dir, f"start_{suffix}.png")
                subtitle = "start"
            elif file_prefix == "end":
                path = os.path.join(task_debug_dir, f"end_{suffix}.png")
                subtitle = "end"
            else:
                path = os.path.join(settle_vis_dir, f"{file_prefix}_{suffix}.png")
                step_idx = int(file_prefix) - 1
                pos_l2 = float(settle_history[step_idx]["target_pos_l2_robot"])
                subtitle = f"step {display_name}  L_pos={pos_l2:.3f}"
            if not os.path.exists(path):
                continue
            img = _load_vis_image(path)
            if img.size != (img_w, img_h):
                img = img.resize((img_w, img_h))
            canvas.paste(img, (x0, y0))
            draw.text((x0, y0 + img_h + 4), f"{suffix}", fill=(20, 20, 20))
            draw.text((x0, y0 + img_h + 20), subtitle, fill=(20, 20, 20))

    if output_path is None:
        output_path = os.path.join(task_debug_dir, "settle_summary.png")
    canvas.save(output_path)


def execute_absolute_robot_target(
    env: RealmEnvironmentDynamic,
    target_robot_pose6: np.ndarray,
    gripper_command: float,
    max_settle_steps: int,
    settle_pos_tol: float,
    settle_rot_tol: float,
    vis_dir: str | None = None,
    include_controller_debug: bool = False,
    use_fast_inner_step: bool = True,
    force_full_steps: bool = False,
    capture_step_obs: bool = False,
    cancel_event=None,
    contact_safety_effort_scale: float = DEFAULT_CONTACT_SAFETY_EFFORT_SCALE,
    freeze_arm_on_contact: bool = False,
    contact_stop_checker=None,
    retreat_on_contact: bool = False,
    contact_retreat_distance_m: float = 0.006,
):
    target_robot_pose6 = np.asarray(target_robot_pose6, dtype=np.float32).reshape(6)
    gripper_command = float(gripper_command)

    def _make_env_action(robot_pose6: np.ndarray) -> np.ndarray:
        action = np.concatenate(
            [np.asarray(robot_pose6, dtype=np.float32).reshape(6), np.array([gripper_command], dtype=np.float32)],
            axis=0,
        ).astype(np.float32)
        action[-1] = gripper_to_env_command(action[-1])
        return action

    env_action = _make_env_action(target_robot_pose6)

    settle_history = []
    curr_task_progression = 0.0
    terminated = False
    truncated = False
    info = {}
    obs = None
    vis_records = []
    fast_inner_steps_executed = 0
    full_inner_steps_executed = 0
    fast_inner_fallback_steps = 0
    contact_safety_steps = 0
    contact_safety_was_active = False
    arm_frozen_on_contact = False
    arm_frozen_contact_step = None
    arm_frozen_contact_pose6 = None
    arm_frozen_contact_reason = None
    arm_contact_retreat_applied = False
    arm_contact_retreat_commanded_pose6 = None
    arm_contact_retreat_actual_pose6 = None

    def _get_current_robot_pose6():
        ee_pos, ee_rot = env.get_ee_pose()
        ee_pos_np = ee_pos.cpu().numpy() if hasattr(ee_pos, "cpu") else np.array(ee_pos)
        ee_rot_np = ee_rot.cpu().numpy() if hasattr(ee_rot, "cpu") else np.array(ee_rot)
        ee_euler_np = Rot.from_quat(ee_rot_np).as_euler("xyz")
        ee_pose_world6 = np.concatenate([ee_pos_np, ee_euler_np]).astype(np.float32)
        return env._world2robot(ee_pose_world6).astype(np.float32)

    def _set_arm_effort_scale(scale):
        arm_controller = getattr(env.robot, "_controllers", {}).get("arm_0")
        if arm_controller is not None and hasattr(arm_controller, "set_contact_safety_effort_scale"):
            arm_controller.set_contact_safety_effort_scale(scale)

    def _build_z_safe_waypoints(current_robot_pose6: np.ndarray):
        """Avoid sweeping through objects when changing height and xy together."""
        z_delta = float(target_robot_pose6[2] - current_robot_pose6[2])
        z_eps = 1e-4
        if abs(z_delta) <= z_eps:
            return [("direct", target_robot_pose6.copy())]
        if z_delta < 0.0:
            xy_first = target_robot_pose6.copy()
            xy_first[2] = current_robot_pose6[2]
            return [
                ("xy_before_lower_z", xy_first),
                ("lower_z_to_target", target_robot_pose6.copy()),
            ]
        z_first = current_robot_pose6.copy()
        z_first[2] = target_robot_pose6[2]
        return [
            ("raise_z_before_xy", z_first),
            ("xy_after_raise_z", target_robot_pose6.copy()),
        ]

    def _allocate_waypoint_steps(waypoints, total_steps: int):
        total_steps = max(1, int(total_steps))
        if len(waypoints) <= 1:
            return [total_steps]
        first_steps = max(1, (total_steps + 1) // 2)
        second_steps = max(1, total_steps - first_steps)
        return [first_steps, second_steps]

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
    enable_fast_path = bool(use_fast_inner_step and vis_dir is None and not include_controller_debug)
    initial_robot_pose6 = _get_current_robot_pose6()
    waypoints = _build_z_safe_waypoints(initial_robot_pose6)
    waypoint_steps = _allocate_waypoint_steps(waypoints, max_settle_steps)
    settle_step_idx = 0
    try:
        for waypoint_idx, ((waypoint_name, waypoint_target_pose6), num_waypoint_steps) in enumerate(
            zip(waypoints, waypoint_steps)
        ):
            waypoint_action = _make_env_action(waypoint_target_pose6)
            env_action = waypoint_action
            for waypoint_repeat_idx in range(max(1, int(num_waypoint_steps))):
                if cancel_event is not None and cancel_event.is_set():
                    break
                settle_step_idx += 1
                pre_robot_pose6 = _get_current_robot_pose6()
                pre_target_z_error = float(target_robot_pose6[2] - pre_robot_pose6[2])
                pre_waypoint_z_error = float(waypoint_target_pose6[2] - pre_robot_pose6[2])
                effort_scale = float(np.clip(float(contact_safety_effort_scale), 0.0, 1.0))
                contact_safety_active = effort_scale < 1.0
                _set_arm_effort_scale(effort_scale)
                contact_safety_was_active = contact_safety_was_active or bool(contact_safety_active)
                if contact_safety_active:
                    contact_safety_steps += 1

                is_last_iter = waypoint_repeat_idx == max(1, int(num_waypoint_steps)) - 1
                if enable_fast_path and not is_last_iter:
                    fast_result = env.control_step_fast(waypoint_action)
                    fast_inner_steps_executed += 1
                    if bool(fast_result.get("used_fallback_step", False)):
                        fast_inner_fallback_steps += 1
                    obs = fast_result["obs"]
                    if fast_result["task_progression"] is not None:
                        curr_task_progression = fast_result["task_progression"]
                    terminated = fast_result["terminated"]
                    truncated = fast_result["truncated"]
                    info = fast_result["info"]
                else:
                    full_inner_steps_executed += 1
                    obs, curr_task_progression, terminated, truncated, info = env.step(waypoint_action)
                    if vis_dir is not None:
                        base_im, _, base_im_second, _, wrist_im, _, _ = extract_from_obs(obs, robot_name=env.robot.name)
                        Image.fromarray(base_im).save(os.path.join(vis_dir, f"{settle_step_idx:03d}_base.png"))
                        Image.fromarray(base_im_second).save(
                            os.path.join(vis_dir, f"{settle_step_idx:03d}_second.png")
                        )
                        Image.fromarray(wrist_im).save(os.path.join(vis_dir, f"{settle_step_idx:03d}_wrist.png"))
                step_robot_pose6 = _get_current_robot_pose6()
                step_err6, step_pos_l2, step_rot_l2 = pose_error_components(target_robot_pose6, step_robot_pose6)
                waypoint_err6, waypoint_pos_l2, waypoint_rot_l2 = pose_error_components(
                    waypoint_target_pose6, step_robot_pose6
                )
                record = {
                    "settle_step": int(settle_step_idx),
                    "waypoint_index": int(waypoint_idx),
                    "waypoint_name": waypoint_name,
                    "waypoint_target_robot_pose6": waypoint_target_pose6.copy(),
                    "waypoint_error6_robot": waypoint_err6.copy(),
                    "waypoint_pos_l2_robot": float(waypoint_pos_l2),
                    "waypoint_rot_l2_robot": float(waypoint_rot_l2),
                    "robot_pose6": step_robot_pose6.copy(),
                    "target_error6_robot": step_err6.copy(),
                    "target_pos_l2_robot": float(step_pos_l2),
                    "target_rot_l2_robot": float(step_rot_l2),
                    "pre_target_z_error_robot": float(pre_target_z_error),
                    "pre_waypoint_z_error_robot": float(pre_waypoint_z_error),
                    "contact_safety_active": bool(contact_safety_active),
                    "contact_safety_effort_scale": float(effort_scale),
                }
                if capture_step_obs and obs is not None:
                    step_base_im, _, step_base_im_second, _, step_wrist_im, _, step_gripper_state = extract_from_obs(
                        obs, robot_name=env.robot.name
                    )
                    step_raw_gripper = get_raw_gripper_position_from_obs(obs, robot_name=env.robot.name)
                    record["obs_images"] = {
                        "base": step_base_im.copy(),
                        "second": step_base_im_second.copy(),
                        "wrist": step_wrist_im.copy(),
                    }
                    record["gripper_state_model"] = float(step_gripper_state)
                    record["raw_gripper_state"] = None if step_raw_gripper is None else float(step_raw_gripper)
                    record["gripper_state_normalized"] = (
                        float(np.clip(float(step_gripper_state), 0.0, 1.0))
                        if step_raw_gripper is None
                        else float(normalize_raw_gripper_position(step_raw_gripper))
                    )
                if include_controller_debug:
                    arm_controller = getattr(env.robot, "_controllers", {}).get("arm_0")
                    controller_debug = {}
                    if arm_controller is not None:
                        controller_debug = to_jsonable(getattr(arm_controller, "last_control_debug", {}))
                    record["controller_debug"] = controller_debug
                settle_history.append(record)
                vis_records.append(
                    {
                        "settle_step": int(settle_step_idx),
                        "waypoint_index": int(waypoint_idx),
                        "waypoint_name": waypoint_name,
                        "target_pos_l2_robot": float(step_pos_l2),
                        "target_rot_l2_robot": float(step_rot_l2),
                        "waypoint_pos_l2_robot": float(waypoint_pos_l2),
                        "waypoint_rot_l2_robot": float(waypoint_rot_l2),
                    }
                )
                if (
                    not arm_frozen_on_contact
                    and bool(freeze_arm_on_contact)
                    and contact_stop_checker is not None
                    and obs is not None
                ):
                    try:
                        if bool(contact_stop_checker(obs)):
                            arm_frozen_on_contact = True
                            arm_frozen_contact_step = int(settle_step_idx)
                            arm_frozen_contact_pose6 = step_robot_pose6.copy()
                            arm_frozen_contact_reason = "contact_stop_checker_triggered"
                            if bool(retreat_on_contact):
                                actual_delta6 = pose_delta6(pre_robot_pose6, step_robot_pose6)
                                actual_xyz = np.asarray(actual_delta6[:3], dtype=np.float32)
                                actual_xyz_norm = float(np.linalg.norm(actual_xyz))
                                retreat_xyz = np.zeros(3, dtype=np.float32)
                                if actual_xyz_norm > 1.0e-6:
                                    retreat_xyz = -actual_xyz / actual_xyz_norm * min(
                                        actual_xyz_norm, float(max(contact_retreat_distance_m, 0.0))
                                    )
                                else:
                                    retreat_xyz[2] = float(max(contact_retreat_distance_m, 0.0))
                                retreat_target_pose6 = step_robot_pose6.copy()
                                retreat_target_pose6[:3] += retreat_xyz.astype(np.float32)
                                arm_contact_retreat_commanded_pose6 = retreat_target_pose6.astype(np.float32).copy()
                                retreat_action = _make_env_action(retreat_target_pose6)
                                obs, curr_task_progression, terminated, truncated, info = env.step(retreat_action)
                                env_action = retreat_action
                                arm_contact_retreat_applied = True
                                arm_contact_retreat_actual_pose6 = _get_current_robot_pose6().astype(np.float32).copy()
                                arm_frozen_contact_pose6 = arm_contact_retreat_actual_pose6.astype(np.float32).copy()
                            record["arm_frozen_on_contact"] = True
                            record["arm_frozen_contact_step"] = int(settle_step_idx)
                            record["arm_frozen_contact_pose6"] = step_robot_pose6.copy()
                            record["arm_frozen_contact_reason"] = arm_frozen_contact_reason
                            record["arm_contact_retreat_applied"] = bool(arm_contact_retreat_applied)
                            record["arm_contact_retreat_commanded_pose6"] = (
                                None
                                if arm_contact_retreat_commanded_pose6 is None
                                else arm_contact_retreat_commanded_pose6.astype(np.float32).copy()
                            )
                            record["arm_contact_retreat_actual_pose6"] = (
                                None
                                if arm_contact_retreat_actual_pose6 is None
                                else arm_contact_retreat_actual_pose6.astype(np.float32).copy()
                            )
                            break
                    except Exception as exc:
                        record["arm_frozen_on_contact_check_error"] = str(exc)
                if terminated or truncated:
                    break
                if (
                    not force_full_steps
                    and waypoint_pos_l2 <= settle_pos_tol
                    and waypoint_rot_l2 <= settle_rot_tol
                ):
                    break
            if cancel_event is not None and cancel_event.is_set():
                break
            if terminated or truncated:
                break
            if arm_frozen_on_contact:
                break
    finally:
        _set_arm_effort_scale(1.0)

    if obs is None and not (cancel_event is not None and cancel_event.is_set()):
        obs = env.get_current_obs()
        curr_task_progression = env.recompute_task_progression(obs)

    return {
        "obs": obs,
        "env_action": env_action,
        "settle_history": settle_history,
        "cancelled": bool(cancel_event is not None and cancel_event.is_set()),
        "task_progression": float(curr_task_progression),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
        "vis_records": vis_records,
        "used_fast_inner_step": bool(enable_fast_path),
        "fast_inner_steps_executed": int(fast_inner_steps_executed),
        "full_inner_steps_executed": int(full_inner_steps_executed),
        "fast_inner_fallback_steps": int(fast_inner_fallback_steps),
        "contact_safety_active": bool(contact_safety_was_active),
        "contact_safety_steps": int(contact_safety_steps),
        "contact_safety_effort_scale": float(contact_safety_effort_scale),
        "arm_frozen_on_contact": bool(arm_frozen_on_contact),
        "arm_frozen_contact_step": arm_frozen_contact_step,
        "arm_frozen_contact_pose6": None
        if arm_frozen_contact_pose6 is None
        else arm_frozen_contact_pose6.astype(np.float32).copy(),
        "arm_frozen_contact_reason": arm_frozen_contact_reason,
        "arm_contact_retreat_applied": bool(arm_contact_retreat_applied),
        "arm_contact_retreat_commanded_pose6": None
        if arm_contact_retreat_commanded_pose6 is None
        else arm_contact_retreat_commanded_pose6.astype(np.float32).copy(),
        "arm_contact_retreat_actual_pose6": None
        if arm_contact_retreat_actual_pose6 is None
        else arm_contact_retreat_actual_pose6.astype(np.float32).copy(),
        "z_safe_waypoints": [
            {
                "name": name,
                "target_robot_pose6": pose.copy(),
                "allocated_settle_steps": int(steps),
            }
            for (name, pose), steps in zip(waypoints, waypoint_steps)
        ],
    }


def execute_gripper_only(
    env: RealmEnvironmentDynamic,
    gripper_command: float,
    steps: int = 1,
    hold_robot_pose6: np.ndarray | None = None,
    downpress_z_robot: float = 0.0,
    vis_dir: str | None = None,
    capture_step_obs: bool = False,
    cancel_event=None,
):
    """Drive the gripper while holding the current EE target fixed."""
    gripper_command = float(np.clip(float(gripper_command), 0.0, 1.0))
    env_gripper_command = float(gripper_to_env_command(gripper_command))
    downpress_z_robot = float(downpress_z_robot)
    steps = max(1, int(steps))
    obs = None
    curr_task_progression = 0.0
    terminated = False
    truncated = False
    info = {}
    used_fallback = False
    target_joint_pos = None
    gripper_dof_idx = None
    actual_joint_pos = None
    rendered_after_step = False
    env_action = None
    gripper_execution_mode = "apply_action_hold_current_pose"
    hold_robot_pose6_base = None
    hold_robot_pose6_target = None
    initial_env_action = None
    adaptive_downpress_release = False
    adaptive_downpress_release_step = None
    adaptive_downpress_actual_z_robot = None
    adaptive_downpress_required_z_robot = None
    settle_history = []
    vis_records = []
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    def _get_current_robot_pose6() -> np.ndarray:
        ee_pos, ee_rot = env.get_ee_pose()
        ee_pos_np = ee_pos.cpu().numpy() if hasattr(ee_pos, "cpu") else np.asarray(ee_pos)
        ee_rot_np = ee_rot.cpu().numpy() if hasattr(ee_rot, "cpu") else np.asarray(ee_rot)
        ee_pose_world6 = np.concatenate([ee_pos_np, Rot.from_quat(ee_rot_np).as_euler("xyz")]).astype(np.float32)
        return env._world2robot(ee_pose_world6).astype(np.float32)

    def _read_gripper_joint_pos(indices):
        joint_pos = env.robot.get_joint_positions()
        if torch.is_tensor(joint_pos):
            return joint_pos[indices].detach().cpu().numpy().astype(np.float32)
        return np.asarray(joint_pos)[indices.detach().cpu().numpy()].astype(np.float32)

    try:
        gripper_controller = getattr(env.robot, "_controllers", {}).get("gripper_0")
        if gripper_controller is None:
            raise AttributeError("robot has no gripper_0 controller")
        gripper_dof_idx = getattr(gripper_controller, "dof_idx", None)
        if gripper_dof_idx is None:
            gripper_dof_idx = getattr(env.robot, "_gripper_control_idx", None)
        if gripper_dof_idx is None:
            raise AttributeError("could not find gripper dof indices")
        if torch.is_tensor(gripper_dof_idx):
            gripper_dof_idx = gripper_dof_idx.to(device=og.sim.device)
        else:
            gripper_dof_idx = torch.as_tensor(gripper_dof_idx, dtype=torch.long, device=og.sim.device)

        hold_robot_pose6_base = (
            _get_current_robot_pose6()
            if hold_robot_pose6 is None
            else np.asarray(hold_robot_pose6, dtype=np.float32).reshape(6).copy()
        )
        hold_robot_pose6_target = hold_robot_pose6_base.astype(np.float32).copy()
        hold_robot_pose6_target[2] += downpress_z_robot
        env_action = np.concatenate(
            [hold_robot_pose6_target, np.atleast_1d(np.array([env_gripper_command], dtype=np.float32))],
            axis=0,
        ).astype(np.float32)
        initial_env_action = env_action.copy()
        # Use the full env.step() path so gripper command scaling, smooth / binary
        # behavior, mimic fingers, grasp-state bookkeeping, and camera refresh match
        # normal execution. The arm target is the current absolute EE pose.
        executed_steps = 0
        downpress_check_step = max(1, min(4, steps // 3))
        downpress_progress_threshold = 0.25
        adaptive_downpress_required_z_robot = (
            float(downpress_z_robot) * float(downpress_progress_threshold)
            if downpress_z_robot < -1.0e-6
            else None
        )
        for step_idx in range(steps):
            if cancel_event is not None and cancel_event.is_set():
                break
            obs, curr_task_progression, terminated, truncated, step_info = env.step(env_action)
            if isinstance(step_info, dict):
                info.update(step_info)
            executed_steps += 1
            current_robot_pose6_for_record = _get_current_robot_pose6()
            step_pos_l2 = float(
                np.linalg.norm(current_robot_pose6_for_record[:3] - hold_robot_pose6_target[:3])
            )
            step_rot_l2 = float(
                np.linalg.norm(pose_delta6(current_robot_pose6_for_record, hold_robot_pose6_target)[3:6])
            )
            record = {
                "settle_step": int(executed_steps),
                "waypoint_index": 0,
                "waypoint_name": "gripper_only",
                "waypoint_target_robot_pose6": hold_robot_pose6_target.copy(),
                "waypoint_error6_robot": pose_delta6(current_robot_pose6_for_record, hold_robot_pose6_target),
                "waypoint_pos_l2_robot": float(step_pos_l2),
                "waypoint_rot_l2_robot": float(step_rot_l2),
                "robot_pose6": current_robot_pose6_for_record.copy(),
                "target_error6_robot": pose_delta6(current_robot_pose6_for_record, hold_robot_pose6_target),
                "target_pos_l2_robot": float(step_pos_l2),
                "target_rot_l2_robot": float(step_rot_l2),
                "gripper_command_normalized": float(gripper_command),
                "gripper_command_env": float(env_gripper_command),
                "downpress_z_robot": float(downpress_z_robot),
            }
            if vis_dir is not None or capture_step_obs:
                step_base_im, _, step_base_im_second, _, step_wrist_im, _, step_gripper_state = extract_from_obs(
                    obs, robot_name=env.robot.name
                )
                if vis_dir is not None:
                    Image.fromarray(step_base_im).save(os.path.join(vis_dir, f"{executed_steps:03d}_base.png"))
                    Image.fromarray(step_base_im_second).save(
                        os.path.join(vis_dir, f"{executed_steps:03d}_second.png")
                    )
                    Image.fromarray(step_wrist_im).save(os.path.join(vis_dir, f"{executed_steps:03d}_wrist.png"))
                if capture_step_obs:
                    step_raw_gripper = get_raw_gripper_position_from_obs(obs, robot_name=env.robot.name)
                    record["obs_images"] = {
                        "base": step_base_im.copy(),
                        "second": step_base_im_second.copy(),
                        "wrist": step_wrist_im.copy(),
                    }
                    record["gripper_state_model"] = float(step_gripper_state)
                    record["raw_gripper_state"] = None if step_raw_gripper is None else float(step_raw_gripper)
                    record["gripper_state_normalized"] = (
                        float(np.clip(float(step_gripper_state), 0.0, 1.0))
                        if step_raw_gripper is None
                        else float(normalize_raw_gripper_position(step_raw_gripper))
                    )
            settle_history.append(record)
            vis_records.append(
                {
                    "settle_step": int(executed_steps),
                    "waypoint_index": 0,
                    "waypoint_name": "gripper_only",
                    "target_pos_l2_robot": float(step_pos_l2),
                    "target_rot_l2_robot": float(step_rot_l2),
                    "waypoint_pos_l2_robot": float(step_pos_l2),
                    "waypoint_rot_l2_robot": float(step_rot_l2),
                }
            )
            if (
                downpress_z_robot < -1.0e-6
                and not adaptive_downpress_release
                and step_idx + 1 >= downpress_check_step
            ):
                current_robot_pose6 = _get_current_robot_pose6()
                actual_downpress_z = float(current_robot_pose6[2] - hold_robot_pose6_base[2])
                adaptive_downpress_actual_z_robot = actual_downpress_z
                # If the EE cannot move down because the table / object blocks it,
                # stop pushing the arm downward and spend the remaining cycles on
                # the gripper command itself. Otherwise the arm objective can keep
                # fighting contact while the fingers never finish closing.
                if actual_downpress_z > float(adaptive_downpress_required_z_robot):
                    adaptive_downpress_release = True
                    adaptive_downpress_release_step = int(step_idx + 1)
                    env_action = np.concatenate(
                        [
                            current_robot_pose6.astype(np.float32),
                            np.atleast_1d(np.array([env_gripper_command], dtype=np.float32)),
                        ],
                        axis=0,
                    ).astype(np.float32)
                    info["gripper_adaptive_downpress_release"] = True
                    info["gripper_adaptive_downpress_release_step"] = int(adaptive_downpress_release_step)
                    info["gripper_adaptive_downpress_actual_z_robot"] = float(actual_downpress_z)
                    info["gripper_adaptive_downpress_required_z_robot"] = float(
                        adaptive_downpress_required_z_robot
                    )
            if terminated or truncated:
                break
        try:
            actual_joint_pos = _read_gripper_joint_pos(gripper_dof_idx)
            controller_control = getattr(gripper_controller, "_control", None)
            if controller_control is not None:
                info["gripper_controller_control"] = (
                    controller_control.detach().cpu().numpy().astype(np.float32)
                    if torch.is_tensor(controller_control)
                    else np.asarray(controller_control, dtype=np.float32)
                )
        except Exception as joint_exc:
            info["gripper_joint_read_error"] = str(joint_exc)
        try:
            og.sim.render()
            rendered_after_step = True
        except Exception as render_exc:
            info["gripper_only_render_error"] = str(render_exc)
        if obs is None:
            obs = env.get_current_obs()
            curr_task_progression = env.recompute_task_progression(obs)
    except Exception as exc:
        info["gripper_only_apply_action_error"] = str(exc)
        try:
            gripper_execution_mode = "direct_joint_target_fallback"
            command_tensor = torch.as_tensor([env_gripper_command], dtype=torch.float32, device=og.sim.device)
            if hasattr(gripper_controller, "_preprocess_command"):
                target_joint_pos = gripper_controller._preprocess_command(command_tensor)
            else:
                target_joint_pos = command_tensor
            target_joint_pos = target_joint_pos.to(device=og.sim.device, dtype=torch.float32).reshape(-1)
            env.robot.set_joint_positions(target_joint_pos, indices=gripper_dof_idx, drive=True)
            executed_steps = 0
            for _ in range(steps):
                if cancel_event is not None and cancel_event.is_set():
                    break
                og.sim.step()
                executed_steps += 1
            try:
                actual_joint_pos = _read_gripper_joint_pos(gripper_dof_idx)
            except Exception as joint_exc:
                info["gripper_joint_read_error"] = str(joint_exc)
            try:
                og.sim.render()
                rendered_after_step = True
            except Exception as render_exc:
                info["gripper_only_render_error"] = str(render_exc)
            obs = env.get_current_obs()
            curr_task_progression = env.recompute_task_progression(obs)
        except Exception as fallback_exc:
            used_fallback = True
            fallback_target_pose6 = (
                _get_current_robot_pose6()
                if hold_robot_pose6_base is None
                else hold_robot_pose6_base.astype(np.float32).copy()
            )
            fallback_target_pose6 = fallback_target_pose6.astype(np.float32).copy()
            fallback_target_pose6[2] += downpress_z_robot
            fallback_result = execute_absolute_robot_target(
                env=env,
                target_robot_pose6=fallback_target_pose6,
                gripper_command=gripper_command,
                max_settle_steps=steps,
                settle_pos_tol=DEFAULT_VJEPA_SETTLE_POS_TOL,
                settle_rot_tol=DEFAULT_VJEPA_SETTLE_ROT_TOL,
                vis_dir=vis_dir,
                capture_step_obs=capture_step_obs,
                cancel_event=cancel_event,
            )
            obs = fallback_result["obs"]
            curr_task_progression = fallback_result["task_progression"]
            terminated = fallback_result["terminated"]
            truncated = fallback_result["truncated"]
            info = fallback_result["info"]
            info = dict(info or {})
            info["gripper_only_apply_action_error"] = str(exc)
            info["gripper_only_direct_joint_fallback_error"] = str(fallback_exc)

    if obs is None:
        obs = env.get_current_obs()
        curr_task_progression = env.recompute_task_progression(obs)

    return {
        "obs": obs,
        "task_progression": float(curr_task_progression),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
        "settle_history": settle_history,
        "vis_records": vis_records,
        "used_fast_inner_step": False,
        "fast_inner_steps_executed": 0,
        "full_inner_steps_executed": int(locals().get("executed_steps", 0))
        if gripper_execution_mode == "apply_action_hold_current_pose"
        else 0,
        "fast_inner_fallback_steps": 0,
        "contact_safety_active": False,
        "contact_safety_steps": 0,
        "contact_safety_effort_scale": 1.0,
        "env_action": None if env_action is None else env_action.copy(),
        "initial_env_action": None if initial_env_action is None else initial_env_action.copy(),
        "gripper_hold_robot_pose6_before_downpress": None
        if hold_robot_pose6_base is None
        else hold_robot_pose6_base.copy(),
        "gripper_hold_robot_pose6": None
        if hold_robot_pose6_target is None
        else hold_robot_pose6_target.copy(),
        "gripper_downpress_z_robot": float(downpress_z_robot),
        "gripper_command_normalized": float(gripper_command),
        "gripper_command_env": float(env_gripper_command),
        "gripper_execution_mode": gripper_execution_mode,
        "gripper_target_joint_pos": None
        if target_joint_pos is None
        else target_joint_pos.detach().cpu().numpy().astype(np.float32),
        "gripper_dof_idx": None
        if gripper_dof_idx is None
        else gripper_dof_idx.detach().cpu().numpy().astype(np.int64),
        "gripper_actual_joint_pos": actual_joint_pos,
        "gripper_rendered_after_step": bool(rendered_after_step),
        "steps": int(locals().get("executed_steps", steps)),
        "used_fallback_absolute_pose": bool(used_fallback),
        "adaptive_downpress_release": bool(adaptive_downpress_release),
        "adaptive_downpress_release_step": adaptive_downpress_release_step,
        "adaptive_downpress_actual_z_robot": adaptive_downpress_actual_z_robot,
        "adaptive_downpress_required_z_robot": adaptive_downpress_required_z_robot,
    }


def apply_camera_delta_to_world_pose6(
    current_ee_world6: np.ndarray,
    delta_action_camera: np.ndarray,
    cam_quat_world: np.ndarray,
) -> np.ndarray:
    current_ee_world6 = np.asarray(current_ee_world6, dtype=np.float32).reshape(6)
    delta_action_camera = np.asarray(delta_action_camera, dtype=np.float32).reshape(7)
    cam_rot_world = Rot.from_quat(np.asarray(cam_quat_world, dtype=np.float64))
    current_rot_world = Rot.from_euler("xyz", current_ee_world6[3:6], degrees=False)
    delta_rot_camera = Rot.from_euler("xyz", delta_action_camera[3:6], degrees=False)

    delta_xyz_world = cam_rot_world.apply(delta_action_camera[:3])
    delta_rot_world = cam_rot_world * delta_rot_camera * cam_rot_world.inv()
    target_xyz_world = current_ee_world6[:3] + delta_xyz_world.astype(np.float32)
    target_rot_world = delta_rot_world * current_rot_world
    target_rpy_world = target_rot_world.as_euler("xyz", degrees=False).astype(np.float32)
    return np.concatenate([target_xyz_world.astype(np.float32), target_rpy_world], axis=0)


def format_vjepa_action_for_env(
    env: RealmEnvironmentDynamic,
    current_ee_world6: np.ndarray,
    delta_action_camera: np.ndarray,
    cam_quat_world: np.ndarray,
) -> np.ndarray:
    delta_action_camera = np.asarray(delta_action_camera, dtype=np.float32).reshape(7)
    target_ee_world6 = apply_camera_delta_to_world_pose6(
        current_ee_world6=current_ee_world6,
        delta_action_camera=delta_action_camera,
        cam_quat_world=cam_quat_world,
    )
    target_pose6_robot = env._world2robot(target_ee_world6).astype(np.float32)
    env_action = np.concatenate([target_pose6_robot, delta_action_camera[6:7]], axis=0).astype(np.float32)
    env_action[-1] = gripper_to_env_command(env_action[-1])
    return env_action


def load_task_cfg(config_path: str, task_cfg_path: str) -> dict:
    task_cfg_full_path = os.path.join(config_path, "tasks", task_cfg_path)
    with open(task_cfg_full_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_task_cfg_path(task: str, scene_profile: str, task_variant: str = DEFAULT_TASK_VARIANT) -> str:
    if scene_profile == "realm_droid10":
        variant_name = task_variant or DEFAULT_TASK_VARIANT
        task_cfg_path = f"REALM_DROID10/{task}/{variant_name}.yaml"
        full_path = os.path.join("/app/realm/config", "tasks", task_cfg_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"No REALM_DROID10 config for task `{task}` variant `{variant_name}`. Expected: {full_path}"
            )
        return task_cfg_path
    if scene_profile == "impact":
        if task_variant not in (None, DEFAULT_TASK_VARIANT):
            raise FileNotFoundError(
                f"Task variants are currently only generated for REALM_DROID10, got scene_profile={scene_profile}."
            )
        impact_task = IMPACT_TASK_NAME_MAP.get(task, task)
        task_cfg_path = f"IMPACT/{impact_task}/default.yaml"
        full_path = os.path.join("/app/realm/config", "tasks", task_cfg_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"No IMPACT config for task `{task}`. Expected: {full_path}"
            )
        return task_cfg_path
    raise ValueError(f"Unsupported scene_profile: {scene_profile}")


def make_case_dir_name(
    task: str,
    perturbation: str,
    state_frame: str,
    scene_profile: str,
    task_variant: str = DEFAULT_TASK_VARIANT,
) -> str:
    suffix = "" if scene_profile == "realm_droid10" else f"_{scene_profile}"
    variant_suffix = "" if task_variant in (None, DEFAULT_TASK_VARIANT) else f"_{task_variant}"
    return f"{task}{variant_suffix}_{perturbation}_{state_frame}_frame{suffix}"


def make_planning_output_dir_name(
    task: str,
    perturbation: str,
    state_frame: str,
    scene_profile: str,
    goal_mode: str,
    task_variant: str = DEFAULT_TASK_VARIANT,
) -> str:
    case_name = make_case_dir_name(task, perturbation, state_frame, scene_profile, task_variant)
    return f"{case_name}_{goal_mode}"


def get_planning_output_dir(
    model_type: str,
    task: str,
    perturbation: str,
    state_frame: str,
    scene_profile: str,
    goal_mode: str,
    task_variant: str = DEFAULT_TASK_VARIANT,
    timestamp: str | None = None,
    output_namespace: str | None = None,
) -> str:
    safe_model_type = str(output_namespace or model_type).strip().lower()
    safe_task = str(task).strip()
    dir_name = make_planning_output_dir_name(
        task=task,
        perturbation=perturbation,
        state_frame=state_frame,
        scene_profile=scene_profile,
        goal_mode=goal_mode,
        task_variant=task_variant,
    )
    base_dir = os.path.join(REALM_OUTPUT_ROOT, safe_model_type, safe_task, dir_name)
    if timestamp:
        return os.path.join(base_dir, str(timestamp))
    return base_dir


def get_base_camera_world_pose(env: RealmEnvironmentDynamic, task_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    pose_name = (
        env.get_external_camera_pose_name("cam1")
        if hasattr(env, "get_external_camera_pose_name")
        else task_cfg["camera_extrinsics"]["cam1"] if "camera_extrinsics" in task_cfg else "default"
    )
    cam_pos, cam_quat = env.construct_ext_cam_pose_by_name(pose_name, env.robot_pos, env.robot_rot_rad)
    return np.asarray(cam_pos, dtype=np.float32), np.asarray(cam_quat, dtype=np.float32)


def get_second_camera_world_pose(env: RealmEnvironmentDynamic, task_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(env, "get_external_camera_pose_name"):
        ext_cam2_pose = env.get_external_camera_pose_name("cam2")
    else:
        ext_cam1_pose = task_cfg["camera_extrinsics"]["cam1"] if "camera_extrinsics" in task_cfg else "default"
        if "camera_extrinsics" in task_cfg and "cam2" in task_cfg["camera_extrinsics"]:
            ext_cam2_pose = task_cfg["camera_extrinsics"]["cam2"]
        else:
            ext_cam2_pose = "default" if ext_cam1_pose == "CP3" else "CP3"
    cam_pos, cam_quat = env.construct_ext_cam_pose_by_name(ext_cam2_pose, env.robot_pos, env.robot_rot_rad)
    return np.asarray(cam_pos, dtype=np.float32), np.asarray(cam_quat, dtype=np.float32)


def get_wrist_camera_world_pose(env: RealmEnvironmentDynamic) -> tuple[np.ndarray, np.ndarray]:
    sensor_candidates = []
    for attr in ("sensors", "_sensors"):
        sensors = getattr(env.robot, attr, None)
        if isinstance(sensors, dict):
            sensor_candidates.extend(sensors.items())
    for name, sensor in sensor_candidates:
        name_l = str(name).lower()
        if "camera" not in name_l:
            continue
        if "gripper" in name_l or "wrist" in name_l or "hand" in name_l:
            if hasattr(sensor, "get_position_orientation"):
                pos, quat = sensor.get_position_orientation()
                pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
                quat = quat.cpu().numpy() if hasattr(quat, "cpu") else np.asarray(quat)
                return np.asarray(pos, dtype=np.float32), np.asarray(quat, dtype=np.float32)

    for link_name in ("gripper_link_camera", "camera_link", "camera_base_link", "panda_hand"):
        link = env.robot.links.get(link_name)
        if link is None:
            continue
        pos, quat = link.get_position_orientation()
        pos = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
        quat = quat.cpu().numpy() if hasattr(quat, "cpu") else np.asarray(quat)
        return np.asarray(pos, dtype=np.float32), np.asarray(quat, dtype=np.float32)

    raise RuntimeError("Unable to locate wrist camera pose from robot sensors or links.")


def get_camera_extrinsics_pose6(
    cam_pos_world: np.ndarray,
    cam_quat_world: np.ndarray,
    camera_frame_convention: str = "usd",
) -> np.ndarray:
    tf_world_cam = camera_pose_to_transform(cam_pos_world, cam_quat_world)
    tf_world_cam = convert_camera_transform_convention(tf_world_cam, convention=camera_frame_convention)
    return transform_to_pose6(tf_world_cam)


def world_pose_to_camera_pose6(
    ee_pose_world6: np.ndarray,
    cam_pos_world: np.ndarray,
    cam_quat_world: np.ndarray,
    camera_frame_convention: str = "usd",
) -> np.ndarray:
    tf_world_ee = pose6_to_transform(ee_pose_world6.astype(np.float64))
    tf_world_cam = camera_pose_to_transform(cam_pos_world.astype(np.float64), cam_quat_world.astype(np.float64))
    tf_world_cam = convert_camera_transform_convention(tf_world_cam, convention=camera_frame_convention)
    tf_cam_ee = np.linalg.inv(tf_world_cam) @ tf_world_ee
    return transform_to_pose6(tf_cam_ee)


def pose_in_parent_frame(child_pose6: np.ndarray, parent_pose6: np.ndarray) -> np.ndarray:
    tf_parent = pose6_to_transform(parent_pose6.astype(np.float64))
    tf_child = pose6_to_transform(child_pose6.astype(np.float64))
    tf_rel = np.linalg.inv(tf_parent) @ tf_child
    return transform_to_pose6(tf_rel)


def camera_delta_to_robot_delta(action7_camera: np.ndarray, cam_quat_world: np.ndarray, robot_yaw_world: float) -> np.ndarray:
    action7_camera = np.asarray(action7_camera, dtype=np.float32).reshape(7)
    rot_world_cam = Rot.from_quat(cam_quat_world.astype(np.float64))
    rot_world_robot = Rot.from_euler("z", float(robot_yaw_world), degrees=False)
    rot_robot_cam = rot_world_robot.inv() * rot_world_cam

    delta_xyz_robot = rot_robot_cam.apply(action7_camera[:3])
    delta_rot_cam = Rot.from_euler("xyz", action7_camera[3:6], degrees=False)
    delta_rot_robot = rot_robot_cam * delta_rot_cam * rot_robot_cam.inv()
    delta_rpy_robot = delta_rot_robot.as_euler("xyz", degrees=False).astype(np.float32)

    return np.concatenate([delta_xyz_robot.astype(np.float32), delta_rpy_robot, action7_camera[6:7]], axis=0)


def get_view_world_poses(env: RealmEnvironmentDynamic, task_cfg: dict) -> dict:
    cam1_pos, cam1_quat = get_base_camera_world_pose(env, task_cfg)
    cam2_pos, cam2_quat = get_second_camera_world_pose(env, task_cfg)
    wrist_pos, wrist_quat = get_wrist_camera_world_pose(env)
    return {
        "exterior_image_1_left": (cam1_pos, cam1_quat),
        "exterior_image_2_left": (cam2_pos, cam2_quat),
        "wrist_image_left": (wrist_pos, wrist_quat),
    }


def select_goal_targets(goal_targets, goal_mode: str):
    if not goal_targets:
        return goal_targets

    if goal_mode == "all":
        return goal_targets

    if goal_mode == "end":
        for target in reversed(goal_targets):
            if target.get("name") == "end" or target.get("stage") == "SUCCESS":
                return [target]
        return [goal_targets[-1]]

    if goal_mode == "first":
        goal_targets_sorted = sorted(goal_targets, key=lambda x: (int(x.get("step", 10**9)), str(x.get("name", ""))))
        return [goal_targets_sorted[0]]

    raise ValueError(f"Unsupported goal_mode: {goal_mode}")


def _safe_current_grasping(env: RealmEnvironmentDynamic, obs) -> bool:
    if obs is None:
        return False
    try:
        return bool(env.check_grasp_condition(obs))
    except Exception:
        return False


def _slot_response_executed_gripper_phase(model_response: dict) -> bool:
    for phase in model_response.get("cem_phase_outputs") or []:
        if phase.get("cem_phase") == "wrist_gripper" and not bool(phase.get("skipped", False)):
            return True
    return False


def slot_three_view_losses_stable_for_gripper(
    step_traces,
    min_pose_steps_before_gripper=DEFAULT_SLOT_GRIPPER_MIN_POSE_STEPS_BEFORE_GRIPPER,
    stable_loss_window=DEFAULT_SLOT_GRIPPER_STABLE_LOSS_WINDOW,
    fixed_view_loss_range=DEFAULT_SLOT_GRIPPER_STABLE_FIXED_VIEW_LOSS_RANGE,
    wrist_view_loss_range=DEFAULT_SLOT_GRIPPER_STABLE_WRIST_VIEW_LOSS_RANGE,
):
    """Decide whether the next SLOT step should run gripper search.

    Skip gripper for the first few outer steps, then trigger it when the
    three objective losses have mostly stalled. The gripper-slot loss is not
    used as a precondition because gripper search is the action that should
    reduce that loss.
    """
    if not step_traces:
        return {
            "stable": False,
            "reason": "no_step_trace",
            "min_pose_steps_before_gripper": int(min_pose_steps_before_gripper),
        }
    current_trace = step_traces[-1]
    current_step = int(current_trace.get("step", len(step_traces) - 1))
    if current_step < int(min_pose_steps_before_gripper):
        return {
            "stable": False,
            "reason": "within_initial_gripper_skip_window",
            "step": int(current_step),
            "min_pose_steps_before_gripper": int(min_pose_steps_before_gripper),
        }
    if current_trace.get("slot_planning_request") != "pose_only":
        return {
            "stable": False,
            "reason": "current_step_is_not_pose_only",
            "step": int(current_step),
            "slot_planning_request": current_trace.get("slot_planning_request"),
            "min_pose_steps_before_gripper": int(min_pose_steps_before_gripper),
        }

    pose_traces = [
        trace
        for trace in step_traces
        if trace.get("slot_planning_request") == "pose_only"
        and isinstance(trace.get("server_per_view_objective_losses"), dict)
    ]
    stable_loss_window = max(1, int(stable_loss_window))
    if len(pose_traces) < stable_loss_window:
        return {
            "stable": False,
            "reason": "insufficient_pose_loss_window",
            "step": int(current_step),
            "pose_loss_window": int(len(pose_traces)),
            "required_pose_loss_window": int(stable_loss_window),
            "min_pose_steps_before_gripper": int(min_pose_steps_before_gripper),
        }

    recent_traces = pose_traces[-stable_loss_window:]
    objective_views = [
        "exterior_image_1_left",
        "exterior_image_2_left",
        "wrist_image_left",
    ]
    loss_history_by_view = {}
    missing_views = []
    for view_name in objective_views:
        values = []
        for trace in recent_traces:
            losses = trace.get("server_per_view_objective_losses") or {}
            value = losses.get(view_name)
            if value is None:
                break
            values.append(float(value))
        if len(values) != len(recent_traces):
            missing_views.append(view_name)
        else:
            loss_history_by_view[view_name] = values
    if missing_views:
        return {
            "stable": False,
            "reason": "missing_recent_objective_loss_views",
            "step": int(current_step),
            "missing_views": missing_views,
            "pose_loss_window": int(len(recent_traces)),
            "min_pose_steps_before_gripper": int(min_pose_steps_before_gripper),
        }

    loss_ranges = {
        view_name: float(np.max(values) - np.min(values))
        for view_name, values in loss_history_by_view.items()
    }
    skip_reasons = []
    for view_name in ("exterior_image_1_left", "exterior_image_2_left"):
        if loss_ranges[view_name] > float(fixed_view_loss_range):
            skip_reasons.append(f"{view_name}_loss_not_stalled")
    if loss_ranges["wrist_image_left"] > float(wrist_view_loss_range):
        skip_reasons.append("wrist_image_left_loss_not_stalled")
    stable = len(skip_reasons) == 0
    return {
        "stable": bool(stable),
        "reason": "three_view_objective_losses_stalled" if stable else "+".join(skip_reasons),
        "step": int(current_step),
        "min_pose_steps_before_gripper": int(min_pose_steps_before_gripper),
        "pose_loss_window": int(len(recent_traces)),
        "stable_loss_window": int(stable_loss_window),
        "fixed_view_loss_range_threshold": float(fixed_view_loss_range),
        "wrist_view_loss_range_threshold": float(wrist_view_loss_range),
        "loss_history_by_view": loss_history_by_view,
        "loss_range_by_view": loss_ranges,
        "skip_reasons": skip_reasons,
    }


def _task_has_stage(env: RealmEnvironmentDynamic, stage: str) -> bool:
    task_progression = getattr(env, "task_progression", None)
    return isinstance(task_progression, dict) and stage in task_progression


def _raw_stage_condition_reached(env: RealmEnvironmentDynamic, stage: str, obs=None) -> bool:
    checker = getattr(env, "success_conditions", {}).get(stage)
    if checker is None or obs is None:
        return False
    try:
        return bool(checker(obs))
    except Exception:
        return False


def _reach_gripper_touch_reached(env: RealmEnvironmentDynamic) -> bool:
    if getattr(env, "task_type", None) != "reach":
        return False
    try:
        main_objects = getattr(env, "main_objects", None)
        finger_links = list(getattr(env, "robot_finger_links", []) or [])
        if not main_objects or not finger_links:
            return False
        obj = main_objects[0]
        touching_links = env._object_contacting_links(obj, finger_links)
        return bool(len(touching_links) > 0)
    except Exception:
        return False


def _reach_motion_evidence_reached(env: RealmEnvironmentDynamic) -> bool:
    """Inference-only fallback for reach tasks when contact likely displaced the object.

    Some reach rollouts visibly bump the target and move / rotate it, but the
    simulator's Touching state never latches. Treat sufficiently large object
    motion as evidence of contact so reach evaluation is less brittle.
    """
    if getattr(env, "task_type", None) != "reach":
        return False
    try:
        main_objects = getattr(env, "main_objects", None)
        if not main_objects:
            return False
        obj = main_objects[0]
        pos_curr, quat_curr = obj.get_position_orientation()
        pos_curr_np = pos_curr.detach().cpu().numpy() if hasattr(pos_curr, "detach") else np.asarray(pos_curr)
        quat_curr_np = (
            quat_curr.detach().cpu().numpy() if hasattr(quat_curr, "detach") else np.asarray(quat_curr)
        )
        pos_curr_np = np.asarray(pos_curr_np, dtype=np.float64).reshape(3)
        quat_curr_np = np.asarray(quat_curr_np, dtype=np.float64).reshape(4)

        pos_orig = getattr(env, "mo_pos_orig", None)
        quat_orig = getattr(env, "mo_rot_orig", None)
        bbox_orig = getattr(env, "mo_bbox_orig", None)
        if pos_orig is None or quat_orig is None:
            return False
        pos_orig_np = pos_orig.detach().cpu().numpy() if hasattr(pos_orig, "detach") else np.asarray(pos_orig)
        quat_orig_np = quat_orig.detach().cpu().numpy() if hasattr(quat_orig, "detach") else np.asarray(quat_orig)
        pos_orig_np = np.asarray(pos_orig_np, dtype=np.float64).reshape(3)
        quat_orig_np = np.asarray(quat_orig_np, dtype=np.float64).reshape(4)

        pos_diff_m = float(np.linalg.norm(pos_curr_np - pos_orig_np))
        rot_diff_rad = float((Rot.from_quat(quat_orig_np).inv() * Rot.from_quat(quat_curr_np)).magnitude())

        bbox_np = None if bbox_orig is None else np.asarray(bbox_orig, dtype=np.float64).reshape(-1)
        max_extent_m = float(np.max(np.abs(bbox_np))) if bbox_np is not None and bbox_np.size > 0 else 0.0
        pos_threshold_m = max(0.008, 0.08 * max_extent_m)
        rot_threshold_rad = 0.14
        return bool(pos_diff_m >= pos_threshold_m or rot_diff_rad >= rot_threshold_rad)
    except Exception:
        return False


def _reach_proximity_evidence_reached(env: RealmEnvironmentDynamic, distance_threshold_m: float = 0.02) -> bool:
    """Inference-only fallback: close enough to count as reach for evaluation."""
    if getattr(env, "task_type", None) != "reach":
        return False
    try:
        main_objects = getattr(env, "main_objects", None)
        if not main_objects:
            return False
        obj = main_objects[0]
        obj_pos, _ = obj.get_position_orientation()
        obj_pos_np = obj_pos.detach().cpu().numpy() if hasattr(obj_pos, "detach") else np.asarray(obj_pos)
        obj_pos_np = np.asarray(obj_pos_np, dtype=np.float64).reshape(3)

        finger_links = list(getattr(env, "robot_finger_links", []) or [])
        if len(finger_links) >= 2:
            finger_positions = []
            for link in finger_links[:2]:
                link_pos, _ = link.get_position_orientation()
                link_pos_np = (
                    link_pos.detach().cpu().numpy() if hasattr(link_pos, "detach") else np.asarray(link_pos)
                )
                finger_positions.append(np.asarray(link_pos_np, dtype=np.float64).reshape(3))
            gripper_tip_pos_np = np.mean(np.stack(finger_positions, axis=0), axis=0)
        else:
            ee_pos, _ = env.get_ee_pose()
            ee_pos_np = ee_pos.detach().cpu().numpy() if hasattr(ee_pos, "detach") else np.asarray(ee_pos)
            gripper_tip_pos_np = np.asarray(ee_pos_np, dtype=np.float64).reshape(3)

        return bool(np.linalg.norm(gripper_tip_pos_np - obj_pos_np) <= float(distance_threshold_m))
    except Exception:
        return False


def physical_stage_reached(
    env: RealmEnvironmentDynamic,
    stage: str,
    curr_task_progression: float,
    obs=None,
    validated_stages=None,
) -> bool:
    if not stage:
        return False
    if stage == ROTATE_POST_GRASP_VALIDATION_STAGE:
        return _safe_current_grasping(env, obs)
    validated_stages = set(validated_stages or [])
    if stage == "SUCCESS":
        task_progression = getattr(env, "task_progression", None)
        if not isinstance(task_progression, dict):
            return float(curr_task_progression) >= 1.0
        if any(
            post_stage in task_progression
            and physical_stage_reached(
                env,
                post_stage,
                curr_task_progression,
                obs=obs,
                validated_stages=validated_stages,
            )
            for post_stage in POST_CARRY_RELEASE_STAGES
        ):
            return True
        return all(
            physical_stage_reached(
                env,
                stage_name,
                curr_task_progression,
                obs=obs,
                validated_stages=validated_stages,
            )
            for stage_name in task_progression.keys()
        )
    task_progression = getattr(env, "task_progression", None)
    if task_progression is None:
        return False
    base_reached = bool(task_progression.get(stage, False))
    if not base_reached and stage == "REACH":
        base_reached = (
            _reach_gripper_touch_reached(env)
            or _reach_motion_evidence_reached(env)
            or _reach_proximity_evidence_reached(env)
        )
    if stage in POST_CARRY_RELEASE_STAGES:
        base_reached = base_reached or _raw_stage_condition_reached(env, stage, obs=obs)
    if not base_reached:
        return False
    if stage in validated_stages:
        return True

    # Carry stages are only valid if the object is still physically grasped now.
    # This prevents a dropped object plus a correctly moving EE from advancing
    # the MPC state machine.
    if stage in ACTIVE_GRASP_REQUIRED_STAGES:
        return _safe_current_grasping(env, obs)

    # Release/place is the final physical evidence for put/stack. It must be
    # allowed to pass directly because the object may already be released into
    # the target before the intermediate MOVE_CLOSE flag is externally observed.
    if stage in POST_CARRY_RELEASE_STAGES:
        return True

    return True


def get_main_object_pose_debug(env: RealmEnvironmentDynamic) -> dict:
    """Best-effort pose/rotation diagnostic for the task's main object."""
    try:
        main_objects = getattr(env, "main_objects", None)
        if not main_objects:
            return {"available": False, "reason": "no_main_objects"}
        obj = main_objects[0]
        pos, quat = obj.get_position_orientation()
        pos_np = pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
        quat_np = quat.cpu().numpy() if hasattr(quat, "cpu") else np.asarray(quat)
        pos_np = np.asarray(pos_np, dtype=np.float32).reshape(3)
        quat_np = np.asarray(quat_np, dtype=np.float32).reshape(4)
        debug = {
            "available": True,
            "name": getattr(obj, "name", None),
            "pos_world": pos_np.copy(),
            "quat_xyzw": quat_np.copy(),
            "pos_diff_m": None,
            "rot_diff_rad": None,
        }
        orig_pos = getattr(env, "mo_pos_orig", None)
        if orig_pos is not None:
            orig_pos_np = (
                orig_pos.cpu().numpy() if hasattr(orig_pos, "cpu") else np.asarray(orig_pos)
            )
            orig_pos_np = np.asarray(orig_pos_np, dtype=np.float64).reshape(3)
            debug["pos_diff_m"] = float(
                np.linalg.norm(pos_np.astype(np.float64) - orig_pos_np)
            )
        orig_quat = getattr(env, "mo_rot_orig", None)
        if orig_quat is not None:
            orig_quat_np = (
                orig_quat.cpu().numpy() if hasattr(orig_quat, "cpu") else np.asarray(orig_quat)
            )
            orig_quat_np = np.asarray(orig_quat_np, dtype=np.float64).reshape(4)
            rel_rot = Rot.from_quat(orig_quat_np).inv() * Rot.from_quat(
                quat_np.astype(np.float64)
            )
            debug["rot_diff_rad"] = float(rel_rot.magnitude())
        return debug
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def get_gripper_search_height_gate_debug(
    env: RealmEnvironmentDynamic,
    robot_pose6: np.ndarray,
    current_goal_target: dict | None,
) -> dict:
    """Return whether gripper search is allowed based on current EE height.

    The gate prevents premature gripper-only search while the end effector is
    still far above the object's graspable vertical band.
    """
    debug = {
        "allowed": True,
        "reason": "no_height_gate_needed",
        "current_z_robot": None,
        "goal_z_robot": None,
        "goal_margin_m": float(DEFAULT_VJEPA_GRIPPER_GOAL_Z_MARGIN_M),
        "max_gripper_search_z_robot": None,
    }
    try:
        robot_pose6_np = np.asarray(robot_pose6, dtype=np.float32).reshape(-1)
        if robot_pose6_np.shape[0] < 3:
            debug["allowed"] = False
            debug["reason"] = "invalid_robot_pose6"
            return debug
        debug["current_z_robot"] = float(robot_pose6_np[2])

        goal_z_robot = None
        if isinstance(current_goal_target, dict) and current_goal_target.get("ee_state7_robot") is not None:
            goal_state7_robot = np.asarray(current_goal_target["ee_state7_robot"], dtype=np.float32).reshape(-1)
            if goal_state7_robot.shape[0] >= 3:
                goal_z_robot = float(goal_state7_robot[2])
                debug["goal_z_robot"] = goal_z_robot

        if goal_z_robot is None:
            debug["allowed"] = True
            debug["reason"] = "no_goal_z_robot"
            return debug
        max_gripper_search_z = float(goal_z_robot + float(DEFAULT_VJEPA_GRIPPER_GOAL_Z_MARGIN_M))
        debug["max_gripper_search_z_robot"] = max_gripper_search_z
        debug["allowed"] = bool(float(robot_pose6_np[2]) <= float(max_gripper_search_z))
        debug["reason"] = "below_gripper_search_height_gate" if debug["allowed"] else "above_gripper_search_height_gate"
        return debug
    except Exception as exc:
        debug["allowed"] = True
        debug["reason"] = f"height_gate_error:{exc}"
        return debug


def get_vjepa_direct_gripper_close_debug(
    env: RealmEnvironmentDynamic,
    current_goal_target: dict | None,
    close_distance_threshold_m: float = 0.025,
    center_axis_threshold_m: float = 0.012,
    approach_offset_threshold_m: float = 0.035,
) -> dict:
    """Return whether the target object is centered enough to directly close the gripper.

    This is a lightweight inference-time heuristic for VJEPA planners. It avoids
    a separate gripper-search phase and only closes when the target object center
    sits near the midpoint between the two finger links.
    """
    debug = {
        "allowed": False,
        "reason": "unavailable",
        "object_name": None,
        "distance_to_finger_midpoint_m": None,
        "center_axis_offset_m": None,
        "approach_offset_m": None,
        "finger_gap_m": None,
        "finger_midpoint_world": None,
        "object_center_world": None,
        "reference_center_source": None,
        "goal_z_robot": None,
    }
    try:
        finger_links = list(getattr(env, "robot_finger_links", []) or [])
        if len(finger_links) < 2:
            debug["reason"] = "insufficient_finger_links"
            return debug
        if not isinstance(current_goal_target, dict) or current_goal_target.get("ee_state7_robot") is None:
            debug["reason"] = "missing_goal_pose"
            return debug
        goal_state7_robot = np.asarray(current_goal_target["ee_state7_robot"], dtype=np.float32).reshape(-1)
        if goal_state7_robot.shape[0] < 3:
            debug["reason"] = "invalid_goal_pose_shape"
            return debug
        debug["object_name"] = current_goal_target.get("name")
        reference_pose_world = np.asarray(env._robot2world(goal_state7_robot[:6]), dtype=np.float64).reshape(-1)
        obj_pos_np = reference_pose_world[:3].astype(np.float64).copy()
        debug["object_center_world"] = obj_pos_np.astype(np.float32).copy()
        debug["reference_center_source"] = "goal_ee_state7_robot"

        finger_positions = []
        for link in finger_links[:2]:
            link_pos, _ = link.get_position_orientation()
            link_pos_np = link_pos.detach().cpu().numpy() if hasattr(link_pos, "detach") else np.asarray(link_pos)
            finger_positions.append(np.asarray(link_pos_np, dtype=np.float64).reshape(3))
        finger_a, finger_b = finger_positions
        finger_mid = 0.5 * (finger_a + finger_b)
        debug["finger_midpoint_world"] = finger_mid.astype(np.float32).copy()

        finger_delta = finger_b - finger_a
        finger_gap = float(np.linalg.norm(finger_delta))
        debug["finger_gap_m"] = finger_gap
        if finger_gap <= 1.0e-6:
            debug["reason"] = "degenerate_finger_gap"
            return debug
        finger_axis = finger_delta / finger_gap

        ee_pos, _ = env.get_ee_pose()
        ee_pos_np = ee_pos.detach().cpu().numpy() if hasattr(ee_pos, "detach") else np.asarray(ee_pos)
        ee_pos_np = np.asarray(ee_pos_np, dtype=np.float64).reshape(3)
        approach_vec = finger_mid - ee_pos_np
        approach_norm = float(np.linalg.norm(approach_vec))
        if approach_norm <= 1.0e-6:
            debug["reason"] = "degenerate_approach_axis"
            return debug
        approach_axis = approach_vec / approach_norm

        offset = obj_pos_np - finger_mid
        distance_to_mid = float(np.linalg.norm(offset))
        center_axis_offset = float(abs(np.dot(offset, finger_axis)))
        approach_offset = float(abs(np.dot(offset, approach_axis)))
        debug["distance_to_finger_midpoint_m"] = distance_to_mid
        debug["center_axis_offset_m"] = center_axis_offset
        debug["approach_offset_m"] = approach_offset

        debug["goal_z_robot"] = float(goal_state7_robot[2])

        centered_between_fingers = center_axis_offset <= max(float(center_axis_threshold_m), 0.35 * finger_gap)
        close_to_midpoint = distance_to_mid <= max(float(close_distance_threshold_m), 0.55 * finger_gap)
        close_in_approach = approach_offset <= float(approach_offset_threshold_m)
        debug["allowed"] = bool(centered_between_fingers and close_to_midpoint and close_in_approach)
        debug["reason"] = (
            "object_centered_between_fingers"
            if debug["allowed"]
            else "object_not_centered_between_fingers"
        )
        return debug
    except Exception as exc:
        debug["allowed"] = False
        debug["reason"] = f"direct_close_error:{exc}"
        return debug


def get_vjepa_gripper_search_distance_gate_debug(
    env: RealmEnvironmentDynamic,
    current_goal_target: dict | None,
    max_distance_threshold_m: float = 0.10,
) -> dict:
    """Allow VJEPA gripper search only when the reference target is near the gripper."""
    debug = {
        "allowed": False,
        "reason": "unavailable",
        "object_name": None,
        "distance_to_finger_midpoint_m": None,
        "max_distance_threshold_m": float(max_distance_threshold_m),
        "finger_midpoint_world": None,
        "reference_center_world": None,
        "reference_center_source": None,
        "goal_z_robot": None,
    }
    try:
        finger_links = list(getattr(env, "robot_finger_links", []) or [])
        if len(finger_links) < 2:
            debug["reason"] = "insufficient_finger_links"
            return debug
        if not isinstance(current_goal_target, dict) or current_goal_target.get("ee_state7_robot") is None:
            debug["reason"] = "missing_goal_pose"
            return debug
        goal_state7_robot = np.asarray(current_goal_target["ee_state7_robot"], dtype=np.float32).reshape(-1)
        if goal_state7_robot.shape[0] < 3:
            debug["reason"] = "invalid_goal_pose_shape"
            return debug
        debug["object_name"] = current_goal_target.get("name")
        debug["goal_z_robot"] = float(goal_state7_robot[2])
        reference_pose_world = np.asarray(env._robot2world(goal_state7_robot[:6]), dtype=np.float64).reshape(-1)
        reference_center_world = reference_pose_world[:3].astype(np.float64).copy()
        debug["reference_center_world"] = reference_center_world.astype(np.float32).copy()
        debug["reference_center_source"] = "goal_ee_state7_robot"

        finger_positions = []
        for link in finger_links[:2]:
            link_pos, _ = link.get_position_orientation()
            link_pos_np = link_pos.detach().cpu().numpy() if hasattr(link_pos, "detach") else np.asarray(link_pos)
            finger_positions.append(np.asarray(link_pos_np, dtype=np.float64).reshape(3))
        finger_mid = 0.5 * (finger_positions[0] + finger_positions[1])
        debug["finger_midpoint_world"] = finger_mid.astype(np.float32).copy()

        distance_to_mid = float(np.linalg.norm(reference_center_world - finger_mid))
        debug["distance_to_finger_midpoint_m"] = distance_to_mid
        debug["allowed"] = bool(distance_to_mid <= float(max_distance_threshold_m))
        debug["reason"] = "within_distance_gate" if debug["allowed"] else "outside_distance_gate"
        return debug
    except Exception as exc:
        debug["allowed"] = False
        debug["reason"] = f"distance_gate_error:{exc}"
        return debug


def selected_goal_stage_reached(
    env: RealmEnvironmentDynamic,
    goal_state: dict,
    curr_task_progression: float,
    goal_idx: int = 0,
    obs=None,
    validated_stages=None,
) -> bool:
    target_stages = list(goal_state.get("target_stages", []))
    if not target_stages or not (0 <= int(goal_idx) < len(target_stages)):
        return False
    target_stage = target_stages[int(goal_idx)]
    if physical_stage_reached(
        env,
        target_stage,
        curr_task_progression,
        obs=obs,
        validated_stages=validated_stages,
    ):
        return True
    if target_stage == "MOVE_CLOSE":
        return any(
            _task_has_stage(env, post_stage)
            and physical_stage_reached(
                env,
                post_stage,
                curr_task_progression,
                obs=obs,
                validated_stages=validated_stages,
            )
            for post_stage in POST_CARRY_RELEASE_STAGES
        )
    return False


def load_success_case_data(success_case_dir: str, goal_mode: str = "all"):
    manifest_path = os.path.join(success_case_dir, "manifest.json")
    goal_targets_path = os.path.join(success_case_dir, "goal_targets.pt")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing success-case manifest: {manifest_path}")
    if not os.path.exists(goal_targets_path):
        raise FileNotFoundError(f"Missing success-case goal_targets: {goal_targets_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    goal_targets = torch.load(goal_targets_path, map_location="cpu")
    goal_targets = select_goal_targets(goal_targets, goal_mode=goal_mode)
    goal_images = []
    for target in goal_targets:
        view_images = {}
        for view_name in VJEPA_VIEW_ORDER:
            suffix = VJEPA_VIEW_FILE_SUFFIX[view_name]
            img_path = os.path.join(success_case_dir, f"{target['name']}_{suffix}.png")
            if os.path.exists(img_path):
                view_images[view_name] = np.array(Image.open(img_path).convert("RGB"))
        if not view_images:
            expected = os.path.join(success_case_dir, f"{target['name']}_base.png")
            raise FileNotFoundError(f"Missing goal image for target `{target['name']}`: {expected}")
        goal_images.append(view_images)
    available_views = [
        view_name for view_name in VJEPA_VIEW_ORDER
        if all(view_name in target_images for target_images in goal_images)
    ]
    if not available_views:
        raise ValueError(
            "No common camera views are available across selected goal targets. "
            "The released successful_case target package must contain consistent multiview images."
        )
    manifest_camera_extrinsics6 = manifest.get("camera_extrinsics6")
    if manifest_camera_extrinsics6 is None:
        manifest_camera_extrinsics6 = manifest.get("camera_extrinsics6_robot")
    if manifest_camera_extrinsics6 is None:
        manifest_camera_extrinsics6 = manifest.get("camera_extrinsics6_world")
    if manifest_camera_extrinsics6 is None:
        raise KeyError(
            "Successful case manifest is missing camera extrinsics. "
            "Expected one of: camera_extrinsics6, camera_extrinsics6_robot, camera_extrinsics6_world."
        )
    goal_state = {
        "success_case_dir": success_case_dir,
        "target_names": [target["name"] for target in goal_targets],
        "target_stages": [target["stage"] for target in goal_targets],
        "camera_extrinsics6": np.array(manifest_camera_extrinsics6, dtype=np.float32),
        "camera_frame_convention": manifest.get("camera_frame_convention"),
        "camera_extrinsics6_by_view": {
            k: np.asarray(v, dtype=np.float32)
            for k, v in (manifest.get("camera_extrinsics6_by_view") or {}).items()
        },
        "camera_pose6_by_view": {
            k: np.asarray(v, dtype=np.float32)
            for k, v in (manifest.get("camera_pose6_by_view") or {}).items()
        },
        "goal_targets": goal_targets,
        "available_views": available_views,
        "fixed_camera_presets": list(manifest.get("fixed_camera_presets") or []),
        "actual_external_camera_presets": dict(manifest.get("actual_external_camera_presets") or {}),
    }
    return manifest, goal_targets, goal_images, goal_state


def load_success_rollout_trace(success_case_dir: str):
    trace_path = os.path.join(success_case_dir, "rollout_trace.pt")
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Missing successful rollout trace: {trace_path}")
    return torch.load(trace_path, map_location="cpu")


def build_multiview_obs_payload(
    base_im,
    base_im_second,
    wrist_im,
    gripper_state,
    camera_pose6_by_view,
    camera_extrinsics_by_view,
    camera_frame_convention="usd",
    completed_stages=None,
):
    image_by_view = {
        "exterior_image_1_left": base_im,
        "exterior_image_2_left": base_im_second,
        "wrist_image_left": wrist_im,
    }
    views = {}
    for view_name, pose6_for_inference in camera_pose6_by_view.items():
        image = image_by_view.get(view_name)
        if image is None:
            continue
        state7 = np.concatenate(
            [pose6_for_inference, np.atleast_1d(np.array(gripper_state, dtype=np.float32))],
            axis=0,
        ).astype(np.float32)
        views[view_name] = {
            "image": image,
            "robot_state": state7,
            "cartesian_position": state7,
            "camera_extrinsics": np.asarray(camera_extrinsics_by_view[view_name], dtype=np.float32).reshape(-1),
            "camera_frame_convention": camera_frame_convention,
        }
    return {
        "views": views,
        "gripper_state": np.atleast_1d(np.array(gripper_state, dtype=np.float32)),
        "camera_frame_convention": camera_frame_convention,
        "completed_stages": list(completed_stages or []),
    }


def build_current_multiview_context(env, task_cfg, obs, camera_frame_convention="usd", gripper_model_units="normalized"):
    base_im, _, base_im_second, _, wrist_im, _, gripper_state = extract_from_obs(obs, robot_name=env.robot.name)
    raw_gripper_state = get_raw_gripper_position_from_obs(obs, robot_name=env.robot.name)
    gripper_state_normalized = float(np.clip(float(gripper_state), 0.0, 1.0))
    gripper_state_raw_width = None
    if camera_frame_convention in {"opencv", "usd_to_opencv", "droid"} and raw_gripper_state is not None:
        gripper_state_raw_width = float(np.clip(raw_gripper_state, 0.0, 0.05))
        gripper_state_normalized = normalize_raw_gripper_position(raw_gripper_state)
        gripper_state = gripper_state_for_model(raw_gripper_state, gripper_model_units)
    ee_pos, ee_rot = env.get_ee_pose()
    ee_pos_np = ee_pos.cpu().numpy() if hasattr(ee_pos, "cpu") else np.asarray(ee_pos)
    ee_rot_np = ee_rot.cpu().numpy() if hasattr(ee_rot, "cpu") else np.asarray(ee_rot)
    ee_euler_np = Rot.from_quat(ee_rot_np).as_euler("xyz")
    ee_pose_world = np.concatenate([ee_pos_np, ee_euler_np]).astype(np.float32)
    robot_pose6 = env._world2robot(ee_pose_world).astype(np.float32)

    view_world_poses = get_view_world_poses(env, task_cfg)
    camera_extrinsics_by_view = {}
    camera_pose6_by_view = {}
    for view_name, (cam_pos_world, cam_quat_world) in view_world_poses.items():
        camera_world_pose6 = get_camera_extrinsics_pose6(
            cam_pos_world,
            cam_quat_world,
            camera_frame_convention=camera_frame_convention,
        )
        camera_extrinsics_by_view[view_name] = env._world2robot(camera_world_pose6).astype(np.float32)
        camera_pose6_by_view[view_name] = pose_in_parent_frame(
            robot_pose6.astype(np.float32),
            camera_extrinsics_by_view[view_name],
        )

    payload = build_multiview_obs_payload(
        base_im=base_im,
        base_im_second=base_im_second,
        wrist_im=wrist_im,
        gripper_state=gripper_state,
        camera_pose6_by_view=camera_pose6_by_view,
        camera_extrinsics_by_view=camera_extrinsics_by_view,
        camera_frame_convention=camera_frame_convention,
    )
    return {
        "payload": payload,
        "base_im": base_im,
        "base_im_second": base_im_second,
        "wrist_im": wrist_im,
        "gripper_state": float(np.asarray(gripper_state).reshape(-1)[0]),
        "gripper_state_model": float(np.asarray(gripper_state).reshape(-1)[0]),
        "gripper_state_normalized": float(gripper_state_normalized),
        "gripper_state_raw_width": None if gripper_state_raw_width is None else float(gripper_state_raw_width),
        "robot_pose6": robot_pose6,
        "camera_pose6_by_view": camera_pose6_by_view,
        "camera_extrinsics_by_view": camera_extrinsics_by_view,
        "view_world_poses": view_world_poses,
    }


DEFAULT_SLOT_TRANSLATION_XY_CAP = 0.03
DEFAULT_SLOT_TRANSLATION_Z_CAP = 0.3
DEFAULT_SLOT_TRANSLATION_XY_STD = 0.01
DEFAULT_SLOT_TRANSLATION_Z_STD = 0.1
DEFAULT_SLOT_ROTATION_CAP = 0.15
DEFAULT_SLOT_ROTATION_STD = 0.05


def slot_action_range_config(
    scale=1.0,
    xy_cap=None,
    z_cap=None,
    xy_std=None,
    z_std=None,
    rot_cap=None,
    rot_std=None,
):
    scale = float(max(scale, 0.0))
    return {
        "translation_cap_xyz": np.array(
            [
                DEFAULT_SLOT_TRANSLATION_XY_CAP if xy_cap is None else float(xy_cap),
                DEFAULT_SLOT_TRANSLATION_XY_CAP if xy_cap is None else float(xy_cap),
                DEFAULT_SLOT_TRANSLATION_Z_CAP if z_cap is None else float(z_cap),
            ],
            dtype=np.float32,
        )
        * scale,
        "translation_std_xyz": np.array(
            [
                DEFAULT_SLOT_TRANSLATION_XY_STD if xy_std is None else float(xy_std),
                DEFAULT_SLOT_TRANSLATION_XY_STD if xy_std is None else float(xy_std),
                DEFAULT_SLOT_TRANSLATION_Z_STD if z_std is None else float(z_std),
            ],
            dtype=np.float32,
        )
        * scale,
        "rotation_cap": float(DEFAULT_SLOT_ROTATION_CAP if rot_cap is None else rot_cap) * scale,
        "rotation_std": float(DEFAULT_SLOT_ROTATION_STD if rot_std is None else rot_std) * scale,
    }


def cap_delta6_by_slot_action_range(delta6, action_range):
    delta6 = np.asarray(delta6, dtype=np.float32).reshape(6).copy()
    translation_cap_xyz = np.asarray(action_range["translation_cap_xyz"], dtype=np.float32).reshape(3)
    delta6[:3] = np.clip(delta6[:3], -translation_cap_xyz, translation_cap_xyz)
    rot_cap = float(action_range["rotation_cap"])
    rot_norm = float(np.linalg.norm(delta6[3:6]))
    if rot_norm > rot_cap > 0.0:
        delta6[3:6] *= rot_cap / max(rot_norm, 1.0e-8)
    return delta6.astype(np.float32)


def sample_delta6_from_slot_action_range(rng, action_range, use_std=False):
    translation_key = "translation_std_xyz" if use_std else "translation_cap_xyz"
    rotation_key = "rotation_std" if use_std else "rotation_cap"
    translation_bounds = np.asarray(action_range[translation_key], dtype=np.float32).reshape(3)
    delta_xyz = rng.uniform(-translation_bounds, translation_bounds).astype(np.float32)
    delta_rpy = rng.normal(size=3).astype(np.float32)
    norm = float(np.linalg.norm(delta_rpy))
    if norm < 1.0e-8:
        delta_rpy[:] = 0.0
        delta_rpy[0] = 1.0
        norm = 1.0
    delta_rpy = delta_rpy / norm * float(rng.uniform(0.0, float(action_range[rotation_key])))
    return np.concatenate([delta_xyz, delta_rpy], axis=0).astype(np.float32)


def sample_nearby_robot_pose6(rng, start_robot_pose6, translation_radius, rotation_radius):
    start_robot_pose6 = np.asarray(start_robot_pose6, dtype=np.float32).reshape(6)
    delta_xyz = rng.normal(size=3).astype(np.float32)
    delta_xyz /= max(float(np.linalg.norm(delta_xyz)), 1.0e-6)
    delta_xyz *= float(rng.uniform(0.0, translation_radius))
    delta_rpy = rng.normal(size=3).astype(np.float32)
    delta_rpy /= max(float(np.linalg.norm(delta_rpy)), 1.0e-6)
    delta_rpy *= float(rng.uniform(0.0, rotation_radius))
    target = start_robot_pose6.copy()
    target[:3] += delta_xyz
    target[3:6] = angle_wrap(target[3:6] + delta_rpy)
    return target.astype(np.float32), np.concatenate([delta_xyz, delta_rpy], axis=0).astype(np.float32)


def sample_nearby_gripper_state(rng, start_gripper_state, gripper_radius):
    start = float(np.clip(float(np.asarray(start_gripper_state).reshape(-1)[0]), 0.0, 1.0))
    radius = float(max(gripper_radius, 0.0))
    if radius >= 1.0:
        target = float(rng.uniform(0.0, 1.0))
    else:
        target = float(np.clip(start + rng.uniform(-radius, radius), 0.0, 1.0))
    return target, float(target - start)


def build_z_levels_xy_circle_warmup_waypoints(
    start_robot_pose6,
    start_gripper_state,
    xy_radius,
    z_offsets,
    points_per_z=4,
    random_gripper=False,
    gripper_min=0.0,
    gripper_max=1.0,
    fixed_gripper_state=None,
    rng=None,
):
    """Build z-level xy-circle warmup targets, then return to the start pose."""
    start_robot_pose6 = np.asarray(start_robot_pose6, dtype=np.float32).reshape(6)
    start_gripper_state = float(np.clip(float(np.asarray(start_gripper_state).reshape(-1)[0]), 0.0, 1.0))
    xy_radius = float(max(xy_radius, 0.0))
    z_offsets = [float(x) for x in np.asarray(z_offsets, dtype=np.float32).reshape(-1)]
    if len(z_offsets) < 1:
        raise ValueError("Expected at least one SLOT warmup z offset.")
    points_per_z = max(1, int(points_per_z))
    random_gripper = bool(random_gripper)
    gripper_min = float(np.clip(gripper_min, 0.0, 1.0))
    gripper_max = float(np.clip(gripper_max, gripper_min, 1.0))
    fixed_gripper_state = (
        None
        if fixed_gripper_state is None
        else float(np.clip(float(fixed_gripper_state), 0.0, 1.0))
    )
    rng = np.random.default_rng() if rng is None else rng
    waypoints = []
    xy_angles = np.linspace(0.0, 2.0 * np.pi, num=points_per_z, endpoint=False, dtype=np.float32)
    for z_offset in z_offsets:
        for phase in xy_angles:
            delta_xyz = np.array(
                [
                    xy_radius * np.cos(phase),
                    xy_radius * np.sin(phase),
                    z_offset,
                ],
                dtype=np.float32,
            )
            delta_rpy = np.zeros(3, dtype=np.float32)
            target_robot_pose6 = start_robot_pose6.copy()
            target_robot_pose6[:3] += delta_xyz
            target_robot_pose6[3:6] = angle_wrap(target_robot_pose6[3:6] + delta_rpy)
            target_gripper_state = (
                float(rng.uniform(gripper_min, gripper_max))
                if random_gripper
                else float(start_gripper_state if fixed_gripper_state is None else fixed_gripper_state)
            )
            waypoints.append(
                {
                    "target_robot_pose6": target_robot_pose6.astype(np.float32),
                    "sampled_delta_robot6": np.concatenate([delta_xyz, delta_rpy], axis=0).astype(np.float32),
                    "target_gripper_state": float(target_gripper_state),
                    "sampled_delta_gripper": float(target_gripper_state - start_gripper_state),
                    "z_level": float(z_offset),
                    "xy_angle_rad": float(phase),
                    "return_to_start": False,
                }
            )
    waypoints.append(
        {
            "target_robot_pose6": start_robot_pose6.astype(np.float32).copy(),
            "sampled_delta_robot6": np.zeros(6, dtype=np.float32),
            "target_gripper_state": float(start_gripper_state if fixed_gripper_state is None else fixed_gripper_state),
            "sampled_delta_gripper": float(
                (start_gripper_state if fixed_gripper_state is None else fixed_gripper_state) - start_gripper_state
            ),
            "z_level": 0.0,
            "xy_angle_rad": None,
            "return_to_start": True,
        }
    )
    return waypoints


def summarize_step_traces(step_traces):
    if not step_traces:
        return {}

    action_actual_cosines = []
    actual_goal_cosines = []
    robot_target_xyz_errs = []
    camera_target_xyz_errs = []
    robot_delta_cosines = []
    camera_delta_cosines = []
    final_target_pos_errs = []
    final_target_rot_errs = []
    settle_steps_used = []
    timing_extract_obs_s = []
    timing_save_step_images_s = []
    timing_client_infer_s = []
    timing_execute_target_s = []
    timing_record_step_s = []
    timing_post_obs_extract_s = []
    timing_outer_step_total_s = []
    timing_server_predict_total_s = []
    timing_server_optimize_action_s = []
    timing_server_parse_obs_s = []
    timing_server_append_buffers_s = []
    timing_server_build_context_s = []
    timing_server_cem_sampling_s = []
    timing_server_cem_eval_total_s = []
    timing_server_cem_sort_update_s = []
    timing_server_eval_cpu_pose_build_s = []
    timing_server_eval_history_materialize_s = []
    timing_server_eval_history_select_s = []
    timing_server_eval_view_prep_s = []
    timing_server_eval_predictor_forward_s = []
    timing_server_eval_action_tensor_to_gpu_s = []
    timing_server_eval_action_cache_to_cpu_s = []
    timing_server_eval_full_actions_cat_s = []
    timing_server_eval_rep_flatten_s = []
    timing_server_eval_predict_sequence_total_s = []
    timing_server_eval_predict_sequence_input_prep_s = []
    timing_server_eval_predict_sequence_forward_s = []
    timing_server_eval_predict_sequence_layer_norm_s = []
    timing_server_eval_predict_sequence_reshape_s = []
    timing_server_eval_pred_extract_s = []
    timing_server_eval_loss_total_s = []
    timing_server_eval_loss_to_cpu_s = []
    server_transport_solves_reused = []
    server_fixed_view_parallel_pairs = []
    server_fixed_view_parallel_warmups = []
    used_fast_inner_step = []
    fast_inner_steps_executed = []
    full_inner_steps_executed = []
    fast_inner_fallback_steps = []
    commanded_vs_trace_delta_cosines = []
    commanded_to_trace_target_delta_pos = []
    commanded_to_trace_target_delta_rot = []
    commanded_to_goal_target_delta_pos = []
    commanded_to_goal_target_delta_rot = []
    server_best_losses = []
    server_best_l1_losses = []
    server_losses_by_view = {}
    server_transport_losses_by_view = {}
    server_objective_losses_by_view = {}

    for x in step_traces:
        if x.get("action_actual_cosine_camera") is not None:
            action_actual_cosines.append(float(x["action_actual_cosine_camera"]))
        if x.get("actual_goal_cosine_camera") is not None:
            actual_goal_cosines.append(float(x["actual_goal_cosine_camera"]))

        if x.get("trace_post_robot_pose6") is not None and x.get("next_robot_pose6") is not None:
            tgt = np.asarray(x["trace_post_robot_pose6"][:3], dtype=np.float32)
            nxt = np.asarray(x["next_robot_pose6"][:3], dtype=np.float32)
            cur = np.asarray(x["robot_pose6"][:3], dtype=np.float32)
            robot_target_xyz_errs.append(float(np.linalg.norm(tgt - nxt)))
            targ_delta = tgt - cur
            act_delta = nxt - cur
            denom = float(np.linalg.norm(targ_delta) * np.linalg.norm(act_delta))
            if denom > 1e-8:
                robot_delta_cosines.append(float(np.dot(targ_delta, act_delta) / denom))

        if x.get("trace_post_state7_camera") is not None and x.get("next_camera_pose6") is not None:
            tgt = np.asarray(x["trace_post_state7_camera"][:3], dtype=np.float32)
            nxt = np.asarray(x["next_camera_pose6"][:3], dtype=np.float32)
            cur = np.asarray(x["camera_pose6"][:3], dtype=np.float32)
            camera_target_xyz_errs.append(float(np.linalg.norm(tgt - nxt)))
            targ_delta = tgt - cur
            act_delta = nxt - cur
            denom = float(np.linalg.norm(targ_delta) * np.linalg.norm(act_delta))
            if denom > 1e-8:
                camera_delta_cosines.append(float(np.dot(targ_delta, act_delta) / denom))

        if x.get("commanded_vs_trace_delta_cosine_robot_xyz") is not None:
            commanded_vs_trace_delta_cosines.append(float(x["commanded_vs_trace_delta_cosine_robot_xyz"]))
        if x.get("commanded_to_trace_target_distance_change") is not None:
            commanded_to_trace_target_delta_pos.append(
                float(x["commanded_to_trace_target_distance_change"]["delta_pos_l2_robot"])
            )
            commanded_to_trace_target_delta_rot.append(
                float(x["commanded_to_trace_target_distance_change"]["delta_rot_l2_robot"])
            )
        if x.get("commanded_to_goal_target_distance_change") is not None:
            commanded_to_goal_target_delta_pos.append(
                float(x["commanded_to_goal_target_distance_change"]["delta_pos_l2_robot"])
            )
            commanded_to_goal_target_delta_rot.append(
                float(x["commanded_to_goal_target_distance_change"]["delta_rot_l2_robot"])
            )
        if x.get("server_best_loss") is not None:
            server_best_losses.append(float(x["server_best_loss"]))
        if x.get("server_best_l1_loss") is not None:
            server_best_l1_losses.append(float(x["server_best_l1_loss"]))
        for view_name, value in (x.get("server_per_view_losses") or {}).items():
            server_losses_by_view.setdefault(view_name, []).append(float(value))
        for view_name, value in (x.get("server_per_view_transport_losses") or {}).items():
            server_transport_losses_by_view.setdefault(view_name, []).append(float(value))
        for view_name, value in (x.get("server_per_view_objective_losses") or {}).items():
            server_objective_losses_by_view.setdefault(view_name, []).append(float(value))

        if "final_target_pos_l2_robot" in x:
            final_target_pos_errs.append(float(x["final_target_pos_l2_robot"]))
        if "final_target_rot_l2_robot" in x:
            final_target_rot_errs.append(float(x["final_target_rot_l2_robot"]))
        if "settle_steps_used" in x:
            settle_steps_used.append(float(x["settle_steps_used"]))
        if x.get("timing_extract_obs_s") is not None:
            timing_extract_obs_s.append(float(x["timing_extract_obs_s"]))
        if x.get("timing_save_step_images_s") is not None:
            timing_save_step_images_s.append(float(x["timing_save_step_images_s"]))
        if x.get("timing_client_infer_s") is not None:
            timing_client_infer_s.append(float(x["timing_client_infer_s"]))
        if x.get("timing_execute_target_s") is not None:
            timing_execute_target_s.append(float(x["timing_execute_target_s"]))
        if x.get("timing_record_step_s") is not None:
            timing_record_step_s.append(float(x["timing_record_step_s"]))
        if x.get("timing_post_obs_extract_s") is not None:
            timing_post_obs_extract_s.append(float(x["timing_post_obs_extract_s"]))
        if x.get("timing_outer_step_total_s") is not None:
            timing_outer_step_total_s.append(float(x["timing_outer_step_total_s"]))
        server_timing = x.get("server_timing")
        if server_timing is not None:
            if server_timing.get("parse_obs_s") is not None:
                timing_server_parse_obs_s.append(float(server_timing["parse_obs_s"]))
            if server_timing.get("append_buffers_s") is not None:
                timing_server_append_buffers_s.append(float(server_timing["append_buffers_s"]))
            if server_timing.get("build_context_s") is not None:
                timing_server_build_context_s.append(float(server_timing["build_context_s"]))
            if server_timing.get("predict_total_s") is not None:
                timing_server_predict_total_s.append(float(server_timing["predict_total_s"]))
            if server_timing.get("optimize_action_s") is not None:
                timing_server_optimize_action_s.append(float(server_timing["optimize_action_s"]))
            optimize_detail = server_timing.get("optimize_detail")
            if optimize_detail is not None:
                if optimize_detail.get("cem_sampling_s") is not None:
                    timing_server_cem_sampling_s.append(float(optimize_detail["cem_sampling_s"]))
                if optimize_detail.get("cem_eval_total_s") is not None:
                    timing_server_cem_eval_total_s.append(float(optimize_detail["cem_eval_total_s"]))
                if optimize_detail.get("cem_sort_update_s") is not None:
                    timing_server_cem_sort_update_s.append(float(optimize_detail["cem_sort_update_s"]))
                if optimize_detail.get("evaluate_cpu_pose_build_s") is not None:
                    timing_server_eval_cpu_pose_build_s.append(float(optimize_detail["evaluate_cpu_pose_build_s"]))
                if optimize_detail.get("evaluate_history_materialize_s") is not None:
                    timing_server_eval_history_materialize_s.append(float(optimize_detail["evaluate_history_materialize_s"]))
                if optimize_detail.get("evaluate_history_select_s") is not None:
                    timing_server_eval_history_select_s.append(float(optimize_detail["evaluate_history_select_s"]))
                if optimize_detail.get("evaluate_view_prep_s") is not None:
                    timing_server_eval_view_prep_s.append(float(optimize_detail["evaluate_view_prep_s"]))
                if optimize_detail.get("evaluate_predictor_forward_s") is not None:
                    timing_server_eval_predictor_forward_s.append(float(optimize_detail["evaluate_predictor_forward_s"]))
                if optimize_detail.get("evaluate_action_tensor_to_gpu_s") is not None:
                    timing_server_eval_action_tensor_to_gpu_s.append(float(optimize_detail["evaluate_action_tensor_to_gpu_s"]))
                if optimize_detail.get("evaluate_action_cache_to_cpu_s") is not None:
                    timing_server_eval_action_cache_to_cpu_s.append(float(optimize_detail["evaluate_action_cache_to_cpu_s"]))
                if optimize_detail.get("evaluate_full_actions_cat_s") is not None:
                    timing_server_eval_full_actions_cat_s.append(float(optimize_detail["evaluate_full_actions_cat_s"]))
                if optimize_detail.get("evaluate_rep_flatten_s") is not None:
                    timing_server_eval_rep_flatten_s.append(float(optimize_detail["evaluate_rep_flatten_s"]))
                if optimize_detail.get("evaluate_predict_sequence_total_s") is not None:
                    timing_server_eval_predict_sequence_total_s.append(float(optimize_detail["evaluate_predict_sequence_total_s"]))
                if optimize_detail.get("evaluate_predict_sequence_input_prep_s") is not None:
                    timing_server_eval_predict_sequence_input_prep_s.append(float(optimize_detail["evaluate_predict_sequence_input_prep_s"]))
                if optimize_detail.get("evaluate_predict_sequence_forward_s") is not None:
                    timing_server_eval_predict_sequence_forward_s.append(float(optimize_detail["evaluate_predict_sequence_forward_s"]))
                if optimize_detail.get("evaluate_predict_sequence_layer_norm_s") is not None:
                    timing_server_eval_predict_sequence_layer_norm_s.append(float(optimize_detail["evaluate_predict_sequence_layer_norm_s"]))
                if optimize_detail.get("evaluate_predict_sequence_reshape_s") is not None:
                    timing_server_eval_predict_sequence_reshape_s.append(float(optimize_detail["evaluate_predict_sequence_reshape_s"]))
                if optimize_detail.get("evaluate_pred_extract_s") is not None:
                    timing_server_eval_pred_extract_s.append(float(optimize_detail["evaluate_pred_extract_s"]))
                if optimize_detail.get("evaluate_loss_total_s") is not None:
                    timing_server_eval_loss_total_s.append(float(optimize_detail["evaluate_loss_total_s"]))
                if optimize_detail.get("evaluate_loss_to_cpu_s") is not None:
                    timing_server_eval_loss_to_cpu_s.append(float(optimize_detail["evaluate_loss_to_cpu_s"]))
                if optimize_detail.get("transport_solves_reused") is not None:
                    server_transport_solves_reused.append(float(optimize_detail["transport_solves_reused"]))
                if optimize_detail.get("fixed_view_parallel_pairs") is not None:
                    server_fixed_view_parallel_pairs.append(float(optimize_detail["fixed_view_parallel_pairs"]))
                if optimize_detail.get("fixed_view_parallel_warmups") is not None:
                    server_fixed_view_parallel_warmups.append(float(optimize_detail["fixed_view_parallel_warmups"]))
        if x.get("used_fast_inner_step") is not None:
            used_fast_inner_step.append(float(bool(x["used_fast_inner_step"])))
        if x.get("fast_inner_steps_executed") is not None:
            fast_inner_steps_executed.append(float(x["fast_inner_steps_executed"]))
        if x.get("full_inner_steps_executed") is not None:
            full_inner_steps_executed.append(float(x["full_inner_steps_executed"]))
        if x.get("fast_inner_fallback_steps") is not None:
            fast_inner_fallback_steps.append(float(x["fast_inner_fallback_steps"]))

    def _stats(values):
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float32)
        return {
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    if not server_objective_losses_by_view and server_losses_by_view:
        server_objective_losses_by_view = {
            view_name: list(values) for view_name, values in server_losses_by_view.items()
        }

    return {
        "num_steps": len(step_traces),
        "action_actual_cosine_camera": _stats(action_actual_cosines),
        "actual_goal_cosine_camera": _stats(actual_goal_cosines),
        "robot_target_xyz_error": _stats(robot_target_xyz_errs),
        "camera_target_xyz_error": _stats(camera_target_xyz_errs),
        "robot_delta_cosine": _stats(robot_delta_cosines),
        "camera_delta_cosine": _stats(camera_delta_cosines),
        "commanded_vs_trace_delta_cosine_robot_xyz": _stats(commanded_vs_trace_delta_cosines),
        "commanded_to_trace_target_delta_pos_l2_robot": _stats(commanded_to_trace_target_delta_pos),
        "commanded_to_trace_target_delta_rot_l2_robot": _stats(commanded_to_trace_target_delta_rot),
        "commanded_to_goal_target_delta_pos_l2_robot": _stats(commanded_to_goal_target_delta_pos),
        "commanded_to_goal_target_delta_rot_l2_robot": _stats(commanded_to_goal_target_delta_rot),
        "server_best_loss": _stats(server_best_losses),
        "server_best_l1_loss": _stats(server_best_l1_losses),
        "server_per_view_losses": {
            view_name: _stats(values)
            for view_name, values in server_losses_by_view.items()
        },
        "server_per_view_transport_losses": {
            view_name: _stats(values)
            for view_name, values in server_transport_losses_by_view.items()
        },
        "server_per_view_objective_losses": {
            view_name: _stats(values)
            for view_name, values in server_objective_losses_by_view.items()
        },
        # Legacy aliases for downstream consumers that expect generic loss names.
        "mean_loss": _stats(server_best_l1_losses if server_best_l1_losses else server_best_losses),
        "per_view_losses": {
            view_name: _stats(values)
            for view_name, values in (server_losses_by_view or server_objective_losses_by_view).items()
        },
        "view_names": sorted(
            set(server_losses_by_view.keys())
            | set(server_transport_losses_by_view.keys())
            | set(server_objective_losses_by_view.keys())
        ),
        "final_target_pos_l2_robot": _stats(final_target_pos_errs),
        "final_target_rot_l2_robot": _stats(final_target_rot_errs),
        "settle_steps_used": _stats(settle_steps_used),
        "timing_extract_obs_s": _stats(timing_extract_obs_s),
        "timing_save_step_images_s": _stats(timing_save_step_images_s),
        "timing_client_infer_s": _stats(timing_client_infer_s),
        "timing_execute_target_s": _stats(timing_execute_target_s),
        "timing_record_step_s": _stats(timing_record_step_s),
        "timing_post_obs_extract_s": _stats(timing_post_obs_extract_s),
        "timing_outer_step_total_s": _stats(timing_outer_step_total_s),
        "timing_server_parse_obs_s": _stats(timing_server_parse_obs_s),
        "timing_server_append_buffers_s": _stats(timing_server_append_buffers_s),
        "timing_server_build_context_s": _stats(timing_server_build_context_s),
        "timing_server_predict_total_s": _stats(timing_server_predict_total_s),
        "timing_server_optimize_action_s": _stats(timing_server_optimize_action_s),
        "timing_server_cem_sampling_s": _stats(timing_server_cem_sampling_s),
        "timing_server_cem_eval_total_s": _stats(timing_server_cem_eval_total_s),
        "timing_server_cem_sort_update_s": _stats(timing_server_cem_sort_update_s),
        "timing_server_eval_cpu_pose_build_s": _stats(timing_server_eval_cpu_pose_build_s),
        "timing_server_eval_history_materialize_s": _stats(timing_server_eval_history_materialize_s),
        "timing_server_eval_history_select_s": _stats(timing_server_eval_history_select_s),
        "timing_server_eval_view_prep_s": _stats(timing_server_eval_view_prep_s),
        "timing_server_eval_predictor_forward_s": _stats(timing_server_eval_predictor_forward_s),
        "timing_server_eval_action_tensor_to_gpu_s": _stats(timing_server_eval_action_tensor_to_gpu_s),
        "timing_server_eval_action_cache_to_cpu_s": _stats(timing_server_eval_action_cache_to_cpu_s),
        "timing_server_eval_full_actions_cat_s": _stats(timing_server_eval_full_actions_cat_s),
        "timing_server_eval_rep_flatten_s": _stats(timing_server_eval_rep_flatten_s),
        "timing_server_eval_predict_sequence_total_s": _stats(timing_server_eval_predict_sequence_total_s),
        "timing_server_eval_predict_sequence_input_prep_s": _stats(timing_server_eval_predict_sequence_input_prep_s),
        "timing_server_eval_predict_sequence_forward_s": _stats(timing_server_eval_predict_sequence_forward_s),
        "timing_server_eval_predict_sequence_layer_norm_s": _stats(timing_server_eval_predict_sequence_layer_norm_s),
        "timing_server_eval_predict_sequence_reshape_s": _stats(timing_server_eval_predict_sequence_reshape_s),
        "timing_server_eval_pred_extract_s": _stats(timing_server_eval_pred_extract_s),
        "timing_server_eval_loss_total_s": _stats(timing_server_eval_loss_total_s),
        "timing_server_eval_loss_to_cpu_s": _stats(timing_server_eval_loss_to_cpu_s),
        "server_transport_solves_reused": _stats(server_transport_solves_reused),
        "server_fixed_view_parallel_pairs": _stats(server_fixed_view_parallel_pairs),
        "server_fixed_view_parallel_warmups": _stats(server_fixed_view_parallel_warmups),
        "used_fast_inner_step_ratio": _stats(used_fast_inner_step),
        "fast_inner_steps_executed": _stats(fast_inner_steps_executed),
        "full_inner_steps_executed": _stats(full_inner_steps_executed),
        "fast_inner_fallback_steps": _stats(fast_inner_fallback_steps),
    }


def plot_step_view_losses(step_traces, save_path, loss_key="server_per_view_objective_losses"):
    from scripts.plot_view_losses import plot_view_losses_from_step_traces

    return plot_view_losses_from_step_traces(step_traces, save_path, loss_key=loss_key)


def _observed_loss_maps_from_record_response(record_response):
    if not isinstance(record_response, dict):
        return {}, {}, {}
    return (
        record_response.get("observed_per_view_losses") or record_response.get("per_view_losses") or {},
        record_response.get("observed_per_view_transport_losses")
        or record_response.get("per_view_transport_losses")
        or {},
        record_response.get("observed_per_view_objective_losses")
        or record_response.get("per_view_objective_losses")
        or {},
    )


def resolve_success_case_dir(
    task: str,
    perturbation: str,
    state_frame: str,
    scene_profile: str,
    success_case_dir: str | None,
    task_variant: str = DEFAULT_TASK_VARIANT,
) -> str:
    if success_case_dir:
        return success_case_dir
    dir_name = make_case_dir_name(task, perturbation, state_frame, scene_profile, task_variant)
    return os.path.join(DEFAULT_SUCCESS_CASE_ROOT, dir_name)


def run_controller_goal_delta_diagnostic(
    task_id=0,
    perturbation_id=0,
    state_frame="camera",
    success_case_dir=DEFAULT_SUCCESS_CASE_DIR,
    goal_mode="first",
    scene_profile="realm_droid10",
    task_variant=DEFAULT_TASK_VARIANT,
    max_settle_steps=DEFAULT_VJEPA_MAX_SETTLE_STEPS,
    settle_pos_tol=DEFAULT_VJEPA_SETTLE_POS_TOL,
    settle_rot_tol=DEFAULT_VJEPA_SETTLE_ROT_TOL,
    enable_settle_debug=DEFAULT_ENABLE_SETTLE_DEBUG,
    fixed_camera_presets=None,
):
    _ensure_omnigibson_runtime_initialized()
    task = SUPPORTED_TASKS[task_id]
    perturbation = SUPPORTED_PERTURBATIONS[perturbation_id]
    task_debug_dir = get_planning_output_dir(
        model_type="controller_goal_delta",
        task=task,
        perturbation=perturbation,
        state_frame=state_frame,
        scene_profile=scene_profile,
        goal_mode=goal_mode,
        task_variant=task_variant,
    )
    os.makedirs(task_debug_dir, exist_ok=True)

    auto_goal_dir = resolve_success_case_dir(task, perturbation, state_frame, scene_profile, success_case_dir, task_variant)
    manifest, goal_targets, goal_images, goal_state = load_success_case_data(auto_goal_dir, goal_mode=goal_mode)
    validate_success_case_manifest(
        manifest=manifest,
        goal_targets=goal_targets,
        goal_mode=goal_mode,
        task=task,
        perturbation=perturbation,
        state_frame=state_frame,
        scene_profile=scene_profile,
        task_variant=task_variant,
        success_case_dir=auto_goal_dir,
        fixed_camera_presets=fixed_camera_presets,
    )
    if not goal_targets:
        raise ValueError("No goal targets available for controller diagnostic.")
    target = goal_targets[0]

    task_cfg_path = get_task_cfg_path(task, scene_profile, task_variant)
    task_cfg = load_task_cfg("/app/realm/config", task_cfg_path)
    env = RealmEnvironmentDynamic(
        config_path="/app/realm/config",
        task_cfg_path=task_cfg_path,
        perturbations=[perturbation],
        multi_view=True,
        no_rendering=False,
        rendering_mode="rt",
        robot="DROID_ee_control",
        camera_extrinsics_override=fixed_camera_presets,
    )

    obs, _ = env.reset()
    obs, rew, terminated, truncated, info = env.warmup(obs)

    base_im, _, base_im_second, _, wrist_im, _robot_state, gripper_state = extract_from_obs(obs, robot_name=env.robot.name)
    Image.fromarray(base_im).save(os.path.join(task_debug_dir, "start_base.png"))
    Image.fromarray(base_im_second).save(os.path.join(task_debug_dir, "start_second.png"))
    Image.fromarray(wrist_im).save(os.path.join(task_debug_dir, "start_wrist.png"))

    ee_pos, ee_rot = env.get_ee_pose()
    ee_pos_np = ee_pos.cpu().numpy() if hasattr(ee_pos, "cpu") else np.array(ee_pos)
    ee_rot_np = ee_rot.cpu().numpy() if hasattr(ee_rot, "cpu") else np.array(ee_rot)
    ee_euler_np = Rot.from_quat(ee_rot_np).as_euler("xyz")
    ee_pose_world6 = np.concatenate([ee_pos_np, ee_euler_np]).astype(np.float32)
    current_robot_pose6 = env._world2robot(ee_pose_world6).astype(np.float32)

    view_world_poses = get_view_world_poses(env, task_cfg)
    ref_cam_pos_world, ref_cam_quat_world = view_world_poses["exterior_image_1_left"]
    camera_world_pose6 = get_camera_extrinsics_pose6(ref_cam_pos_world, ref_cam_quat_world)
    current_camera_extrinsics6 = env._world2robot(camera_world_pose6).astype(np.float32)
    current_camera_pose6 = pose_in_parent_frame(current_robot_pose6, current_camera_extrinsics6)

    goal_state7_camera = np.asarray(target["ee_state7_camera"], dtype=np.float32).reshape(-1)
    goal_state7_robot = np.asarray(target["ee_state7_robot"], dtype=np.float32).reshape(-1)
    current_state7_camera = np.concatenate(
        [current_camera_pose6, np.atleast_1d(np.array(gripper_state, dtype=np.float32))],
        axis=0,
    ).astype(np.float32)

    delta_pose6_camera = pose_delta6(current_camera_pose6, goal_state7_camera[:6])
    delta_gripper = float(goal_state7_camera[6] - current_state7_camera[6])
    action_camera = np.concatenate(
        [delta_pose6_camera, np.array([delta_gripper], dtype=np.float32)],
        axis=0,
    ).astype(np.float32)
    target_ee_world6 = apply_camera_delta_to_world_pose6(
        current_ee_world6=ee_pose_world6,
        delta_action_camera=action_camera,
        cam_quat_world=ref_cam_quat_world,
    )
    commanded_target_robot_pose6 = env._world2robot(target_ee_world6).astype(np.float32)
    execution_result = execute_absolute_robot_target(
        env=env,
        target_robot_pose6=commanded_target_robot_pose6,
        gripper_command=float(action_camera[6]),
        max_settle_steps=max_settle_steps,
        settle_pos_tol=settle_pos_tol,
        settle_rot_tol=settle_rot_tol,
        vis_dir=os.path.join(task_debug_dir, "settle_vis") if enable_settle_debug else None,
        include_controller_debug=enable_settle_debug,
    )
    obs = execution_result["obs"]
    env_action = execution_result["env_action"]

    conversion_alignment = {
        "commanded_vs_goal_target_error6_robot": pose_error_components(goal_state7_robot[:6], commanded_target_robot_pose6)[0],
        "commanded_vs_goal_target_pos_l2_robot": pose_error_components(goal_state7_robot[:6], commanded_target_robot_pose6)[1],
        "commanded_vs_goal_target_rot_l2_robot": pose_error_components(goal_state7_robot[:6], commanded_target_robot_pose6)[2],
    }
    settle_history = execution_result["settle_history"]
    curr_task_progression = execution_result["task_progression"]
    terminated = execution_result["terminated"]
    truncated = execution_result["truncated"]
    info = execution_result["info"]
    for record in settle_history:
        step_robot_pose6 = np.asarray(record["robot_pose6"], dtype=np.float32)
        err6_goal, pos_l2_goal, rot_l2_goal = pose_error_components(goal_state7_robot[:6], step_robot_pose6)
        record["goal_error6_robot"] = err6_goal.copy()
        record["goal_pos_l2_robot"] = float(pos_l2_goal)
        record["goal_rot_l2_robot"] = float(rot_l2_goal)

    end_base_im, _, end_base_im_second, _, end_wrist_im, _, end_gripper_state = extract_from_obs(obs, robot_name=env.robot.name)
    Image.fromarray(end_base_im).save(os.path.join(task_debug_dir, "end_base.png"))
    Image.fromarray(end_base_im_second).save(os.path.join(task_debug_dir, "end_second.png"))
    Image.fromarray(end_wrist_im).save(os.path.join(task_debug_dir, "end_wrist.png"))

    end_ee_pos, end_ee_rot = env.get_ee_pose()
    end_ee_pos_np = end_ee_pos.cpu().numpy() if hasattr(end_ee_pos, "cpu") else np.array(end_ee_pos)
    end_ee_rot_np = end_ee_rot.cpu().numpy() if hasattr(end_ee_rot, "cpu") else np.array(end_ee_rot)
    end_ee_euler_np = Rot.from_quat(end_ee_rot_np).as_euler("xyz")
    end_ee_pose_world6 = np.concatenate([end_ee_pos_np, end_ee_euler_np]).astype(np.float32)
    end_robot_pose6 = env._world2robot(end_ee_pose_world6).astype(np.float32)
    end_camera_pose6 = world_pose_to_camera_pose6(end_ee_pose_world6, ref_cam_pos_world, ref_cam_quat_world)

    result = {
        "mode": "controller_goal_delta_diagnostic",
        "task": task,
        "perturbation": perturbation,
        "scene_profile": scene_profile,
        "success_case_dir": auto_goal_dir,
        "goal_name": target.get("name"),
        "goal_stage": target.get("stage"),
        "initial_robot_pose6": current_robot_pose6.copy(),
        "initial_camera_pose6": current_camera_pose6.copy(),
        "goal_state7_robot": goal_state7_robot.copy(),
        "goal_state7_camera": goal_state7_camera.copy(),
        "action_camera": action_camera.copy(),
        "commanded_target_robot_pose6": commanded_target_robot_pose6.copy(),
        "env_action": env_action.copy(),
        "conversion_alignment": conversion_alignment,
        "final_robot_pose6": end_robot_pose6.copy(),
        "final_camera_pose6": end_camera_pose6.copy(),
        "final_target_error6_robot": pose_error_components(commanded_target_robot_pose6, end_robot_pose6)[0],
        "final_target_pos_l2_robot": pose_error_components(commanded_target_robot_pose6, end_robot_pose6)[1],
        "final_target_rot_l2_robot": pose_error_components(commanded_target_robot_pose6, end_robot_pose6)[2],
        "final_goal_error6_robot": pose_error_components(goal_state7_robot[:6], end_robot_pose6)[0],
        "final_goal_pos_l2_robot": pose_error_components(goal_state7_robot[:6], end_robot_pose6)[1],
        "final_goal_rot_l2_robot": pose_error_components(goal_state7_robot[:6], end_robot_pose6)[2],
        "task_progression": float(curr_task_progression),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "settle_pos_tol": float(settle_pos_tol),
        "settle_rot_tol": float(settle_rot_tol),
        "settle_steps_used": len(settle_history),
        "settle_history": settle_history,
        "settle_vis_dir": os.path.join(task_debug_dir, "settle_vis") if enable_settle_debug else None,
        "enable_settle_debug": bool(enable_settle_debug),
        "gripper_state_start": float(gripper_state),
        "gripper_state_end": float(end_gripper_state),
    }

    if enable_settle_debug:
        create_settle_visual_summary(
            task_debug_dir=task_debug_dir,
            settle_vis_dir=os.path.join(task_debug_dir, "settle_vis"),
            settle_history=settle_history,
        )

    with open(os.path.join(task_debug_dir, "controller_goal_delta_result.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(result), f, ensure_ascii=False, indent=2)
    torch.save(result, os.path.join(task_debug_dir, "controller_goal_delta_result.pt"))
    print(f"Controller-only diagnostic saved to {task_debug_dir}", flush=True)


def validate_success_case_manifest(
    manifest: dict,
    goal_targets,
    goal_mode: str,
    task: str,
    perturbation: str,
    state_frame: str,
    scene_profile: str,
    success_case_dir: str,
    task_variant: str = DEFAULT_TASK_VARIANT,
    fixed_camera_presets=None,
):
    manifest_task = manifest.get("task")
    manifest_perturbation = manifest.get("perturbation")
    manifest_frame = manifest.get("state_frame")
    manifest_num_targets = manifest.get("num_targets")
    manifest_scene_profile = manifest.get("scene_profile")
    manifest_task_variant = manifest.get("task_variant", DEFAULT_TASK_VARIANT)

    if manifest_task is not None and manifest_task != task:
        raise ValueError(
            f"Success-case task mismatch: expected `{task}`, got `{manifest_task}` from {success_case_dir}"
        )
    if manifest_perturbation is not None and manifest_perturbation != perturbation:
        raise ValueError(
            f"Success-case perturbation mismatch: expected `{perturbation}`, got `{manifest_perturbation}` from {success_case_dir}"
        )
    if manifest_frame is not None and manifest_frame != state_frame:
        raise ValueError(
            f"Success-case state_frame mismatch: expected `{state_frame}`, got `{manifest_frame}` from {success_case_dir}"
        )
    manifest_target_names = list(manifest.get("target_names", []))
    selected_target_names = [target.get("name") for target in goal_targets]
    if goal_mode == "all":
        if manifest_num_targets is not None and int(manifest_num_targets) != len(goal_targets):
            raise ValueError(
                f"Success-case num_targets mismatch: manifest={manifest_num_targets}, loaded={len(goal_targets)}"
            )
    else:
        for name in selected_target_names:
            if manifest_target_names and name not in manifest_target_names:
                raise ValueError(
                    f"Selected goal target `{name}` is not listed in manifest target_names={manifest_target_names}"
                )
    if manifest_scene_profile is not None and manifest_scene_profile != scene_profile:
        raise ValueError(
            f"Success-case scene_profile mismatch: expected `{scene_profile}`, got `{manifest_scene_profile}` from {success_case_dir}"
        )
    if manifest_task_variant is not None and manifest_task_variant != task_variant:
        raise ValueError(
            f"Success-case task_variant mismatch: expected `{task_variant}`, got `{manifest_task_variant}` from {success_case_dir}"
        )
    if fixed_camera_presets:
        manifest_camera_presets = manifest.get("fixed_camera_presets") or []
        if list(manifest_camera_presets) != list(fixed_camera_presets):
            raise ValueError(
                "Success-case fixed_camera_presets mismatch: "
                f"expected `{list(fixed_camera_presets)}`, got `{manifest_camera_presets}` from {success_case_dir}"
            )


def run_vjepa_planning(
    port=8000,
    max_steps=20,
    horizon=8,
    task_id=0,
    perturbation_id=0,
    state_frame="camera",
    success_case_dir=DEFAULT_SUCCESS_CASE_DIR,
    goal_mode="first",
    scene_profile="realm_droid10",
    task_variant=DEFAULT_TASK_VARIANT,
    action_repeat=DEFAULT_VJEPA_ACTION_REPEAT,
    replay_mode="robot_absolute",
    max_settle_steps=DEFAULT_VJEPA_MAX_SETTLE_STEPS,
    settle_pos_tol=DEFAULT_VJEPA_SETTLE_POS_TOL,
    settle_rot_tol=DEFAULT_VJEPA_SETTLE_ROT_TOL,
    enable_settle_debug=DEFAULT_ENABLE_SETTLE_DEBUG,
    model_type="vjepa",
    slot_model_gripper_units="normalized",
    env=None,
    task_cfg=None,
    task_cfg_path=None,
    cancel_event=None,
    output_run_suffix=None,
    output_namespace=None,
    fixed_camera_presets=None,
    rotate_deterministic_after_grasp=True,
    rotate_deterministic_step_rad=DEFAULT_ROTATE_DETERMINISTIC_STEP_RAD,
    rotate_deterministic_axis=DEFAULT_ROTATE_DETERMINISTIC_AXIS,
    rotate_deterministic_overshoot_rad=DEFAULT_ROTATE_DETERMINISTIC_OVERSHOOT_RAD,
):
    _ensure_omnigibson_runtime_initialized()
    def cancel_requested():
        return cancel_event is not None and cancel_event.is_set()

    if replay_mode != "robot_absolute":
        print(
            f"Requested replay_mode={replay_mode}, but online VJEPA planning now executes "
            "server-provided absolute robot targets. Forcing robot_absolute semantics.",
            flush=True,
        )
        replay_mode = "robot_absolute"
    
    client = InferenceClient(model_type, host="127.0.0.1", port=port)
    
    task = SUPPORTED_TASKS[task_id]
    task_is_reach = str(task).startswith("reach_")
    perturbation = SUPPORTED_PERTURBATIONS[perturbation_id]
    rotate_axis_to_index = {"roll": 3, "pitch": 4, "yaw": 5, "x": 3, "y": 4, "z": 5}
    rotate_axis_name = str(rotate_deterministic_axis or DEFAULT_ROTATE_DETERMINISTIC_AXIS).lower()
    if rotate_axis_name not in rotate_axis_to_index:
        raise ValueError(
            f"Unsupported rotate_deterministic_axis={rotate_deterministic_axis}; "
            f"expected one of {sorted(rotate_axis_to_index)}"
        )
    rotate_axis_index = rotate_axis_to_index[rotate_axis_name]
    rotate_step_rad = abs(float(rotate_deterministic_step_rad))
    rotate_overshoot_rad = max(0.0, float(rotate_deterministic_overshoot_rad))
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_timestamp = (
        run_timestamp
        if output_run_suffix is None or str(output_run_suffix).strip() == ""
        else f"{run_timestamp}_{str(output_run_suffix).strip()}"
    )
    task_debug_dir = get_planning_output_dir(
        model_type=model_type,
        task=task,
        perturbation=perturbation,
        state_frame=state_frame,
        scene_profile=scene_profile,
        goal_mode=goal_mode,
        task_variant=task_variant,
        timestamp=run_dir_timestamp,
        output_namespace=output_namespace,
    )
    os.makedirs(task_debug_dir, exist_ok=True)
    
    print(f"==================================================", flush=True)
    print(
        f"Starting {model_type.upper()} Planning for Task: {task} | Perturbation: {perturbation} "
        f"| Variant: {task_variant} | fixed_camera_presets={list(fixed_camera_presets or [])}",
        flush=True,
    )
    print(f"Output directory: {task_debug_dir}", flush=True)
    print(f"==================================================", flush=True)
    
    # 1. Load the goal from successful_case
    auto_goal_dir = resolve_success_case_dir(
        task,
        perturbation,
        state_frame,
        scene_profile,
        success_case_dir,
        task_variant,
    )
    success_rollout_trace = None
    try:
        manifest, goal_targets, goal_images, goal_state = load_success_case_data(
            auto_goal_dir,
            goal_mode=goal_mode,
        )
        try:
            success_rollout_trace = load_success_rollout_trace(auto_goal_dir)
        except FileNotFoundError:
            success_rollout_trace = None
        validate_success_case_manifest(
            manifest=manifest,
            goal_targets=goal_targets,
            goal_mode=goal_mode,
            task=task,
            perturbation=perturbation,
            state_frame=state_frame,
            scene_profile=scene_profile,
            success_case_dir=auto_goal_dir,
            task_variant=task_variant,
            fixed_camera_presets=fixed_camera_presets,
        )
    except Exception as e:
        print(f"Error: Could not load successful case data at {auto_goal_dir}: {e}", flush=True)
        return
        
    try:
        print(
            f"Sending goal image to {model_type.upper()} server at port {port} from {auto_goal_dir}..."
            f" scene_profile={scene_profile} goal_mode={goal_mode} targets={goal_state['target_names']}"
            f" views={goal_state['available_views']}",
            flush=True,
        )
        client.set_goal(goal_images, goal_state=goal_state)
        print(f"Successfully set {model_type.upper()} goal!", flush=True)
    except Exception as e:
        print(f"Error setting goal: {e}", flush=True)
        return

    # 2. Instantiate or reuse env
    if task_cfg_path is None:
        task_cfg_path = get_task_cfg_path(task, scene_profile, task_variant)
    if task_cfg is None:
        task_cfg = load_task_cfg("/app/realm/config", task_cfg_path)
    if env is None:
        env = RealmEnvironmentDynamic(
            config_path="/app/realm/config",
            task_cfg_path=task_cfg_path,
            perturbations=[perturbation],
            multi_view=True,
            no_rendering=False,
            rendering_mode="rt",
            robot="DROID_ee_control",
            camera_extrinsics_override=fixed_camera_presets,
        )
    
    obs, reset_info = env.reset()
    obs, rew, terminated, truncated, info = env.warmup(obs)
    task_is_reach = bool(task_is_reach or getattr(env, "task_type", None) == "reach")
    try:
        robot_visual_diagnostics_after_warmup = (
            env.collect_robot_visual_diagnostics(reason="after_warmup_before_extra_repair")
            if hasattr(env, "collect_robot_visual_diagnostics")
            else None
        )
        extra_visual_repair = (
            env.force_robot_gripper_visuals_visible(reason="after_warmup_extra_repair")
            if hasattr(env, "force_robot_gripper_visuals_visible")
            else None
        )
        robot_visual_diagnostics_after_extra_repair = (
            env.collect_robot_visual_diagnostics(reason="after_warmup_after_extra_repair")
            if hasattr(env, "collect_robot_visual_diagnostics")
            else None
        )
        gripper_visual_debug = {
            "reset_info_gripper_visual_repair": (reset_info or {}).get("gripper_visual_repair"),
            "env_last_gripper_visual_repair": getattr(env, "last_gripper_visual_repair", None),
            "robot_visual_diagnostics_after_warmup": robot_visual_diagnostics_after_warmup,
            "extra_visual_repair": extra_visual_repair,
            "robot_visual_diagnostics_after_extra_repair": robot_visual_diagnostics_after_extra_repair,
        }
        with open(os.path.join(task_debug_dir, "gripper_visual_debug.json"), "w", encoding="utf-8") as f:
            json.dump(to_jsonable(gripper_visual_debug), f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"Warning: failed to save gripper visual debug: {exc}", flush=True)
    if cancel_requested():
        print("Planning cancelled after env reset/warmup.", flush=True)
        return {
            "success": False,
            "status": "cancelled",
            "task_progression": 0.0,
            "strict_task_success": False,
            "output_dir": task_debug_dir,
            "num_steps": 0,
            "run_timestamp": run_timestamp,
            "run_dir_timestamp": run_dir_timestamp,
        }
    dump_robot_camera_observations(
        obs,
        robot_name=env.robot.name,
        output_dir=os.path.join(task_debug_dir, "robot_camera_debug"),
        prefix="start",
    )
    view_world_poses = get_view_world_poses(env, task_cfg)
    robot_yaw_world = float(env.robot_rot_rad[2])
    manifest_camera_frame_convention = goal_state.get("camera_frame_convention")
    phase_based_model_types = {"slot", "vjepa-droid", "vjepa_droid", "vjepa-query-predictor", "vjepa_query_predictor"}
    vjepa_phase_model_types = {"vjepa-droid", "vjepa_droid", "vjepa-query-predictor", "vjepa_query_predictor"}
    phase_based_model = model_type in phase_based_model_types
    vjepa_phase_model = model_type in vjepa_phase_model_types
    camera_frame_convention = (
        str(manifest_camera_frame_convention)
        if manifest_camera_frame_convention
        else ("opencv" if model_type == "slot" else "usd")
    )

    t = 0
    final_task_progression = 0.0
    strict_task_success = False
    cancelled = False
    
    # Save input goal images
    for i, img_map in enumerate(goal_images):
        target_name = goal_state["target_names"][i] if i < len(goal_state["target_names"]) else f"goal_{i}"
        for view_name, img in img_map.items():
            suffix = VJEPA_VIEW_FILE_SUFFIX[view_name]
            Image.fromarray(img).save(os.path.join(task_debug_dir, f"input_{target_name}_{suffix}.png"))
    with open(os.path.join(task_debug_dir, "goal_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(manifest), f, ensure_ascii=False, indent=2)
    
    # Create subdirectories for different image modalities
    os.makedirs(os.path.join(task_debug_dir, "base_cam"), exist_ok=True)
    os.makedirs(os.path.join(task_debug_dir, "second_cam"), exist_ok=True)
    os.makedirs(os.path.join(task_debug_dir, "wrist_cam"), exist_ok=True)
    if enable_settle_debug:
        os.makedirs(os.path.join(task_debug_dir, "settle_steps"), exist_ok=True)
    image_writer = AsyncImageWriter(max_workers=2, max_pending=48)
    
    all_poses = []
    step_traces = []
    reached_stages = set()
    validated_stages = set()
    active_goal_idx = 0
    post_grasp_tightened = False
    grasp_hold_gripper_command = None
    slot_gripper_hold_command = None
    slot_release_gripper_until_next_search = False
    slot_next_planning_phase = "pose"
    slot_cached_pose_obs_for_release = None
    slot_gripper_release_restore_pose6 = None
    slot_mask_vis_saved = False
    slot_goal_mask_vis_saved = False
    slot_warmup_payload_sequence = None
    slot_temporal_initial_payload = None
    slot_warmup_records = []
    slot_gripper_calibration_rounds_done = 0
    rotate_deterministic_hold_robot_pose6 = None
    rotate_deterministic_hold_goal_idx = None
    rotate_deterministic_direction_sign = None
    rotate_deterministic_accumulated_abs_rad = 0.0

    slot_warmup_config = None
    warmup_model_active = False
    warmup_log_prefix = "[WARMUP]"

    def flush_trace_artifacts(cancelled_flag=False):
        """Persist traces incrementally so diagnostics survive timeouts / interrupts."""
        try:
            torch.save(all_poses, os.path.join(task_debug_dir, "all_poses.pt"))
            torch.save(step_traces, os.path.join(task_debug_dir, "step_traces.pt"))
            step_trace_summary = summarize_step_traces(step_traces)
            step_trace_summary["run_timestamp"] = run_timestamp
            step_trace_summary["run_dir_timestamp"] = run_dir_timestamp
            step_trace_summary["output_dir"] = task_debug_dir
            step_trace_summary["task_variant"] = task_variant
            step_trace_summary["task_cfg_path"] = task_cfg_path
            step_trace_summary["fixed_camera_presets"] = list(fixed_camera_presets or [])
            step_trace_summary["cancelled"] = bool(cancelled_flag or cancel_requested())
            step_trace_summary["actual_external_camera_presets"] = dict(
                getattr(env, "external_camera_pose_names", {})
            )
            with open(os.path.join(task_debug_dir, "step_trace_summary.json"), "w", encoding="utf-8") as f:
                json.dump(to_jsonable(step_trace_summary), f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"Warning: failed to flush trace artifacts: {exc}", flush=True)

    def run_post_grasp_tighten_and_lift(trace_record, current_robot_pose6, reason):
        nonlocal obs, curr_task_progression, final_task_progression, cancelled
        nonlocal post_grasp_tightened, grasp_hold_gripper_command

        t_tighten_start = time.perf_counter()
        pre_post_grasp_robot_pose6 = np.asarray(current_robot_pose6, dtype=np.float32).reshape(6).copy()
        post_grasp_rollback_sim_state = None
        post_grasp_rollback_python_state = None
        try:
            post_grasp_rollback_sim_state = save_sim_state(og)
            post_grasp_rollback_python_state = save_python_task_state(env)
            trace_record["post_grasp_rollback_state_saved"] = True
        except Exception as exc:
            trace_record["post_grasp_rollback_state_saved"] = False
            trace_record["post_grasp_rollback_state_error"] = str(exc)
        tighten_target_robot_pose6 = pre_post_grasp_robot_pose6.copy()
        tighten_target_robot_pose6[2] += float(POST_GRASP_DOWNWARD_OFFSET_Z)
        tighten_result = execute_absolute_robot_target(
            env=env,
            target_robot_pose6=tighten_target_robot_pose6,
            gripper_command=POST_GRASP_CLOSE_COMMAND,
            max_settle_steps=POST_GRASP_CLOSE_STEPS,
            settle_pos_tol=settle_pos_tol,
            settle_rot_tol=settle_rot_tol,
            vis_dir=None,
            include_controller_debug=False,
            force_full_steps=True,
            cancel_event=cancel_event,
        )
        if cancel_requested():
            cancelled = True
            trace_record["post_grasp_tighten_cancelled"] = True
            return {"cancelled": True, "robot_pose6": pre_post_grasp_robot_pose6}

        post_grasp_tightened = True
        obs = tighten_result["obs"]
        curr_task_progression = tighten_result["task_progression"]
        final_task_progression = float(curr_task_progression)
        post_tighten_task_progression = float(curr_task_progression)
        tighten_duration_s = float(time.perf_counter() - t_tighten_start)
        post_tighten_raw_gripper = get_raw_gripper_position_from_obs(obs, robot_name=env.robot.name)
        if post_tighten_raw_gripper is None:
            post_tighten_gripper_normalized = POST_GRASP_CLOSE_COMMAND
        else:
            post_tighten_gripper_normalized = normalize_raw_gripper_position(post_tighten_raw_gripper)
        post_tighten_ee_pos, post_tighten_ee_rot = env.get_ee_pose()
        post_tighten_ee_pos_np = (
            post_tighten_ee_pos.cpu().numpy()
            if hasattr(post_tighten_ee_pos, "cpu")
            else np.asarray(post_tighten_ee_pos)
        )
        post_tighten_ee_rot_np = (
            post_tighten_ee_rot.cpu().numpy()
            if hasattr(post_tighten_ee_rot, "cpu")
            else np.asarray(post_tighten_ee_rot)
        )
        post_tighten_ee_euler_np = Rot.from_quat(post_tighten_ee_rot_np).as_euler("xyz")
        post_tighten_ee_pose_world = np.concatenate(
            [post_tighten_ee_pos_np, post_tighten_ee_euler_np]
        ).astype(np.float32)
        post_tighten_robot_pose6 = env._world2robot(post_tighten_ee_pose_world).astype(np.float32)
        post_grasp_validation_stage = get_post_grasp_validation_stage(env)
        if post_grasp_validation_stage is None:
            grasp_hold_gripper_command = float(
                np.clip(post_tighten_gripper_normalized + POST_GRASP_HOLD_EXTRA_CLOSE, 0.0, 1.0)
            )
            trace_record["post_grasp_tighten"] = {
                "reason": str(reason),
                "gripper_command_normalized": float(POST_GRASP_CLOSE_COMMAND),
                "downward_offset_z_robot": float(POST_GRASP_DOWNWARD_OFFSET_Z),
                "target_robot_pose6_before_downward_offset": pre_post_grasp_robot_pose6.astype(np.float32).copy(),
                "target_robot_pose6": tighten_target_robot_pose6.astype(np.float32).copy(),
                "settle_steps": int(POST_GRASP_CLOSE_STEPS),
                "force_full_steps": True,
                "duration_s": float(tighten_duration_s),
                "task_progression": float(post_tighten_task_progression),
                "settle_history": tighten_result["settle_history"],
                "env_action": tighten_result["env_action"].copy(),
                "post_tighten_gripper_normalized": float(post_tighten_gripper_normalized),
                "post_tighten_raw_gripper": None if post_tighten_raw_gripper is None else float(post_tighten_raw_gripper),
                "hold_extra_close": float(POST_GRASP_HOLD_EXTRA_CLOSE),
                "grasp_hold_gripper_command": float(grasp_hold_gripper_command),
                "validation_stage": None,
            }
            trace_record["post_grasp_lift"] = {
                "reason": str(reason),
                "skipped": True,
                "skip_reason": "task_has_no_lift_validation_stage_after_grasp",
                "validation_stage": None,
                "lift_succeeded": True,
            }
            print(
                f"Selected GRASP reached during {reason} at step {t}; task has no post-GRASP lift "
                f"validation stage, so holding future gripper at {grasp_hold_gripper_command:.3f}.",
                flush=True,
            )
            validated_stages.add("GRASP")
            reached_stages.add("GRASP")
            return {
                "cancelled": False,
                "robot_pose6": post_tighten_robot_pose6,
                "lift_succeeded": True,
                "validation_stage": None,
            }
        lift_target_robot_pose6 = post_tighten_robot_pose6.astype(np.float32).copy()
        post_grasp_lift_offset_z = get_post_grasp_lift_offset_z(post_grasp_validation_stage)
        lift_target_robot_pose6[2] += float(post_grasp_lift_offset_z)
        t_lift_start = time.perf_counter()
        lift_result = execute_absolute_robot_target(
            env=env,
            target_robot_pose6=lift_target_robot_pose6,
            gripper_command=POST_GRASP_CLOSE_COMMAND,
            max_settle_steps=POST_GRASP_LIFT_STEPS,
            settle_pos_tol=settle_pos_tol,
            settle_rot_tol=settle_rot_tol,
            vis_dir=None,
            include_controller_debug=False,
            force_full_steps=True,
            cancel_event=cancel_event,
        )
        if cancel_requested():
            cancelled = True
            trace_record["post_grasp_lift_cancelled"] = True
            return {"cancelled": True, "robot_pose6": post_tighten_robot_pose6}

        obs = lift_result["obs"]
        curr_task_progression = lift_result["task_progression"]
        final_task_progression = float(curr_task_progression)
        post_lift_ee_pos, post_lift_ee_rot = env.get_ee_pose()
        post_lift_ee_pos_np = (
            post_lift_ee_pos.cpu().numpy()
            if hasattr(post_lift_ee_pos, "cpu")
            else np.asarray(post_lift_ee_pos)
        )
        post_lift_ee_rot_np = (
            post_lift_ee_rot.cpu().numpy()
            if hasattr(post_lift_ee_rot, "cpu")
            else np.asarray(post_lift_ee_rot)
        )
        post_lift_ee_euler_np = Rot.from_quat(post_lift_ee_rot_np).as_euler("xyz")
        post_lift_ee_pose_world = np.concatenate([post_lift_ee_pos_np, post_lift_ee_euler_np]).astype(
            np.float32
        )
        post_lift_robot_pose6 = env._world2robot(post_lift_ee_pose_world).astype(np.float32)
        post_lift_raw_gripper = get_raw_gripper_position_from_obs(obs, robot_name=env.robot.name)
        if post_lift_raw_gripper is None:
            post_lift_gripper_normalized = POST_GRASP_CLOSE_COMMAND
        else:
            post_lift_gripper_normalized = normalize_raw_gripper_position(post_lift_raw_gripper)
        grasp_hold_gripper_command = float(
            np.clip(post_lift_gripper_normalized + POST_GRASP_HOLD_EXTRA_CLOSE, 0.0, 1.0)
        )
        post_grasp_lift_succeeded = physical_stage_reached(
            env,
            post_grasp_validation_stage,
            curr_task_progression,
            obs=obs,
            validated_stages=validated_stages,
        )
        trace_record["post_grasp_tighten"] = {
            "reason": str(reason),
            "gripper_command_normalized": float(POST_GRASP_CLOSE_COMMAND),
            "downward_offset_z_robot": float(POST_GRASP_DOWNWARD_OFFSET_Z),
            "target_robot_pose6_before_downward_offset": pre_post_grasp_robot_pose6.astype(np.float32).copy(),
            "target_robot_pose6": tighten_target_robot_pose6.astype(np.float32).copy(),
            "settle_steps": int(POST_GRASP_CLOSE_STEPS),
            "force_full_steps": True,
            "duration_s": float(tighten_duration_s),
            "task_progression": float(post_tighten_task_progression),
            "settle_history": tighten_result["settle_history"],
            "env_action": tighten_result["env_action"].copy(),
            "post_tighten_gripper_normalized": float(post_tighten_gripper_normalized),
            "post_tighten_raw_gripper": None if post_tighten_raw_gripper is None else float(post_tighten_raw_gripper),
            "hold_extra_close": float(POST_GRASP_HOLD_EXTRA_CLOSE),
            "grasp_hold_gripper_command": float(grasp_hold_gripper_command),
            "validation_stage": post_grasp_validation_stage,
        }
        trace_record["post_grasp_lift"] = {
            "reason": str(reason),
            "validation_stage": post_grasp_validation_stage,
            "gripper_command_normalized": float(POST_GRASP_CLOSE_COMMAND),
            "lift_offset_z_robot": float(post_grasp_lift_offset_z),
            "target_robot_pose6_before_lift": post_tighten_robot_pose6.astype(np.float32).copy(),
            "target_robot_pose6": lift_target_robot_pose6.astype(np.float32).copy(),
            "actual_robot_pose6_after_lift": post_lift_robot_pose6.astype(np.float32).copy(),
            "settle_steps": int(POST_GRASP_LIFT_STEPS),
            "force_full_steps": True,
            "duration_s": float(time.perf_counter() - t_lift_start),
            "task_progression": float(curr_task_progression),
            "settle_history": lift_result["settle_history"],
            "env_action": lift_result["env_action"].copy(),
            "post_lift_gripper_normalized": float(post_lift_gripper_normalized),
            "post_lift_raw_gripper": None if post_lift_raw_gripper is None else float(post_lift_raw_gripper),
            "lift_succeeded": bool(post_grasp_lift_succeeded),
        }
        if not post_grasp_lift_succeeded:
            rollback_ok = False
            rollback_error = None
            if post_grasp_rollback_sim_state is not None:
                try:
                    load_sim_state(og, post_grasp_rollback_sim_state)
                    load_python_task_state(env, post_grasp_rollback_python_state)
                    og.sim.render()
                    obs = env.get_current_obs()
                    curr_task_progression = env.recompute_task_progression(obs)
                    cleared_stages = clear_failed_grasp_candidate_flags(env)
                    task_progression = getattr(env, "task_progression", None)
                    if isinstance(task_progression, dict):
                        curr_task_progression = float(
                            sum(1 for value in task_progression.values() if bool(value))
                            / max(1, len(task_progression))
                        )
                    final_task_progression = float(curr_task_progression)
                    rollback_ok = True
                except Exception as exc:
                    rollback_error = str(exc)
            else:
                rollback_error = "rollback snapshot unavailable"
                cleared_stages = []
            post_grasp_tightened = False
            grasp_hold_gripper_command = None
            validated_stages.discard("GRASP")
            validated_stages.discard("LIFT_SLIGHT")
            validated_stages.discard("LIFT_LARGE")
            validated_stages.discard(ROTATE_POST_GRASP_VALIDATION_STAGE)
            reached_stages.discard("GRASP")
            reached_stages.discard("LIFT_SLIGHT")
            reached_stages.discard("LIFT_LARGE")
            reached_stages.discard(ROTATE_POST_GRASP_VALIDATION_STAGE)
            trace_record["selected_goal_reached_now"] = False
            trace_record["post_grasp_lift_failed_rollback"] = {
                "rollback_ok": bool(rollback_ok),
                "rollback_error": rollback_error,
                "cleared_task_stages": list(cleared_stages),
            }
            print(
                f"Selected GRASP reached during {reason} at step {t}, but {post_grasp_validation_stage} failed; "
                f"rollback_ok={rollback_ok}. Continuing grasp attempts.",
                flush=True,
            )
            return {
                "cancelled": False,
                "robot_pose6": pre_post_grasp_robot_pose6,
            "lift_succeeded": False,
            "validation_stage": post_grasp_validation_stage,
            "rollback_ok": bool(rollback_ok),
        }
        print(
            f"Selected GRASP reached during {reason} at step {t}; lowered z by "
            f"{POST_GRASP_DOWNWARD_OFFSET_Z:.3f}, tightened gripper with command="
            f"{POST_GRASP_CLOSE_COMMAND:.3f}, then "
            f"{describe_post_grasp_validation(post_grasp_validation_stage, post_grasp_lift_offset_z)}; "
            f"holding future gripper at {grasp_hold_gripper_command:.3f}.",
            flush=True,
        )
        validated_stages.add("GRASP")
        reached_stages.add("GRASP")
        validated_stages.add(post_grasp_validation_stage)
        reached_stages.add(post_grasp_validation_stage)
        return {
            "cancelled": False,
            "robot_pose6": post_lift_robot_pose6,
            "lift_succeeded": True,
            "validation_stage": post_grasp_validation_stage,
        }

    def run_slot_gripper_calibration_round():
        nonlocal obs, curr_task_progression, final_task_progression, cancelled, t
        nonlocal slot_gripper_calibration_rounds_done, slot_next_planning_phase
        nonlocal slot_release_gripper_until_next_search, slot_gripper_release_restore_pose6
        nonlocal active_goal_idx
        if not phase_based_model:
            return False
        gripper_min = float((slot_warmup_config or {}).get("gripper_min", 0.0))
        gripper_max = float((slot_warmup_config or {}).get("gripper_max", 0.95))
        gripper_min = float(np.clip(gripper_min, 0.0, 1.0))
        gripper_max = float(np.clip(max(gripper_max, gripper_min), 0.0, 1.0))
        angles = np.linspace(
            gripper_min,
            gripper_max,
            DEFAULT_SLOT_GRIPPER_CALIBRATION_SAMPLES,
            dtype=np.float32,
        )
        round_idx = int(slot_gripper_calibration_rounds_done)
        print(
            f"[SLOT] gripper calibration round {round_idx + 1}/"
            f"{DEFAULT_SLOT_GRIPPER_CALIBRATION_ROUNDS}: executing "
            f"{len(angles)} gripper angles {np.round(angles, 4).tolist()}",
            flush=True,
        )
        calibration_base_context = build_current_multiview_context(
            env,
            task_cfg,
            obs,
            camera_frame_convention=camera_frame_convention,
            gripper_model_units=slot_model_gripper_units,
        )
        calibration_base_robot_pose6 = np.asarray(
            calibration_base_context["robot_pose6"], dtype=np.float32
        ).reshape(6)
        calibration_base_gripper_state = float(calibration_base_context["gripper_state"])
        calibration_downpress_max_z = float((slot_warmup_config or {}).get("gripper_downpress_max_z", 0.05))
        for angle_idx, gripper_angle in enumerate(angles.tolist()):
            if t >= max_steps:
                break
            if cancel_requested():
                cancelled = True
                break
            t_outer_step_start = time.perf_counter()
            pre_context = build_current_multiview_context(
                env,
                task_cfg,
                obs,
                camera_frame_convention=camera_frame_convention,
                gripper_model_units=slot_model_gripper_units,
            )
            pre_robot_pose6 = np.asarray(pre_context["robot_pose6"], dtype=np.float32).reshape(6)
            pre_gripper_state = float(pre_context["gripper_state"])
            gripper_angle_delta = float(gripper_angle) - float(pre_gripper_state)
            calibration_downpress_z_robot = (
                -float(calibration_downpress_max_z) * float(np.clip(gripper_angle, 0.0, 1.0))
                if abs(gripper_angle_delta) > 1e-4
                else 0.0
            )
            calibration_target_robot_pose6 = calibration_base_robot_pose6.astype(np.float32).copy()
            calibration_target_robot_pose6[2] += float(calibration_downpress_z_robot)
            t_execute_start = time.perf_counter()
            execution_result = execute_gripper_only(
                env=env,
                gripper_command=float(gripper_angle),
                steps=max(1, int(max_settle_steps)),
                hold_robot_pose6=calibration_base_robot_pose6,
                downpress_z_robot=float(calibration_downpress_z_robot),
                cancel_event=cancel_event,
            )
            timing_execute_target_s = time.perf_counter() - t_execute_start
            if cancel_requested():
                cancelled = True
                break
            obs = execution_result["obs"]
            curr_task_progression = execution_result["task_progression"]
            final_task_progression = float(curr_task_progression)
            post_context = build_current_multiview_context(
                env,
                task_cfg,
                obs,
                camera_frame_convention=camera_frame_convention,
                gripper_model_units=slot_model_gripper_units,
            )
            post_robot_pose6 = np.asarray(post_context["robot_pose6"], dtype=np.float32).reshape(6)
            calibration_actual_delta6 = pose_delta6(calibration_base_robot_pose6, post_robot_pose6)
            calibration_commanded_delta6 = pose_delta6(
                calibration_base_robot_pose6,
                calibration_target_robot_pose6,
            )
            all_poses.append(
                {
                    "step": int(t),
                    "ee_pos": post_robot_pose6[:3].astype(np.float32).copy(),
                    "ee_rot": post_robot_pose6[3:6].astype(np.float32).copy(),
                    "mode": "slot_gripper_calibration",
                }
            )
            fake_model_response = {
                "best_target_robot_pose6": calibration_target_robot_pose6.astype(np.float32).copy(),
                "predicted_next_state7_camera": np.concatenate(
                    [
                        np.asarray(pre_context["camera_pose6_by_view"]["wrist_image_left"], dtype=np.float32).reshape(6),
                        np.array([float(gripper_angle)], dtype=np.float32),
                    ],
                    axis=0,
                ),
                "slot_planning_request": "gripper_calibration",
                "cem_phase_outputs": [
                    {
                        "cem_phase": "gripper_calibration",
                        "target_robot_pose6": calibration_target_robot_pose6.astype(np.float32).copy(),
                        "gripper_delta": float(gripper_angle) - pre_gripper_state,
                        "gripper_downpress_z_robot": float(calibration_downpress_z_robot),
                        "gripper_downpress_base_robot_pose6": calibration_base_robot_pose6.astype(np.float32).copy(),
                        "skipped": False,
                    }
                ],
            }
            t_save_step_images_start = time.perf_counter()
            slot_mask_panels = (
                get_slot_mask_panels_for_payload(client, post_context["payload"])
                if model_type == "slot"
                else {}
            )
            image_writer.submit(
                os.path.join(task_debug_dir, "base_cam", f"{t:04d}.png"),
                append_slot_mask_panel(post_context["base_im"], slot_mask_panels.get("exterior_image_1_left")),
            )
            image_writer.submit(
                os.path.join(task_debug_dir, "second_cam", f"{t:04d}.png"),
                append_slot_mask_panel(post_context["base_im_second"], slot_mask_panels.get("exterior_image_2_left")),
            )
            image_writer.flush()
            save_wrist_image_with_cem_phase_outputs(
                os.path.join(task_debug_dir, "wrist_cam", f"{t:04d}.png"),
                append_slot_mask_panel(post_context["wrist_im"], slot_mask_panels.get("wrist_image_left")),
                fake_model_response,
            )
            timing_save_step_images_s = time.perf_counter() - t_save_step_images_start
            t_record_start = time.perf_counter()
            try:
                record_response = client.record_step_with_action(
                    {
                        "pre_obs": pre_context["payload"],
                        "post_obs": post_context["payload"],
                        "gripper_history_only": True,
                        "history_source": f"gripper_calibration_round_{round_idx}",
                    }
                )
            except Exception as exc:
                record_response = {"error": str(exc)}
                print(f"SLOT gripper calibration record issue: {exc}", flush=True)
            timing_record_step_s = time.perf_counter() - t_record_start
            step_traces.append(
                {
                    "step": int(t),
                    "mode": "online_slot_gripper_calibration",
                    "slot_planning_phase": "gripper",
                    "slot_planning_request": "gripper_calibration",
                    "slot_gripper_calibration_round": int(round_idx),
                    "slot_gripper_calibration_angle_index": int(angle_idx),
                    "slot_gripper_calibration_angles": angles.astype(np.float32).copy(),
                    "robot_pose6": pre_robot_pose6.astype(np.float32).copy(),
                    "next_robot_pose6": post_robot_pose6.astype(np.float32).copy(),
                    "slot_gripper_downpress_base_robot_pose6": calibration_base_robot_pose6.astype(np.float32).copy(),
                    "slot_gripper_downpress_base_gripper_state": float(calibration_base_gripper_state),
                    "slot_gripper_downpress_z_robot": float(calibration_downpress_z_robot),
                    "slot_gripper_downpress_actual_delta6_robot": calibration_actual_delta6.astype(
                        np.float32
                    ).copy(),
                    "slot_gripper_downpress_commanded_delta6_robot": calibration_commanded_delta6.astype(
                        np.float32
                    ).copy(),
                    "slot_gripper_downpress_actual_z_robot": float(calibration_actual_delta6[2]),
                    "slot_gripper_downpress_actual_xy_l2_robot": float(
                        np.linalg.norm(calibration_actual_delta6[:2])
                    ),
                    "slot_gripper_downpress_actual_rot_l2_robot": float(
                        np.linalg.norm(calibration_actual_delta6[3:6])
                    ),
                    "target_robot_pose6": calibration_target_robot_pose6.astype(np.float32).copy(),
                    "commanded_target_robot_pose6": calibration_target_robot_pose6.astype(np.float32).copy(),
                    "scoring_target_robot_pose6": calibration_base_robot_pose6.astype(np.float32).copy(),
                    "camera_pose6_by_view": {
                        k: v.astype(np.float32).copy()
                        for k, v in pre_context["camera_pose6_by_view"].items()
                    },
                    "next_camera_pose6_by_view": {
                        k: v.astype(np.float32).copy()
                        for k, v in post_context["camera_pose6_by_view"].items()
                    },
                    "camera_extrinsics6_by_view": {
                        k: v.astype(np.float32).copy()
                        for k, v in pre_context["camera_extrinsics_by_view"].items()
                    },
                    "next_camera_extrinsics6_by_view": {
                        k: v.astype(np.float32).copy()
                        for k, v in post_context["camera_extrinsics_by_view"].items()
                    },
                    "gripper_state_model": float(pre_context["gripper_state"]),
                    "next_gripper_state_model": float(post_context["gripper_state"]),
                    "gripper_state_normalized": float(pre_context["gripper_state_normalized"]),
                    "next_gripper_state_normalized": float(post_context["gripper_state_normalized"]),
                    "gripper_target_state_model": float(gripper_angle),
                    "gripper_target_state_normalized": float(gripper_angle),
                    "gripper_command_env": float(gripper_to_env_command(float(gripper_angle))),
                    "env_action": execution_result.get("env_action"),
                    "initial_env_action": execution_result.get("initial_env_action"),
                    "settle_history": execution_result.get("settle_history", []),
                    "settle_steps_used": int(execution_result.get("steps", max_settle_steps)),
                    "full_inner_steps_executed": int(
                        execution_result.get("full_inner_steps_executed", execution_result.get("steps", 0))
                    ),
                    "gripper_only_execution": True,
                    "gripper_only_execution_mode": execution_result.get("gripper_execution_mode"),
                    "gripper_only_env_action": execution_result.get("env_action"),
                    "gripper_only_hold_robot_pose6_before_downpress": execution_result.get(
                        "gripper_hold_robot_pose6_before_downpress"
                    ),
                    "gripper_only_hold_robot_pose6": execution_result.get("gripper_hold_robot_pose6"),
                    "gripper_only_downpress_z_robot": execution_result.get("gripper_downpress_z_robot"),
                    "gripper_only_adaptive_downpress_release": bool(
                        execution_result.get("adaptive_downpress_release", False)
                    ),
                    "gripper_only_adaptive_downpress_release_step": execution_result.get(
                        "adaptive_downpress_release_step"
                    ),
                    "gripper_only_adaptive_downpress_actual_z_robot": execution_result.get(
                        "adaptive_downpress_actual_z_robot"
                    ),
                    "gripper_only_adaptive_downpress_required_z_robot": execution_result.get(
                        "adaptive_downpress_required_z_robot"
                    ),
                    "gripper_only_target_joint_pos": execution_result.get("gripper_target_joint_pos"),
                    "gripper_only_dof_idx": execution_result.get("gripper_dof_idx"),
                    "gripper_only_actual_joint_pos": execution_result.get("gripper_actual_joint_pos"),
                    "gripper_only_controller_control": (
                        execution_result.get("info", {}).get("gripper_controller_control")
                        if isinstance(execution_result.get("info"), dict)
                        else None
                    ),
                    "gripper_only_rendered_after_step": bool(
                        execution_result.get("gripper_rendered_after_step", False)
                    ),
                    "slot_record_response": record_response,
                    "server_per_view_losses": record_response.get("per_view_losses"),
                    "server_per_view_transport_losses": record_response.get("per_view_transport_losses"),
                    "server_per_view_objective_losses": record_response.get("per_view_objective_losses"),
                    "observed_per_view_losses": record_response.get("per_view_losses"),
                    "observed_per_view_transport_losses": record_response.get("per_view_transport_losses"),
                    "observed_per_view_objective_losses": record_response.get("per_view_objective_losses"),
                    "server_per_view_gripper_slot_transport_losses": record_response.get(
                        "per_view_gripper_slot_transport_losses"
                    ),
                    "server_per_view_other_slot_mean_transport_losses": record_response.get(
                        "per_view_other_slot_mean_transport_losses"
                    ),
                    "candidate_diagnostics_summary": None,
                    "cem_phase_outputs": fake_model_response["cem_phase_outputs"],
                    "goal_idx": int(active_goal_idx),
                    "goal_name": None
                    if goal_state is None or active_goal_idx >= len(goal_state.get("target_names", []))
                    else goal_state["target_names"][active_goal_idx],
                    "saved_observation_timing": "post_execution",
                    "timing_extract_obs_s": 0.0,
                    "timing_client_infer_s": 0.0,
                    "timing_execute_target_s": float(timing_execute_target_s),
                    "timing_post_obs_extract_s": 0.0,
                    "timing_save_step_images_s": float(timing_save_step_images_s),
                    "timing_record_step_s": float(timing_record_step_s),
                    "timing_outer_step_total_s": float(time.perf_counter() - t_outer_step_start),
                    "server_timing": None,
                    "validated_stages": sorted(validated_stages),
                }
            )
            calibration_goal_reached = selected_goal_stage_reached(
                env,
                goal_state,
                curr_task_progression,
                goal_idx=active_goal_idx,
                obs=obs,
                validated_stages=validated_stages,
            )
            target_names = list(goal_state.get("target_names", [])) if goal_state is not None else []
            target_stages = list(goal_state.get("target_stages", [])) if goal_state is not None else []
            reached_stage = (
                target_stages[active_goal_idx]
                if 0 <= active_goal_idx < len(target_stages)
                else "UNKNOWN"
            )
            reached_name = (
                target_names[active_goal_idx]
                if 0 <= active_goal_idx < len(target_names)
                else f"goal_{active_goal_idx}"
            )
            step_traces[-1]["selected_goal_reached_now"] = bool(calibration_goal_reached)
            step_traces[-1]["selected_goal_reached_stage"] = reached_stage if calibration_goal_reached else None
            step_traces[-1]["selected_goal_reached_name"] = reached_name if calibration_goal_reached else None
            if calibration_goal_reached and reached_stage == "GRASP":
                Image.fromarray(post_context["base_im"]).save(
                    os.path.join(task_debug_dir, "reached_stage_GRASP.png")
                )
                Image.fromarray(post_context["base_im_second"]).save(
                    os.path.join(task_debug_dir, "reached_stage_GRASP_second.png")
                )
                Image.fromarray(post_context["wrist_im"]).save(
                    os.path.join(task_debug_dir, "reached_stage_GRASP_wrist.png")
                )
                validated_stages.add("GRASP")
                reached_stages.add("GRASP")
                step_traces[-1]["slot_gripper_calibration_hit_grasp"] = True
                step_traces[-1]["reached_stage_grasp_saved_images"] = {
                    "base": "reached_stage_GRASP.png",
                    "second": "reached_stage_GRASP_second.png",
                    "wrist": "reached_stage_GRASP_wrist.png",
                    "timing": "pre_post_grasp_tighten_and_lift",
                }
                post_grasp_result = run_post_grasp_tighten_and_lift(
                    step_traces[-1],
                    post_robot_pose6,
                    reason="slot_gripper_calibration",
                )
                slot_gripper_calibration_rounds_done += 1
                slot_next_planning_phase = "pose"
                slot_release_gripper_until_next_search = False
                slot_gripper_release_restore_pose6 = None
                flush_trace_artifacts(cancelled_flag=cancelled)
                t += 1
                if post_grasp_result.get("cancelled", False):
                    return {"handled": True, "stop": True}
                if not post_grasp_result.get("lift_succeeded", True):
                    step_traces[-1]["slot_gripper_calibration_hit_grasp_but_lift_failed"] = True
                    step_traces[-1]["selected_goal_reached_now"] = False
                    validated_stages.discard("GRASP")
                    validated_stages.discard("LIFT_SLIGHT")
                    reached_stages.discard("GRASP")
                    reached_stages.discard("LIFT_SLIGHT")
                    flush_trace_artifacts(cancelled_flag=cancelled)
                    return {"handled": True, "stop": False}
                if goal_mode == "first":
                    step_traces[-1]["stop_reason"] = "gripper_calibration_reached_first_goal_grasp"
                    flush_trace_artifacts(cancelled_flag=cancelled)
                    return {"handled": True, "stop": True}
                next_goal_idx = active_goal_idx + 1
                if next_goal_idx < len(goal_state.get("goal_targets", [])):
                    try:
                        advance_response = client.advance_goal(goal_idx=next_goal_idx)
                        active_goal_idx = int(advance_response.get("goal_idx", next_goal_idx))
                        step_traces[-1]["advanced_goal_response"] = advance_response
                        if model_type == "slot" and hasattr(client, "reset_loss_tracking"):
                            client.reset_loss_tracking()
                    except Exception as exc:
                        step_traces[-1]["advance_goal_error"] = str(exc)
                        print(f"Warning: failed to advance goal after calibration GRASP: {exc}", flush=True)
                    flush_trace_artifacts(cancelled_flag=cancelled)
                    return {"handled": True, "stop": False}
                step_traces[-1]["stop_reason"] = "gripper_calibration_reached_final_grasp_goal"
                flush_trace_artifacts(cancelled_flag=cancelled)
                return {"handled": True, "stop": True}
            flush_trace_artifacts(cancelled_flag=cancelled)
            t += 1
        slot_gripper_calibration_rounds_done += 1
        slot_next_planning_phase = "pose"
        slot_release_gripper_until_next_search = True
        try:
            latest_context = build_current_multiview_context(
                env,
                task_cfg,
                obs,
                camera_frame_convention=camera_frame_convention,
                gripper_model_units=slot_model_gripper_units,
            )
            slot_gripper_release_restore_pose6 = np.asarray(latest_context["robot_pose6"], dtype=np.float32).reshape(6)
        except Exception:
            slot_gripper_release_restore_pose6 = None
        return True

    try:
        slot_warmup_config = client.get_warmup_config()
    except Exception:
        slot_warmup_config = None
    warmup_model_active = bool(
        slot_warmup_config
        and int(slot_warmup_config.get("num_frames", 0)) > 0
        and hasattr(client, "score_observations_to_goal")
    )
    if warmup_model_active:
        warmup_log_prefix = f"[{str(model_type).upper()} warmup]"
        warmup_image_dir = os.path.join(task_debug_dir, "warmup_images")
        os.makedirs(warmup_image_dir, exist_ok=True)
        initial_context = build_current_multiview_context(
            env,
            task_cfg,
            obs,
            camera_frame_convention=camera_frame_convention,
            gripper_model_units=slot_model_gripper_units,
        )
        # Preserve the real observation from before any SLOT warmup motion.
        # A3 uses this independently from the warmup sequence for causal
        # cold-start padding; other planners simply ignore the extra field.
        slot_temporal_initial_payload = initial_context["payload"]
        warmup_contexts = []
        warmup_fixed_gripper_state = (
            1.0 if task_is_reach else slot_warmup_config.get("fixed_gripper_state")
        )
        warmup_random_gripper = (
            False if task_is_reach else slot_warmup_config.get("random_gripper", False)
        )
        warmup_waypoints = build_z_levels_xy_circle_warmup_waypoints(
            start_robot_pose6=initial_context["robot_pose6"],
            start_gripper_state=initial_context["gripper_state_normalized"],
            xy_radius=slot_warmup_config["xy_radius"],
            z_offsets=slot_warmup_config["z_offsets"],
            points_per_z=slot_warmup_config.get("points_per_z", 4),
            random_gripper=warmup_random_gripper,
            gripper_min=slot_warmup_config.get("gripper_min", 0.0),
            gripper_max=slot_warmup_config.get("gripper_max", 1.0),
            fixed_gripper_state=warmup_fixed_gripper_state,
        )
        with open(os.path.join(task_debug_dir, "slot_warmup_waypoints.json"), "w", encoding="utf-8") as f:
            json.dump(
                to_jsonable(
                    {
                        "model_type": str(model_type),
                        "slot_warmup_config": slot_warmup_config,
                        "task_is_reach": bool(task_is_reach),
                        "effective_random_gripper": bool(warmup_random_gripper),
                        "effective_fixed_gripper_state": None
                        if warmup_fixed_gripper_state is None
                        else float(warmup_fixed_gripper_state),
                        "target_gripper_state_normalized": [
                            float(wp["target_gripper_state"]) for wp in warmup_waypoints
                        ],
                        "sampled_delta_gripper_normalized": [
                            float(wp["sampled_delta_gripper"]) for wp in warmup_waypoints
                        ],
                        "target_robot_pose6": [wp["target_robot_pose6"] for wp in warmup_waypoints],
                    }
                ),
                f,
                ensure_ascii=False,
                indent=2,
            )
        slot_warmup_settle_steps = int(slot_warmup_config.get("settle_steps", 8))
        for warmup_idx, waypoint in enumerate(warmup_waypoints):
            if cancel_requested():
                print(f"Planning cancelled during {model_type} warmup at warmup={warmup_idx}.", flush=True)
                cancelled = True
                break
            warmup_execution = execute_absolute_robot_target(
                env,
                target_robot_pose6=waypoint["target_robot_pose6"],
                gripper_command=waypoint["target_gripper_state"],
                max_settle_steps=slot_warmup_settle_steps,
                settle_pos_tol=settle_pos_tol,
                settle_rot_tol=settle_rot_tol,
                vis_dir=None,
                include_controller_debug=False,
                use_fast_inner_step=True,
                cancel_event=cancel_event,
            )
            if cancel_requested():
                print(f"Planning cancelled during {model_type} warmup execution at warmup={warmup_idx}.", flush=True)
                cancelled = True
                break
            obs = warmup_execution["obs"] if warmup_execution["obs"] is not None else env.get_current_obs()
            warmup_context = build_current_multiview_context(
                env,
                task_cfg,
                obs,
                camera_frame_convention=camera_frame_convention,
                gripper_model_units=slot_model_gripper_units,
            )
            if warmup_idx < 2:
                dump_robot_camera_observations(
                    obs,
                    robot_name=env.robot.name,
                    output_dir=os.path.join(task_debug_dir, "robot_camera_debug"),
                    prefix=f"warmup_{warmup_idx:02d}",
                )
            warmup_contexts.append(warmup_context)
            for suffix, image in {
                "base": warmup_context["base_im"],
                "second": warmup_context["base_im_second"],
                "wrist": warmup_context["wrist_im"],
            }.items():
                image_writer.submit(os.path.join(warmup_image_dir, f"warmup_{warmup_idx:02d}_{suffix}.png"), image)
            actual_delta_robot6 = (warmup_context["robot_pose6"] - initial_context["robot_pose6"]).astype(np.float32)
            slot_warmup_records.append(
                {
                    "warmup_idx": int(warmup_idx),
                    "mode": "z_levels_xy_circle",
                    "sampled_delta_robot6": waypoint["sampled_delta_robot6"].copy(),
                    "sampled_delta_gripper_normalized": float(waypoint["sampled_delta_gripper"]),
                    "z_level": waypoint.get("z_level"),
                    "xy_angle_rad": waypoint.get("xy_angle_rad"),
                    "return_to_start": bool(waypoint.get("return_to_start", False)),
                    "target_robot_pose6": waypoint["target_robot_pose6"].copy(),
                    "target_gripper_state_normalized": float(waypoint["target_gripper_state"]),
                    "actual_robot_pose6": warmup_context["robot_pose6"].copy(),
                    "actual_gripper_state_model": float(warmup_context["gripper_state_model"]),
                    "actual_gripper_state_normalized": float(warmup_context["gripper_state_normalized"]),
                    "actual_delta_robot6": actual_delta_robot6.copy(),
                    "actual_delta_gripper_model": float(
                        warmup_context["gripper_state_model"] - initial_context["gripper_state_model"]
                    ),
                    "actual_delta_gripper_normalized": float(
                        warmup_context["gripper_state_normalized"] - initial_context["gripper_state_normalized"]
                    ),
                    "settle_steps_used": int(len(warmup_execution["settle_history"])),
                }
            )
            print(
                f"{warmup_log_prefix} "
                f"warmup={warmup_idx:02d} mode=z_levels_xy_circle "
                f"target_delta_pos={np.linalg.norm(waypoint['sampled_delta_robot6'][:3]):.4f} "
                f"actual_delta_pos={np.linalg.norm(actual_delta_robot6[:3]):.4f} "
                f"target_gripper={float(waypoint['target_gripper_state']):.3f} "
                f"actual_gripper={float(warmup_context['gripper_state_normalized']):.3f}",
                flush=True,
            )

        slot_warmup_payload_sequence = [context["payload"] for context in warmup_contexts]
        slot_warmup_goal_loss_response = None
        try:
            if cancelled:
                print(f"{warmup_log_prefix} skipped warmup goal-loss annotation because planning was cancelled.", flush=True)
            else:
                image_writer.flush()
                slot_warmup_goal_loss_response = client.score_observations_to_goal(
                    {
                        "obs_sequence": slot_warmup_payload_sequence,
                        "goal_idx": int(active_goal_idx),
                    }
                )
                warmup_loss_records = (slot_warmup_goal_loss_response or {}).get("records") or []
                view_suffix_by_name = {
                    "exterior_image_1_left": "base",
                    "exterior_image_2_left": "second",
                    "wrist_image_left": "wrist",
                }
                for loss_record in warmup_loss_records:
                    loss_idx = int(loss_record.get("obs_idx", -1))
                    if 0 <= loss_idx < len(slot_warmup_records):
                        slot_warmup_records[loss_idx]["goal_loss"] = loss_record
                    available_loss_views = (
                        loss_record.get("per_view_objective_losses")
                        or loss_record.get("per_view_l1_losses")
                        or loss_record.get("per_view_losses")
                        or {}
                    )
                    for view_name, suffix in view_suffix_by_name.items():
                        if view_name not in available_loss_views:
                            continue
                        l1_loss = (loss_record.get("per_view_l1_losses") or {}).get(view_name)
                        if l1_loss is None:
                            l1_loss = (loss_record.get("per_view_losses") or {}).get(view_name)
                        transport_loss = (loss_record.get("per_view_transport_losses") or {}).get(view_name)
                        objective_loss = (loss_record.get("per_view_objective_losses") or {}).get(view_name)
                        annotate_image_with_text_lines(
                            os.path.join(warmup_image_dir, f"warmup_{loss_idx:02d}_{suffix}.png"),
                            [
                                f"warmup {loss_idx:02d} -> goal[{int(loss_record.get('goal_idx', active_goal_idx))}]",
                                None if l1_loss is None else f"L1={float(l1_loss):.4f}",
                                None if transport_loss is None else f"OT/IoU={float(transport_loss):.4f}",
                                None if objective_loss is None else f"Obj={float(objective_loss):.4f}",
                            ],
                        )
                print(
                    f"{warmup_log_prefix} "
                    f"annotated {len(warmup_loss_records)} warmup frames with goal losses "
                    f"(goal_idx={active_goal_idx})",
                    flush=True,
                )
        except Exception as exc:
            print(f"{warmup_log_prefix} Failed to annotate warmup goal losses: {exc}", flush=True)
        with open(os.path.join(task_debug_dir, "slot_warmup_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                to_jsonable(
                    {
                        "model_type": str(model_type),
                        "camera_frame_convention": camera_frame_convention,
                        "slot_warmup_mode": "z_levels_xy_circle",
                        "slot_warmup_num_frames": int(len(warmup_contexts)),
                        "slot_warmup_config": slot_warmup_config,
                        "task_is_reach": bool(task_is_reach),
                        "effective_random_gripper": bool(warmup_random_gripper),
                        "effective_fixed_gripper_state": None
                        if warmup_fixed_gripper_state is None
                        else float(warmup_fixed_gripper_state),
                        "slot_model_gripper_units": slot_model_gripper_units,
                        "slot_warmup_settle_steps": int(slot_warmup_settle_steps),
                        "slot_warmup_goal_loss_response": slot_warmup_goal_loss_response,
                        "slot_warmup_records": slot_warmup_records,
                    }
                ),
                f,
                ensure_ascii=False,
                indent=2,
            )
    
    # 3. Run Evaluation Loop
    while t < max_steps and not cancelled:
        if cancel_requested():
            print(f"Planning cancelled before outer step {t}.", flush=True)
            cancelled = True
            break
        t_outer_step_start = time.perf_counter()
        slot_pre_step_release_summary = None
        if (
            model_type == "slot"
            and slot_next_planning_phase == "pose"
            and slot_release_gripper_until_next_search
            and grasp_hold_gripper_command is None
        ):
            cached_release_obs = slot_cached_pose_obs_for_release
            release_restore_pose6 = (
                np.asarray(slot_gripper_release_restore_pose6, dtype=np.float32).reshape(6)
                if slot_gripper_release_restore_pose6 is not None
                else None
            )
            if release_restore_pose6 is None:
                ee_pos_release, ee_rot_release = env.get_ee_pose()
                ee_pos_release_np = (
                    ee_pos_release.cpu().numpy() if hasattr(ee_pos_release, "cpu") else np.asarray(ee_pos_release)
                )
                ee_rot_release_np = (
                    ee_rot_release.cpu().numpy() if hasattr(ee_rot_release, "cpu") else np.asarray(ee_rot_release)
                )
                ee_pose_world_release = np.concatenate(
                    [ee_pos_release_np, Rot.from_quat(ee_rot_release_np).as_euler("xyz")]
                ).astype(np.float32)
                release_restore_pose6 = env._world2robot(ee_pose_world_release).astype(np.float32)
            t_release_start = time.perf_counter()
            release_result = execute_absolute_robot_target(
                env=env,
                target_robot_pose6=release_restore_pose6,
                gripper_command=0.0,
                max_settle_steps=max_settle_steps,
                settle_pos_tol=settle_pos_tol,
                settle_rot_tol=settle_rot_tol,
                vis_dir=None,
                include_controller_debug=False,
                cancel_event=cancel_event,
            )
            obs = release_result["obs"]
            curr_task_progression = release_result["task_progression"]
            final_task_progression = float(curr_task_progression)
            slot_release_gripper_until_next_search = False
            slot_gripper_release_restore_pose6 = None
            slot_cached_pose_obs_for_release = None
            slot_pre_step_release_summary = {
                "executed": True,
                "reused_cached_pose_observation": False,
                "cached_pose_observation_was_available": bool(cached_release_obs is not None),
                "restore_pose6_robot": release_restore_pose6.astype(np.float32).copy(),
                "duration_s": float(time.perf_counter() - t_release_start),
                "release_restore_steps": len(release_result.get("settle_history", [])),
                "used_fallback_absolute_pose": bool(release_result.get("used_fallback_absolute_pose", False)),
            }
            if cancel_requested():
                print(f"Planning cancelled after SLOT pre-step gripper release at step {t}.", flush=True)
                cancelled = True
                break
        t_extract_start = time.perf_counter()
        base_im, base_depth, base_im_second, base_depth_second, wrist_im, robot_state, gripper_state = extract_from_obs(obs, robot_name=env.robot.name)
        raw_gripper_state = get_raw_gripper_position_from_obs(obs, robot_name=env.robot.name)
        gripper_state_normalized = float(np.clip(float(gripper_state), 0.0, 1.0))
        gripper_state_raw_width = None
        if model_type == "slot" and raw_gripper_state is not None:
            gripper_state_raw_width = float(np.clip(raw_gripper_state, 0.0, 0.05))
            gripper_state_normalized = normalize_raw_gripper_position(raw_gripper_state)
            gripper_state = gripper_state_for_model(raw_gripper_state, slot_model_gripper_units)
        ee_pos, ee_rot = env.get_ee_pose()
        _ee_pos = ee_pos.cpu().numpy() if hasattr(ee_pos, 'cpu') else np.array(ee_pos)
        _ee_rot = ee_rot.cpu().numpy() if hasattr(ee_rot, 'cpu') else np.array(ee_rot)
        _ee_euler = Rot.from_quat(_ee_rot).as_euler('xyz')
        _ee_pose_world = np.concatenate([_ee_pos, _ee_euler])
        robot_pose6 = env._world2robot(_ee_pose_world).astype(np.float32)
        view_world_poses = get_view_world_poses(env, task_cfg)
        camera_extrinsics_by_view = {}
        camera_pose6_by_view = {}
        for view_name, (cam_pos_world, cam_quat_world) in view_world_poses.items():
            camera_world_pose6 = get_camera_extrinsics_pose6(
                cam_pos_world,
                cam_quat_world,
                camera_frame_convention=camera_frame_convention,
            )
            camera_extrinsics_by_view[view_name] = env._world2robot(camera_world_pose6).astype(np.float32)
            camera_pose6_by_view[view_name] = pose_in_parent_frame(
                robot_pose6.astype(np.float32),
                camera_extrinsics_by_view[view_name],
            )
        camera_pose6 = camera_pose6_by_view["exterior_image_1_left"]
        timing_extract_obs_s = time.perf_counter() - t_extract_start
        
        # Step sequence images are saved after execution so they reflect the physical result,
        # not the observation used for planning.
        timing_save_step_images_s = 0.0
        main_object_debug = get_main_object_pose_debug(env)
        
        step_pose = {
            "step": t,
            "ee_pos": ee_pos.cpu().numpy() if hasattr(ee_pos, 'cpu') else np.array(ee_pos),
            "ee_rot": ee_rot.cpu().numpy() if hasattr(ee_rot, 'cpu') else np.array(ee_rot),
            "main_object": main_object_debug,
        }
        all_poses.append(step_pose)
        
        ref_cam_quat_world = view_world_poses["exterior_image_1_left"][1]
        pose6_for_trace = camera_pose6
        completed_stages_for_mpc = [
            stage
            for stage in (env.task_progression.keys() if env.task_progression is not None else [])
            if physical_stage_reached(
                env,
                stage,
                curr_task_progression=0.0,
                obs=obs,
                validated_stages=validated_stages,
            )
        ]
        current_goal_target = None
        if goal_state is not None:
            goal_targets_local = goal_state.get("goal_targets", [])
            if goal_targets_local and 0 <= active_goal_idx < len(goal_targets_local):
                current_goal_target = goal_targets_local[active_goal_idx]
        current_goal_stage_name = None if current_goal_target is None else current_goal_target.get("stage")
        # Do not treat SUCCESS as a release command before it is actually
        # reached. Some generated put cases only expose GRASP + SUCCESS goals;
        # skipping the post-grasp hold during that second goal opens the gripper
        # while the robot is still carrying the object. Explicit release is
        # handled after MOVE_CLOSE / PLACE_* stage completion below.
        is_release_goal_stage = current_goal_stage_name in POST_CARRY_RELEASE_STAGES
        task_progression_keys = set(env.task_progression.keys()) if env.task_progression is not None else set()
        rotate_deterministic_base_active = bool(
            phase_based_model
            and rotate_deterministic_after_grasp
            and "ROTATED" in task_progression_keys
            and "GRASP" in set(completed_stages_for_mpc)
        )
        rotate_deterministic_active = bool(
            rotate_deterministic_base_active
            and float(rotate_deterministic_accumulated_abs_rad)
            < float(DEFAULT_ROTATE_DETERMINISTIC_FULL_TURN_RAD)
        )
        if rotate_deterministic_base_active:
            if (
                rotate_deterministic_hold_robot_pose6 is None
                or rotate_deterministic_hold_goal_idx != int(active_goal_idx)
            ):
                rotate_deterministic_hold_robot_pose6 = robot_pose6.astype(np.float32).copy()
                rotate_deterministic_hold_goal_idx = int(active_goal_idx)
                rotate_deterministic_direction_sign = None
                rotate_deterministic_accumulated_abs_rad = 0.0
                print(
                    f"[{str(model_type).upper()}] Locked deterministic rotate hold pose:"
                    f" goal_idx={rotate_deterministic_hold_goal_idx}"
                    f" pose6={np.round(rotate_deterministic_hold_robot_pose6, 4).tolist()}",
                    flush=True,
                )
        else:
            rotate_deterministic_hold_robot_pose6 = None
            rotate_deterministic_hold_goal_idx = None
            rotate_deterministic_direction_sign = None
            rotate_deterministic_accumulated_abs_rad = 0.0
        slot_current_planning_phase = slot_next_planning_phase if phase_based_model else None
        if vjepa_phase_model:
            slot_current_planning_phase = "pose"
            slot_next_planning_phase = "pose"
        if phase_based_model and ("GRASP" in set(completed_stages_for_mpc) or is_release_goal_stage):
            slot_current_planning_phase = "pose"
            slot_next_planning_phase = "pose"
        if phase_based_model and task_is_reach:
            slot_current_planning_phase = "pose"
            slot_next_planning_phase = "pose"
        slot_planning_request = None
        if phase_based_model:
            slot_planning_request = "gripper_only" if slot_current_planning_phase == "gripper" else "pose_only"
            if rotate_deterministic_active:
                slot_planning_request = "score_only"
        if (
            phase_based_model
            and slot_planning_request == "gripper_only"
            and slot_gripper_calibration_rounds_done < DEFAULT_SLOT_GRIPPER_CALIBRATION_ROUNDS
        ):
            calibration_result = run_slot_gripper_calibration_round()
            if isinstance(calibration_result, dict) and calibration_result.get("stop", False):
                break
            continue
        t_infer_start = time.perf_counter()
        model_response = client.infer(
            instruction="",
            base_im=base_im,
            base_im_second=base_im_second,
            wrist_im=wrist_im,
            robot_state=robot_state,
            gripper_state=gripper_state,
            ee_control=True,
            cartesian_position=camera_pose6_by_view,
            camera_extrinsics=camera_extrinsics_by_view,
            camera_frame_convention=camera_frame_convention,
            pre_obs_sequence=slot_warmup_payload_sequence if (phase_based_model and t == 0) else None,
            temporal_initial_obs=slot_temporal_initial_payload if (phase_based_model and t == 0) else None,
            completed_stages=completed_stages_for_mpc,
            slot_planning_request=slot_planning_request,
            current_robot_pose6=robot_pose6,
        )
        timing_client_infer_s = time.perf_counter() - t_infer_start
        if cancel_requested():
            print(f"Planning cancelled after model inference at step {t}.", flush=True)
            cancelled = True
            break
        slot_pose_only_response = model_response if slot_planning_request == "pose_only" else None
        slot_pose_execution_summary = None
        slot_gripper_obs_updated_after_pose = False
        if phase_based_model and t == 0 and not model_response.get("cem_phase_outputs"):
            print(
                "[SLOT] Warning: model_response has no cem_phase_outputs; "
                "wrist_cam annotation will mark it as missing. Restart serve_slot.py if it is still running old code.",
                flush=True,
            )
        slot_executed_gripper_phase = (
            phase_based_model and _slot_response_executed_gripper_phase(model_response)
        )
        if model_type == "slot" and not slot_mask_vis_saved and hasattr(client, "plot_slot_masks"):
            slot_mask_path = os.path.abspath(os.path.join(task_debug_dir, "slot_masks_step0000.png"))
            try:
                mask_response = client.plot_slot_masks(slot_mask_path)
                vis_name = None if mask_response is None else mask_response.get("visualization")
                print(
                    f"Saved SLOT warmup/goal mask visualization to {slot_mask_path}"
                    f" visualization={vis_name}",
                    flush=True,
                )
            except Exception as exc:
                print(f"Failed to save SLOT mask visualization: {exc}", flush=True)
            slot_mask_vis_saved = True
        if model_type == "slot" and not slot_goal_mask_vis_saved:
            try:
                goal_panels_by_goal = get_goal_slot_mask_panels(client)
                saved_goal_mask_count = save_goal_images_with_slot_masks(
                    task_debug_dir=task_debug_dir,
                    goal_images=goal_images,
                    goal_state=goal_state,
                    goal_panels_by_goal=goal_panels_by_goal,
                )
                if saved_goal_mask_count > 0:
                    print(
                        "Saved SLOT goal mask visualizations to base_cam/second_cam/wrist_cam "
                        f"count={saved_goal_mask_count}.",
                        flush=True,
                    )
                else:
                    print(
                        "Warning: SLOT goal mask visualization returned no savable panels.",
                        flush=True,
                    )
            except Exception as exc:
                print(f"Failed to save SLOT goal mask visualizations: {exc}", flush=True)
            slot_goal_mask_vis_saved = True
        commanded_target_robot_pose6 = np.asarray(
            model_response["best_target_robot_pose6"], dtype=np.float32
        ).reshape(-1)
        if commanded_target_robot_pose6.shape[0] != 6:
            raise ValueError(
                f"Expected best_target_robot_pose6 with 6 dims, got {commanded_target_robot_pose6.shape}"
            )
        scoring_target_robot_pose6 = commanded_target_robot_pose6.astype(np.float32).copy()
        slot_gripper_execution_downpress_z_robot = 0.0
        slot_gripper_server_downpress_z_robot = 0.0
        slot_gripper_downpress_base_robot_pose6 = None
        rotate_deterministic_delta_robot6 = None
        rotate_deterministic_goal_axis_error = None
        rotate_deterministic_raw_goal_axis_error = None
        rotate_deterministic_actual_axis_delta = None
        if phase_based_model:
            predicted_next_state7_camera = np.asarray(
                model_response["predicted_next_state7_camera"], dtype=np.float32
            ).reshape(-1)
            if predicted_next_state7_camera.shape[0] != 7:
                raise ValueError(
                    "SLOT response must include predicted_next_state7_camera with 7 dims, "
                    f"got {predicted_next_state7_camera.shape}"
                )
            action = None
            action_robot_frame = None
            target_ee_world6 = None
            target_robot_pose6 = commanded_target_robot_pose6.copy()
            model_gripper_target_state = float(predicted_next_state7_camera[6])
            gripper_target_state = model_gripper_to_normalized(
                model_gripper_target_state,
                slot_model_gripper_units,
            )
            gripper_policy = "world_model_state"
            gripper_command = gripper_target_state
            vjepa_direct_gripper_close_debug = None
            vjepa_direct_gripper_close_triggered = False
            if slot_planning_request == "score_only" and rotate_deterministic_active:
                if rotate_deterministic_hold_robot_pose6 is None:
                    rotate_deterministic_hold_robot_pose6 = robot_pose6.astype(np.float32).copy()
                    rotate_deterministic_hold_goal_idx = int(active_goal_idx)
                    rotate_deterministic_direction_sign = None
                    rotate_deterministic_accumulated_abs_rad = 0.0
                target_robot_pose6 = robot_pose6.astype(np.float32).copy()
                target_robot_pose6[:3] = rotate_deterministic_hold_robot_pose6[:3]
                rotate_sign = rotate_deterministic_direction_sign
                if current_goal_target is not None and current_goal_target.get("ee_state7_robot") is not None:
                    goal_state7_for_rotate = np.asarray(
                        current_goal_target["ee_state7_robot"], dtype=np.float32
                    ).reshape(-1)
                    if goal_state7_for_rotate.shape[0] >= 6:
                        raw_goal_axis_error = float(
                            angle_wrap(goal_state7_for_rotate[rotate_axis_index] - robot_pose6[rotate_axis_index])
                        )
                        rotate_deterministic_raw_goal_axis_error = raw_goal_axis_error
                        rotate_deterministic_goal_axis_error = raw_goal_axis_error
                        if rotate_sign is None and abs(raw_goal_axis_error) > 1.0e-4:
                            rotate_sign = float(np.sign(raw_goal_axis_error))
                if rotate_sign is None:
                    rotate_sign = 1.0
                rotate_deterministic_direction_sign = float(rotate_sign)
                remaining_full_turn = max(
                    0.0,
                    float(DEFAULT_ROTATE_DETERMINISTIC_FULL_TURN_RAD)
                    - float(rotate_deterministic_accumulated_abs_rad),
                )
                rotate_step = float(rotate_sign) * min(rotate_step_rad, remaining_full_turn)
                target_robot_pose6[rotate_axis_index] = float(
                    angle_wrap(target_robot_pose6[rotate_axis_index] + rotate_step)
                )
                commanded_target_robot_pose6 = target_robot_pose6.astype(np.float32).copy()
                scoring_target_robot_pose6 = robot_pose6.astype(np.float32).copy()
                rotate_deterministic_delta_robot6 = pose_delta6(robot_pose6, target_robot_pose6)
                gripper_command = float(gripper_state_normalized)
                gripper_policy = f"{gripper_policy}+slot_rotate_deterministic_hold"
            elif slot_planning_request == "pose_only":
                gripper_command = 0.0
                gripper_policy = f"{gripper_policy}+slot_pose_open"
                if task_is_reach:
                    gripper_command = 1.0
                    gripper_policy = f"{gripper_policy}+reach_closed"
                elif vjepa_phase_model and "GRASP" not in set(completed_stages_for_mpc) and not is_release_goal_stage:
                    vjepa_direct_gripper_close_debug = {
                        "allowed": False,
                        "reason": "disabled_use_distance_gated_gripper_search",
                    }
            elif slot_planning_request == "gripper_only":
                slot_gripper_server_downpress_z_robot = float(
                    model_response.get("gripper_execution_downpress_z_robot", 0.0)
                )
                slot_gripper_downpress_base_robot_pose6 = robot_pose6.astype(np.float32).copy()
                target_robot_pose6 = slot_gripper_downpress_base_robot_pose6.copy()
                gripper_delta_for_downpress = float(gripper_command) - float(gripper_state_normalized)
                slot_gripper_execution_downpress_z_robot = (
                    slot_gripper_server_downpress_z_robot if abs(gripper_delta_for_downpress) > 1e-4 else 0.0
                )
                target_robot_pose6[2] += float(slot_gripper_execution_downpress_z_robot)
                commanded_target_robot_pose6 = target_robot_pose6.astype(np.float32).copy()
                gripper_policy = f"{gripper_policy}+slot_gripper_eval_no_downpress_exec_downpress"
        else:
            action = np.asarray(model_response["action"], dtype=np.float32).reshape(-1)
            action_robot_frame = camera_delta_to_robot_delta(action, ref_cam_quat_world, robot_yaw_world)
            target_ee_world6 = apply_camera_delta_to_world_pose6(
                current_ee_world6=_ee_pose_world.astype(np.float32),
                delta_action_camera=action,
                cam_quat_world=ref_cam_quat_world,
            )
            target_robot_pose6 = env._world2robot(target_ee_world6).astype(np.float32)
            predicted_next_state7_camera = None
            model_gripper_target_state = None
            gripper_target_state = None
            gripper_policy = "model_action"
            gripper_command = float(action[6])
        if phase_based_model:
            slot_pressure_hold_allowed = (
                "GRASP" in set(validated_stages)
                or "GRASP" in set(completed_stages_for_mpc)
                or grasp_hold_gripper_command is not None
                or slot_gripper_hold_command is not None
            )
            if (
                slot_gripper_hold_command is not None
                and slot_pressure_hold_allowed
                and not is_release_goal_stage
            ):
                gripper_command = float(slot_gripper_hold_command)
                gripper_policy = f"{gripper_policy}+slot_pressure_hold"
        if task_is_reach and not is_release_goal_stage:
            gripper_command = 1.0
            if "reach_closed" not in str(gripper_policy):
                gripper_policy = f"{gripper_policy}+reach_closed"
        if grasp_hold_gripper_command is not None and not is_release_goal_stage:
            gripper_command_before_grasp_hold = float(gripper_command)
            gripper_command = float(np.clip(max(gripper_command, grasp_hold_gripper_command), 0.0, 1.0))
            gripper_policy = f"{gripper_policy}+post_grasp_hold"
        else:
            gripper_command_before_grasp_hold = None
        put_move_close_z_constraint = None
        if (
            getattr(env, "task_type", None) == "put"
            and current_goal_stage_name in {"MOVE_CLOSE", "SUCCESS", "PLACE_INTO", "PLACE_ONTO"}
            and float(commanded_target_robot_pose6[2]) < float(PUT_MOVE_CLOSE_MIN_Z_ROBOT)
        ):
            original_z = float(commanded_target_robot_pose6[2])
            commanded_target_robot_pose6 = commanded_target_robot_pose6.astype(np.float32).copy()
            commanded_target_robot_pose6[2] = float(PUT_MOVE_CLOSE_MIN_Z_ROBOT)
            target_robot_pose6 = target_robot_pose6.astype(np.float32).copy()
            target_robot_pose6[2] = float(PUT_MOVE_CLOSE_MIN_Z_ROBOT)
            put_move_close_z_constraint = {
                "applied": True,
                "stage": current_goal_stage_name,
                "task_type": getattr(env, "task_type", None),
                "original_z_robot": original_z,
                "min_z_robot": float(PUT_MOVE_CLOSE_MIN_Z_ROBOT),
                "clamped_z_robot": float(commanded_target_robot_pose6[2]),
            }
        new_action = np.concatenate(
            [commanded_target_robot_pose6, np.atleast_1d(np.array([gripper_command], dtype=np.float32))],
            axis=0,
        ).astype(np.float32)
        new_action[-1] = gripper_to_env_command(new_action[-1])
        executed_action_for_record = None
        if model_type != "slot":
            model_response_action = model_response.get("action")
            if model_response_action is not None:
                executed_action_for_record = np.asarray(model_response_action, dtype=np.float32).reshape(-1).copy()
            elif action is not None:
                executed_action_for_record = action.astype(np.float32).copy()
            elif predicted_next_state7_camera is not None:
                predicted_next_state7_camera = np.asarray(
                    predicted_next_state7_camera,
                    dtype=np.float32,
                ).reshape(-1)
                if predicted_next_state7_camera.shape[0] == 7:
                    executed_action_for_record = np.concatenate(
                        [
                            pose_delta6(pose6_for_trace, predicted_next_state7_camera[:6]).astype(np.float32),
                            np.array(
                                [float(predicted_next_state7_camera[6]) - float(gripper_state)],
                                dtype=np.float32,
                            ),
                        ],
                        axis=0,
                    ).astype(np.float32)
        trace_step = success_rollout_trace[t] if success_rollout_trace is not None and t < len(success_rollout_trace) else None
        goal_delta_xyz_camera = None
        action_goal_cosine = None
        if current_goal_target is not None and current_goal_target.get("ee_state7_camera") is not None:
            goal_state7_camera = np.asarray(current_goal_target["ee_state7_camera"], dtype=np.float32)
            goal_delta_xyz_camera = goal_state7_camera[:3] - pose6_for_trace[:3]
            if action is not None:
                action_goal_cosine = cosine_similarity(action[:3], goal_delta_xyz_camera)
        commanded_delta_robot6 = pose_delta6(robot_pose6, commanded_target_robot_pose6)
        slot_gripper_downpress_z_robot = (
            float(slot_gripper_execution_downpress_z_robot)
            if phase_based_model and slot_planning_request == "gripper_only"
            else 0.0
        )
        if rotate_deterministic_active and rotate_deterministic_delta_robot6 is not None:
            deterministic_phase_output = {
                "cem_phase": "deterministic_rotate",
                "view_loss_mode": "score_only",
                "eval_view_names": list(VJEPA_VIEW_ORDER),
                "updates_execution_target": True,
                "target_robot_pose6": commanded_target_robot_pose6.astype(np.float32).copy(),
                "scoring_target_robot_pose6": scoring_target_robot_pose6.astype(np.float32).copy(),
                "delta_pose6_robot": rotate_deterministic_delta_robot6.astype(np.float32).copy(),
                "axis": rotate_axis_name,
                "axis_index": int(rotate_axis_index),
                "step_rad": float(rotate_step_rad),
                "overshoot_rad": float(rotate_overshoot_rad),
                "full_turn_rad": float(DEFAULT_ROTATE_DETERMINISTIC_FULL_TURN_RAD),
                "direction_sign": None
                if rotate_deterministic_direction_sign is None
                else float(rotate_deterministic_direction_sign),
                "accumulated_abs_rad_before_step": float(rotate_deterministic_accumulated_abs_rad),
                "actual_axis_delta_after_step": rotate_deterministic_actual_axis_delta,
                "hold_robot_pose6": None
                if rotate_deterministic_hold_robot_pose6 is None
                else rotate_deterministic_hold_robot_pose6.astype(np.float32).copy(),
                "hold_goal_idx": None
                if rotate_deterministic_hold_goal_idx is None
                else int(rotate_deterministic_hold_goal_idx),
                "raw_goal_axis_error_before_step": rotate_deterministic_raw_goal_axis_error,
                "goal_axis_error_before_step": rotate_deterministic_goal_axis_error,
                "best_loss": model_response.get("best_loss"),
                "current_selection_loss": model_response.get("best_loss"),
                "predicted_loss_improvement": 0.0,
                "noop_due_to_no_predicted_improvement": False,
                "top_pose_candidates": [
                    {
                        "rank": 0,
                        "target_robot_pose6": commanded_target_robot_pose6.astype(np.float32).copy(),
                        "delta_pose6_robot": rotate_deterministic_delta_robot6.astype(np.float32).copy(),
                        "loss": model_response.get("best_loss"),
                    }
                ],
            }
            model_response["cem_phase_outputs"] = [deterministic_phase_output]
        trace_target_delta_robot6 = None
        commanded_vs_trace_delta_cosine_robot_xyz = None
        commanded_to_trace_target_distance_change = None
        if trace_step is not None:
            trace_post_robot_pose6 = np.asarray(trace_step["post"]["robot_pose6"], dtype=np.float32).reshape(-1)
            trace_target_delta_robot6 = pose_delta6(robot_pose6, trace_post_robot_pose6)
            commanded_vs_trace_delta_cosine_robot_xyz = cosine_similarity(
                commanded_delta_robot6[:3],
                trace_target_delta_robot6[:3],
            )
            commanded_to_trace_target_distance_change = compute_target_distance_change(
                current_pose6=robot_pose6,
                commanded_pose6=commanded_target_robot_pose6,
                target_pose6=trace_post_robot_pose6,
            )
        commanded_to_goal_target_distance_change = None
        goal_state7_robot = None
        if current_goal_target is not None and current_goal_target.get("ee_state7_robot") is not None:
            goal_state7_robot = np.asarray(current_goal_target["ee_state7_robot"], dtype=np.float32).reshape(-1)
            commanded_to_goal_target_distance_change = compute_target_distance_change(
                current_pose6=robot_pose6,
                commanded_pose6=commanded_target_robot_pose6,
                target_pose6=goal_state7_robot[:6],
            )
        step_traces.append(
            {
                "step": int(t),
                "mode": f"online_{model_type}_absolute_target",
                "state_frame": "camera",
                "ee_state7": np.concatenate(
                    [pose6_for_trace, np.atleast_1d(np.array(gripper_state, dtype=np.float32))],
                    axis=0,
                ).astype(np.float32),
                "robot_pose6": robot_pose6.astype(np.float32).copy(),
                "main_object": main_object_debug,
                "camera_pose6": camera_pose6.astype(np.float32).copy(),
                "camera_pose6_by_view": {k: v.astype(np.float32).copy() for k, v in camera_pose6_by_view.items()},
                "camera_extrinsics6": camera_extrinsics_by_view["exterior_image_1_left"].astype(np.float32).copy(),
                "camera_frame_convention": camera_frame_convention,
                "camera_extrinsics6_by_view": {
                    k: v.astype(np.float32).copy() for k, v in camera_extrinsics_by_view.items()
                },
                "raw_action": None if action is None else action.copy(),
                "robot_frame_action": None if action_robot_frame is None else action_robot_frame.astype(np.float32).copy(),
                "predicted_next_state7_camera": None
                if predicted_next_state7_camera is None
                else predicted_next_state7_camera.astype(np.float32).copy(),
                "gripper_state_model": float(gripper_state),
                "gripper_state_normalized": float(gripper_state_normalized),
                "gripper_state_raw_width": None if gripper_state_raw_width is None else float(gripper_state_raw_width),
                "slot_model_gripper_units": slot_model_gripper_units,
                "gripper_delta_model": None if action is None else float(action[6]),
                "model_gripper_target_state": None
                if model_gripper_target_state is None
                else float(model_gripper_target_state),
                "gripper_target_state_model": None
                if model_gripper_target_state is None
                else float(model_gripper_target_state),
                "gripper_target_state_normalized": None if gripper_target_state is None else float(gripper_target_state),
                "gripper_policy": gripper_policy,
                "gripper_command_before_grasp_hold": gripper_command_before_grasp_hold,
                "grasp_hold_gripper_command": grasp_hold_gripper_command,
                "grasp_hold_applied": bool(grasp_hold_gripper_command is not None and not is_release_goal_stage),
                "grasp_hold_skipped_for_release": bool(grasp_hold_gripper_command is not None and is_release_goal_stage),
                "put_move_close_z_constraint": put_move_close_z_constraint,
                "slot_executed_gripper_phase": bool(slot_executed_gripper_phase),
                "vjepa_direct_gripper_close_triggered": bool(vjepa_direct_gripper_close_triggered)
                if phase_based_model
                else False,
                "vjepa_direct_gripper_close_debug": vjepa_direct_gripper_close_debug
                if phase_based_model
                else None,
                "slot_gripper_hold_command_before_exec": None
                if slot_gripper_hold_command is None
                else float(slot_gripper_hold_command),
                "slot_release_gripper_until_next_search_before_exec": bool(slot_release_gripper_until_next_search),
                "executed_gripper_delta_model": float(model_gripper_target_state) - float(gripper_state)
                if model_gripper_target_state is not None
                else None,
                "executed_gripper_delta_normalized": float(gripper_command) - float(gripper_state_normalized),
                "gripper_state_raw": float(gripper_state),
                "gripper_delta_raw": None if action is None else float(action[6]),
                "gripper_target_state_raw": None
                if model_gripper_target_state is None
                else float(model_gripper_target_state),
                "gripper_command_env": float(new_action[-1]),
                "slot_gripper_downpress_z_robot": float(slot_gripper_downpress_z_robot),
                "slot_gripper_server_downpress_z_robot": float(slot_gripper_server_downpress_z_robot),
                "slot_gripper_downpress_base_robot_pose6": None
                if slot_gripper_downpress_base_robot_pose6 is None
                else slot_gripper_downpress_base_robot_pose6.astype(np.float32).copy(),
                "commanded_delta_robot6": commanded_delta_robot6.astype(np.float32).copy(),
                "trace_target_delta_robot6": None
                if trace_target_delta_robot6 is None
                else trace_target_delta_robot6.astype(np.float32).copy(),
                "commanded_vs_trace_delta_cosine_robot_xyz": commanded_vs_trace_delta_cosine_robot_xyz,
                "commanded_to_trace_target_distance_change": commanded_to_trace_target_distance_change,
                "goal_state7_robot": None if goal_state7_robot is None else goal_state7_robot.astype(np.float32).copy(),
                "commanded_to_goal_target_distance_change": commanded_to_goal_target_distance_change,
                "server_reference_view": model_response.get("reference_view"),
                "server_best_loss": None if model_response.get("best_loss") is None else float(model_response["best_loss"]),
                "server_best_l1_loss": None
                if model_response.get("best_l1_loss") is None
                else float(model_response["best_l1_loss"]),
                "server_per_view_losses": model_response.get("per_view_losses"),
                "server_per_view_transport_losses": model_response.get("per_view_transport_losses"),
                "server_per_view_objective_losses": model_response.get("per_view_objective_losses"),
                "server_per_view_gripper_slot_transport_losses": model_response.get(
                    "per_view_gripper_slot_transport_losses"
                ),
                "server_per_view_other_slot_mean_transport_losses": model_response.get(
                    "per_view_other_slot_mean_transport_losses"
                ),
                "server_per_view_gripper_mixed_transport_losses": model_response.get(
                    "per_view_gripper_mixed_transport_losses"
                ),
                "server_gripper_slot_index_by_view": model_response.get("gripper_slot_index_by_view"),
                "server_gripper_slot_debug_by_view": model_response.get("gripper_slot_debug_by_view"),
                "server_mask_transport_loss": model_response.get("mask_transport_loss"),
                "candidate_diagnostics_summary": model_response.get("candidate_diagnostics_summary"),
                "cem_phase_outputs": model_response.get("cem_phase_outputs"),
                "candidate_diagnostics": model_response.get("candidate_diagnostics"),
                "slot_planning_phase": slot_current_planning_phase,
                "slot_planning_request": slot_planning_request,
                "rotate_deterministic_active": bool(rotate_deterministic_active),
                "rotate_deterministic_axis": rotate_axis_name,
                "rotate_deterministic_axis_index": int(rotate_axis_index),
                "rotate_deterministic_step_rad": float(rotate_step_rad),
                "rotate_deterministic_overshoot_rad": float(rotate_overshoot_rad),
                "rotate_deterministic_delta_robot6": None
                if rotate_deterministic_delta_robot6 is None
                else rotate_deterministic_delta_robot6.astype(np.float32).copy(),
                "rotate_deterministic_hold_robot_pose6": None
                if rotate_deterministic_hold_robot_pose6 is None
                else rotate_deterministic_hold_robot_pose6.astype(np.float32).copy(),
                "rotate_deterministic_hold_goal_idx": None
                if rotate_deterministic_hold_goal_idx is None
                else int(rotate_deterministic_hold_goal_idx),
                "rotate_deterministic_raw_goal_axis_error": rotate_deterministic_raw_goal_axis_error,
                "rotate_deterministic_goal_axis_error": rotate_deterministic_goal_axis_error,
                "rotate_deterministic_direction_sign": None
                if rotate_deterministic_direction_sign is None
                else float(rotate_deterministic_direction_sign),
                "rotate_deterministic_full_turn_rad": float(DEFAULT_ROTATE_DETERMINISTIC_FULL_TURN_RAD),
                "rotate_deterministic_accumulated_abs_rad": float(
                    rotate_deterministic_accumulated_abs_rad
                ),
                "rotate_deterministic_actual_axis_delta": rotate_deterministic_actual_axis_delta,
                "slot_pre_step_release_summary": slot_pre_step_release_summary,
                "slot_cached_pose_obs_for_release_available": bool(slot_cached_pose_obs_for_release is not None),
                "slot_gripper_obs_updated_after_pose": bool(slot_gripper_obs_updated_after_pose),
                "slot_pose_only_response": slot_pose_only_response,
                "slot_pose_execution_summary": slot_pose_execution_summary,
                "target_ee_world6": None if target_ee_world6 is None else target_ee_world6.astype(np.float32).copy(),
                "target_robot_pose6": target_robot_pose6.astype(np.float32).copy(),
                "scoring_target_robot_pose6": scoring_target_robot_pose6.astype(np.float32).copy(),
                "commanded_target_robot_pose6": commanded_target_robot_pose6.astype(np.float32).copy(),
                "env_action": new_action.astype(np.float32).copy(),
                "trace_step": None if trace_step is None else int(trace_step["step"]),
                "trace_action_source": "slot_rotate_deterministic_axis_step"
                if rotate_deterministic_active
                else f"{model_type}_pose_cem_absolute_target",
                "replay_mode": replay_mode,
                "trace_pre_state7_camera": None
                if trace_step is None
                else np.asarray(trace_step["pre"]["ee_state7_camera"], dtype=np.float32).copy(),
                "trace_post_state7_camera": None
                if trace_step is None
                else np.asarray(trace_step["post"]["ee_state7_camera"], dtype=np.float32).copy(),
                "trace_post_robot_pose6": None
                if trace_step is None
                else np.asarray(trace_step["post"]["robot_pose6"], dtype=np.float32).copy(),
                "goal_idx": int(active_goal_idx),
                "goal_name": None if current_goal_target is None else current_goal_target.get("name"),
                "goal_stage": None if current_goal_target is None else current_goal_target.get("stage"),
                "goal_state7_camera": None
                if current_goal_target is None
                else np.asarray(current_goal_target.get("ee_state7_camera"), dtype=np.float32),
                "goal_delta_xyz_camera": None
                if goal_delta_xyz_camera is None
                else goal_delta_xyz_camera.astype(np.float32).copy(),
                "action_goal_cosine_camera": action_goal_cosine,
                "ee_pos_world": np.array(_ee_pos, dtype=np.float32).copy(),
                "ee_euler_world": np.array(_ee_euler, dtype=np.float32).copy(),
                "camera_pose_world_by_view": {
                    k: {
                        "pos": pos.astype(np.float32).copy(),
                        "quat": quat.astype(np.float32).copy(),
                    }
                    for k, (pos, quat) in view_world_poses.items()
                },
                "gripper_state": float(gripper_state),
            }
        )
        if t == 0:
            trace_state7 = np.concatenate(
                [pose6_for_trace, np.atleast_1d(np.array(gripper_state, dtype=np.float32))],
                axis=0,
            ).astype(np.float32)
            print(
                f"{model_type.upper()} online planning:"
                f" robot_pose6={np.round(robot_pose6, 4).tolist()}"
                f" camera_pose6={np.round(camera_pose6, 4).tolist()}"
                f" camera_extrinsics6={np.round(camera_extrinsics_by_view['exterior_image_1_left'], 4).tolist()}"
                f" ee_state7={np.round(trace_state7, 4).tolist()}"
                f" predicted_action_camera={None if action is None else np.round(action, 4).tolist()}"
                f" robot_frame_action={None if action_robot_frame is None else np.round(action_robot_frame, 4).tolist()}"
                f" predicted_next_state7_camera={None if predicted_next_state7_camera is None else np.round(predicted_next_state7_camera, 4).tolist()}"
                f" best_target_robot_pose6={np.round(commanded_target_robot_pose6, 4).tolist()}"
                f" server_best_loss={model_response.get('best_loss')}"
                f" server_best_l1_loss={model_response.get('best_l1_loss')}"
                f" server_transport={model_response.get('per_view_transport_losses')}"
                f" env_action={np.round(new_action, 4).tolist()}",
                flush=True,
            )

        save_execution_step_images = bool(
            enable_settle_debug
            or rotate_deterministic_active
            or (phase_based_model and slot_planning_request == "gripper_only")
        )
        settle_vis_dir = (
            os.path.join(task_debug_dir, "settle_steps", f"{t:04d}")
            if save_execution_step_images
            else None
        )
        t_execute_start = time.perf_counter()
        freeze_slot_pose_arm_on_contact = bool(
            model_type == "slot"
            and slot_planning_request == "pose_only"
            and getattr(env, "task_type", None) == "pick"
            and "pick_can_of_soda" in str(task_cfg_path)
        )
        if phase_based_model and slot_planning_request == "gripper_only":
            slot_gripper_release_restore_pose6 = robot_pose6.astype(np.float32).copy()
            execution_result = execute_gripper_only(
                env=env,
                gripper_command=gripper_command,
                steps=max(1, int(max_settle_steps)),
                hold_robot_pose6=slot_gripper_downpress_base_robot_pose6
                if slot_gripper_downpress_base_robot_pose6 is not None
                else robot_pose6,
                downpress_z_robot=float(slot_gripper_execution_downpress_z_robot),
                vis_dir=settle_vis_dir,
                cancel_event=cancel_event,
            )
            execution_result["gripper_only_execution"] = True
        else:
            execution_result = execute_absolute_robot_target(
                env=env,
                target_robot_pose6=commanded_target_robot_pose6,
                gripper_command=gripper_command,
                max_settle_steps=max_settle_steps,
                settle_pos_tol=settle_pos_tol,
                settle_rot_tol=settle_rot_tol,
                vis_dir=settle_vis_dir,
                include_controller_debug=enable_settle_debug,
                cancel_event=cancel_event,
                freeze_arm_on_contact=freeze_slot_pose_arm_on_contact,
                contact_stop_checker=(
                    (
                        lambda current_obs: (
                            bool(env.check_touch_condition(current_obs))
                            or bool(env.check_collisions()[1])
                        )
                    )
                    if freeze_slot_pose_arm_on_contact
                    else None
                ),
                retreat_on_contact=freeze_slot_pose_arm_on_contact,
            )
            execution_result["gripper_only_execution"] = False
        timing_execute_target_s = time.perf_counter() - t_execute_start
        if cancel_requested():
            print(f"Planning cancelled after target execution at step {t}.", flush=True)
            cancelled = True
            break
        obs = execution_result["obs"]
        curr_task_progression = execution_result["task_progression"]
        final_task_progression = float(curr_task_progression)
        terminated = execution_result["terminated"]
        truncated = execution_result["truncated"]
        info = execution_result["info"]
        settle_history = execution_result["settle_history"]
        selected_goal_reached_now = selected_goal_stage_reached(
            env,
            goal_state,
            curr_task_progression,
            goal_idx=active_goal_idx,
            obs=obs,
            validated_stages=validated_stages,
        )
        next_ee_pos, next_ee_rot = env.get_ee_pose()
        next_ee_pos_np = next_ee_pos.cpu().numpy() if hasattr(next_ee_pos, "cpu") else np.array(next_ee_pos)
        next_ee_rot_np = next_ee_rot.cpu().numpy() if hasattr(next_ee_rot, "cpu") else np.array(next_ee_rot)
        next_ee_euler_np = Rot.from_quat(next_ee_rot_np).as_euler("xyz")
        next_ee_pose_world = np.concatenate([next_ee_pos_np, next_ee_euler_np]).astype(np.float32)
        next_robot_pose6 = env._world2robot(next_ee_pose_world).astype(np.float32)
        next_camera_pose6 = world_pose_to_camera_pose6(
            next_ee_pose_world,
            view_world_poses["exterior_image_1_left"][0],
            view_world_poses["exterior_image_1_left"][1],
            camera_frame_convention=camera_frame_convention,
        )
        if rotate_deterministic_active:
            rotate_deterministic_actual_axis_delta = float(
                abs(angle_wrap(next_robot_pose6[rotate_axis_index] - robot_pose6[rotate_axis_index]))
            )
            rotate_deterministic_accumulated_abs_rad = min(
                float(DEFAULT_ROTATE_DETERMINISTIC_FULL_TURN_RAD),
                float(rotate_deterministic_accumulated_abs_rad)
                + rotate_deterministic_actual_axis_delta,
            )
            if (
                rotate_deterministic_accumulated_abs_rad
                < float(DEFAULT_ROTATE_DETERMINISTIC_FULL_TURN_RAD) - 1.0e-4
            ):
                selected_goal_reached_now = False
            if step_traces:
                step_traces[-1]["rotate_deterministic_actual_axis_delta"] = (
                    rotate_deterministic_actual_axis_delta
                )
                step_traces[-1]["rotate_deterministic_accumulated_abs_rad"] = float(
                    rotate_deterministic_accumulated_abs_rad
                )
        actual_delta_xyz_camera = next_camera_pose6[:3] - pose6_for_trace[:3]
        next_main_object_debug = get_main_object_pose_debug(env)
        step_traces[-1]["next_robot_pose6"] = next_robot_pose6.copy()
        step_traces[-1]["next_camera_pose6"] = next_camera_pose6.astype(np.float32).copy()
        step_traces[-1]["actual_delta_xyz_camera"] = actual_delta_xyz_camera.astype(np.float32).copy()
        step_traces[-1]["next_main_object"] = next_main_object_debug
        gripper_downpress_base = step_traces[-1].get("slot_gripper_downpress_base_robot_pose6")
        if gripper_downpress_base is not None:
            gripper_downpress_base = np.asarray(gripper_downpress_base, dtype=np.float32).reshape(6)
            gripper_downpress_actual_delta6 = pose_delta6(gripper_downpress_base, next_robot_pose6)
            gripper_downpress_commanded_delta6 = pose_delta6(
                gripper_downpress_base,
                commanded_target_robot_pose6,
            )
            step_traces[-1]["slot_gripper_downpress_actual_delta6_robot"] = (
                gripper_downpress_actual_delta6.astype(np.float32).copy()
            )
            step_traces[-1]["slot_gripper_downpress_commanded_delta6_robot"] = (
                gripper_downpress_commanded_delta6.astype(np.float32).copy()
            )
            step_traces[-1]["slot_gripper_downpress_actual_z_robot"] = float(
                gripper_downpress_actual_delta6[2]
            )
            step_traces[-1]["slot_gripper_downpress_actual_xy_l2_robot"] = float(
                np.linalg.norm(gripper_downpress_actual_delta6[:2])
            )
            step_traces[-1]["slot_gripper_downpress_actual_rot_l2_robot"] = float(
                np.linalg.norm(gripper_downpress_actual_delta6[3:6])
            )
        else:
            step_traces[-1]["slot_gripper_downpress_actual_delta6_robot"] = None
            step_traces[-1]["slot_gripper_downpress_commanded_delta6_robot"] = None
            step_traces[-1]["slot_gripper_downpress_actual_z_robot"] = None
            step_traces[-1]["slot_gripper_downpress_actual_xy_l2_robot"] = None
            step_traces[-1]["slot_gripper_downpress_actual_rot_l2_robot"] = None
        step_traces[-1]["action_repeat"] = int(action_repeat)
        step_traces[-1]["max_settle_steps"] = int(max_settle_steps)
        step_traces[-1]["settle_pos_tol"] = float(settle_pos_tol)
        step_traces[-1]["settle_rot_tol"] = float(settle_rot_tol)
        step_traces[-1]["settle_history"] = settle_history
        step_traces[-1]["settle_vis_dir"] = settle_vis_dir
        step_traces[-1]["enable_settle_debug"] = bool(enable_settle_debug)
        step_traces[-1]["used_fast_inner_step"] = bool(execution_result["used_fast_inner_step"])
        step_traces[-1]["fast_inner_steps_executed"] = int(execution_result["fast_inner_steps_executed"])
        step_traces[-1]["full_inner_steps_executed"] = int(execution_result["full_inner_steps_executed"])
        step_traces[-1]["fast_inner_fallback_steps"] = int(execution_result["fast_inner_fallback_steps"])
        step_traces[-1]["gripper_only_execution"] = bool(execution_result.get("gripper_only_execution", False))
        step_traces[-1]["gripper_only_used_fallback_absolute_pose"] = bool(
            execution_result.get("used_fallback_absolute_pose", False)
        )
        step_traces[-1]["gripper_only_target_joint_pos"] = execution_result.get("gripper_target_joint_pos")
        step_traces[-1]["gripper_only_dof_idx"] = execution_result.get("gripper_dof_idx")
        step_traces[-1]["gripper_only_actual_joint_pos"] = execution_result.get("gripper_actual_joint_pos")
        step_traces[-1]["gripper_only_execution_mode"] = execution_result.get("gripper_execution_mode")
        step_traces[-1]["gripper_only_env_action"] = execution_result.get("env_action")
        step_traces[-1]["gripper_only_initial_env_action"] = execution_result.get("initial_env_action")
        step_traces[-1]["gripper_only_adaptive_downpress_release"] = bool(
            execution_result.get("adaptive_downpress_release", False)
        )
        step_traces[-1]["gripper_only_adaptive_downpress_release_step"] = execution_result.get(
            "adaptive_downpress_release_step"
        )
        step_traces[-1]["gripper_only_adaptive_downpress_actual_z_robot"] = execution_result.get(
            "adaptive_downpress_actual_z_robot"
        )
        step_traces[-1]["gripper_only_adaptive_downpress_required_z_robot"] = execution_result.get(
            "adaptive_downpress_required_z_robot"
        )
        step_traces[-1]["arm_frozen_on_contact"] = bool(execution_result.get("arm_frozen_on_contact", False))
        step_traces[-1]["arm_frozen_contact_step"] = execution_result.get("arm_frozen_contact_step")
        step_traces[-1]["arm_frozen_contact_pose6"] = execution_result.get("arm_frozen_contact_pose6")
        step_traces[-1]["arm_frozen_contact_reason"] = execution_result.get("arm_frozen_contact_reason")
        step_traces[-1]["gripper_only_controller_control"] = (
            info.get("gripper_controller_control") if isinstance(info, dict) else None
        )
        step_traces[-1]["gripper_only_rendered_after_step"] = bool(
            execution_result.get("gripper_rendered_after_step", False)
        )
        step_traces[-1]["contact_safety_active"] = bool(execution_result.get("contact_safety_active", False))
        step_traces[-1]["contact_safety_steps"] = int(execution_result.get("contact_safety_steps", 0))
        step_traces[-1]["contact_safety_effort_scale"] = float(
            execution_result.get("contact_safety_effort_scale", DEFAULT_CONTACT_SAFETY_EFFORT_SCALE)
        )
        final_err6, final_pos_l2, final_rot_l2 = pose_error_components(commanded_target_robot_pose6, next_robot_pose6)
        step_traces[-1]["final_target_error6_robot"] = final_err6.copy()
        step_traces[-1]["final_target_pos_l2_robot"] = float(final_pos_l2)
        step_traces[-1]["final_target_rot_l2_robot"] = float(final_rot_l2)
        step_traces[-1]["settle_steps_used"] = len(settle_history)
        step_traces[-1]["selected_goal_reached_now"] = bool(selected_goal_reached_now)
        step_traces[-1]["action_actual_cosine_camera"] = (
            None if action is None else cosine_similarity(action[:3], actual_delta_xyz_camera)
        )
        step_traces[-1]["actual_goal_cosine_camera"] = (
            None
            if goal_delta_xyz_camera is None
            else cosine_similarity(actual_delta_xyz_camera, goal_delta_xyz_camera)
        )

        next_view_world_poses = get_view_world_poses(env, task_cfg)
        next_camera_extrinsics_by_view = {}
        next_camera_pose6_by_view = {}
        for view_name, (cam_pos_world, cam_quat_world) in next_view_world_poses.items():
            camera_world_pose6 = get_camera_extrinsics_pose6(
                cam_pos_world,
                cam_quat_world,
                camera_frame_convention=camera_frame_convention,
            )
            next_camera_extrinsics_by_view[view_name] = env._world2robot(camera_world_pose6).astype(np.float32)
            next_camera_pose6_by_view[view_name] = pose_in_parent_frame(
                next_robot_pose6.astype(np.float32),
                next_camera_extrinsics_by_view[view_name],
            )
        t_post_obs_extract_start = time.perf_counter()
        next_base_im, _, next_base_im_second, _, next_wrist_im, _, next_gripper_state = extract_from_obs(
            obs, robot_name=env.robot.name
        )
        next_raw_gripper_state = get_raw_gripper_position_from_obs(obs, robot_name=env.robot.name)
        next_gripper_state_normalized = float(np.clip(float(next_gripper_state), 0.0, 1.0))
        next_gripper_state_raw_width = None
        if model_type == "slot" and next_raw_gripper_state is not None:
            next_gripper_state_raw_width = float(np.clip(next_raw_gripper_state, 0.0, 0.05))
            next_gripper_state_normalized = normalize_raw_gripper_position(next_raw_gripper_state)
            next_gripper_state = gripper_state_for_model(next_raw_gripper_state, slot_model_gripper_units)
        step_traces[-1]["next_gripper_state_model"] = float(next_gripper_state)
        step_traces[-1]["next_gripper_state_normalized"] = float(next_gripper_state_normalized)
        step_traces[-1]["next_gripper_state_raw_width"] = (
            None if next_gripper_state_raw_width is None else float(next_gripper_state_raw_width)
        )
        step_traces[-1]["next_gripper_state_raw"] = float(next_gripper_state)
        if phase_based_model and (slot_executed_gripper_phase or vjepa_direct_gripper_close_triggered):
            gripper_actual_closure_delta = float(next_gripper_state_normalized) - float(gripper_state_normalized)
            # Keep the old closure-blocked heuristic only as a diagnostic. A failed actuator
            # command can also look like "commanded close but measured gripper did not close",
            # so pressure / hold should use the same physical grasp criterion as GRASP.
            gripper_closure_blocked = (
                gripper_command is not None
                and float(gripper_command) > 0.05
                and (float(gripper_command) - float(next_gripper_state_normalized)) > 0.05
                and float(next_gripper_state_normalized) > 0.02
                and gripper_actual_closure_delta > 0.005
            )
            gripper_grasp_condition = bool(_safe_current_grasping(env, obs))
            gripper_pressure_detected = gripper_grasp_condition
            if gripper_pressure_detected:
                # Once the physical gripper reports pressure / grasp contact,
                # keep that closure through subsequent pose steps. Otherwise the
                # next pose search's default open command immediately drops the
                # object before GRASP validation can stabilize it.
                slot_gripper_hold_command = float(
                    np.clip(
                        max(float(gripper_command), float(next_gripper_state_normalized)),
                        0.0,
                        1.0,
                    )
                )
                slot_release_gripper_until_next_search = False
                step_traces[-1]["slot_gripper_released_after_no_pressure"] = False
                step_traces[-1]["slot_gripper_release_deferred_to_next_pose"] = bool(
                    slot_release_gripper_until_next_search
                )
                step_traces[-1]["slot_gripper_pressure_hold_deferred_until_grasp"] = False
            else:
                slot_gripper_hold_command = None
                slot_release_gripper_until_next_search = bool(
                    gripper_command is not None and float(gripper_command) > 0.05 and not is_release_goal_stage
                )
                step_traces[-1]["slot_gripper_released_after_no_pressure"] = False
                step_traces[-1]["slot_gripper_release_deferred_to_next_pose"] = bool(
                    slot_release_gripper_until_next_search
                )
                step_traces[-1]["slot_gripper_release_steps"] = 0
                step_traces[-1]["slot_gripper_pressure_hold_deferred_until_grasp"] = False
            slot_next_planning_phase = "pose"
            step_traces[-1]["slot_gripper_pressure_detected"] = bool(gripper_pressure_detected)
            step_traces[-1]["slot_gripper_pressure_grasp_condition"] = bool(gripper_grasp_condition)
            step_traces[-1]["slot_gripper_pressure_closure_blocked"] = False
            step_traces[-1]["slot_gripper_pressure_closure_blocked_diagnostic"] = bool(gripper_closure_blocked)
            step_traces[-1]["slot_gripper_actual_closure_delta"] = float(gripper_actual_closure_delta)
            step_traces[-1]["slot_gripper_activation_stability"] = None
            step_traces[-1]["slot_gripper_hold_command_after_exec"] = (
                None if slot_gripper_hold_command is None else float(slot_gripper_hold_command)
            )
            step_traces[-1]["slot_release_gripper_until_next_search_after_exec"] = bool(
                slot_release_gripper_until_next_search
            )
        elif phase_based_model:
            if task_is_reach:
                slot_next_planning_phase = "pose"
                slot_cached_pose_obs_for_release = None
                slot_gripper_release_restore_pose6 = None
                step_traces[-1]["slot_gripper_activation_stability"] = {
                    "stable": False,
                    "reason": "reach_task_gripper_search_disabled",
                }
            elif vjepa_phase_model:
                gripper_distance_gate = get_vjepa_gripper_search_distance_gate_debug(
                    env=env,
                    current_goal_target=current_goal_target,
                )
                if (
                    slot_planning_request == "pose_only"
                    and "GRASP" not in set(completed_stages_for_mpc)
                    and gripper_distance_gate["allowed"]
                ):
                    slot_next_planning_phase = "gripper"
                    slot_cached_pose_obs_for_release = obs
                    slot_gripper_release_restore_pose6 = None
                else:
                    slot_next_planning_phase = "pose"
                    slot_cached_pose_obs_for_release = None
                    slot_gripper_release_restore_pose6 = None
                step_traces[-1]["slot_gripper_activation_stability"] = {
                    "stable": bool(gripper_distance_gate["allowed"]),
                    "reason": gripper_distance_gate.get("reason"),
                    "distance_to_finger_midpoint_m": gripper_distance_gate.get("distance_to_finger_midpoint_m"),
                    "max_distance_threshold_m": gripper_distance_gate.get("max_distance_threshold_m"),
                    "mode": "vjepa_distance_gate",
                }
                step_traces[-1]["slot_gripper_activation_height_gate"] = gripper_distance_gate
            elif slot_planning_request == "pose_only" and "GRASP" not in set(completed_stages_for_mpc):
                gripper_stability = slot_three_view_losses_stable_for_gripper(step_traces)
                gripper_height_gate = get_gripper_search_height_gate_debug(
                    env=env,
                    robot_pose6=robot_pose6,
                    current_goal_target=current_goal_target,
                )
                if gripper_stability["stable"] and gripper_height_gate["allowed"]:
                    slot_next_planning_phase = "gripper"
                    slot_cached_pose_obs_for_release = obs
                    slot_gripper_release_restore_pose6 = None
                else:
                    slot_next_planning_phase = "pose"
                    slot_cached_pose_obs_for_release = None
                    slot_gripper_release_restore_pose6 = None
                step_traces[-1]["slot_gripper_activation_stability"] = gripper_stability
                step_traces[-1]["slot_gripper_activation_height_gate"] = gripper_height_gate
            else:
                slot_next_planning_phase = "pose"
                if "GRASP" in set(completed_stages_for_mpc) or is_release_goal_stage:
                    slot_cached_pose_obs_for_release = None
                    slot_gripper_release_restore_pose6 = None
                step_traces[-1]["slot_gripper_activation_stability"] = None
                step_traces[-1]["slot_gripper_activation_height_gate"] = None
            step_traces[-1]["slot_gripper_pressure_detected"] = None
            step_traces[-1]["slot_gripper_pressure_grasp_condition"] = None
            step_traces[-1]["slot_gripper_pressure_closure_blocked"] = None
            step_traces[-1]["slot_gripper_pressure_closure_blocked_diagnostic"] = None
            step_traces[-1]["slot_gripper_actual_closure_delta"] = None
            step_traces[-1]["slot_gripper_released_after_no_pressure"] = False
            step_traces[-1]["slot_gripper_release_deferred_to_next_pose"] = False
            step_traces[-1]["slot_gripper_release_steps"] = 0
            step_traces[-1]["slot_gripper_hold_command_after_exec"] = (
                None if slot_gripper_hold_command is None else float(slot_gripper_hold_command)
            )
            step_traces[-1]["slot_release_gripper_until_next_search_after_exec"] = bool(
                slot_release_gripper_until_next_search
            )
        if phase_based_model:
            step_traces[-1]["slot_next_planning_phase_after_exec"] = slot_next_planning_phase
            step_traces[-1]["slot_cached_pose_obs_for_release_after_exec"] = bool(
                slot_cached_pose_obs_for_release is not None
            )
            step_traces[-1]["slot_gripper_release_restore_pose6_after_exec"] = (
                None
                if slot_gripper_release_restore_pose6 is None
                else slot_gripper_release_restore_pose6.astype(np.float32).copy()
            )
        timing_post_obs_extract_s = time.perf_counter() - t_post_obs_extract_start
        pre_obs_payload = build_multiview_obs_payload(
            base_im=base_im,
            base_im_second=base_im_second,
            wrist_im=wrist_im,
            gripper_state=gripper_state,
            camera_pose6_by_view=camera_pose6_by_view,
            camera_extrinsics_by_view=camera_extrinsics_by_view,
            camera_frame_convention=camera_frame_convention,
        )
        post_obs_payload = build_multiview_obs_payload(
            base_im=next_base_im,
            base_im_second=next_base_im_second,
            wrist_im=next_wrist_im,
            gripper_state=next_gripper_state,
            camera_pose6_by_view=next_camera_pose6_by_view,
            camera_extrinsics_by_view=next_camera_extrinsics_by_view,
            camera_frame_convention=camera_frame_convention,
        )
        t_save_step_images_start = time.perf_counter()
        slot_mask_panels = (
            get_slot_mask_panels_for_payload(client, post_obs_payload)
            if model_type == "slot"
            else {}
        )
        image_writer.submit(
            os.path.join(task_debug_dir, "base_cam", f"{t:04d}.png"),
            append_slot_mask_panel(next_base_im, slot_mask_panels.get("exterior_image_1_left")),
        )
        image_writer.submit(
            os.path.join(task_debug_dir, "second_cam", f"{t:04d}.png"),
            append_slot_mask_panel(next_base_im_second, slot_mask_panels.get("exterior_image_2_left")),
        )
        image_writer.flush()
        save_wrist_image_with_cem_phase_outputs(
            os.path.join(task_debug_dir, "wrist_cam", f"{t:04d}.png"),
            append_slot_mask_panel(next_wrist_im, slot_mask_panels.get("wrist_image_left")),
            model_response,
        )
        timing_save_step_images_s += time.perf_counter() - t_save_step_images_start
        step_traces[-1]["saved_observation_timing"] = "post_execution"
        t_record_start = time.perf_counter()
        try:
            record_payload = {
                "pre_obs": pre_obs_payload,
                "post_obs": post_obs_payload,
            }
            if executed_action_for_record is not None:
                record_payload["action"] = executed_action_for_record.astype(np.float32)
            record_response = client.record_step_with_action(record_payload)
            step_traces[-1][f"{model_type}_record_response"] = record_response
            (
                observed_per_view_losses,
                observed_per_view_transport_losses,
                observed_per_view_objective_losses,
            ) = _observed_loss_maps_from_record_response(record_response)
            step_traces[-1]["observed_per_view_losses"] = observed_per_view_losses
            step_traces[-1]["observed_per_view_transport_losses"] = observed_per_view_transport_losses
            step_traces[-1]["observed_per_view_objective_losses"] = observed_per_view_objective_losses
        except Exception as e:
            print(f"Teacher-forced latent recording issue: {e}", flush=True)
            step_traces[-1][f"{model_type}_record_error"] = str(e)
        timing_record_step_s = time.perf_counter() - t_record_start

        if settle_vis_dir is not None:
            create_settle_visual_summary(
                task_debug_dir=task_debug_dir,
                settle_vis_dir=settle_vis_dir,
                settle_history=settle_history,
                output_path=os.path.join(settle_vis_dir, "settle_summary.png"),
            )

        if env.task_progression is not None:
            for stage in env.task_progression.keys():
                # A single contact-based GRASP signal is only a candidate. For
                # thin / easy-to-push objects it can fire while the gripper is
                # merely touching the object. Treat GRASP as reached only after
                # the explicit post-grasp lift validation below succeeds.
                if stage == "GRASP" and stage not in validated_stages:
                    continue
                if physical_stage_reached(
                    env,
                    stage,
                    curr_task_progression,
                    obs=obs,
                    validated_stages=validated_stages,
                ) and stage not in reached_stages:
                    validated_stages.add(stage)
                    reached_stages.add(stage)
                    if stage in LOGGED_PHYSICAL_STAGES:
                        Image.fromarray(next_base_im).save(os.path.join(task_debug_dir, f"reached_stage_{stage}.png"))
                        print(f"Reached physical stage: {stage} at step {t}!", flush=True)

        post_move_release_from_success_goal = False
        if (
            current_goal_stage_name == "SUCCESS"
            and task_has_post_carry_release_stage(env)
            and "MOVE_CLOSE" in set(validated_stages)
            and not any(
                physical_stage_reached(
                    env,
                    post_stage,
                    curr_task_progression,
                    obs=obs,
                    validated_stages=validated_stages,
                )
                for post_stage in POST_CARRY_RELEASE_STAGES
            )
        ):
            # Some success-case manifests only expose GRASP + SUCCESS goals.
            # In those cases SUCCESS means "move close, then release"; once the
            # physical MOVE_CLOSE stage is validated, run the same forced release
            # path used by explicit MOVE_CLOSE targets instead of continuing CEM.
            post_move_release_from_success_goal = True
            selected_goal_reached_now = True
            step_traces[-1]["selected_goal_reached_now"] = True
            step_traces[-1]["selected_goal_reached_by_success_goal_move_close"] = True

        if selected_goal_reached_now:
            target_names = list(goal_state.get("target_names", []))
            target_stages = list(goal_state.get("target_stages", []))
            reached_stage = target_stages[active_goal_idx] if active_goal_idx < len(target_stages) else "UNKNOWN"
            reached_name = target_names[active_goal_idx] if active_goal_idx < len(target_names) else f"goal_{active_goal_idx}"
            stop_after_move_close_release = False
            post_move_close_extra_step_traces = []
            if reached_stage == "GRASP" and not post_grasp_tightened:
                Image.fromarray(next_base_im).save(os.path.join(task_debug_dir, "reached_stage_GRASP.png"))
                Image.fromarray(next_base_im_second).save(
                    os.path.join(task_debug_dir, "reached_stage_GRASP_second.png")
                )
                Image.fromarray(next_wrist_im).save(os.path.join(task_debug_dir, "reached_stage_GRASP_wrist.png"))
                step_traces[-1]["reached_stage_grasp_saved_images"] = {
                    "base": "reached_stage_GRASP.png",
                    "second": "reached_stage_GRASP_second.png",
                    "wrist": "reached_stage_GRASP_wrist.png",
                    "timing": "pre_post_grasp_tighten_and_lift",
                }
                post_grasp_rollback_sim_state = None
                post_grasp_rollback_python_state = None
                try:
                    post_grasp_rollback_sim_state = save_sim_state(og)
                    post_grasp_rollback_python_state = save_python_task_state(env)
                    step_traces[-1]["post_grasp_rollback_state_saved"] = True
                except Exception as exc:
                    step_traces[-1]["post_grasp_rollback_state_saved"] = False
                    step_traces[-1]["post_grasp_rollback_state_error"] = str(exc)
                    print(f"Warning: failed to snapshot sim before post-grasp lift: {exc}", flush=True)
                t_tighten_start = time.perf_counter()
                pre_post_grasp_robot_pose6 = next_robot_pose6.astype(np.float32).copy()
                tighten_target_robot_pose6 = next_robot_pose6.astype(np.float32).copy()
                tighten_target_robot_pose6[2] += float(POST_GRASP_DOWNWARD_OFFSET_Z)
                tighten_result = execute_absolute_robot_target(
                    env=env,
                    target_robot_pose6=tighten_target_robot_pose6,
                    gripper_command=POST_GRASP_CLOSE_COMMAND,
                    max_settle_steps=POST_GRASP_CLOSE_STEPS,
                    settle_pos_tol=settle_pos_tol,
                    settle_rot_tol=settle_rot_tol,
                    vis_dir=None,
                    include_controller_debug=False,
                    force_full_steps=True,
                    cancel_event=cancel_event,
                )
                if cancel_requested():
                    print(f"Planning cancelled during post-grasp tighten at step {t}.", flush=True)
                    cancelled = True
                    break
                post_grasp_tightened = True
                obs = tighten_result["obs"]
                curr_task_progression = tighten_result["task_progression"]
                final_task_progression = float(curr_task_progression)
                post_tighten_task_progression = float(curr_task_progression)
                tighten_duration_s = float(time.perf_counter() - t_tighten_start)
                post_tighten_raw_gripper = get_raw_gripper_position_from_obs(obs, robot_name=env.robot.name)
                if post_tighten_raw_gripper is None:
                    post_tighten_gripper_normalized = POST_GRASP_CLOSE_COMMAND
                else:
                    post_tighten_gripper_normalized = normalize_raw_gripper_position(post_tighten_raw_gripper)
                (
                    post_tighten_base_im,
                    _,
                    post_tighten_base_im_second,
                    _,
                    post_tighten_wrist_im,
                    _,
                    _,
                ) = extract_from_obs(obs, robot_name=env.robot.name)
                post_tighten_image_paths = {
                    "base": os.path.join("base_cam", f"post_grasp_tighten_step{t:04d}.png"),
                    "second": os.path.join("second_cam", f"post_grasp_tighten_step{t:04d}.png"),
                    "wrist": os.path.join("wrist_cam", f"post_grasp_tighten_step{t:04d}.png"),
                }
                image_writer.submit(os.path.join(task_debug_dir, post_tighten_image_paths["base"]), post_tighten_base_im)
                image_writer.submit(
                    os.path.join(task_debug_dir, post_tighten_image_paths["second"]),
                    post_tighten_base_im_second,
                )
                image_writer.submit(
                    os.path.join(task_debug_dir, post_tighten_image_paths["wrist"]),
                    post_tighten_wrist_im,
                )
                post_tighten_ee_pos, post_tighten_ee_rot = env.get_ee_pose()
                post_tighten_ee_pos_np = (
                    post_tighten_ee_pos.cpu().numpy()
                    if hasattr(post_tighten_ee_pos, "cpu")
                    else np.asarray(post_tighten_ee_pos)
                )
                post_tighten_ee_rot_np = (
                    post_tighten_ee_rot.cpu().numpy()
                    if hasattr(post_tighten_ee_rot, "cpu")
                    else np.asarray(post_tighten_ee_rot)
                )
                post_tighten_ee_euler_np = Rot.from_quat(post_tighten_ee_rot_np).as_euler("xyz")
                post_tighten_ee_pose_world = np.concatenate(
                    [post_tighten_ee_pos_np, post_tighten_ee_euler_np]
                ).astype(np.float32)
                post_tighten_robot_pose6 = env._world2robot(post_tighten_ee_pose_world).astype(np.float32)
                post_grasp_validation_stage = get_post_grasp_validation_stage(env)
                if post_grasp_validation_stage is None:
                    next_robot_pose6 = post_tighten_robot_pose6.astype(np.float32).copy()
                    grasp_hold_gripper_command = float(
                        np.clip(post_tighten_gripper_normalized + POST_GRASP_HOLD_EXTRA_CLOSE, 0.0, 1.0)
                    )
                    image_writer.flush()
                    step_traces[-1]["post_grasp_tighten"] = {
                        "gripper_command_normalized": float(POST_GRASP_CLOSE_COMMAND),
                        "downward_offset_z_robot": float(POST_GRASP_DOWNWARD_OFFSET_Z),
                        "target_robot_pose6_before_downward_offset": pre_post_grasp_robot_pose6.astype(np.float32).copy(),
                        "target_robot_pose6": tighten_target_robot_pose6.astype(np.float32).copy(),
                        "settle_steps": int(POST_GRASP_CLOSE_STEPS),
                        "force_full_steps": True,
                        "duration_s": float(tighten_duration_s),
                        "task_progression": float(post_tighten_task_progression),
                        "settle_history": tighten_result["settle_history"],
                        "env_action": tighten_result["env_action"].copy(),
                        "post_tighten_gripper_normalized": float(post_tighten_gripper_normalized),
                        "post_tighten_raw_gripper": None
                        if post_tighten_raw_gripper is None
                        else float(post_tighten_raw_gripper),
                        "hold_extra_close": float(POST_GRASP_HOLD_EXTRA_CLOSE),
                        "grasp_hold_gripper_command": float(grasp_hold_gripper_command),
                        "saved_images": post_tighten_image_paths,
                        "validation_stage": None,
                    }
                    step_traces[-1]["post_grasp_lift"] = {
                        "skipped": True,
                        "skip_reason": "task_has_no_lift_validation_stage_after_grasp",
                        "validation_stage": None,
                        "lift_succeeded": True,
                    }
                    validated_stages.add("GRASP")
                    reached_stages.add("GRASP")
                    print(
                        f"Selected GRASP reached at step {t}; task has no post-GRASP lift validation stage, "
                        f"holding future gripper at {grasp_hold_gripper_command:.3f}.",
                        flush=True,
                    )
                    post_grasp_lift_succeeded = True
                    post_lift_robot_pose6 = post_tighten_robot_pose6.astype(np.float32).copy()
                    post_lift_gripper_normalized = post_tighten_gripper_normalized
                    post_lift_raw_gripper = post_tighten_raw_gripper
                    post_lift_image_paths = {}
                    post_lift_base_im = post_tighten_base_im
                    lift_target_robot_pose6 = post_tighten_robot_pose6.astype(np.float32).copy()
                    post_grasp_lift_offset_z = 0.0
                    t_lift_start = time.perf_counter()
                    lift_result = {
                        "settle_history": [],
                        "env_action": tighten_result["env_action"].copy(),
                    }
                else:
                    lift_target_robot_pose6 = post_tighten_robot_pose6.astype(np.float32).copy()
                    post_grasp_lift_offset_z = get_post_grasp_lift_offset_z(post_grasp_validation_stage)
                    lift_target_robot_pose6[2] += float(post_grasp_lift_offset_z)
                    t_lift_start = time.perf_counter()
                    lift_result = execute_absolute_robot_target(
                        env=env,
                        target_robot_pose6=lift_target_robot_pose6,
                        gripper_command=POST_GRASP_CLOSE_COMMAND,
                        max_settle_steps=POST_GRASP_LIFT_STEPS,
                        settle_pos_tol=settle_pos_tol,
                        settle_rot_tol=settle_rot_tol,
                        vis_dir=None,
                        include_controller_debug=False,
                        force_full_steps=True,
                        cancel_event=cancel_event,
                    )
                    if cancel_requested():
                        print(f"Planning cancelled during post-grasp lift at step {t}.", flush=True)
                        cancelled = True
                        break
                    obs = lift_result["obs"]
                    curr_task_progression = lift_result["task_progression"]
                    final_task_progression = float(curr_task_progression)
                    post_lift_ee_pos, post_lift_ee_rot = env.get_ee_pose()
                    post_lift_ee_pos_np = (
                        post_lift_ee_pos.cpu().numpy()
                        if hasattr(post_lift_ee_pos, "cpu")
                        else np.asarray(post_lift_ee_pos)
                    )
                    post_lift_ee_rot_np = (
                        post_lift_ee_rot.cpu().numpy()
                        if hasattr(post_lift_ee_rot, "cpu")
                        else np.asarray(post_lift_ee_rot)
                    )
                    post_lift_ee_euler_np = Rot.from_quat(post_lift_ee_rot_np).as_euler("xyz")
                    post_lift_ee_pose_world = np.concatenate([post_lift_ee_pos_np, post_lift_ee_euler_np]).astype(
                        np.float32
                    )
                    post_lift_robot_pose6 = env._world2robot(post_lift_ee_pose_world).astype(np.float32)
                    next_robot_pose6 = post_lift_robot_pose6.astype(np.float32).copy()
                    post_lift_raw_gripper = get_raw_gripper_position_from_obs(obs, robot_name=env.robot.name)
                    if post_lift_raw_gripper is None:
                        post_lift_gripper_normalized = POST_GRASP_CLOSE_COMMAND
                    else:
                        post_lift_gripper_normalized = normalize_raw_gripper_position(post_lift_raw_gripper)
                    (
                        post_lift_base_im,
                        _,
                        post_lift_base_im_second,
                        _,
                        post_lift_wrist_im,
                        _,
                        _,
                    ) = extract_from_obs(obs, robot_name=env.robot.name)
                    post_lift_image_paths = {
                        "base": os.path.join("base_cam", f"post_grasp_lift_step{t:04d}.png"),
                        "second": os.path.join("second_cam", f"post_grasp_lift_step{t:04d}.png"),
                        "wrist": os.path.join("wrist_cam", f"post_grasp_lift_step{t:04d}.png"),
                    }
                    image_writer.submit(os.path.join(task_debug_dir, post_lift_image_paths["base"]), post_lift_base_im)
                    image_writer.submit(
                        os.path.join(task_debug_dir, post_lift_image_paths["second"]),
                        post_lift_base_im_second,
                    )
                    image_writer.submit(os.path.join(task_debug_dir, post_lift_image_paths["wrist"]), post_lift_wrist_im)
                    image_writer.flush()
                    grasp_hold_gripper_command = float(
                        np.clip(post_lift_gripper_normalized + POST_GRASP_HOLD_EXTRA_CLOSE, 0.0, 1.0)
                    )
                    post_grasp_lift_succeeded = physical_stage_reached(
                        env,
                        post_grasp_validation_stage,
                        curr_task_progression,
                        obs=obs,
                        validated_stages=validated_stages,
                    )
                step_traces[-1]["post_grasp_tighten"] = {
                    "gripper_command_normalized": float(POST_GRASP_CLOSE_COMMAND),
                    "downward_offset_z_robot": float(POST_GRASP_DOWNWARD_OFFSET_Z),
                    "target_robot_pose6_before_downward_offset": pre_post_grasp_robot_pose6.astype(np.float32).copy(),
                    "target_robot_pose6": tighten_target_robot_pose6.astype(np.float32).copy(),
                    "settle_steps": int(POST_GRASP_CLOSE_STEPS),
                    "force_full_steps": True,
                    "duration_s": float(tighten_duration_s),
                    "task_progression": float(post_tighten_task_progression),
                    "settle_history": tighten_result["settle_history"],
                    "env_action": tighten_result["env_action"].copy(),
                    "post_tighten_gripper_normalized": float(post_tighten_gripper_normalized),
                    "post_tighten_raw_gripper": None if post_tighten_raw_gripper is None else float(post_tighten_raw_gripper),
                    "hold_extra_close": float(POST_GRASP_HOLD_EXTRA_CLOSE),
                    "grasp_hold_gripper_command": float(grasp_hold_gripper_command),
                    "saved_images": post_tighten_image_paths,
                    "validation_stage": post_grasp_validation_stage,
                }
                step_traces[-1]["post_grasp_lift"] = {
                    "validation_stage": post_grasp_validation_stage,
                    "skipped": post_grasp_validation_stage is None,
                    "skip_reason": None
                    if post_grasp_validation_stage is not None
                    else "task_has_no_lift_validation_stage_after_grasp",
                    "gripper_command_normalized": float(POST_GRASP_CLOSE_COMMAND),
                    "lift_offset_z_robot": float(post_grasp_lift_offset_z),
                    "target_robot_pose6_before_lift": post_tighten_robot_pose6.astype(np.float32).copy(),
                    "target_robot_pose6": lift_target_robot_pose6.astype(np.float32).copy(),
                    "actual_robot_pose6_after_lift": post_lift_robot_pose6.astype(np.float32).copy(),
                    "settle_steps": int(POST_GRASP_LIFT_STEPS),
                    "force_full_steps": True,
                    "duration_s": float(time.perf_counter() - t_lift_start),
                    "task_progression": float(curr_task_progression),
                    "settle_history": lift_result["settle_history"],
                    "env_action": lift_result["env_action"].copy(),
                    "post_lift_gripper_normalized": float(post_lift_gripper_normalized),
                    "post_lift_raw_gripper": None if post_lift_raw_gripper is None else float(post_lift_raw_gripper),
                    "lift_succeeded": bool(post_grasp_lift_succeeded),
                    "saved_images": post_lift_image_paths,
                }
                if not post_grasp_lift_succeeded:
                    rollback_ok = False
                    rollback_error = None
                    if post_grasp_rollback_sim_state is not None:
                        try:
                            load_sim_state(og, post_grasp_rollback_sim_state)
                            load_python_task_state(env, post_grasp_rollback_python_state)
                            og.sim.render()
                            obs = env.get_current_obs()
                            curr_task_progression = env.recompute_task_progression(obs)
                            cleared_stages = clear_failed_grasp_candidate_flags(env)
                            task_progression = getattr(env, "task_progression", None)
                            if isinstance(task_progression, dict):
                                curr_task_progression = float(
                                    sum(1 for value in task_progression.values() if bool(value))
                                    / max(1, len(task_progression))
                                )
                            final_task_progression = float(curr_task_progression)
                            rollback_ok = True
                        except Exception as exc:
                            rollback_error = str(exc)
                    else:
                        rollback_error = "rollback snapshot unavailable"
                        cleared_stages = []
                    post_grasp_tightened = False
                    grasp_hold_gripper_command = None
                    if phase_based_model:
                        release_restore_pose6_src = pre_post_grasp_robot_pose6
                        if release_restore_pose6_src is None:
                            release_restore_pose6_src = next_robot_pose6
                        if release_restore_pose6_src is None:
                            release_restore_pose6_src = robot_pose6
                        slot_next_planning_phase = "pose"
                        slot_cached_pose_obs_for_release = None
                        slot_release_gripper_until_next_search = True
                        slot_gripper_release_restore_pose6 = np.asarray(
                            release_restore_pose6_src,
                            dtype=np.float32,
                        ).reshape(6).copy()
                    selected_goal_reached_now = False
                    step_traces[-1]["selected_goal_reached_now"] = False
                    step_traces[-1]["post_grasp_lift_failed_rollback"] = {
                        "rollback_ok": bool(rollback_ok),
                        "rollback_error": rollback_error,
                        "release_gripper_before_next_pose_search": bool(phase_based_model),
                        "cleared_task_stages": list(cleared_stages),
                    }
                    validated_stages.discard("GRASP")
                    validated_stages.discard("LIFT_SLIGHT")
                    validated_stages.discard("LIFT_LARGE")
                    validated_stages.discard(ROTATE_POST_GRASP_VALIDATION_STAGE)
                    reached_stages.discard("GRASP")
                    reached_stages.discard("LIFT_SLIGHT")
                    reached_stages.discard("LIFT_LARGE")
                    reached_stages.discard(ROTATE_POST_GRASP_VALIDATION_STAGE)
                    print(
                        f"Selected GRASP reached at step {t}, but {post_grasp_validation_stage} failed after tighten/lift; "
                        f"rollback_ok={rollback_ok}. Continuing grasp attempts.",
                        flush=True,
                    )
                    step_traces[-1]["timing_extract_obs_s"] = float(timing_extract_obs_s)
                    step_traces[-1]["timing_save_step_images_s"] = float(timing_save_step_images_s)
                    step_traces[-1]["timing_client_infer_s"] = float(timing_client_infer_s)
                    step_traces[-1]["timing_execute_target_s"] = float(timing_execute_target_s)
                    step_traces[-1]["timing_post_obs_extract_s"] = float(timing_post_obs_extract_s)
                    step_traces[-1]["timing_record_step_s"] = float(timing_record_step_s)
                    step_traces[-1]["timing_outer_step_total_s"] = float(time.perf_counter() - t_outer_step_start)
                    step_traces[-1]["server_timing"] = model_response.get("server_timing")
                    step_traces[-1]["validated_stages"] = sorted(validated_stages)
                    flush_trace_artifacts(cancelled_flag=cancelled)
                    t += 1
                    continue
                validated_stages.add("GRASP")
                reached_stages.add("GRASP")
                if post_grasp_validation_stage is not None:
                    validated_stages.add(post_grasp_validation_stage)
                    reached_stages.add(post_grasp_validation_stage)
                    Image.fromarray(post_lift_base_im).save(
                        os.path.join(task_debug_dir, f"reached_stage_{post_grasp_validation_stage}.png")
                    )
                    print(
                        f"Selected GRASP reached at step {t}; lowered z by "
                        f"{POST_GRASP_DOWNWARD_OFFSET_Z:.3f} and tightened gripper with command="
                        f"{POST_GRASP_CLOSE_COMMAND:.3f} for {POST_GRASP_CLOSE_STEPS} forced steps; "
                        f"then {describe_post_grasp_validation(post_grasp_validation_stage, post_grasp_lift_offset_z)} "
                        f"for {POST_GRASP_LIFT_STEPS} forced steps; "
                        f"holding future gripper at {grasp_hold_gripper_command:.3f}.",
                        flush=True,
                    )
            if reached_stage == "MOVE_CLOSE" or post_move_release_from_success_goal:
                t_release_start = time.perf_counter()
                release_target_robot_pose6 = next_robot_pose6.astype(np.float32).copy()
                release_target_robot_pose6[2] += float(POST_MOVE_RELEASE_LIFT_OFFSET_Z)
                grasp_hold_gripper_command = None
                release_step_records = []
                release_env_action = None
                release_base_im_pre = next_base_im
                release_base_im_second_pre = next_base_im_second
                release_wrist_im_pre = next_wrist_im
                release_gripper_state_pre = next_gripper_state
                release_camera_pose6_by_view_pre = next_camera_pose6_by_view
                release_camera_extrinsics_by_view_pre = next_camera_extrinsics_by_view
                for release_step_idx in range(1, int(POST_MOVE_RELEASE_STEPS) + 1):
                    release_pre_obs_payload = build_multiview_obs_payload(
                        base_im=release_base_im_pre,
                        base_im_second=release_base_im_second_pre,
                        wrist_im=release_wrist_im_pre,
                        gripper_state=release_gripper_state_pre,
                        camera_pose6_by_view=release_camera_pose6_by_view_pre,
                        camera_extrinsics_by_view=release_camera_extrinsics_by_view_pre,
                        camera_frame_convention=camera_frame_convention,
                    )
                    release_result = execute_absolute_robot_target(
                        env=env,
                        target_robot_pose6=release_target_robot_pose6,
                        gripper_command=POST_MOVE_RELEASE_COMMAND,
                        max_settle_steps=1,
                        settle_pos_tol=settle_pos_tol,
                        settle_rot_tol=settle_rot_tol,
                        vis_dir=None,
                        include_controller_debug=False,
                        force_full_steps=True,
                        cancel_event=cancel_event,
                    )
                    if cancel_requested():
                        print(
                            f"Planning cancelled during post-move release at step {t}, forced step {release_step_idx}.",
                            flush=True,
                        )
                        cancelled = True
                        break
                    release_env_action = release_result["env_action"].copy()
                    obs = release_result["obs"]
                    curr_task_progression = release_result["task_progression"]
                    final_task_progression = float(curr_task_progression)
                    release_step_ee_pos, release_step_ee_rot = env.get_ee_pose()
                    release_step_ee_pos_np = (
                        release_step_ee_pos.cpu().numpy()
                        if hasattr(release_step_ee_pos, "cpu")
                        else np.array(release_step_ee_pos)
                    )
                    release_step_ee_rot_np = (
                        release_step_ee_rot.cpu().numpy()
                        if hasattr(release_step_ee_rot, "cpu")
                        else np.array(release_step_ee_rot)
                    )
                    release_step_ee_euler_np = Rot.from_quat(release_step_ee_rot_np).as_euler("xyz")
                    release_step_ee_pose_world = np.concatenate(
                        [release_step_ee_pos_np, release_step_ee_euler_np]
                    ).astype(np.float32)
                    release_step_robot_pose6 = env._world2robot(release_step_ee_pose_world).astype(np.float32)
                    release_step_view_world_poses = get_view_world_poses(env, task_cfg)
                    release_step_camera_extrinsics_by_view = {}
                    release_step_camera_pose6_by_view = {}
                    for view_name, (cam_pos_world, cam_quat_world) in release_step_view_world_poses.items():
                        camera_world_pose6 = get_camera_extrinsics_pose6(
                            cam_pos_world,
                            cam_quat_world,
                            camera_frame_convention=camera_frame_convention,
                        )
                        release_step_camera_extrinsics_by_view[view_name] = env._world2robot(
                            camera_world_pose6
                        ).astype(np.float32)
                        release_step_camera_pose6_by_view[view_name] = pose_in_parent_frame(
                            release_step_robot_pose6.astype(np.float32),
                            release_step_camera_extrinsics_by_view[view_name],
                        )
                    (
                        release_base_im,
                        _,
                        release_base_im_second,
                        _,
                        release_wrist_im,
                        _,
                        release_gripper_state,
                    ) = extract_from_obs(obs, robot_name=env.robot.name)
                    release_raw_gripper = get_raw_gripper_position_from_obs(obs, robot_name=env.robot.name)
                    release_gripper_state_normalized = float(np.clip(float(release_gripper_state), 0.0, 1.0))
                    release_gripper_state_raw_width = None
                    if model_type == "slot" and release_raw_gripper is not None:
                        release_gripper_state_raw_width = float(np.clip(release_raw_gripper, 0.0, 0.05))
                        release_gripper_state_normalized = normalize_raw_gripper_position(release_raw_gripper)
                        release_gripper_state = gripper_state_for_model(
                            release_raw_gripper,
                            slot_model_gripper_units,
                        )
                    release_post_obs_payload = build_multiview_obs_payload(
                        base_im=release_base_im,
                        base_im_second=release_base_im_second,
                        wrist_im=release_wrist_im,
                        gripper_state=release_gripper_state,
                        camera_pose6_by_view=release_step_camera_pose6_by_view,
                        camera_extrinsics_by_view=release_step_camera_extrinsics_by_view,
                        camera_frame_convention=camera_frame_convention,
                    )
                    release_record_payload = {
                        "pre_obs": release_pre_obs_payload,
                        "post_obs": release_post_obs_payload,
                    }
                    try:
                        release_record_response = client.record_step_with_action(release_record_payload)
                    except Exception as exc:
                        release_record_response = {"error": str(exc)}
                        print(f"Release forced-step latent recording issue: {exc}", flush=True)
                    (
                        release_observed_per_view_losses,
                        release_observed_per_view_transport_losses,
                        release_observed_per_view_objective_losses,
                    ) = _observed_loss_maps_from_record_response(release_record_response)
                    release_frame_idx = t + release_step_idx
                    image_writer.submit(
                        os.path.join(task_debug_dir, "base_cam", f"{release_frame_idx:04d}.png"),
                        release_base_im,
                    )
                    image_writer.submit(
                        os.path.join(task_debug_dir, "second_cam", f"{release_frame_idx:04d}.png"),
                        release_base_im_second,
                    )
                    image_writer.submit(
                        os.path.join(task_debug_dir, "wrist_cam", f"{release_frame_idx:04d}.png"),
                        release_wrist_im,
                    )
                    release_step_records.append(
                        {
                            "release_step": int(release_step_idx),
                            "saved_sequence_frame_idx": int(release_frame_idx),
                            "mode": "post_move_close_forced_planning_step",
                            "robot_pose6": release_step_robot_pose6.astype(np.float32).copy(),
                            "camera_pose6_by_view": {
                                k: v.astype(np.float32).copy() for k, v in release_step_camera_pose6_by_view.items()
                            },
                            "camera_extrinsics6_by_view": {
                                k: v.astype(np.float32).copy()
                                for k, v in release_step_camera_extrinsics_by_view.items()
                            },
                            "target_robot_pose6": release_target_robot_pose6.astype(np.float32).copy(),
                            "env_action": release_env_action.copy(),
                            "task_progression": float(curr_task_progression),
                            "gripper_state_model": float(release_gripper_state),
                            "gripper_state_normalized": float(release_gripper_state_normalized),
                            "raw_gripper_state": None if release_raw_gripper is None else float(release_raw_gripper),
                            "gripper_state_raw_width": None
                            if release_gripper_state_raw_width is None
                            else float(release_gripper_state_raw_width),
                            f"{model_type}_record_response": release_record_response,
                            "observed_per_view_losses": release_observed_per_view_losses,
                            "observed_per_view_transport_losses": release_observed_per_view_transport_losses,
                            "observed_per_view_objective_losses": release_observed_per_view_objective_losses,
                            "saved_images": {
                                "base": os.path.join("base_cam", f"{release_frame_idx:04d}.png"),
                                "second": os.path.join("second_cam", f"{release_frame_idx:04d}.png"),
                                "wrist": os.path.join("wrist_cam", f"{release_frame_idx:04d}.png"),
                            },
                        }
                    )
                    post_move_close_extra_step_traces.append(
                        {
                            "step": int(release_frame_idx),
                            "mode": "post_move_close_forced_planning_step",
                            "state_frame": "camera",
                            "robot_pose6": release_step_robot_pose6.astype(np.float32).copy(),
                            "camera_pose6": release_step_camera_pose6_by_view[
                                "exterior_image_1_left"
                            ].astype(np.float32).copy(),
                            "camera_pose6_by_view": {
                                k: v.astype(np.float32).copy() for k, v in release_step_camera_pose6_by_view.items()
                            },
                            "camera_extrinsics6": release_step_camera_extrinsics_by_view[
                                "exterior_image_1_left"
                            ].astype(np.float32).copy(),
                            "camera_extrinsics6_by_view": {
                                k: v.astype(np.float32).copy()
                                for k, v in release_step_camera_extrinsics_by_view.items()
                            },
                            "camera_frame_convention": camera_frame_convention,
                            "target_robot_pose6": release_target_robot_pose6.astype(np.float32).copy(),
                            "commanded_target_robot_pose6": release_target_robot_pose6.astype(np.float32).copy(),
                            "env_action": release_env_action.copy(),
                            "gripper_policy": "post_move_close_forced_release",
                            "gripper_command_env": float(release_env_action[-1]),
                            "gripper_target_state_normalized": float(POST_MOVE_RELEASE_COMMAND),
                            "gripper_state_model": float(release_gripper_state),
                            "gripper_state_normalized": float(release_gripper_state_normalized),
                            "gripper_state_raw_width": None
                            if release_gripper_state_raw_width is None
                            else float(release_gripper_state_raw_width),
                            "next_robot_pose6": release_step_robot_pose6.astype(np.float32).copy(),
                            "next_camera_pose6": release_step_camera_pose6_by_view[
                                "exterior_image_1_left"
                            ].astype(np.float32).copy(),
                            "next_gripper_state_model": float(release_gripper_state),
                            "next_gripper_state_normalized": float(release_gripper_state_normalized),
                            "next_gripper_state_raw_width": None
                            if release_gripper_state_raw_width is None
                            else float(release_gripper_state_raw_width),
                            "goal_idx": int(active_goal_idx),
                            "goal_name": reached_name,
                            "goal_stage": "POST_MOVE_CLOSE_RELEASE",
                            "selected_goal_reached_now": False,
                            "strict_task_success_now": release_step_idx == int(POST_MOVE_RELEASE_STEPS),
                            "strict_task_success_by_post_move_close_release": True,
                            "release_step": int(release_step_idx),
                            "release_total_steps": int(POST_MOVE_RELEASE_STEPS),
                            "observed_per_view_losses": release_observed_per_view_losses,
                            "observed_per_view_transport_losses": release_observed_per_view_transport_losses,
                            "observed_per_view_objective_losses": release_observed_per_view_objective_losses,
                            "saved_images": {
                                "base": os.path.join("base_cam", f"{release_frame_idx:04d}.png"),
                                "second": os.path.join("second_cam", f"{release_frame_idx:04d}.png"),
                                "wrist": os.path.join("wrist_cam", f"{release_frame_idx:04d}.png"),
                            },
                            f"{model_type}_record_response": release_record_response,
                        }
                    )
                    release_base_im_pre = release_base_im
                    release_base_im_second_pre = release_base_im_second
                    release_wrist_im_pre = release_wrist_im
                    release_gripper_state_pre = release_gripper_state
                    release_camera_pose6_by_view_pre = release_step_camera_pose6_by_view
                    release_camera_extrinsics_by_view_pre = release_step_camera_extrinsics_by_view
                post_release_raw_gripper = get_raw_gripper_position_from_obs(obs, robot_name=env.robot.name)
                post_release_gripper_normalized = (
                    POST_MOVE_RELEASE_COMMAND
                    if post_release_raw_gripper is None
                    else normalize_raw_gripper_position(post_release_raw_gripper)
                )
                release_ee_pos, release_ee_rot = env.get_ee_pose()
                release_ee_pos_np = (
                    release_ee_pos.cpu().numpy()
                    if hasattr(release_ee_pos, "cpu")
                    else np.array(release_ee_pos)
                )
                release_ee_rot_np = (
                    release_ee_rot.cpu().numpy()
                    if hasattr(release_ee_rot, "cpu")
                    else np.array(release_ee_rot)
                )
                release_ee_euler_np = Rot.from_quat(release_ee_rot_np).as_euler("xyz")
                release_ee_pose_world = np.concatenate([release_ee_pos_np, release_ee_euler_np]).astype(np.float32)
                release_robot_pose6 = env._world2robot(release_ee_pose_world).astype(np.float32)
                release_view_world_poses = get_view_world_poses(env, task_cfg)
                release_camera_extrinsics_by_view = {}
                release_camera_pose6_by_view = {}
                for view_name, (cam_pos_world, cam_quat_world) in release_view_world_poses.items():
                    camera_world_pose6 = get_camera_extrinsics_pose6(
                        cam_pos_world,
                        cam_quat_world,
                        camera_frame_convention=camera_frame_convention,
                    )
                    release_camera_extrinsics_by_view[view_name] = env._world2robot(camera_world_pose6).astype(np.float32)
                    release_camera_pose6_by_view[view_name] = pose_in_parent_frame(
                        release_robot_pose6.astype(np.float32),
                        release_camera_extrinsics_by_view[view_name],
                    )
                release_base_im, _, release_base_im_second, _, release_wrist_im, _, release_gripper_state = extract_from_obs(
                    obs, robot_name=env.robot.name
                )
                image_writer.submit(os.path.join(task_debug_dir, "final_release_base.png"), release_base_im)
                image_writer.submit(os.path.join(task_debug_dir, "final_release_second.png"), release_base_im_second)
                image_writer.submit(os.path.join(task_debug_dir, "final_release_wrist.png"), release_wrist_im)
                step_traces[-1]["post_move_close_release"] = {
                    "gripper_command_normalized": float(POST_MOVE_RELEASE_COMMAND),
                    "lift_offset_z_robot": float(POST_MOVE_RELEASE_LIFT_OFFSET_Z),
                    "trigger_stage": reached_stage,
                    "triggered_from_success_goal_move_close": bool(post_move_release_from_success_goal),
                    "target_robot_pose6_before_lift": next_robot_pose6.astype(np.float32).copy(),
                    "target_robot_pose6": release_target_robot_pose6.astype(np.float32).copy(),
                    "final_robot_pose6": release_robot_pose6.astype(np.float32).copy(),
                    "final_camera_pose6_by_view": {
                        k: v.astype(np.float32).copy() for k, v in release_camera_pose6_by_view.items()
                    },
                    "final_camera_extrinsics6_by_view": {
                        k: v.astype(np.float32).copy() for k, v in release_camera_extrinsics_by_view.items()
                    },
                    "settle_steps": int(POST_MOVE_RELEASE_STEPS),
                    "force_full_steps": True,
                    "duration_s": float(time.perf_counter() - t_release_start),
                    "task_progression": float(curr_task_progression),
                    "env_action": None if release_env_action is None else release_env_action.copy(),
                    "post_release_gripper_normalized": float(post_release_gripper_normalized),
                    "post_release_raw_gripper": None if post_release_raw_gripper is None else float(post_release_raw_gripper),
                    "post_release_gripper_state_model": float(release_gripper_state),
                    "release_step_records": release_step_records,
                    "saved_sequence_frame_indices": [
                        int(record["saved_sequence_frame_idx"]) for record in release_step_records
                    ],
                    "saved_images": {
                        "final_base": "final_release_base.png",
                        "final_second": "final_release_second.png",
                        "final_wrist": "final_release_wrist.png",
                    },
                }
                strict_task_success = True
                stop_after_move_close_release = True
                release_trigger_label = (
                    "SUCCESS goal after validated MOVE_CLOSE"
                    if post_move_release_from_success_goal
                    else "MOVE_CLOSE"
                )
                print(
                    f"Selected {release_trigger_label} reached at step {t}; releasing gripper with command="
                    f"{POST_MOVE_RELEASE_COMMAND:.3f} and lifting robot z by "
                    f"{POST_MOVE_RELEASE_LIFT_OFFSET_Z:.3f} for {POST_MOVE_RELEASE_STEPS} forced steps. "
                    "Stopping without further CEM.",
                    flush=True,
                )
            step_traces[-1]["timing_extract_obs_s"] = float(timing_extract_obs_s)
            step_traces[-1]["timing_save_step_images_s"] = float(timing_save_step_images_s)
            step_traces[-1]["timing_client_infer_s"] = float(timing_client_infer_s)
            step_traces[-1]["timing_execute_target_s"] = float(timing_execute_target_s)
            step_traces[-1]["timing_post_obs_extract_s"] = float(timing_post_obs_extract_s)
            step_traces[-1]["timing_record_step_s"] = float(timing_record_step_s)
            step_traces[-1]["timing_outer_step_total_s"] = float(time.perf_counter() - t_outer_step_start)
            step_traces[-1]["server_timing"] = model_response.get("server_timing")
            if stop_after_move_close_release:
                step_traces[-1]["strict_task_success_now"] = True
                step_traces[-1]["strict_task_success_by_post_move_close_release"] = True
                step_traces[-1]["validated_stages"] = sorted(validated_stages)
                step_traces.extend(post_move_close_extra_step_traces)
                flush_trace_artifacts(cancelled_flag=cancelled)
                break
            if goal_mode == "first":
                print(
                    f"Selected goal `{reached_name}` stage `{reached_stage}` reached at step {t}; "
                    "stopping because goal_mode=first.",
                    flush=True,
                )
                flush_trace_artifacts(cancelled_flag=cancelled)
                break
            next_goal_idx = active_goal_idx + 1
            if next_goal_idx < len(goal_state.get("goal_targets", [])):
                try:
                    advance_response = client.advance_goal(goal_idx=next_goal_idx)
                except Exception as exc:
                    advance_response = {"error": str(exc)}
                    print(f"Failed to advance server goal to idx {next_goal_idx}: {exc}", flush=True)
                if isinstance(advance_response, dict) and advance_response.get("error"):
                    step_traces[-1]["advanced_goal_response"] = advance_response
                    flush_trace_artifacts(cancelled_flag=cancelled)
                    break
                active_goal_idx = int(advance_response.get("goal_idx", next_goal_idx)) if isinstance(advance_response, dict) else next_goal_idx
                step_traces[-1]["advanced_goal_response"] = advance_response
                step_traces[-1]["advanced_goal_idx"] = int(active_goal_idx)
                if model_type == "slot":
                    try:
                        if hasattr(client, "reset_loss_tracking"):
                            client.reset_loss_tracking()
                    except Exception as exc:
                        print(f"Warning: failed to reset SLOT loss tracking after goal advance: {exc}", flush=True)
                print(
                    f"Selected goal `{reached_name}` stage `{reached_stage}` reached at step {t}; "
                    f"advancing to goal_idx={active_goal_idx}.",
                    flush=True,
                )
            else:
                final_goal_success_now = bool(
                    reached_stage == "SUCCESS"
                    or physical_stage_reached(
                        env,
                        "SUCCESS",
                        curr_task_progression,
                        obs=obs,
                        validated_stages=validated_stages,
                    )
                )
                strict_task_success = bool(strict_task_success or final_goal_success_now)
                step_traces[-1]["strict_task_success_now"] = bool(final_goal_success_now)
                step_traces[-1]["strict_task_success_by_final_goal"] = bool(final_goal_success_now)
                step_traces[-1]["validated_stages"] = sorted(validated_stages)
                print(
                    f"Selected goal `{reached_name}` stage `{reached_stage}` reached at step {t}; "
                    "no remaining goal targets.",
                    flush=True,
                )
                flush_trace_artifacts(cancelled_flag=cancelled)
                break

        strict_task_success_now = physical_stage_reached(
            env,
            "SUCCESS",
            curr_task_progression,
            obs=obs,
            validated_stages=validated_stages,
        )
        strict_task_success = bool(strict_task_success or strict_task_success_now)
        step_traces[-1]["strict_task_success_now"] = bool(strict_task_success_now)
        step_traces[-1]["validated_stages"] = sorted(validated_stages)

        if strict_task_success_now:
            step_traces[-1]["timing_extract_obs_s"] = float(timing_extract_obs_s)
            step_traces[-1]["timing_save_step_images_s"] = float(timing_save_step_images_s)
            step_traces[-1]["timing_client_infer_s"] = float(timing_client_infer_s)
            step_traces[-1]["timing_execute_target_s"] = float(timing_execute_target_s)
            step_traces[-1]["timing_post_obs_extract_s"] = float(timing_post_obs_extract_s)
            step_traces[-1]["timing_record_step_s"] = float(timing_record_step_s)
            step_traces[-1]["timing_outer_step_total_s"] = float(time.perf_counter() - t_outer_step_start)
            step_traces[-1]["server_timing"] = model_response.get("server_timing")
            print(f"Task successfully completed at step {t}!")
            flush_trace_artifacts(cancelled_flag=cancelled)
            break

        step_traces[-1]["timing_extract_obs_s"] = float(timing_extract_obs_s)
        step_traces[-1]["timing_save_step_images_s"] = float(timing_save_step_images_s)
        step_traces[-1]["timing_client_infer_s"] = float(timing_client_infer_s)
        step_traces[-1]["timing_execute_target_s"] = float(timing_execute_target_s)
        step_traces[-1]["timing_post_obs_extract_s"] = float(timing_post_obs_extract_s)
        step_traces[-1]["timing_record_step_s"] = float(timing_record_step_s)
        step_traces[-1]["timing_outer_step_total_s"] = float(time.perf_counter() - t_outer_step_start)
        step_traces[-1]["server_timing"] = model_response.get("server_timing")
        flush_trace_artifacts(cancelled_flag=cancelled)
            
        t += 1
        
    image_writer.shutdown()

    flush_trace_artifacts(cancelled_flag=cancelled)
    view_loss_plot_path = os.path.abspath(os.path.join(task_debug_dir, "view_losses_over_steps.png"))
    try:
        if plot_step_view_losses(step_traces, view_loss_plot_path):
            print(f"Saved per-view loss plot to {view_loss_plot_path}")
        else:
            print("Warning: no per-view losses found; skipped view loss plot.", flush=True)
    except Exception as exc:
        print(f"Warning: failed to plot per-view losses to {view_loss_plot_path}: {exc}", flush=True)
    print(f"Debug data saved to {task_debug_dir}")
    
    # Request plotting PCA of latent trajectory
    pca_save_path = os.path.abspath(os.path.join(task_debug_dir, "trajectory_pca.png"))
    if not (cancelled or cancel_requested()) and hasattr(client, "plot_pca"):
        print(f"Plotting PCA of latents to {pca_save_path}...")
        try:
            client.plot_pca(pca_save_path)
        except Exception as exc:
            print(f"Warning: failed to plot PCA to {pca_save_path}: {exc}", flush=True)
    elif cancelled or cancel_requested():
        print("Skipping PCA plot because planning was cancelled.", flush=True)

    debug_trace = (
        client.get_debug_trace()
        if enable_settle_debug and not (cancelled or cancel_requested()) and hasattr(client, "get_debug_trace")
        else None
    )
    if debug_trace is not None:
        with open(os.path.join(task_debug_dir, "server_debug_trace.json"), "w", encoding="utf-8") as f:
            json.dump(to_jsonable(debug_trace), f, ensure_ascii=False, indent=2)
    
    cancelled = bool(cancelled or cancel_requested())
    if not cancelled:
        client.reset()
    planning_result = {
        "success": bool(strict_task_success and not cancelled),
        "status": "cancelled" if cancelled else ("success" if strict_task_success else "completed"),
        "task_progression": float(final_task_progression),
        "strict_task_success": bool(strict_task_success),
        "output_dir": task_debug_dir,
        "num_steps": int(len(step_traces)),
        "run_timestamp": run_timestamp,
        "run_dir_timestamp": run_dir_timestamp,
        "output_namespace": str(output_namespace or model_type),
    }
    try:
        with open(os.path.join(task_debug_dir, "planning_result.json"), "w", encoding="utf-8") as f:
            json.dump(to_jsonable(planning_result), f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"Warning: failed to save planning_result.json: {exc}", flush=True)
    return planning_result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--perturbation_id", type=int, default=0)
    parser.add_argument("--state-frame", type=str, choices=["camera"], default="camera")
    parser.add_argument("--success-case-dir", type=str, default=DEFAULT_SUCCESS_CASE_DIR)
    parser.add_argument("--goal-mode", type=str, choices=["first", "end", "all"], default="first")
    parser.add_argument("--scene-profile", type=str, choices=SUPPORTED_SCENE_PROFILES, default="realm_droid10")
    parser.add_argument("--task-variant", type=str, default=DEFAULT_TASK_VARIANT)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument(
        "--replay-mode",
        type=str,
        choices=["robot_absolute", "camera_delta", "camera_delta_closed_loop"],
        default="robot_absolute",
    )
    parser.add_argument("--max-settle-steps", type=int, default=DEFAULT_VJEPA_MAX_SETTLE_STEPS)
    parser.add_argument("--settle-pos-tol", type=float, default=DEFAULT_VJEPA_SETTLE_POS_TOL)
    parser.add_argument("--settle-rot-tol", type=float, default=DEFAULT_VJEPA_SETTLE_ROT_TOL)
    parser.add_argument("--disable-rotate-deterministic-after-grasp", action="store_true")
    parser.add_argument("--rotate-deterministic-step-rad", type=float, default=DEFAULT_ROTATE_DETERMINISTIC_STEP_RAD)
    parser.add_argument(
        "--rotate-deterministic-overshoot-rad",
        type=float,
        default=DEFAULT_ROTATE_DETERMINISTIC_OVERSHOOT_RAD,
        help="Extra radians added beyond the success-case rotate target to absorb execution under-shoot.",
    )
    parser.add_argument(
        "--rotate-deterministic-axis",
        type=str,
        choices=["roll", "pitch", "yaw", "x", "y", "z"],
        default=DEFAULT_ROTATE_DETERMINISTIC_AXIS,
    )
    parser.add_argument("--enable-settle-debug", action="store_true")
    parser.add_argument("--controller-only-goal-delta", action="store_true")
    parser.add_argument(
        "--fixed-camera-presets",
        nargs=2,
        default=None,
        metavar=("CAM1", "CAM2"),
        help="Override the two fixed external camera presets. Must match the released successful_case targets.",
    )
    args, unknown_args = parser.parse_known_args()
    unknown_user_args = [arg for arg in unknown_args if not arg.startswith("--/")]
    if unknown_user_args:
        parser.error(f"unrecognized arguments: {' '.join(unknown_user_args)}")
    if args.controller_only_goal_delta:
        run_controller_goal_delta_diagnostic(
            task_id=args.task_id,
            perturbation_id=args.perturbation_id,
            state_frame=args.state_frame,
            success_case_dir=args.success_case_dir,
            goal_mode=args.goal_mode,
            scene_profile=args.scene_profile,
            task_variant=args.task_variant,
            max_settle_steps=args.max_settle_steps,
            settle_pos_tol=args.settle_pos_tol,
            settle_rot_tol=args.settle_rot_tol,
            enable_settle_debug=args.enable_settle_debug,
            fixed_camera_presets=args.fixed_camera_presets,
            rotate_deterministic_after_grasp=not args.disable_rotate_deterministic_after_grasp,
            rotate_deterministic_step_rad=args.rotate_deterministic_step_rad,
            rotate_deterministic_axis=args.rotate_deterministic_axis,
            rotate_deterministic_overshoot_rad=args.rotate_deterministic_overshoot_rad,
        )
    else:
        run_vjepa_planning(
            port=args.port,
            task_id=args.task_id,
            perturbation_id=args.perturbation_id,
            state_frame=args.state_frame,
            success_case_dir=args.success_case_dir,
            goal_mode=args.goal_mode,
            scene_profile=args.scene_profile,
            task_variant=args.task_variant,
            max_steps=args.max_steps,
            action_repeat=args.action_repeat,
            replay_mode=args.replay_mode,
            max_settle_steps=args.max_settle_steps,
            settle_pos_tol=args.settle_pos_tol,
            settle_rot_tol=args.settle_rot_tol,
            enable_settle_debug=args.enable_settle_debug,
            fixed_camera_presets=args.fixed_camera_presets,
        )
    _ensure_omnigibson_runtime_initialized()
    og.shutdown()
    sys.exit(0)
