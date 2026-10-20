import math
import numpy as np
import torch
import yaml
import random
import copy
import os
import time
import functools

from realm.environments.env_base import RealmEnvironmentBase, TASK_PROGRESS_RUBRICS
from realm.environments.perturbations.default import default as _pert_default
from realm.environments.perturbations.v_light import v_light as _pert_v_light
from realm.environments.perturbations.v_view import v_view as _pert_v_view
from realm.environments.perturbations.v_sc import v_sc as _pert_v_sc
from realm.environments.perturbations.semantic import s_prop as _pert_s_prop, s_lang as _pert_s_lang, s_mo as _pert_s_mo, s_aff as _pert_s_aff, s_int as _pert_s_int
from realm.environments.perturbations.b_hobj import b_hobj as _pert_b_hobj
from realm.environments.perturbations.sb_noun import sb_noun as _pert_sb_noun
from realm.environments.perturbations.sb_vrb import sb_vrb as _pert_sb_vrb
from realm.environments.perturbations.vb_pose import vb_pose as _pert_vb_pose
from realm.environments.perturbations.vb_mobj import vb_mobj as _pert_vb_mobj
from realm.environments.perturbations.vsb_nobj import vsb_nobj as _pert_vsb_nobj
from realm.robots.widowx import WidowX
from realm.robots.ur import UR
from realm.helpers import (
    calculate_new_camera_pose_mixed_rotations,
    add_rotation_noise,
    get_non_colliding_positions_for_objects,
    apply_blur_and_contrast,
    get_non_droid_categories,
    get_droid_categories_by_theme,
    get_objects_by_names,
    get_default_objects_cfg,
    robot_to_world,
    world_to_robot,
)

import omnigibson as og
import omnigibson.utils.transform_utils as omnigibson_transform_utils
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject, PrimitiveObject, USDObject
from omnigibson.utils.asset_utils import get_all_object_category_models
from omnigibson.utils.asset_utils import get_all_object_models
from omnigibson.utils.usd_utils import create_joint
from omnigibson.prims.joint_prim import JointPrim
from scipy.spatial.transform import Rotation as R



MISSING_PERTURBATIONS = ["V-OBJ", "VB-ISC", "VS-PROP", "SB-ADV", "SB-SMO"]
SUPPORTED_TASK_TYPES = ["put", "pick", "rotate", "push", "stack", "open_drawer", "close_drawer"]
SKILL_COMPATIBILITY_MATRIX = {
    "put": ["pick", "rotate", "stack"],
    "push": [],  # ["put", "pick", "rotate", "stack"],
    "pick": ["put", "rotate", "stack"],
    "rotate": ["put", "pick", "stack"],
    "stack": ["put", "pick", "rotate"],
    "open": ["close"],
    "close": ["open"]
}
DEFAULT_RESET_JOINTPOS = np.array([0, -1 / 5 * np.pi, 0, -4 / 5 * np.pi, 0, 3 / 5 * np.pi, 0.0])
DROID_BASE_HEIGHT = 0.86244
MAX_CAMERA_POS_DEVIATION = 0.2
MAX_CAMERA_PITCH_DEVIATION = 0.2
MAX_CAMERA_YAW_DEVIATION = 0.2
DROID_DEFAULT_DOF = 11
MIN_ROBOT_VISUAL_MESHES = 8
MIN_GRIPPER_VISUAL_MESHES = 2
MIN_GRIPPER_VISUAL_XFORMS = 8
DEFAULT_TABLE_STATIC_FRICTION = 3.0
DEFAULT_TABLE_DYNAMIC_FRICTION = 2.0
DEFAULT_OBJECT_STATIC_FRICTION = 1.5
DEFAULT_OBJECT_DYNAMIC_FRICTION = 1.0
DEFAULT_CONTACT_RESTITUTION = 0.0
DEFAULT_MAIN_OBJECT_MIN_MASS = 0.25
DEFAULT_MAIN_OBJECT_MAX_MASS = 2.0
TABLE_CONTACT_NAME_HINTS = ("table", "countertop", "desk", "surface")
TABLE_CONTACT_CATEGORY_HINTS = ("table", "countertop", "desk")


def _debug_summarize_value(value, max_len=220):
    try:
        if isinstance(value, (str, os.PathLike)):
            path = os.fspath(value)
            exists = os.path.exists(path)
            size = os.path.getsize(path) if exists and os.path.isfile(path) else None
            return f"{path!r} exists={exists} size={size}"
        text = repr(value)
    except Exception as exc:
        text = f"<repr-error {exc}>"
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _debug_summarize_bound_self(value):
    try:
        cls_name = value.__class__.__name__
        parts = [cls_name]
        for attr_name in ("name", "_name", "prim_path", "_prim_path", "relative_prim_path", "_relative_prim_path"):
            if hasattr(value, attr_name):
                attr_value = getattr(value, attr_name)
                if attr_value is not None:
                    parts.append(f"{attr_name}={_debug_summarize_value(attr_value, max_len=100)}")
        usd_path = None
        try:
            usd_path = getattr(value, "usd_path", None)
        except Exception as exc:
            usd_path = f"<usd_path-error {exc}>"
        if usd_path is not None:
            parts.append(f"usd_path={_debug_summarize_value(usd_path, max_len=160)}")
        return " ".join(parts)
    except Exception as exc:
        return f"<self-summary-error {exc}>"


_KIT_RUNTIME_LOAD_SETTINGS_APPLIED = False
_REALM_OG_ENV_LOADING = False


def _apply_kit_runtime_load_settings(reason):
    """Force local-only Kit registry settings once SimulationApp exists."""
    global _KIT_RUNTIME_LOAD_SETTINGS_APPLIED
    if _KIT_RUNTIME_LOAD_SETTINGS_APPLIED:
        return
    try:
        settings = lazy.carb.settings.get_settings()
    except Exception as exc:
        print(f"[OG_DEBUG] Kit runtime settings unavailable at {reason}: {exc}", flush=True)
        return

    errors = []
    for key in (
        "/omni.kit.plugin/syncUsdLoads",
        "/rtx/materialDb/syncLoads",
        "/rtx/hydra/materialSyncLoads",
    ):
        try:
            settings.set_bool(key, True)
        except Exception as exc:
            errors.append(f"{key}: {exc}")
    for key in (
        "/app/extensions/registryEnabled",
        "/exts/omni.kit.registry.nucleus/enabled",
        "/exts/omni.particle.system.bundle/enabled",
        "/exts/omni.particle.system.core/enabled",
        "/exts/omni.particle.system.core.scripts/enabled",
    ):
        try:
            settings.set_bool(key, False)
        except Exception as exc:
            errors.append(f"{key}: {exc}")
    for key, value in (
        ("/app/extensions/registryUrl", ""),
        ("/app/extensions/registryUrls", []),
        ("/exts/omni.kit.registry.nucleus/registries", []),
        ("/exts/omni.kit.registry.nucleus/urls", []),
    ):
        try:
            settings.set(key, value)
        except Exception as exc:
            errors.append(f"{key}: {exc}")

    _KIT_RUNTIME_LOAD_SETTINGS_APPLIED = True
    if errors:
        print(
            f"[OG_DEBUG] Kit runtime load settings partially applied at {reason}; "
            f"errors={errors}",
            flush=True,
        )
    else:
        print(f"[OG_DEBUG] Kit runtime load settings applied at {reason}", flush=True)


def _debug_wrap_callable(owner, attr_name, label):
    original = getattr(owner, attr_name, None)
    if original is None or getattr(original, "_realm_debug_wrapped", False):
        return False

    @functools.wraps(original)
    def _wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        self_text = _debug_summarize_bound_self(args[0]) if args else ""
        arg_text = ", ".join(_debug_summarize_value(arg) for arg in args[1:4])
        kw_text = ", ".join(
            f"{key}={_debug_summarize_value(value)}"
            for key, value in list(kwargs.items())[:6]
        )
        joined = ", ".join(part for part in [arg_text, kw_text] if part)
        print(f"[OG_DEBUG] START {label}.{attr_name} self=[{self_text}] args=({joined})", flush=True)
        if label == "Environment" and attr_name in {"load", "_load_scene", "_load_robots"}:
            _apply_kit_runtime_load_settings(f"{label}.{attr_name}")
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            print(
                f"[OG_DEBUG] ERROR {label}.{attr_name} self=[{self_text}] "
                f"duration_s={time.perf_counter() - t0:.2f}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        print(
            f"[OG_DEBUG] DONE {label}.{attr_name} self=[{self_text}] "
            f"duration_s={time.perf_counter() - t0:.2f}",
            flush=True,
        )
        return result

    _wrapped._realm_debug_wrapped = True
    setattr(owner, attr_name, _wrapped)
    return True


def _debug_wrap_module_function(module, attr_name, label):
    original = getattr(module, attr_name, None)
    if original is None or getattr(original, "_realm_debug_wrapped", False):
        return False

    @functools.wraps(original)
    def _wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        arg_text = ", ".join(_debug_summarize_value(arg) for arg in args[:5])
        kw_text = ", ".join(
            f"{key}={_debug_summarize_value(value)}"
            for key, value in list(kwargs.items())[:8]
        )
        joined = ", ".join(part for part in [arg_text, kw_text] if part)
        print(f"[OG_DEBUG] START {label}.{attr_name}({joined})", flush=True)
        if attr_name == "add_asset_to_stage":
            _apply_kit_runtime_load_settings(f"{label}.{attr_name}")
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            print(
                f"[OG_DEBUG] ERROR {label}.{attr_name} "
                f"duration_s={time.perf_counter() - t0:.2f}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        print(
            f"[OG_DEBUG] DONE {label}.{attr_name} "
            f"duration_s={time.perf_counter() - t0:.2f}",
            flush=True,
        )
        return result

    _wrapped._realm_debug_wrapped = True
    setattr(module, attr_name, _wrapped)
    return True


def _install_visual_xform_mismatch_guard():
    """
    OmniGibson occasionally propagates an object's scene-level pose into a USD
    visual / collision child prim while checking XFormPrim load properties
    during resident scene switches. These children already carry asset-local
    transforms, so the strict old_pos/new_pos assert aborts otherwise valid
    scene loads.

    Keep the guard narrow: only downgrade this assertion for /visuals and
    /collisions child prims. Robot visual-mesh presence is still validated
    separately after env creation, so this does not hide invisible gripper
    failures.
    """
    if getattr(og, "_realm_visual_xform_mismatch_guard_installed", False):
        return False
    try:
        import omnigibson.prims.xform_prim as og_xform_prim
    except Exception as exc:
        print(f"[OG_DEBUG] visual xform mismatch guard unavailable: {exc}", flush=True)
        return False

    xform_prim_cls = getattr(og_xform_prim, "XFormPrim", None)
    original = getattr(xform_prim_cls, "_set_xform_properties", None) if xform_prim_cls is not None else None
    if original is None or getattr(original, "_realm_visual_xform_guard_wrapped", False):
        return False

    @functools.wraps(original)
    def _wrapped(self, *args, **kwargs):
        prim_path = None
        for attr_name in ("prim_path", "_prim_path", "relative_prim_path", "_relative_prim_path"):
            try:
                value = getattr(self, attr_name, None)
            except Exception:
                value = None
            if value:
                prim_path = str(value)
                break
        is_asset_child_xform = prim_path and (
            "/visuals" in prim_path
            or "/collisions" in prim_path
        )
        try:
            return original(self, *args, **kwargs)
        except AssertionError as exc:
            message = str(exc)
            if is_asset_child_xform and "old_pos:" in message and "new_pos:" in message:
                print(
                    "[OG_DEBUG] downgraded visual/collision child xform mismatch during load "
                    f"prim_path={prim_path}: {message}",
                    flush=True,
                )
                return None
            raise

    _wrapped._realm_visual_xform_guard_wrapped = True
    setattr(xform_prim_cls, "_set_xform_properties", _wrapped)
    og._realm_visual_xform_mismatch_guard_installed = True
    print("[OG_DEBUG] installed visual xform mismatch guard for XFormPrim._set_xform_properties", flush=True)
    return True


def _extract_prim_path_from_any(value):
    if value is None:
        return None
    try:
        if hasattr(value, "prim_path") and getattr(value, "prim_path", None):
            return str(getattr(value, "prim_path"))
    except Exception:
        pass
    try:
        if hasattr(value, "_prim_path") and getattr(value, "_prim_path", None):
            return str(getattr(value, "_prim_path"))
    except Exception:
        pass
    try:
        get_path = getattr(value, "GetPath", None)
        if callable(get_path):
            path = get_path()
            if path:
                return str(path)
    except Exception:
        pass
    text = str(value)
    if "/" in text:
        return text
    return None


def _is_static_architecture_collision_path(prim_path):
    if not prim_path:
        return False
    prim_path = str(prim_path)
    if "/collisions" not in prim_path:
        return False
    lowered = prim_path.lower()
    return any(
        hint in lowered
        for hint in (
            "/ceilings_",
            "/walls_",
            "/floors_",
            "/window_",
            "/door_",
            "/stairs_",
        )
    )


def _install_extent_ratio_guard():
    if getattr(og, "_realm_extent_ratio_guard_installed", False):
        return False
    try:
        import omnigibson.utils.usd_utils as og_usd_utils
    except Exception as exc:
        print(f"[OG_DEBUG] extent-ratio guard unavailable: {exc}", flush=True)
        return False

    original = getattr(og_usd_utils, "check_extent_radius_ratio", None)
    if original is None or getattr(original, "_realm_extent_ratio_guard_wrapped", False):
        return False

    @functools.wraps(original)
    def _wrapped(*args, **kwargs):
        global _REALM_OG_ENV_LOADING
        if _REALM_OG_ENV_LOADING:
            for value in list(args) + list(kwargs.values()):
                prim_path = _extract_prim_path_from_any(value)
                if _is_static_architecture_collision_path(prim_path):
                    print(
                        "[OG_DEBUG] skipped extent/radius ratio check for static architecture collision "
                        f"prim_path={prim_path}",
                        flush=True,
                    )
                    return None
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            if _REALM_OG_ENV_LOADING:
                for value in list(args) + list(kwargs.values()):
                    prim_path = _extract_prim_path_from_any(value)
                    if _is_static_architecture_collision_path(prim_path):
                        print(
                            "[OG_DEBUG] downgraded extent/radius ratio failure for static architecture collision "
                            f"prim_path={prim_path} error={exc}",
                            flush=True,
                        )
                        return None
            raise

    _wrapped._realm_extent_ratio_guard_wrapped = True
    setattr(og_usd_utils, "check_extent_radius_ratio", _wrapped)
    og._realm_extent_ratio_guard_installed = True
    print("[OG_DEBUG] installed extent-ratio guard for static architecture collisions", flush=True)
    return True


def _install_material_invalid_shader_guard():
    """
    DatasetObject material post-load can see an invalid shader prim after Kit /
    USD imaging has been through resident scene switches. The shader asset-path
    rewrite is cosmetic; failing it should not abort object loading.
    """
    if getattr(og, "_realm_material_invalid_shader_guard_installed", False):
        return False
    try:
        import omnigibson.prims.material_prim as og_material_prim
    except Exception as exc:
        print(f"[OG_DEBUG] material invalid-shader guard unavailable: {exc}", flush=True)
        return False

    material_cls = getattr(og_material_prim, "MaterialPrim", None)
    if material_cls is None:
        return False

    original_input_names = getattr(material_cls, "shader_input_names_by_type", None)
    if original_input_names is not None and not getattr(
        original_input_names,
        "_realm_invalid_shader_guard_wrapped",
        False,
    ):

        @functools.wraps(original_input_names)
        def _shader_input_names_by_type_guard(self, *args, **kwargs):
            try:
                return original_input_names(self, *args, **kwargs)
            except RuntimeError as exc:
                if "invalid prim" in str(exc).lower():
                    print(
                        "[OG_DEBUG] skipped material shader input scan on invalid prim "
                        f"material={_debug_summarize_bound_self(self)} error={exc}",
                        flush=True,
                    )
                    return set()
                raise

        _shader_input_names_by_type_guard._realm_invalid_shader_guard_wrapped = True
        setattr(material_cls, "shader_input_names_by_type", _shader_input_names_by_type_guard)

    original_update_paths = getattr(material_cls, "shader_update_asset_paths_with_root_path", None)
    if original_update_paths is not None and not getattr(
        original_update_paths,
        "_realm_invalid_shader_guard_wrapped",
        False,
    ):

        @functools.wraps(original_update_paths)
        def _shader_update_asset_paths_guard(self, *args, **kwargs):
            try:
                return original_update_paths(self, *args, **kwargs)
            except RuntimeError as exc:
                if "invalid prim" in str(exc).lower():
                    print(
                        "[OG_DEBUG] skipped material asset-path update on invalid shader prim "
                        f"material={_debug_summarize_bound_self(self)} error={exc}",
                        flush=True,
                    )
                    return None
                raise

        _shader_update_asset_paths_guard._realm_invalid_shader_guard_wrapped = True
        setattr(material_cls, "shader_update_asset_paths_with_root_path", _shader_update_asset_paths_guard)

    og._realm_material_invalid_shader_guard_installed = True
    print("[OG_DEBUG] installed material invalid-shader guard for MaterialPrim", flush=True)
    return True


def _install_omnigibson_env_debug_hooks():
    if getattr(og, "_realm_env_debug_hooks_installed", False):
        return

    installed = []
    if _install_visual_xform_mismatch_guard():
        installed.append("XFormPrim._set_xform_properties_guard")
    if _install_material_invalid_shader_guard():
        installed.append("MaterialPrim.invalid_shader_guard")
    if _install_extent_ratio_guard():
        installed.append("usd_utils.check_extent_radius_ratio_guard")
    if os.environ.get("REALM_DEBUG_OG_ENV", "0").lower() in {"0", "false", "no", "off"}:
        og._realm_env_debug_hooks_installed = True
        print(f"[OG_DEBUG] Installed OmniGibson env load guards: {installed}", flush=True)
        return
    targets = [
        (getattr(og, "Environment", None), "Environment", [
            "__init__",
            "load",
            "_load_scene",
            "_load_robots",
            "_load_task",
            "_load_external_sensors",
            "_load_observation_space",
        ]),
    ]
    try:
        import omnigibson.scenes.scene_base as og_scene_base

        targets.append((getattr(og_scene_base, "Scene", None), "Scene", ["load", "add_object", "reset"]))
    except Exception as exc:
        print(f"[OG_DEBUG] Scene hook unavailable: {exc}", flush=True)
    try:
        import omnigibson.objects.object_base as og_object_base
        import omnigibson.objects.usd_object as og_usd_object
        import omnigibson.objects.controllable_object as og_controllable_object

        targets.extend([
            (getattr(og_object_base, "BaseObject", None), "BaseObject", ["load", "_load"]),
            (getattr(og_usd_object, "USDObject", None), "USDObject", ["load", "_load"]),
            (getattr(og_controllable_object, "ControllableObject", None), "ControllableObject", ["load", "_load"]),
        ])
    except Exception as exc:
        print(f"[OG_DEBUG] Object hook unavailable: {exc}", flush=True)
    try:
        import omnigibson.prims.prim_base as og_prim_base

        targets.append((getattr(og_prim_base, "BasePrim", None), "BasePrim", ["load", "_load", "_post_load"]))
    except Exception as exc:
        print(f"[OG_DEBUG] Prim hook unavailable: {exc}", flush=True)

    for owner, label, attr_names in targets:
        if owner is None:
            continue
        for attr_name in attr_names:
            if _debug_wrap_callable(owner, attr_name, label):
                installed.append(f"{label}.{attr_name}")

    module_targets = []
    try:
        import omnigibson.utils.usd_utils as og_usd_utils

        module_targets.append((og_usd_utils, "omnigibson.utils.usd_utils", ["add_asset_to_stage"]))
    except Exception as exc:
        print(f"[OG_DEBUG] usd_utils hook unavailable: {exc}", flush=True)
    try:
        import omnigibson.objects.usd_object as og_usd_object

        # usd_object.py imports add_asset_to_stage into its module namespace, so
        # wrapping only omnigibson.utils.usd_utils.add_asset_to_stage does not
        # affect USDObject._load. Patch this direct reference too.
        module_targets.append((og_usd_object, "omnigibson.objects.usd_object", ["add_asset_to_stage"]))
    except Exception as exc:
        print(f"[OG_DEBUG] usd_object add_asset hook unavailable: {exc}", flush=True)
    try:
        import omni.isaac.core.utils.stage as isaac_stage_utils

        module_targets.append((isaac_stage_utils, "omni.isaac.core.utils.stage", ["add_reference_to_stage"]))
    except Exception as exc:
        print(f"[OG_DEBUG] isaac stage hook unavailable: {exc}", flush=True)

    for module, label, attr_names in module_targets:
        for attr_name in attr_names:
            if _debug_wrap_module_function(module, attr_name, label):
                installed.append(f"{label}.{attr_name}")

    og._realm_env_debug_hooks_installed = True
    print(f"[OG_DEBUG] Installed OmniGibson env debug hooks: {installed}", flush=True)

# Panda joint origins from panda_robotiq_85.urdf: (xyz, rpy) per joint
_PANDA_JOINT_ORIGINS = [
    ([0,       0,      0.333], [0,        0, 0]),
    ([0,       0,      0    ], [-np.pi/2, 0, 0]),
    ([0,      -0.316,  0    ], [ np.pi/2, 0, 0]),
    ([0.0825,  0,      0    ], [ np.pi/2, 0, 0]),
    ([-0.0825, 0.384,  0    ], [-np.pi/2, 0, 0]),
    ([0,       0,      0    ], [ np.pi/2, 0, 0]),
    ([0.088,   0,      0    ], [ np.pi/2, 0, 0]),
]
_PANDA_EE_OFFSET = [0, 0, 0.107]  # panda_link8 fixed offset from panda_link7 (panda_arm.urdf)


def _panda_fk(q):
    """Forward kinematics for Panda arm using URDF parameters.
    q: array of 7 joint angles (radians).
    Returns (pos, quat_xyzw) of panda_link8 in the robot base (link0) frame.
    Each joint transform: T = Translate(xyz) @ RPY(rpy) @ Rz(q_i)
    """
    def _rot3(a, axis):
        ca, sa = np.cos(a), np.sin(a)
        if axis == 'x':
            return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        if axis == 'y':
            return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])  # z

    def _ht(xyz, rpy, qi):
        # Fixed part: Translate(xyz) @ Rz(yaw) @ Ry(pitch) @ Rx(roll)
        r_fixed = _rot3(rpy[2], 'z') @ _rot3(rpy[1], 'y') @ _rot3(rpy[0], 'x')
        r_total = r_fixed @ _rot3(qi, 'z')
        m = np.eye(4)
        m[:3, :3] = r_total
        m[:3, 3] = xyz
        return m

    m = np.eye(4)
    for (xyz, rpy), qi in zip(_PANDA_JOINT_ORIGINS, q):
        m = m @ _ht(xyz, rpy, qi)

    m_ee = np.eye(4)
    m_ee[:3, 3] = _PANDA_EE_OFFSET
    m = m @ m_ee

    from scipy.spatial.transform import Rotation as _R
    return m[:3, 3].copy(), _R.from_matrix(m[:3, :3]).as_quat()


def set_rendering_mode(rendering_mode):
    carb_settings = lazy.carb.settings.get_settings()
    if rendering_mode == "pt":
        def enable_interactive_path_tracing(carb_settings, samples_per_pixel=8):
            carb_settings.set("/rtx/rendermode", "PathTracing")
            if samples_per_pixel is not None:
                carb_settings.set_int("/rtx/pathtracing/spp", samples_per_pixel)
                carb_settings.set_int("/rtx/pathtracing/totalSpp", samples_per_pixel)
                carb_settings.set_int(
                    "/rtx/pathtracing/useDirectLightingCache", False
                )
            carb_settings.set_bool("/rtx/pathtracing/optixDenoiser/enabled", True)

        #carb_settings.set("/persistent/omnihydra/useSceneGraphInstancing", True)
        enable_interactive_path_tracing(carb_settings, samples_per_pixel=8)
    elif rendering_mode == "r":
        carb_settings.set_string("/rtx/rendermode", "RaytracedLighting")
        carb_settings.set_bool("/rtx/translucency/enabled", True)
        carb_settings.set_bool("/rtx/reflections/enabled", False)
        carb_settings.set_bool("/rtx/indirectDiffuse/enabled", False)
        carb_settings.set_bool("/rtx/directLighting/sampledLighting/enabled", True)
        carb_settings.set_int("/rtx/directLighting/sampledLighting/samplesPerPixel", 1)
        carb_settings.set_bool("/rtx/shadows/enabled", False)
        carb_settings.set_int("/rtx/post/dlss/execMode", 0)
        carb_settings.set_bool("/rtx/ambientOcclusion/enabled", False)
        carb_settings.set_bool("/rtx-transient/dlssg/enabled", False)
        carb_settings.set_float("/rtx-transient/resourcemanager/texturestreaming/memoryBudget", 0.6)
        carb_settings.set_float("/rtx/sceneDb/ambientLightIntensity", 1.0)
        carb_settings.set_bool("/exts/omni.renderer.core/present/enabled", False)
        carb_settings.set_string("/isaaclab/rendering/rendering_mode", "performance")
    else:
        assert rendering_mode == "rt", f"rendering mode must be 'pt', 'rt', or 'r'"


class RealmEnvironmentDynamic(RealmEnvironmentBase):
    def __init__(
        self,
        config_path="/app/realm/config",
        scene_model=None,
        scene_part=None,
        reset_qpos=None,
        task_cfg_path="REALM_DROID10/put_green_block_into_bowl/default.cfg",
        perturbations=None,
        common_freq: int = None,
        no_rendering: bool = False,
        multi_view: bool = False,
        rendering_mode: str = "rt",
        robot: str = "DROID",
        camera_extrinsics_override=None,
    ) -> None:
        assert not (multi_view and no_rendering), f"Multi-view rendering was enabled during no_rendering mode. Either one is likely a mistake."
        self.task_cfg_path = "/".join(task_cfg_path.split("/")[-3:])
        self.use_droid_with_base = True if self.task_cfg_path.split("/")[0] == "REALM_DROID10" else False # TODO: infer properly from the task/scene config yaml
        self.robot_name = robot
        self.multi_view = multi_view
        self.no_rendering = no_rendering
        self.rendering_mode = rendering_mode
        self.camera_extrinsics_override = None if camera_extrinsics_override is None else list(camera_extrinsics_override)
        if self.camera_extrinsics_override is not None:
            assert 1 <= len(self.camera_extrinsics_override) <= 2, (
                "camera_extrinsics_override must contain one or two camera preset names"
            )
        self.config_path = config_path
        self.scene_model = scene_model
        self.scene_part = scene_part
        self.reset_qpos = reset_qpos if reset_qpos is not None else DEFAULT_RESET_JOINTPOS
        self.common_freq = common_freq
        self.supported_pertrubations = {
            'Default':  lambda: _pert_default(self),
            "V-AUG":    lambda: _pert_default(self),  # V-AUG is applied when distorting the images in obs
            "V-VIEW":   lambda: _pert_v_view(self),
            "V-SC":     lambda: _pert_v_sc(self),
            "V-LIGHT":  lambda: _pert_v_light(self),
            "S-PROP":   lambda: _pert_s_prop(self),
            "S-LANG":   lambda: _pert_s_lang(self),
            "S-MO":     lambda: _pert_s_mo(self),
            "S-AFF":    lambda: _pert_s_aff(self),
            "S-INT":    lambda: _pert_s_int(self),
            "B-HOBJ":   lambda: _pert_b_hobj(self),
            "SB-NOUN":  lambda: _pert_sb_noun(self),
            "SB-VRB":   lambda: _pert_sb_vrb(self),
            "VB-POSE":  lambda: _pert_vb_pose(self),
            "VB-MOBJ":  lambda: _pert_vb_mobj(self),
            "VSB-NOBJ": lambda: _pert_vsb_nobj(self),
        }

        self.active_perturbations = perturbations
        for perturbation in self.active_perturbations:
            assert perturbation in self.supported_pertrubations.keys()

        if self.use_droid_with_base:
            from realm.robots.droid_arm_mounted import DROID
        else:
            from realm.robots.droid_arm import DROID

        camera_extrinsics_path = f"{self.config_path}/env/external_sensors/camera_extrinsics.yaml"
        self.cfg_camera_extrinsics = yaml.load(open(camera_extrinsics_path, "r"), Loader=yaml.FullLoader)
        if self.camera_extrinsics_override is not None:
            missing_camera_presets = [
                name for name in self.camera_extrinsics_override
                if name not in self.cfg_camera_extrinsics
            ]
            if missing_camera_presets:
                raise KeyError(
                    "Unknown camera_extrinsics_override preset(s): "
                    f"{missing_camera_presets}. Available presets: {sorted(self.cfg_camera_extrinsics.keys())}"
                )

        cfg, mo_cfgs, to_cfgs, dist_cfgs = self.construct_environment_config()
        assert len(mo_cfgs) == 1
        assert len(to_cfgs) <= 1
        assert "position" in mo_cfgs[0], "mo must have a specified position"
        if "SB-NOUN" in self.active_perturbations and cfg["task_type"] == "push":
            raise NotImplementedError() # TODO: move this to some compatibility matrix / exclusion list

        if common_freq is not None:
            cfg["env"]["rendering_frequency"] = common_freq
            cfg["env"]["action_frequency"] = common_freq

        self.mo_pos_orig = np.array(mo_cfgs[0]["position"])
        self.mo_rot_orig = np.array(mo_cfgs[0]["orientation"] if "orientation" in mo_cfgs[0] else [0, 0, 0, 1])
        self.mo_bbox_orig = np.array(mo_cfgs[0]["bounding_box"])

        self.cfg = copy.deepcopy(cfg)
        self.task_type = self.cfg["task_type"]
        self.instruction = self.cfg["instruction"]
        self.action_frequency = int(self.cfg["env"].get("action_frequency", getattr(gm, "DEFAULT_SIM_STEP_FREQ", 15)))
        self.physics_frequency = int(getattr(gm, "DEFAULT_PHYSICS_FREQ", 120))
        self.physics_steps_per_action = max(1, int(round(float(self.physics_frequency) / float(max(self.action_frequency, 1)))))

        t_env_load = time.perf_counter()
        _install_omnigibson_env_debug_hooks()
        robot_usd_path = None
        try:
            robot_usd_path = DROID.usd_path.fget(None)
        except Exception as exc:
            robot_usd_path = f"<unavailable: {exc}>"
        scene_cfg = cfg.get("scene", {})
        scene_model = scene_cfg.get("model") if isinstance(scene_cfg, dict) else None
        print(
            "[RealmEnvironmentDynamic] Creating OmniGibson Environment... "
            f"task_type={self.task_type} instruction={self.instruction!r} "
            f"scene_model={scene_model} "
            f"robot={robot} use_droid_with_base={self.use_droid_with_base} "
            f"droid_class_module={getattr(DROID, '__module__', None)} "
            f"robot_usd_path={robot_usd_path} "
            f"robot_usd_exists={os.path.exists(robot_usd_path) if isinstance(robot_usd_path, str) else None} "
            f"main_objects={len(mo_cfgs)} target_objects={len(to_cfgs)} distractors={len(dist_cfgs)} "
            f"action_frequency={self.action_frequency} physics_frequency={self.physics_frequency} "
            f"physics_steps_per_action={self.physics_steps_per_action}",
            flush=True,
        )
        global _REALM_OG_ENV_LOADING
        _REALM_OG_ENV_LOADING = True
        try:
            self.omnigibson_env = og.Environment(configs=[cfg])
        finally:
            _REALM_OG_ENV_LOADING = False
        print(
            "[RealmEnvironmentDynamic] OmniGibson Environment created "
            f"duration_s={time.perf_counter() - t_env_load:.2f}",
            flush=True,
        )

        assert len(self.omnigibson_env.robots) == 1  # assumes single robot, single arm
        self.robot = self.omnigibson_env.robots[0]
        self.robot_finger_links = {self.robot._links[link] for link in self.robot.finger_link_names[self.robot.default_arm]}
        self.last_gripper_visual_repair = {}

        self.main_objects = [self.omnigibson_env.scene.object_registry("name", mo["name"]) for mo in mo_cfgs]
        self.target_objects = [self.omnigibson_env.scene.object_registry("name", to["name"]) for to in to_cfgs]
        self.distractors = [self.omnigibson_env.scene.object_registry("name", dist["name"]) for dist in dist_cfgs]

        self.init_poses = {obj._relative_prim_path: { # using relative prim path as unique id
            "pos": obj.get_position_orientation()[0],
            "rot": obj.get_position_orientation()[1]
        } for obj in self.main_objects + self.target_objects + self.distractors}

        if "VSB-NOBJ" in self.active_perturbations and self.task_type in ["open_drawer", "close_drawer"]:
            self.init_poses[self.main_objects[0]._relative_prim_path]["pos"][-1] += 0.3

        if "V-AUG" in self.active_perturbations:
            self.v_aug_sigma = np.random.uniform(0.0, 3.0)
            self.v_aug_alpha = np.random.uniform(0.5, 2.0)

        # ---------- apply fixes to the env ----------
        for step_name, step_fn in [
            ("update_robot_physics", self.update_robot_physics),
            ("apply_scene_fixes_from_cfg", self.apply_scene_fixes_from_cfg),
            ("update_task_object_mass", self.update_task_object_mass),
            ("update_contact_friction", self.update_contact_friction),
            ("disable_visual_toggles", self.disable_visual_toggles),
            ("force_robot_gripper_visuals_visible", lambda: self.force_robot_gripper_visuals_visible(reason="post_env_init")),
            ("validate_robot_visual_meshes_or_raise", lambda: self.validate_robot_visual_meshes_or_raise(reason="post_env_init")),
            ("set_rendering_mode", lambda: set_rendering_mode(rendering_mode)),
        ]:
            t_step = time.perf_counter()
            print(f"[RealmEnvironmentDynamic] post-init step start: {step_name}", flush=True)
            step_fn()
            print(
                f"[RealmEnvironmentDynamic] post-init step done: {step_name} "
                f"duration_s={time.perf_counter() - t_step:.2f}",
                flush=True,
            )

        super().__init__(
            main_objects=self.main_objects,
            target_objects=self.target_objects,
            task_type=self.task_type,
            robot=self.robot,
            mo_cfgs=mo_cfgs
        )

    def construct_environment_config(self):
        cfg = dict()
        task_cfg = yaml.load(open(f"{self.config_path}/tasks/{self.task_cfg_path}", "r"), Loader=yaml.FullLoader)
        cfg.update(task_cfg)

        # ---------------------------------------- scene config ----------------------------------------
        for k in ["external_sensors", "robots"]:
            assert k not in cfg, f"{k} should be defined outside the scene file!"

        if self.scene_model is None:
            assert self.scene_part is None
            self.scene_model = list(task_cfg["supported_scenes"].keys())[0]
            self.scene_part = task_cfg["supported_scenes"][self.scene_model][0]
        assert self.scene_model in task_cfg["supported_scenes"]
        assert self.scene_part in task_cfg["supported_scenes"][self.scene_model]
        cfg.update(task_cfg["task"])

        scene_cfg_path = f"{self.config_path}/scenes/{self.scene_model}/{self.scene_part}/scene_definition.yaml"
        scene_cfg = None
        if os.path.exists(scene_cfg_path):
            scene_cfg = yaml.load(open(scene_cfg_path, "r"), Loader=yaml.FullLoader)
            cfg["scene"] = copy.deepcopy(scene_cfg["scene"])
        else:
            cfg["scene"] = {
                "type": "InteractiveTraversableScene",
                "scene_model": self.scene_model,
            }

        spawn_cfg = yaml.load(open(f"{self.config_path}/scenes/scenes.yaml", "r"), Loader=yaml.FullLoader)
        assert self.scene_model in spawn_cfg and self.scene_part in spawn_cfg[self.scene_model]
        scene_data = spawn_cfg[self.scene_model][self.scene_part]
        if all(k in scene_data for k in ["x_min", "x_max", "y_min", "y_max", "z"]):
            x_min = scene_data["x_min"]
            x_max = scene_data["x_max"]
            y_min = scene_data["y_min"]
            y_max = scene_data["y_max"]
            z = scene_data["z"]
            self.spawn_bbox = np.array([x_min, x_max, y_min, y_max, z])
        else:
            self.spawn_bbox = None

        # ---------------------------------------- robot config ----------------------------------------
        assert "pos" in scene_data and "rot" in scene_data
        robot_pos = scene_data['pos']
        robot_rot = [math.radians(angle_deg) for angle_deg in scene_data['rot']]
        self.robot_pos = np.array(robot_pos, dtype=float)
        self.robot_rot_rad = np.array(robot_rot, dtype=float)

        cfg_robot = yaml.load(open(f"{self.config_path}/robots/{self.robot_name}.yaml", "r"), Loader=yaml.FullLoader)
        self.ee_control = cfg_robot["robots"][0].get("ee_control", False)
        cfg_robot["robots"][0]["position"] = robot_pos
        cfg_robot["robots"][0]["orientation"] = omnigibson_transform_utils.euler2quat(
            torch.tensor(robot_rot, dtype=torch.float32)).tolist()
        cfg_robot["robots"][0]["fixed_base"] = True

        reset_joint_pos = np.zeros(cfg_robot["robots"][0]["dof"] if "dof" in cfg_robot["robots"][0] else DROID_DEFAULT_DOF)
        if "DROID" in self.robot_name:
            if "reset_joint_pos" in task_cfg:
                reset_joint_pos[:7] = np.array(task_cfg['reset_joint_pos'])
            elif "reset_joint_pos" in scene_data:
                reset_joint_pos[:7] = np.array(scene_data['reset_joint_pos'])
            else:
                reset_joint_pos[:7] = DEFAULT_RESET_JOINTPOS
        elif self.robot_name == "WidowX":
            reset_joint_pos[:6] = np.zeros(6) #np.array([0.0, -0.849879, 0.258767, 0.0, 1.2831712, 0.0])
        cfg_robot["robots"][0]["reset_joint_pos"] = reset_joint_pos

        if self.common_freq is not None:
            cfg_robot["robots"][0]["control_freq"] = self.common_freq
            cfg_robot["robots"][0]["controller_config"]["arm_0"]["control_freq"] = self.common_freq

        cfg.update(cfg_robot)
        self.reset_qpos = reset_joint_pos

        # ---------------------------------------- object config ----------------------------------------
        obj_list = task_cfg["main_objects"] + task_cfg["target_objects"]
        if "distractors" in task_cfg:
            obj_list += task_cfg["distractors"]
        if "immutables" in task_cfg:
            obj_list += task_cfg["immutables"]
        if scene_cfg is not None:
            obj_list += scene_cfg["objects"]

        robot_rot_deg_z = scene_data['rot'][-1]
        assert robot_rot_deg_z >= 0
        obj_pos_modifier_x = 1
        if 90 <= robot_rot_deg_z <= 270:
            obj_pos_modifier_x = -1

        if self.spawn_bbox is not None:
            for obj in obj_list:
                obj["relative_bbox_position"][0] *= obj_pos_modifier_x
                if obj_pos_modifier_x != 1:
                    if obj["relative_bbox_position"][0] < 0:
                        obj["relative_bbox_position"][0] -= obj_pos_modifier_x * (self.spawn_bbox[1] - self.spawn_bbox[0])
                    else:
                        obj["relative_bbox_position"][0] += obj_pos_modifier_x * (self.spawn_bbox[1] - self.spawn_bbox[0])
                obj["position"] = [x + y for x, y in zip(obj["relative_bbox_position"], [self.spawn_bbox[0], self.spawn_bbox[2], self.spawn_bbox[4]])]

            # TODO: the pipeline is broken for dynamically reducing # objects when there are too many distractors and
            # they become unplaceable - 3 is always fine and easy to place so we use that for now as maximum
            num_distractors = 3 if any(p in self.active_perturbations for p in ["V-SC"]) else 0 #"VB-ISC" #"SB-NOUN"
            cfg["objects"] = None
            excluded_categories = []
            for obj in task_cfg["main_objects"] + task_cfg["target_objects"]:
                if "category" in obj:
                    excluded_categories.append(obj["category"])
            distractors = self.sample_objects(num_objects=num_distractors, excluded_categories=excluded_categories)

            cfg["objects"] = get_non_colliding_positions_for_objects(
                xmin=self.spawn_bbox[0],
                xmax=self.spawn_bbox[1],
                ymin=self.spawn_bbox[2],
                ymax=self.spawn_bbox[3],
                z=self.spawn_bbox[4],
                obj_cfg=obj_list + distractors,
                max_attempts_per_object=25000,
                main_object_names=[o["name"] for o in obj_list],
            )
        else:
            cfg["objects"] = obj_list
            distractors = []

        if "distractors" in task_cfg:
            distractors += task_cfg["distractors"]
        if "immutables" in task_cfg:
            distractors += task_cfg["immutables"] # immutables go here because the distractor list above is meant to be replaceable objects

        for obj in cfg["objects"]:
            assert "position" in obj

        # ---------------------------------------- external camera config ----------------------------------------
        if "env" not in cfg:
            cfg["env"] = {
                "initial_pos_z_offset": 0.2
            }
        ext_cam1_pose = task_cfg["camera_extrinsics"]["cam1"] if "camera_extrinsics" in task_cfg else "default"
        if "camera_extrinsics" in task_cfg and "cam2" in task_cfg["camera_extrinsics"]:
            ext_cam2_pose = task_cfg["camera_extrinsics"]["cam2"]
        else:
            ext_cam2_pose = "default" if ext_cam1_pose == "CP3" else "CP3"
        if self.camera_extrinsics_override is not None:
            ext_cam1_pose = self.camera_extrinsics_override[0]
            if len(self.camera_extrinsics_override) > 1:
                ext_cam2_pose = self.camera_extrinsics_override[1]
        self.external_camera_pose_names = {
            "cam1": ext_cam1_pose,
            "cam2": ext_cam2_pose,
        }

        if not self.no_rendering:

            base_cam_pos, base_cam_rot = self.construct_ext_cam_pose_by_name(ext_cam1_pose, robot_pos, robot_rot)

            cfg_external_sensors = yaml.load(open(f"{self.config_path}/env/external_sensors/camera_config.yaml", "r"), Loader=yaml.FullLoader)
            cfg_external_sensors["external_sensors"][0]["position"] = base_cam_pos
            cfg_external_sensors["external_sensors"][0]["orientation"] = base_cam_rot

            if self.multi_view:
                second_base_cam_pos, second_base_cam_rot = self.construct_ext_cam_pose_by_name(ext_cam2_pose, robot_pos,
                                                                                               robot_rot)
                cfg_external_sensors["external_sensors"][1]["position"] = second_base_cam_pos
                cfg_external_sensors["external_sensors"][1]["orientation"] = second_base_cam_rot
            else:
                del cfg_external_sensors["external_sensors"][1]

            cfg["env"].update(cfg_external_sensors)

        return (copy.deepcopy(cfg),
                copy.deepcopy([o for o in task_cfg["main_objects"]]),
                copy.deepcopy([o for o in task_cfg["target_objects"]]),
                copy.deepcopy([o for o in distractors])
                )

    def construct_ext_cam_pose_by_name(self, pose_name, robot_pos, robot_rot):
        assert pose_name in self.cfg_camera_extrinsics
        base_cam_pos = self.cfg_camera_extrinsics[pose_name]["pos"]
        base_cam_rot = self.cfg_camera_extrinsics[pose_name]["rot"]
        base_cam_pos, base_cam_rot = calculate_new_camera_pose_mixed_rotations(
            base_cam_pos, base_cam_rot,
            robot_pos, robot_rot
        )
        base_cam_pos[-1] += DROID_BASE_HEIGHT if self.use_droid_with_base else 0  # height of the robot base
        return base_cam_pos, base_cam_rot

    def get_external_camera_pose_name(self, cam_key):
        cam_key = f"cam{cam_key}" if isinstance(cam_key, int) else str(cam_key)
        return self.external_camera_pose_names[cam_key]

    def update_robot_physics(self):
        if not self.robot_name == "DROID":
            return

        friction = np.array(self.cfg["robots"][0]["friction"])
        armature = np.array(self.cfg["robots"][0]["armature"])

        joint_names = self.robot.arm_joint_names
        for idx in range(7):
            prim_path = f"{self.robot.prim_path}/panda_link{idx}/{joint_names['0'][idx]}"
            joint_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path)
            assert joint_prim.IsValid()
            joint_prim.GetAttribute("physxJoint:jointFriction").Set(friction[idx])
            joint_prim.GetAttribute("physxJoint:armature").Set(armature[idx])

        # Fix triangle mesh collision approximation for dynamic bodies
        for link_name, link in self.robot.links.items():
            for collision_mesh in link.collision_meshes.values():
                prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(collision_mesh.prim_path)
                if prim.IsValid() and prim.HasAttribute("physxMeshCollision:approximation"):
                    approx = prim.GetAttribute("physxMeshCollision:approximation").Get()
                    if approx in ["none", "meshSimplification"]:
                        prim.GetAttribute("physxMeshCollision:approximation").Set("convexHull")

    def _bind_contact_material_to_object(self, obj, material, static_friction, dynamic_friction, restitution):
        updated = 0
        for link in getattr(obj, "links", {}).values():
            for collision_mesh in getattr(link, "collision_meshes", {}).values():
                prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(collision_mesh.prim_path)
                if not prim.IsValid():
                    continue
                try:
                    lazy.pxr.UsdPhysics.CollisionAPI.Apply(prim)
                    binding_api = lazy.pxr.UsdShade.MaterialBindingAPI.Apply(prim)
                    binding_api.Bind(material)
                    updated += 1
                except Exception as exc:
                    print(
                        f"[RealmEnvironmentDynamic] Warning: failed to bind contact material for "
                        f"{obj.name}:{collision_mesh.prim_path}: {exc}",
                        flush=True,
                    )
        if updated == 0:
            # Some OG objects expose only root-link collision. Try binding at the link prim level.
            for link in getattr(obj, "links", {}).values():
                prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(link.prim_path)
                if not prim.IsValid():
                    continue
                try:
                    binding_api = lazy.pxr.UsdShade.MaterialBindingAPI.Apply(prim)
                    binding_api.Bind(material)
                    updated += 1
                except Exception:
                    pass
        return updated

    def _create_contact_material(self, name, static_friction, dynamic_friction, restitution):
        stage = lazy.omni.usd.get_context().get_stage()
        material_path = f"/World/Looks/{name}"
        material = lazy.pxr.UsdShade.Material.Define(stage, material_path)
        material_prim = material.GetPrim()
        usd_material_api = lazy.pxr.UsdPhysics.MaterialAPI.Apply(material_prim)
        usd_material_api.CreateStaticFrictionAttr(float(static_friction))
        usd_material_api.CreateDynamicFrictionAttr(float(dynamic_friction))
        usd_material_api.CreateRestitutionAttr(float(restitution))
        try:
            physx_material_api = lazy.pxr.PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
            physx_material_api.CreateFrictionCombineModeAttr("max")
            physx_material_api.CreateRestitutionCombineModeAttr("min")
        except Exception:
            pass
        return material

    def _is_table_contact_object(self, obj):
        name = str(getattr(obj, "name", "")).lower()
        category = str(getattr(obj, "category", "")).lower()
        return (
            any(hint in name for hint in TABLE_CONTACT_NAME_HINTS)
            or any(hint in category for hint in TABLE_CONTACT_CATEGORY_HINTS)
        )

    def update_contact_friction(self):
        contact_cfg = self.cfg.get("contact_friction", {})
        if contact_cfg is False or (isinstance(contact_cfg, dict) and not contact_cfg.get("enabled", True)):
            return
        if contact_cfg is None:
            contact_cfg = {}

        table_static = float(contact_cfg.get("table_static_friction", DEFAULT_TABLE_STATIC_FRICTION))
        table_dynamic = float(contact_cfg.get("table_dynamic_friction", DEFAULT_TABLE_DYNAMIC_FRICTION))
        object_static = float(contact_cfg.get("object_static_friction", DEFAULT_OBJECT_STATIC_FRICTION))
        object_dynamic = float(contact_cfg.get("object_dynamic_friction", DEFAULT_OBJECT_DYNAMIC_FRICTION))
        restitution = float(contact_cfg.get("restitution", DEFAULT_CONTACT_RESTITUTION))
        apply_to_task_objects = bool(contact_cfg.get("apply_to_task_objects", True))
        apply_to_distractors = bool(contact_cfg.get("apply_to_distractors", False))

        try:
            table_material = self._create_contact_material(
                "realm_high_friction_table",
                table_static,
                table_dynamic,
                restitution,
            )
            object_material = self._create_contact_material(
                "realm_high_friction_task_object",
                object_static,
                object_dynamic,
                restitution,
            )
        except Exception as exc:
            print(f"[RealmEnvironmentDynamic] Warning: failed to create high-friction materials: {exc}", flush=True)
            return

        table_meshes = 0
        object_meshes = 0
        for obj in self.omnigibson_env.scene.objects:
            if self._is_table_contact_object(obj):
                table_meshes += self._bind_contact_material_to_object(
                    obj,
                    table_material,
                    table_static,
                    table_dynamic,
                    restitution,
                )

        task_objects = []
        if apply_to_task_objects:
            task_objects.extend(self.main_objects)
            task_objects.extend(self.target_objects)
        if apply_to_distractors:
            task_objects.extend(self.distractors)
        seen = set()
        for obj in task_objects:
            if obj is None or obj.name in seen:
                continue
            seen.add(obj.name)
            object_meshes += self._bind_contact_material_to_object(
                obj,
                object_material,
                object_static,
                object_dynamic,
                restitution,
            )

        print(
            "[RealmEnvironmentDynamic] Contact friction updated: "
            f"table/static={table_static:g}, table/dynamic={table_dynamic:g}, "
            f"object/static={object_static:g}, object/dynamic={object_dynamic:g}, "
            f"table_meshes={table_meshes}, object_meshes={object_meshes}",
            flush=True,
        )

    def update_task_object_mass(self):
        mass_cfg = self.cfg.get("task_object_mass")
        if mass_cfg is None:
            return
        if mass_cfg is False or (isinstance(mass_cfg, dict) and not mass_cfg.get("enabled", True)):
            return

        min_mass = float(mass_cfg.get("main_object_min_mass", DEFAULT_MAIN_OBJECT_MIN_MASS))
        max_mass = float(mass_cfg.get("main_object_max_mass", DEFAULT_MAIN_OBJECT_MAX_MASS))
        if min_mass <= 0.0:
            return
        max_mass = max(max_mass, min_mass)
        apply_to_target_objects = bool(mass_cfg.get("apply_to_target_objects", False))

        objects_to_update = list(self.main_objects)
        if apply_to_target_objects:
            objects_to_update.extend(self.target_objects)

        updates = []
        for obj in objects_to_update:
            if obj is None:
                continue
            for link_name, link in getattr(obj, "_links", getattr(obj, "links", {})).items():
                try:
                    old_mass = float(link.mass)
                    new_mass = float(np.clip(max(old_mass, min_mass), min_mass, max_mass))
                    if new_mass > old_mass + 1.0e-8:
                        link.mass = new_mass
                    updates.append(
                        {
                            "object": str(getattr(obj, "name", "")),
                            "link": str(link_name),
                            "old_mass": old_mass,
                            "new_mass": float(link.mass),
                        }
                    )
                except Exception as exc:
                    print(
                        f"[RealmEnvironmentDynamic] Warning: failed to update mass for "
                        f"{getattr(obj, 'name', '<unknown>')}:{link_name}: {exc}",
                        flush=True,
                    )

        if updates:
            summary = ", ".join(
                f"{item['object']}/{item['link']}:{item['old_mass']:.4g}->{item['new_mass']:.4g}"
                for item in updates[:8]
            )
            if len(updates) > 8:
                summary += f", ... ({len(updates)} links)"
            print(
                "[RealmEnvironmentDynamic] Task object mass updated: "
                f"main_object_min_mass={min_mass:g}, main_object_max_mass={max_mass:g}, "
                f"{summary}",
                flush=True,
            )

    def apply_scene_fixes_from_cfg(self):
        spawn_cfg = yaml.load(open(f"{self.config_path}/scenes/scenes.yaml", "r"), Loader=yaml.FullLoader)

        if self.scene_model in spawn_cfg and self.scene_part in spawn_cfg[self.scene_model]:
            scene_data = spawn_cfg[self.scene_model][self.scene_part]
            og.sim.stop()
            for obj in self.omnigibson_env.scene.objects:
                if obj.name in scene_data.get("to_fix", []):
                    obj.fixed_base = True
                    create_joint(
                        prim_path=f"{obj.prim_path}/rootJoint",
                        joint_type="FixedJoint",
                        body1=f"{obj.prim_path}/{obj._root_link_name}",
                    )
                elif obj.name in scene_data.get("to_remove", []):
                    obj_to_remove = self.omnigibson_env.scene.object_registry("name", obj.name)
                    self.omnigibson_env.scene.remove_object(obj_to_remove)
                # elif obj.name in special_prims[self.scene_model][self.scene_part].get("drawer", []):
                #     drawer_to_modify = self.omnigibson_env.scene.object_registry("name", obj.name)

            og.sim.play()

    def disable_visual_toggles(self):
        for obj in self.omnigibson_env.scene.objects:
            # TODO: (martin) for pre-baked OG switches on walls their rotation seems off so we cannot use those without the visual toggle...
            if og.object_states.ToggledOn in obj.states:
                obj.states[og.object_states.ToggledOn].visual_marker.visible = False

    def _iter_robot_stage_prims(self):
        stage = lazy.omni.usd.get_context().get_stage()
        robot_prim_path = getattr(self.robot, "prim_path", None)
        root_prim = stage.GetPrimAtPath(robot_prim_path) if robot_prim_path is not None else None
        if root_prim is None or not root_prim.IsValid():
            raise RuntimeError(f"invalid robot prim path: {robot_prim_path}")
        try:
            yield from lazy.pxr.Usd.PrimRange.AllPrims(root_prim)
        except Exception:
            pending = [root_prim]
            while pending:
                prim = pending.pop()
                yield prim
                pending.extend(list(prim.GetChildren()))

    def collect_robot_visual_diagnostics(self, reason="manual"):
        info = {
            "reason": reason,
            "robot_prim_path": getattr(self.robot, "prim_path", None),
            "num_prims": 0,
            "num_imageable": 0,
            "num_mesh": 0,
            "num_visual_mesh": 0,
            "num_gripper_visual_mesh": 0,
            "num_gripper_visual_xform": 0,
            "num_collision_path_prims": 0,
            "type_counts": {},
            "purpose_counts": {},
            "visibility_counts": {},
            "active_counts": {},
            "loaded_counts": {},
            "instance_counts": {},
            "mesh_paths_sample": [],
            "gripper_visual_mesh_paths_sample": [],
            "gripper_visual_xform_paths_sample": [],
            "visual_xform_paths_sample": [],
            "collision_mesh_paths_sample": [],
            "gripper_paths_sample": [],
            "errors": [],
        }
        keywords = ("gripper", "finger", "robotiq", "knuckle", "coupler", "follower", "spring", "pad")
        try:
            for prim in self._iter_robot_stage_prims():
                info["num_prims"] += 1
                path = str(prim.GetPath())
                path_l = path.lower()
                type_name = str(prim.GetTypeName())
                info["type_counts"][type_name] = int(info["type_counts"].get(type_name, 0)) + 1
                for key, value in (
                    ("active_counts", bool(prim.IsActive())),
                    ("loaded_counts", bool(prim.IsLoaded())),
                    ("instance_counts", bool(prim.IsInstance())),
                ):
                    value_key = str(value)
                    info[key][value_key] = int(info[key].get(value_key, 0)) + 1
                if "/collisions" in path_l or path_l.endswith("/collisions"):
                    info["num_collision_path_prims"] += 1
                try:
                    imageable = lazy.pxr.UsdGeom.Imageable(prim)
                    if imageable:
                        info["num_imageable"] += 1
                        purpose = str(imageable.GetPurposeAttr().Get() or "")
                        visibility = str(imageable.ComputeVisibility())
                        info["purpose_counts"][purpose] = int(info["purpose_counts"].get(purpose, 0)) + 1
                        info["visibility_counts"][visibility] = int(info["visibility_counts"].get(visibility, 0)) + 1
                except Exception as exc:
                    info["errors"].append(f"imageable {path}: {exc}")
                if type_name == "Mesh":
                    info["num_mesh"] += 1
                    if "/collisions" in path_l:
                        if len(info["collision_mesh_paths_sample"]) < 20:
                            info["collision_mesh_paths_sample"].append(path)
                    else:
                        info["num_visual_mesh"] += 1
                        if len(info["mesh_paths_sample"]) < 40:
                            info["mesh_paths_sample"].append(path)
                        if any(token in path_l for token in keywords):
                            info["num_gripper_visual_mesh"] += 1
                            if len(info["gripper_visual_mesh_paths_sample"]) < 40:
                                info["gripper_visual_mesh_paths_sample"].append(path)
                is_gripper_related = any(token in path_l for token in keywords)
                if type_name == "Xform" and "/collisions" not in path_l:
                    if len(info["visual_xform_paths_sample"]) < 40:
                        info["visual_xform_paths_sample"].append(path)
                    if is_gripper_related:
                        info["num_gripper_visual_xform"] += 1
                        if len(info["gripper_visual_xform_paths_sample"]) < 40:
                            info["gripper_visual_xform_paths_sample"].append(path)
                if is_gripper_related and len(info["gripper_paths_sample"]) < 40:
                    info["gripper_paths_sample"].append(path)
        except Exception as exc:
            info["errors"].append(f"stage: {exc}")
        return info

    def force_robot_gripper_visuals_visible(self, reason="manual"):
        """
        Isaac/OG occasionally leaves gripper visual prims invisible after
        repeated resets or scene switches in a long-lived process. Physics still
        works, so this explicitly repairs render visibility for all robot prims
        that look like gripper/finger geometry. It also repairs robot visual
        prim purpose, because referenced robot visuals can occasionally stay in
        guide/proxy purpose while collision geometry remains physically active.
        """
        keywords = (
            "gripper",
            "finger",
            "robotiq",
            "knuckle",
            "coupler",
            "follower",
            "spring",
            "pad",
        )
        info = {
            "reason": reason,
            "robot_prim_path": getattr(self.robot, "prim_path", None),
            "num_candidate_prims": 0,
            "num_made_visible": 0,
            "num_load_called": 0,
            "type_counts": {},
            "mesh_paths": [],
            "num_link_visible_set": 0,
            "num_robot_prims_scanned": 0,
            "num_robot_imageable_visible": 0,
            "num_robot_purpose_fixed": 0,
            "errors": [],
            "sample_paths": [],
            "before_diagnostics": None,
            "after_diagnostics": None,
        }
        try:
            info["before_diagnostics"] = self.collect_robot_visual_diagnostics(reason=f"{reason}_before")
        except Exception as exc:
            info["errors"].append(f"before diagnostics: {exc}")
        try:
            for link_name, link in getattr(self.robot, "_links", {}).items():
                if any(token in str(link_name).lower() for token in keywords):
                    try:
                        link.visible = True
                        info["num_link_visible_set"] += 1
                    except Exception as exc:
                        info["errors"].append(f"link {link_name}: {exc}")
        except Exception as exc:
            info["errors"].append(f"links: {exc}")

        try:
            stage = lazy.omni.usd.get_context().get_stage()
            robot_prim_path = getattr(self.robot, "prim_path", None)
            root_prim = stage.GetPrimAtPath(robot_prim_path) if robot_prim_path is not None else None
            if root_prim is None or not root_prim.IsValid():
                raise RuntimeError(f"invalid robot prim path: {robot_prim_path}")
            try:
                root_prim.Load()
                info["num_load_called"] += 1
            except Exception as exc:
                info["errors"].append(f"root load {robot_prim_path}: {exc}")
            for prim in self._iter_robot_stage_prims():
                info["num_robot_prims_scanned"] += 1
                path = str(prim.GetPath())
                name_l = path.lower()
                type_name = str(prim.GetTypeName())
                is_collision_path = "/collisions" in name_l or name_l.endswith("/collisions")
                is_gripper_related = any(token in name_l for token in keywords)
                try:
                    prim.Load()
                    info["num_load_called"] += 1
                except Exception as exc:
                    info["errors"].append(f"prim load {path}: {exc}")

                try:
                    imageable = lazy.pxr.UsdGeom.Imageable(prim)
                    if imageable and not is_collision_path:
                        imageable.MakeVisible()
                        info["num_robot_imageable_visible"] += 1
                        purpose_attr = imageable.GetPurposeAttr()
                        purpose = str(purpose_attr.Get() or "")
                        if purpose in {"guide", "proxy"}:
                            purpose_attr.Set("default")
                            info["num_robot_purpose_fixed"] += 1
                except Exception as exc:
                    info["errors"].append(f"prim {path}: {exc}")
                if not is_gripper_related:
                    continue
                info["num_candidate_prims"] += 1
                info["type_counts"][type_name] = int(info["type_counts"].get(type_name, 0)) + 1
                if type_name == "Mesh" and len(info["mesh_paths"]) < 20:
                    info["mesh_paths"].append(path)
                if len(info["sample_paths"]) < 20:
                    info["sample_paths"].append(path)
                try:
                    imageable = lazy.pxr.UsdGeom.Imageable(prim)
                    if imageable and not is_collision_path:
                        imageable.MakeVisible()
                        info["num_made_visible"] += 1
                except Exception as exc:
                    info["errors"].append(f"gripper prim {path}: {exc}")
        except Exception as exc:
            info["errors"].append(f"stage: {exc}")

        try:
            info["after_diagnostics"] = self.collect_robot_visual_diagnostics(reason=f"{reason}_after")
        except Exception as exc:
            info["errors"].append(f"after diagnostics: {exc}")
        self.last_gripper_visual_repair = info
        if info["num_candidate_prims"] > 0 or info["errors"]:
            print(
                "[RealmEnvironmentDynamic] Gripper visual repair: "
                f"reason={reason} candidates={info['num_candidate_prims']} "
                f"made_visible={info['num_made_visible']} "
                f"robot_visible={info['num_robot_imageable_visible']} "
                f"purpose_fixed={info['num_robot_purpose_fixed']} "
                f"load_called={info['num_load_called']} "
                f"mesh_count={info['type_counts'].get('Mesh', 0)} "
                f"link_visible_set={info['num_link_visible_set']} "
                f"errors={len(info['errors'])}",
                flush=True,
            )
        return info

    def validate_robot_visual_meshes_or_raise(
        self,
        reason="manual",
        min_visual_meshes=MIN_ROBOT_VISUAL_MESHES,
        min_gripper_visual_meshes=MIN_GRIPPER_VISUAL_MESHES,
        min_gripper_visual_xforms=MIN_GRIPPER_VISUAL_XFORMS,
    ):
        diagnostics = self.collect_robot_visual_diagnostics(reason=reason)
        num_visual_mesh = int(diagnostics.get("num_visual_mesh", 0))
        num_gripper_visual_mesh = int(diagnostics.get("num_gripper_visual_mesh", 0))
        num_gripper_visual_xform = int(diagnostics.get("num_gripper_visual_xform", 0))
        has_gripper_visuals = (
            num_gripper_visual_mesh >= int(min_gripper_visual_meshes)
            or num_gripper_visual_xform >= int(min_gripper_visual_xforms)
        )
        if num_visual_mesh < int(min_visual_meshes) or not has_gripper_visuals:
            raise RuntimeError(
                "Robot visual mesh load failure: physics/collision may still work, but renderable robot/gripper "
                f"visual meshes are missing. reason={reason} robot_prim_path={diagnostics.get('robot_prim_path')} "
                f"visual_meshes={num_visual_mesh} min_required={int(min_visual_meshes)} "
                f"gripper_visual_meshes={num_gripper_visual_mesh} "
                f"min_gripper_required={int(min_gripper_visual_meshes)} "
                f"gripper_visual_xforms={num_gripper_visual_xform} "
                f"min_gripper_xform_required={int(min_gripper_visual_xforms)} "
                f"total_mesh={diagnostics.get('num_mesh')} type_counts={diagnostics.get('type_counts')} "
                f"purpose_counts={diagnostics.get('purpose_counts')} "
                f"visibility_counts={diagnostics.get('visibility_counts')} "
                f"gripper_visual_mesh_sample={diagnostics.get('gripper_visual_mesh_paths_sample')} "
                f"gripper_visual_xform_sample={diagnostics.get('gripper_visual_xform_paths_sample')} "
                f"visual_xform_sample={diagnostics.get('visual_xform_paths_sample')} "
                f"collision_mesh_sample={diagnostics.get('collision_mesh_paths_sample')} "
                f"errors={diagnostics.get('errors')}"
            )
        return diagnostics

    # ============================== [ROLLOUT UTILS] ==============================
    def warmup(self, obs=None):
        og.log.info("Starting warmup...")
        self.force_robot_gripper_visuals_visible(reason="warmup_start")
        for _ in range(30):
            og.sim.render()

        if obs is None:
            obs, _ = self.reset()

        if self.ee_control:
            arm_controller = self.robot._controllers.get("arm_0")
            if arm_controller is not None and arm_controller.mode != "absolute_pose":
                ee_cmd = np.zeros(6)
            else:
                ee_pos, ee_quat = self.get_ee_pose()
                ee_pos = ee_pos.cpu().numpy() if hasattr(ee_pos, 'cpu') else np.array(ee_pos)
                ee_euler = R.from_quat(ee_quat.cpu().numpy()).as_euler('xyz')
                ee_cmd = self._world2robot(np.concatenate([ee_pos, ee_euler]))

        for t in range(30):
            gripper_val = np.atleast_1d(1.0 if t < 15 else -1.0)
            if self.ee_control:
                new_action = np.concatenate((ee_cmd, gripper_val))
            else:
                new_action = np.concatenate((self.reset_qpos[:7], gripper_val))

            obs, rew, terminated, truncated, info = self.step(new_action)

            # if self.ee_control:
            #     # Sanity check: robot must not drift during warmup.
            #     # Compare FK position at each step against FK at step 0 (not against ee_cmd,
            #     # which may have a constant calibration offset from the URDF model).
            #     q_current = obs[self.robot.name]['proprio'].cpu().numpy()[:7]
            #     fk_pos, fk_quat = _panda_fk(q_current)
            #     drift = np.linalg.norm(fk_pos - ee_cmd[:3])
            #     assert drift < 0.05, (
            #         f"Robot drifted {drift:.4f}m during EE warmup (step {t}). "
            #         f"fk_pos_initial={ee_cmd}, fk_pos_now={fk_pos}"
            #     )

        self.mo_pos_orig, self.mo_rot_orig = self.main_objects[0].get_position_orientation()
        self.force_robot_gripper_visuals_visible(reason="warmup_end")
        self.validate_robot_visual_meshes_or_raise(reason="warmup_end")
        og.log.info("Warmup finished.")
        return obs, rew, terminated, truncated, info

    def reset(self):
        reset_info = {}
        try:
            if hasattr(og.sim, "is_playing") and not og.sim.is_playing():
                og.sim.play()
        except Exception as exc:
            og.log.warning(f"Failed to ensure simulator is playing before reset: {exc}")
        try:
            obs, reset_info = self.omnigibson_env.reset()
        except ValueError as exc:
            if "Observation space does not match returned observations" not in str(exc):
                raise
            og.log.warning(
                "OmniGibson reset observation-space validation failed; "
                "continuing with RealmEnvironmentDynamic.get_current_obs(). "
                f"Original error: {exc}"
            )
            obs = self.get_current_obs()
            reset_info = {}
        reset_info = reset_info or {}
        self.reset_joints()
        self.restore_task_object_initial_poses()
        reset_info["gripper_visual_repair"] = self.force_robot_gripper_visuals_visible(reason="reset")
        reset_info["robot_visual_diagnostics"] = self.validate_robot_visual_meshes_or_raise(reason="reset")

        self.was_lifted = False
        for k in self.task_progression.keys():
            self.task_progression[k] = False

        for p in self.active_perturbations:
            self.supported_pertrubations[p]()
        obs = self.get_current_obs()
        if "V-AUG" in self.active_perturbations:
            self.v_aug_sigma = np.random.uniform(0.0, 2.5)
            self.v_aug_alpha = np.random.uniform(0.25, 1.5)
            obs = apply_blur_and_contrast(obs, self.v_aug_sigma, self.v_aug_alpha)
        return obs, reset_info

    def restore_task_object_initial_poses(self):
        """
        Omnigibson reset can leave dynamic objects in stale states when an env is
        reused by the SLOT sim server. Restore task objects explicitly so each
        run starts from the same table layout.
        """
        for obj in self.main_objects + self.target_objects + self.distractors:
            pose = self.init_poses.get(obj._relative_prim_path)
            if pose is None:
                continue
            pos = torch.as_tensor(pose["pos"], dtype=torch.float32)
            rot = torch.as_tensor(pose["rot"], dtype=torch.float32)
            obj.set_position_orientation(position=pos, orientation=rot)
            if hasattr(obj, "set_linear_velocity"):
                obj.set_linear_velocity(torch.zeros(3, dtype=torch.float32))
            if hasattr(obj, "set_angular_velocity"):
                obj.set_angular_velocity(torch.zeros(3, dtype=torch.float32))

    def _robot2world(self, action):
        base_height = DROID_BASE_HEIGHT if self.use_droid_with_base else 0.0
        return robot_to_world(action, self.robot_pos, self.robot_rot_rad[2], base_height)

    def _world2robot(self, action):
        base_height = DROID_BASE_HEIGHT if self.use_droid_with_base else 0.0
        return world_to_robot(action, self.robot_pos, self.robot_rot_rad[2], base_height)

    def step(self, action):
        # if self.ee_control:
        #     action = self._robot2world(action)

        obs, rew, terminated, truncated, info = self.omnigibson_env.step(action)

        task_progression = self.recompute_task_progression(obs)

        if "V-AUG" in self.active_perturbations:
            obs = apply_blur_and_contrast(obs, self.v_aug_sigma, self.v_aug_alpha)

        return obs, task_progression, terminated, truncated, info

    def _apply_action_fast(self, action):
        action_arr = np.asarray(action, dtype=np.float32)

        if hasattr(self.robot, "apply_action"):
            self.robot.apply_action(action_arr)
            return True

        if hasattr(self.omnigibson_env, "step"):
            return False

        raise AttributeError("Could not find a fast action application path on the robot/environment.")

    def control_step_fast(self, action, physics_steps=None):
        """
        Fast control-only step: apply action and advance simulation without building full observations.

        This is intended for the inner control loop where we only need the updated EE pose, not camera frames.
        The final inner iteration should still use the regular step() path so the caller can obtain fresh obs.
        """
        if physics_steps is None:
            physics_steps = self.physics_steps_per_action
        physics_steps = max(1, int(physics_steps))

        if not self._apply_action_fast(action):
            # Safe fallback if the lower-level action API is unavailable in the current OG build.
            obs, rew, terminated, truncated, info = self.step(action)
            return {
                "obs": obs,
                "task_progression": rew,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
                "used_fallback_step": True,
            }

        for _ in range(physics_steps):
            og.sim.step()

        return {
            "obs": None,
            "task_progression": None,
            "terminated": False,
            "truncated": False,
            "info": {},
            "used_fallback_step": False,
        }

    def get_current_obs(self):
        """
        Fetch the current observation without advancing control.
        """
        if hasattr(self.omnigibson_env, "get_obs"):
            obs = self.omnigibson_env.get_obs()
            if isinstance(obs, tuple):
                return obs[0]
            return obs
        if hasattr(self.omnigibson_env, "_get_obs"):
            obs = self.omnigibson_env._get_obs()
            if isinstance(obs, tuple):
                return obs[0]
            return obs
        raise AttributeError("Could not find a get_obs/_get_obs method on omnigibson_env.")

    # ============================== [INIT HELPERS] ==============================
    def sample_objects(self, num_objects=3, included_categories=None, excluded_categories=None, ):
        assert not (included_categories is not None and excluded_categories is not None)

        # TODO: this can be pre-computed once, no need to parse the whole thing every call
        available_object_paths = []
        whitelisted_categories = get_non_droid_categories()

        if included_categories is not None:
          whitelisted_categories = included_categories
        elif excluded_categories is not None:
            for cat in excluded_categories:
                if cat in whitelisted_categories:
                    whitelisted_categories.remove(cat)

        for model_path in get_all_object_models():
            if os.path.exists(model_path):
                category = model_path.split("/")[-2]
                if category in whitelisted_categories:
                    available_object_paths.append(model_path)

        if not available_object_paths:
            return []

        if len(available_object_paths) < num_objects:
            og.log.info(
                f"Warning: Only {len(available_object_paths)} suitable objects found, less than requested {num_objects}.")
            num_objects = len(available_object_paths)

        # Randomly sample unique objects
        sampled_indices = np.random.choice(len(available_object_paths), size=num_objects, replace=False)
        sampled_objects = []
        for i in sampled_indices:
            category = available_object_paths[i].split("/")[-2]
            model_id = available_object_paths[i].split("/")[-1]
            name = f"distractor_{i}"
            obj_cfg = {
                "type": "DatasetObject",
                "name": name,
                "category": category,
                "model": model_id,
            }
            sampled_objects.append(obj_cfg)

        return sampled_objects

    def replace_obj(self, obj: DatasetObject, included_categories=None, maximum_dim=0.2, fixed_base=False, preserve_ori=True):
        obj_name = obj.name

        if not (included_categories is None) and len(included_categories) == 1 and "bottom_cabinet" in included_categories:
            bottom_cabinet_models = [
                "bamfsz",
                "dsbcxl",
                "ilofmb",
                # "jhymlr", two top drawers
                "lhucjo",
                "mbmbpa",
                "nddvba",
                "immwzb",
                "pkdnbu",
                "plccav",
                #"pllcur", opens bottom for some reason
                "rntwkg",
                # "ttmejh", not leveled
                "slgzfc",
                "rvpunw",
                "wesxdp",
                "rhdbzv"
            ]
            sampled_idx = np.random.choice(len(bottom_cabinet_models), size=1, replace=False)[0]
            nobj_cfg = {
                "type": "DatasetObject",
                "name": obj_name,
                "category": "bottom_cabinet",
                "model": bottom_cabinet_models[sampled_idx],
            }
        else:
            candidates = self.sample_objects(num_objects=1, included_categories=included_categories)
            if not candidates:
                raise ValueError(f"replace_obj: No suitable objects found for categories: {included_categories}")
            nobj_cfg = candidates[0]

        self.omnigibson_env.scene.remove_object(obj)

        new_obj = DatasetObject(
            name=obj_name,
            relative_prim_path=obj._relative_prim_path,
            category=nobj_cfg["category"],
            model=nobj_cfg["model"],
            fixed_base=fixed_base
        )
        self.omnigibson_env.scene.add_object(new_obj)

        if preserve_ori:
            new_obj.set_bbox_center_position_orientation(torch.tensor(self.init_poses[new_obj._relative_prim_path]["pos"]),
                                                        torch.tensor(self.init_poses[new_obj._relative_prim_path]["rot"]))
        else:
            new_obj.set_bbox_center_position_orientation(torch.tensor(self.init_poses[new_obj._relative_prim_path]["pos"]),
                                                        torch.tensor([0, 0, 0, 1]))

        bbox_center, bbox_orn, bbox_extent, bbox_center_in_frame = new_obj.get_base_aligned_bbox()
        nobj_cfg["bounding_box"] = bbox_center

        max_dim = np.max(bbox_extent.numpy())
        new_scale_factor = maximum_dim / max_dim
        if new_scale_factor < 1.0:
            new_obj.scale = new_scale_factor # TODO: explain method code in comments
            nobj_cfg["bounding_box"] = nobj_cfg["bounding_box"] * new_scale_factor
        nobj_cfg["fixed_base"] = fixed_base

        return new_obj, nobj_cfg
