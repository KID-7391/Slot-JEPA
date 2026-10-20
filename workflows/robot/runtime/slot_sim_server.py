import argparse
import faulthandler
import gc
import io
import importlib.util
import os
import queue
import json
import subprocess
import sys
import threading
import time
import traceback
import types

import msgpack
import numpy as np
import zmq

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from _output_permissions import configure_output_umask, handoff_result_outputs


configure_output_umask()
OUTPUT_ROOT = os.path.join(REPO_ROOT, "realm", "outputs")

VJEPA_PLANNING_PATH = os.path.join(THIS_DIR, "vjepa_planning.py")
SLOT_PLANNING_PATH = os.path.join(THIS_DIR, "slot_planning.py")
SLOT_SIM_SERVER_PATH = os.path.join(THIS_DIR, "slot_sim_server.py")
VJEPA_DROID_PLANNING_PATH = os.path.join(THIS_DIR, "05_vjepa_droid_planning.py")
VJEPA_QUERY_PREDICTOR_PLANNING_PATH = os.path.join(THIS_DIR, "05_vjepa_query_predictor_planning.py")


def _configure_kit_offline_mode():
    """
    Force Kit/Isaac to use local extensions only. Otherwise, when any optional
    extension (e.g. omni.particle.system.bundle) is missing, Kit tries to sync
    https://ovextensionsprod... and can hang or fail in offline containers.
    """
    env_defaults = {
        "OMNI_KIT_DISABLE_EXT_REGISTRY": "1",
        "OMNI_KIT_DISABLE_EXTENSIONS_REGISTRY": "1",
        "OV_DISABLE_EXT_REGISTRY": "1",
        "OMNI_KIT_REGISTRY_ENABLED": "0",
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)
    # Match successful-case collection by default: reuse the normal Kit cache
    # so materials / RTX resources resolve the same way. Set
    # REALM_ISOLATE_KIT_RUNTIME=1 only when debugging cache-lock contention.
    if os.environ.get("REALM_ISOLATE_KIT_RUNTIME", "0").lower() in {"1", "true", "yes"}:
        kit_runtime_root = os.environ.setdefault(
            "REALM_KIT_RUNTIME_ROOT",
            os.path.join("/tmp", f"realm_kit_runtime_{os.getpid()}"),
        )
        os.environ.setdefault("XDG_CACHE_HOME", os.path.join(kit_runtime_root, "cache"))
        os.environ.setdefault("XDG_CONFIG_HOME", os.path.join(kit_runtime_root, "config"))
        os.environ.setdefault("XDG_DATA_HOME", os.path.join(kit_runtime_root, "data"))
    kit_exts_dir = os.environ.setdefault(
        "REALM_KIT_EXTS_DIR",
        os.path.join(REPO_ROOT, "realm", "kit_exts"),
    )

    kit_flags = [
        "--ext-folder",
        kit_exts_dir,
        # Keep USD/material loads synchronous. Async loads can make OmniGibson
        # read stale/default xforms while loading visual meshes, which then
        # fails xform_prim._set_xform_properties during scene switches.
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
            "[slot_sim_server] Kit offline extension registry flags injected: "
            + " ".join(added),
            flush=True,
        )


_configure_kit_offline_mode()


def _env_flag_is_true(name: str, default: str = "0") -> bool:
    value = str(os.environ.get(name, default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _configure_kit_headless_mode():
    if not _env_flag_is_true("OMNIGIBSON_HEADLESS", "0"):
        print("[slot_sim_server] Kit headless mode disabled (OMNIGIBSON_HEADLESS is falsey)", flush=True)
        return

    os.environ.setdefault("DISPLAY", "")
    os.environ.setdefault("KIT_DISABLE_UI", "1")

    headless_flags = [
        "--no-window",
        "--/app/window/enabled=false",
        "--/app/livestream/enabled=false",
    ]
    existing = set(sys.argv)
    added = []
    for flag in headless_flags:
        if flag not in existing:
            sys.argv.append(flag)
            added.append(flag)
    print(
        "[slot_sim_server] Kit headless flags active "
        f"OMNIGIBSON_HEADLESS={os.environ.get('OMNIGIBSON_HEADLESS')} "
        f"DISPLAY={os.environ.get('DISPLAY')!r} "
        f"KIT_DISABLE_UI={os.environ.get('KIT_DISABLE_UI')} "
        f"added_flags={' '.join(added) if added else '<none>'}",
        flush=True,
    )


_configure_kit_headless_mode()


def _install_omni_particle_stub_if_missing():
    """
    Some Isaac/OmniGibson builds used here do not ship omni.particle, but OG's
    deprecated_utils imports it unconditionally while constructing rigid prim
    views. We do not use particle systems in these SLOT runs, so a minimal stub
    is enough to keep the import path alive without changing real particle
    behavior when the extension actually exists.
    """
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
    print(
        "[slot_sim_server] installed fallback stub for missing omni.particle extension",
        flush=True,
    )


class MsgSerializer:
    @staticmethod
    def to_bytes(data):
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data):
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


def _load_vjepa_planning_module():
    spec = importlib.util.spec_from_file_location("vjepa_planning_shared", VJEPA_PLANNING_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    success_case_root = os.path.abspath(str(module.DEFAULT_SUCCESS_CASE_ROOT))
    if not os.path.isdir(success_case_root):
        raise FileNotFoundError(
            "Released robot goal package is not mounted: "
            f"{success_case_root}. Mount slot_jepa_release/artifacts at /artifacts."
        )
    success_case_count = sum(
        os.path.isfile(os.path.join(success_case_root, name, "manifest.json"))
        for name in os.listdir(success_case_root)
    )
    if success_case_count == 0:
        raise FileNotFoundError(
            f"No successful-case manifests found under {success_case_root}. "
            "The /artifacts mount is empty or points to the wrong release directory."
        )
    print(
        "[slot_sim_server] loaded planning module "
        f"path={VJEPA_PLANNING_PATH} "
        f"has_cem_wrist_annotation={hasattr(module, 'save_wrist_image_with_cem_phase_outputs')} "
        f"success_case_root={success_case_root} success_case_count={success_case_count}",
        flush=True,
    )
    return module


def _set_omnigibson_macros_once(enable_object_states: bool = True, enable_hq_rendering: bool = True):
    # This must run before importing full omnigibson or constructing any env.
    # MacroDict refuses changes after a macro has been read by the simulator.
    import random

    import torch
    from omnigibson.macros import gm

    _install_omni_particle_stub_if_missing()

    gm.DEFAULT_SIM_STEP_FREQ = 15
    gm.DEFAULT_RENDERING_FREQ = 15
    gm.DEFAULT_PHYSICS_FREQ = 120
    gm.ENABLE_TRANSITION_RULES = False
    gm.ENABLE_OBJECT_STATES = bool(enable_object_states)
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_HQ_RENDERING = bool(enable_hq_rendering)
    print(
        "[slot_sim_server] Omnigibson macros: "
        f"DEFAULT_SIM_STEP_FREQ={gm.DEFAULT_SIM_STEP_FREQ} "
        f"DEFAULT_RENDERING_FREQ={gm.DEFAULT_RENDERING_FREQ} "
        f"DEFAULT_PHYSICS_FREQ={gm.DEFAULT_PHYSICS_FREQ} "
        f"ENABLE_OBJECT_STATES={gm.ENABLE_OBJECT_STATES} "
        f"ENABLE_HQ_RENDERING={gm.ENABLE_HQ_RENDERING}",
        flush=True,
    )
    try:
        import carb.settings

        settings = carb.settings.get_settings()
        print(
            "[slot_sim_server] Kit extension settings: "
            f"registryEnabled={settings.get('/app/extensions/registryEnabled')} "
            f"excluded={settings.get('/app/extensions/excluded')}",
            flush=True,
        )
    except Exception as exc:
        print(f"[slot_sim_server] Kit extension settings unavailable before app launch: {exc}", flush=True)

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _log_blocked_extension_status(reason: str):
    blocked_exts = [
        "omni.particle.system.bundle",
        "omni.particle.system.core",
        "omni.particle.system.core.scripts",
        "omni.usd.schema.flow",
        "omni.flowusd",
    ]
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        statuses = {}
        for ext_name in blocked_exts:
            try:
                statuses[ext_name] = manager.get_enabled_extension_id(ext_name)
            except Exception as exc:
                statuses[ext_name] = f"error:{exc}"
        print(f"[slot_sim_server] blocked extension status after {reason}: {statuses}", flush=True)
    except Exception as exc:
        print(f"[slot_sim_server] blocked extension status unavailable after {reason}: {exc}", flush=True)


def _start_watchdog(label: str, timeout_s: float = 420.0, hard_timeout_s=None):
    stop_event = threading.Event()
    try:
        faulthandler.dump_traceback_later(float(timeout_s), repeat=False, file=sys.stdout)
    except Exception:
        pass

    def _worker():
        if stop_event.wait(float(timeout_s)):
            return
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass
        print(
            f"[slot_sim_server] watchdog soft timeout after {timeout_s:.1f}s while {label}; dumping Python stacks",
            flush=True,
        )
        try:
            faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
        except Exception:
            traceback.print_stack()
        if hard_timeout_s is None:
            return
        remaining_s = max(0.0, float(hard_timeout_s) - float(timeout_s))
        if stop_event.wait(remaining_s):
            return
        print(
            f"[slot_sim_server] watchdog hard timeout after {hard_timeout_s:.1f}s while {label}; "
            "forcing process exit because Kit/OmniGibson is still blocked",
            flush=True,
        )
        try:
            faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
        finally:
            # The blocked path is often inside Kit / USD C++ code, so raising in
            # this Python thread cannot interrupt it. Exit hard so the caller can
            # restart the sim server instead of waiting forever.
            os._exit(124)

    thread = threading.Thread(target=_worker, name=f"slot_watchdog_{label}", daemon=True)
    thread.start()
    return stop_event


def _resolve_env_watchdog_timeouts(params, task_cfg):
    soft_timeout_s = float(params.get("env_watchdog_soft_timeout_s", 420.0))
    hard_timeout_s = float(params.get("env_watchdog_hard_timeout_s", 1200.0))
    supported_scenes = task_cfg.get("supported_scenes", {}) if isinstance(task_cfg, dict) else {}
    scene_names = {str(name) for name in supported_scenes.keys()}

    # Some REALM_DROID10 scenes take much longer to finish OmniGibson / Kit
    # environment construction even when they eventually succeed. In practice
    # office_cubicles_left is substantially slower than Pomaria_1_int.
    if "office_cubicles_left" in scene_names:
        soft_timeout_s = max(soft_timeout_s, 600.0)
        hard_timeout_s = max(hard_timeout_s, 1800.0)
    elif "Pomaria_1_int" in scene_names:
        soft_timeout_s = max(soft_timeout_s, 420.0)
        hard_timeout_s = max(hard_timeout_s, 1200.0)

    if hard_timeout_s < soft_timeout_s:
        hard_timeout_s = soft_timeout_s
    return soft_timeout_s, hard_timeout_s


def _ensure_sim_stopped(shared, context, timeout_s=10.0):
    sim = getattr(shared.og, "sim", None)
    if sim is None or not hasattr(sim, "is_stopped"):
        return

    last_error = None
    for attempt in range(3):
        try:
            if sim.is_stopped():
                return
        except Exception as exc:
            last_error = exc

        print(
            f"[slot_sim_server] stopping simulator before {context} "
            f"(attempt {attempt + 1}/3)",
            flush=True,
        )
        try:
            sim.stop()
        except Exception as exc:
            last_error = exc

        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            try:
                if sim.is_stopped():
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(0.05)

    raise RuntimeError(
        f"Simulator did not stop before {context}; cannot safely load a new scene. "
        f"Last error: {last_error}"
    )


def _stage_scene_prims(shared):
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return []
        world = stage.GetPrimAtPath("/World")
        if not world or not world.IsValid():
            return []
        return [str(child.GetPath()) for child in world.GetChildren() if child.GetName().startswith("scene_")]
    except Exception:
        return []


def _cleanup_stage_scene_prims(shared, reason):
    try:
        import omni.kit.app
        import omni.usd

        context = omni.usd.get_context()
        stage = context.get_stage()
        if stage is None:
            print(f"[slot_sim_server] stage cleanup skipped after {reason}: no active stage", flush=True)
            return []
        scene_paths = _stage_scene_prims(shared)
        # Do not remove /World/scene_* prims one by one. Hydra / USD imaging
        # can keep delegates to those prims briefly, and the next environment
        # load may then hit "Accessed schema on invalid prim". Opening a fresh
        # anonymous stage is a harder but cleaner scene switch boundary.
        try:
            new_stage_result = context.new_stage()
        except Exception as exc:
            print(f"[slot_sim_server] warning: omni.usd context.new_stage() failed: {exc}", flush=True)
            new_stage_result = None
        app = omni.kit.app.get_app()
        for _ in range(30):
            try:
                app.update()
            except Exception:
                break
            if not _stage_scene_prims(shared):
                # Keep a couple of extra ticks so Hydra / USD imaging can drop
                # stale delegates before the next Environment.load().
                for _ in range(2):
                    try:
                        app.update()
                    except Exception:
                        break
                break
        print(
            f"[slot_sim_server] stage scene cleanup after {reason}: "
            f"before={scene_paths} new_stage_result={new_stage_result} after={_stage_scene_prims(shared)}",
            flush=True,
        )
        return scene_paths
    except Exception as exc:
        print(f"[slot_sim_server] warning: stage scene cleanup failed after {reason}: {exc}", flush=True)
        return []


def _reset_sim_scene_bookkeeping(shared, reason):
    sim = getattr(shared.og, "sim", None)
    if sim is None:
        return
    changed = {}
    for attr in ("scenes", "_scenes"):
        try:
            value = getattr(sim, attr)
        except Exception:
            continue
        if isinstance(value, list):
            old_len = len(value)
            try:
                value.clear()
                changed[attr] = f"list_clear:{old_len}->0"
            except Exception as exc:
                changed[attr] = f"list_clear_failed:{exc}"
    for attr in (
        "scene_idx",
        "_scene_idx",
        "n_scenes",
        "_n_scenes",
        "scene_counter",
        "_scene_counter",
        "last_scene_edge",
        "_last_scene_edge",
    ):
        try:
            old_value = getattr(sim, attr)
        except Exception:
            continue
        try:
            if "edge" in attr:
                setattr(sim, attr, 0.0)
            else:
                setattr(sim, attr, 0)
            changed[attr] = f"{old_value!r}->{getattr(sim, attr)!r}"
        except Exception as exc:
            changed[attr] = f"set_failed:{exc}"
    if changed:
        print(
            f"[slot_sim_server] reset OmniGibson scene bookkeeping after {reason}: {changed}",
            flush=True,
        )


def _reset_sim_after_env_creation_failure(shared, reason):
    try:
        _ensure_sim_stopped(shared, f"recovering after env creation failure ({reason})", timeout_s=5.0)
    except Exception as exc:
        print(f"[slot_sim_server] warning: failed to stop sim during env-create recovery: {exc}", flush=True)
    _cleanup_stage_scene_prims(shared, reason=f"env_create_failure:{reason}")
    _reset_sim_scene_bookkeeping(shared, reason=f"env_create_failure:{reason}")
    gc.collect()


def _call_first_available(obj, method_names, *, label):
    if obj is None:
        return False
    for method_name in method_names:
        method = getattr(obj, method_name, None)
        if method is None:
            continue
        try:
            method()
            print(f"[slot_sim_server] {label}: called {type(obj).__name__}.{method_name}()", flush=True)
            return True
        except Exception as exc:
            print(
                f"[slot_sim_server] {label}: {type(obj).__name__}.{method_name}() failed: {exc}",
                flush=True,
            )
    return False


def _destroy_env_session(shared, cached, *, reason):
    env = None if cached is None else cached.get("env")
    if env is None:
        return

    print(
        "[slot_sim_server] destroying env session before creating a different env "
        f"reason={reason} task={cached.get('task')} variant={cached.get('task_variant')}",
        flush=True,
    )
    try:
        _ensure_sim_stopped(shared, f"destroying cached env session ({reason})", timeout_s=5.0)
    except Exception as exc:
        print(f"[slot_sim_server] warning: failed to stop sim before env destroy: {exc}", flush=True)

    og_env = getattr(env, "omnigibson_env", None)
    # Do not unload DatasetObjects one by one here. During scene switches,
    # per-object remove/load can leave USD imaging / xform caches briefly
    # pointing at half-unloaded visual prims; the next env load then fails in
    # xform_prim._set_xform_properties with old_pos/new_pos mismatches. A fresh
    # stage boundary below is a cleaner teardown for the resident sim server.

    _call_first_available(og_env, ("close", "shutdown"), label="destroy og env")
    _call_first_available(env, ("close", "shutdown"), label="destroy realm env")
    _cleanup_stage_scene_prims(shared, reason=f"destroy_env_session:{reason}")
    _reset_sim_scene_bookkeeping(shared, reason=f"destroy_env_session:{reason}")

    try:
        env.omnigibson_env = None
    except Exception:
        pass
    try:
        cached["env"] = None
    except Exception:
        pass


def _get_or_create_env_session(shared, session_cache, params):
    task_id = int(params.get("task_id", 0))
    perturbation_id = int(params.get("perturbation_id", 0))
    scene_profile = params.get("scene_profile", "realm_droid10")
    task_variant = params.get("task_variant", getattr(shared, "DEFAULT_TASK_VARIANT", "default"))
    fixed_camera_presets = params.get("fixed_camera_presets")
    fixed_camera_key = tuple(fixed_camera_presets or ())
    task = shared.SUPPORTED_TASKS[task_id]
    perturbation = shared.SUPPORTED_PERTURBATIONS[perturbation_id]
    cache_key = (task_id, perturbation_id, scene_profile, task_variant, fixed_camera_key)
    cached = session_cache.get(cache_key)
    if cached is not None:
        env = cached.get("env")
        repair_info = None
        visual_diag = None
        should_rebuild_cached_env = False
        if env is not None and hasattr(env, "force_robot_gripper_visuals_visible"):
            repair_info = env.force_robot_gripper_visuals_visible(reason="reuse_cached_env_session")
        if env is not None and hasattr(env, "collect_robot_visual_diagnostics"):
            visual_diag = env.collect_robot_visual_diagnostics(reason="reuse_cached_env_session_check")
            num_visual_mesh = int(visual_diag.get("num_visual_mesh", 0))
            num_gripper_visual_mesh = int(visual_diag.get("num_gripper_visual_mesh", 0))
            num_gripper_visual_xform = int(visual_diag.get("num_gripper_visual_xform", 0))
            should_rebuild_cached_env = (
                num_visual_mesh < 8
                or (num_gripper_visual_mesh < 2 and num_gripper_visual_xform < 8)
            )
            if should_rebuild_cached_env:
                raise RuntimeError(
                    "Cached env robot visual mesh load failure. Refusing to continue with invisible gripper/robot "
                    f"visuals. visual_mesh={num_visual_mesh} gripper_visual_mesh={num_gripper_visual_mesh} "
                    f"gripper_visual_xform={num_gripper_visual_xform} "
                    f"diagnostics={visual_diag}"
                )
        print(
            "[slot_sim_server] reusing env session "
            f"task={task} perturbation={perturbation} scene_profile={scene_profile} "
            f"variant={task_variant} fixed_camera_presets={list(fixed_camera_key)} "
            f"gripper_visual_candidates={None if repair_info is None else repair_info.get('num_candidate_prims')} "
            f"gripper_visual_errors={None if repair_info is None else len(repair_info.get('errors', []))} "
            f"robot_mesh={None if visual_diag is None else visual_diag.get('num_mesh')} "
            f"gripper_visual_mesh={None if visual_diag is None else visual_diag.get('num_gripper_visual_mesh')} "
            f"gripper_visual_xform={None if visual_diag is None else visual_diag.get('num_gripper_visual_xform')} "
            f"rebuild={should_rebuild_cached_env}",
            flush=True,
        )
        if cached is not None:
            return cached

    max_env_sessions = int(params.get("max_env_sessions", 1))
    if max_env_sessions <= 1 and session_cache:
        print(
            "[slot_sim_server] dropping cached env sessions before creating a different env "
            f"old_sessions={len(session_cache)}",
            flush=True,
        )
        for old_cached in list(session_cache.values()):
            _destroy_env_session(shared, old_cached, reason="cache_miss_max_env_sessions_1")
        _ensure_sim_stopped(shared, "dropping cached env sessions", timeout_s=5.0)
        session_cache.clear()
        gc.collect()

    print(
        "[slot_sim_server] creating env session "
        f"task={task} perturbation={perturbation} scene_profile={scene_profile} "
        f"variant={task_variant} fixed_camera_presets={list(fixed_camera_key)}",
        flush=True,
    )
    runtime_init = getattr(shared, "_ensure_omnigibson_runtime_initialized", None)
    if callable(runtime_init):
        runtime_init()
    task_cfg_path = shared.get_task_cfg_path(task, scene_profile, task_variant)
    task_cfg = shared.load_task_cfg("/app/realm/config", task_cfg_path)
    env_watchdog_soft_timeout_s, env_watchdog_hard_timeout_s = _resolve_env_watchdog_timeouts(params, task_cfg)
    print(
        "[slot_sim_server] env create watchdog timeouts "
        f"soft={env_watchdog_soft_timeout_s:.1f}s hard={env_watchdog_hard_timeout_s:.1f}s "
        f"supported_scenes={list((task_cfg.get('supported_scenes') or {}).keys())}",
        flush=True,
    )
    _ensure_sim_stopped(shared, "creating a new env session")
    if session_cache:
        # Defensive only; normal max_env_sessions=1 path clears the cache above.
        _cleanup_stage_scene_prims(shared, reason="pre_create_with_nonempty_cache")
        _reset_sim_scene_bookkeeping(shared, reason="pre_create_with_nonempty_cache")
    else:
        stale_scene_prims = _stage_scene_prims(shared)
        if stale_scene_prims:
            print(
                "[slot_sim_server] found stale scene prims before env creation despite empty cache; "
                f"cleaning them: {stale_scene_prims}",
                flush=True,
            )
            _cleanup_stage_scene_prims(shared, reason="pre_create_stale_stage")
            _reset_sim_scene_bookkeeping(shared, reason="pre_create_stale_stage")
    if hasattr(shared, "set_sim_config"):
        shared.set_sim_config(rendering_mode="rt", robot="DROID_ee_control")
    t_env_start = time.perf_counter()
    env_create_attempts = int(params.get("env_create_attempts", 3))
    env_create_attempts = max(1, env_create_attempts)
    env = None
    last_env_create_exc = None
    for env_attempt in range(env_create_attempts):
        if env_attempt > 0:
            print(
                "[slot_sim_server] retrying env creation after load failure "
                f"attempt={env_attempt + 1}/{env_create_attempts}",
                flush=True,
            )
            _reset_sim_after_env_creation_failure(shared, reason=f"retry_{env_attempt}")
            _ensure_sim_stopped(shared, f"retrying env creation attempt {env_attempt + 1}")
            time.sleep(0.5)
        env_watchdog = _start_watchdog(
            f"creating RealmEnvironmentDynamic attempt {env_attempt + 1}/{env_create_attempts}",
            timeout_s=env_watchdog_soft_timeout_s,
            hard_timeout_s=env_watchdog_hard_timeout_s,
        )
        try:
            env = shared.RealmEnvironmentDynamic(
                config_path="/app/realm/config",
                task_cfg_path=task_cfg_path,
                perturbations=[perturbation],
                multi_view=True,
                no_rendering=False,
                rendering_mode="rt",
                robot="DROID_ee_control",
                camera_extrinsics_override=fixed_camera_presets,
            )
            last_env_create_exc = None
            break
        except Exception as exc:
            last_env_create_exc = exc
            print(
                "[slot_sim_server] env creation failed "
                f"attempt={env_attempt + 1}/{env_create_attempts}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            _reset_sim_after_env_creation_failure(shared, reason=f"attempt_{env_attempt + 1}")
            if env_attempt >= env_create_attempts - 1:
                raise
        finally:
            env_watchdog.set()
            try:
                faulthandler.cancel_dump_traceback_later()
            except Exception:
                pass
    if env is None:
        raise RuntimeError(f"Failed to create RealmEnvironmentDynamic: {last_env_create_exc}")
    _log_blocked_extension_status("env session creation")
    print(
        "[slot_sim_server] env session created "
        f"task={task} perturbation={perturbation} variant={task_variant} "
        f"duration_s={time.perf_counter() - t_env_start:.2f}",
        flush=True,
    )
    cached = {
        "env": env,
        "task_cfg": task_cfg,
        "task_cfg_path": task_cfg_path,
        "task": task,
        "perturbation": perturbation,
        "scene_profile": scene_profile,
        "task_variant": task_variant,
        "fixed_camera_presets": list(fixed_camera_key),
    }
    session_cache[cache_key] = cached
    return cached


def _run_slot_planning(shared, session_cache, params):
    if bool(params.get("controller_only_goal_delta", False)):
        return shared.run_controller_goal_delta_diagnostic(
            task_id=int(params.get("task_id", 0)),
            perturbation_id=int(params.get("perturbation_id", 0)),
            state_frame=params.get("state_frame", "camera"),
            success_case_dir=params.get("success_case_dir", shared.DEFAULT_SUCCESS_CASE_DIR),
            goal_mode=params.get("goal_mode", "first"),
            scene_profile=params.get("scene_profile", "realm_droid10"),
            task_variant=params.get("task_variant", getattr(shared, "DEFAULT_TASK_VARIANT", "default")),
            fixed_camera_presets=params.get("fixed_camera_presets"),
            max_settle_steps=int(params.get("max_settle_steps", shared.DEFAULT_VJEPA_MAX_SETTLE_STEPS)),
            settle_pos_tol=float(params.get("settle_pos_tol", shared.DEFAULT_VJEPA_SETTLE_POS_TOL)),
            settle_rot_tol=float(params.get("settle_rot_tol", shared.DEFAULT_VJEPA_SETTLE_ROT_TOL)),
            enable_settle_debug=bool(params.get("enable_settle_debug", False)),
        )

    session = _get_or_create_env_session(shared, session_cache, params)
    model_type = str(params.get("model_type", "slot"))
    return shared.run_vjepa_planning(
        port=int(params.get("port", 8021)),
        task_id=int(params.get("task_id", 0)),
        perturbation_id=int(params.get("perturbation_id", 0)),
        state_frame=params.get("state_frame", "camera"),
        success_case_dir=params.get("success_case_dir", shared.DEFAULT_SUCCESS_CASE_DIR),
        goal_mode=params.get("goal_mode", "first"),
        scene_profile=params.get("scene_profile", "realm_droid10"),
        task_variant=params.get("task_variant", getattr(shared, "DEFAULT_TASK_VARIANT", "default")),
        max_steps=int(params.get("max_steps", 20)),
        action_repeat=int(params.get("action_repeat", 1)),
        replay_mode=params.get("replay_mode", "robot_absolute"),
        max_settle_steps=int(params.get("max_settle_steps", shared.DEFAULT_VJEPA_MAX_SETTLE_STEPS)),
        settle_pos_tol=float(params.get("settle_pos_tol", shared.DEFAULT_VJEPA_SETTLE_POS_TOL)),
        settle_rot_tol=float(params.get("settle_rot_tol", shared.DEFAULT_VJEPA_SETTLE_ROT_TOL)),
        enable_settle_debug=bool(params.get("enable_settle_debug", False)),
        model_type=model_type,
        slot_model_gripper_units=params.get("slot_model_gripper_units", "normalized"),
        env=session["env"],
        task_cfg=session["task_cfg"],
        task_cfg_path=session["task_cfg_path"],
        cancel_event=params.get("_cancel_event"),
        output_run_suffix=f"attempt{int(params.get('retry_attempt', 0)) + 1:02d}",
        output_namespace=(
            "slot"
            if params.get("ablation_variant", "full") == "full"
            else f"slot-{str(params.get('ablation_variant')).replace('_', '-')}"
        ),
        fixed_camera_presets=params.get("fixed_camera_presets"),
        rotate_deterministic_after_grasp=not bool(
            params.get("disable_rotate_deterministic_after_grasp", False)
        ),
        rotate_deterministic_step_rad=float(
            params.get(
                "rotate_deterministic_step_rad",
                getattr(shared, "DEFAULT_ROTATE_DETERMINISTIC_STEP_RAD", 0.2),
            )
        ),
        rotate_deterministic_axis=params.get(
            "rotate_deterministic_axis",
            getattr(shared, "DEFAULT_ROTATE_DETERMINISTIC_AXIS", "yaw"),
        ),
        rotate_deterministic_overshoot_rad=float(
            params.get(
                "rotate_deterministic_overshoot_rad",
                getattr(shared, "DEFAULT_ROTATE_DETERMINISTIC_OVERSHOOT_RAD", 0.25),
            )
        ),
    )


def _planning_script_for_model_type(model_type):
    normalized = str(model_type or "slot").strip().lower().replace("_", "-")
    if normalized == "slot":
        return SLOT_PLANNING_PATH
    if normalized == "vjepa-droid":
        return VJEPA_DROID_PLANNING_PATH
    if normalized == "vjepa-query-predictor":
        return VJEPA_QUERY_PREDICTOR_PLANNING_PATH
    raise ValueError(f"Unsupported planning model_type={model_type!r}")


def _append_cli_arg(cmd, flag, value):
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _run_slot_planning_isolated(params):
    """Run one planning request in a fresh Python / Kit process.

    OmniGibson / Isaac stage teardown is not reliable when switching scenes
    inside one long-lived Kit process. This broker mode keeps the ZMQ control
    endpoint resident, but gives every planning request a clean Kit runtime.
    """
    task_run_id = int(params.get("_task_run_id", 0))
    result_dir = os.path.join("/tmp", "realm_slot_sim_server_results")
    os.makedirs(result_dir, exist_ok=True)
    result_json = os.path.join(result_dir, f"task_run_{task_run_id}_{os.getpid()}.json")

    model_type = str(params.get("model_type", "slot"))
    planning_script_path = _planning_script_for_model_type(model_type)
    cmd = [
        sys.executable,
        planning_script_path,
        "--local-sim",
        "--result-json",
        result_json,
    ]
    _append_cli_arg(cmd, "--port", int(params.get("port", 8011)))
    _append_cli_arg(cmd, "--task_id", int(params.get("task_id", 0)))
    _append_cli_arg(cmd, "--perturbation_id", int(params.get("perturbation_id", 0)))
    _append_cli_arg(cmd, "--state-frame", params.get("state_frame", "camera"))
    if params.get("success_case_dir") is not None:
        _append_cli_arg(cmd, "--success-case-dir", params.get("success_case_dir"))
    _append_cli_arg(cmd, "--goal-mode", params.get("goal_mode", "all"))
    _append_cli_arg(cmd, "--scene-profile", params.get("scene_profile", "realm_droid10"))
    _append_cli_arg(cmd, "--task-variant", params.get("task_variant", "default"))
    _append_cli_arg(cmd, "--ablation-variant", params.get("ablation_variant", "full"))
    _append_cli_arg(cmd, "--max-steps", int(params.get("max_steps", 20)))
    _append_cli_arg(cmd, "--action-repeat", int(params.get("action_repeat", 1)))
    _append_cli_arg(cmd, "--replay-mode", params.get("replay_mode", "robot_absolute"))
    _append_cli_arg(cmd, "--max-settle-steps", int(params.get("max_settle_steps", 4)))
    _append_cli_arg(cmd, "--settle-pos-tol", float(params.get("settle_pos_tol", 0.01)))
    _append_cli_arg(cmd, "--settle-rot-tol", float(params.get("settle_rot_tol", 0.05)))
    _append_cli_arg(cmd, "--rotate-deterministic-step-rad", float(params.get("rotate_deterministic_step_rad", 0.2)))
    _append_cli_arg(
        cmd,
        "--rotate-deterministic-overshoot-rad",
        float(params.get("rotate_deterministic_overshoot_rad", 0.25)),
    )
    _append_cli_arg(cmd, "--rotate-deterministic-axis", params.get("rotate_deterministic_axis", "yaw"))
    if model_type.strip().lower().replace("_", "-") == "slot":
        _append_cli_arg(cmd, "--slot-model-gripper-units", params.get("slot_model_gripper_units", "normalized"))
    _append_cli_arg(cmd, "--retries", int(params.get("retry_max_attempts", 1)))
    fixed_camera_presets = params.get("fixed_camera_presets")
    if fixed_camera_presets:
        cmd.append("--fixed-camera-presets")
        cmd.extend([str(x) for x in fixed_camera_presets])
    if bool(params.get("enable_settle_debug", False)):
        cmd.append("--enable-settle-debug")
    if bool(params.get("controller_only_goal_delta", False)):
        cmd.append("--controller-only-goal-delta")
    if bool(params.get("disable_rotate_deterministic_after_grasp", False)):
        cmd.append("--disable-rotate-deterministic-after-grasp")

    env = os.environ.copy()
    env.setdefault("OMNIGIBSON_HEADLESS", "1")
    cancel_event = params.get("_cancel_event")
    print(
        "[slot_sim_server] isolated planning subprocess starting "
        f"task_run_id={task_run_id} cmd={' '.join(cmd)}",
        flush=True,
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=REPO_ROOT,
        env=env,
    )
    try:
        assert proc.stdout is not None
        while True:
            if cancel_event is not None and cancel_event.is_set() and proc.poll() is None:
                print(
                    f"[slot_sim_server] terminating isolated planning subprocess task_run_id={task_run_id}",
                    flush=True,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(
                        f"[slot_sim_server] killing isolated planning subprocess task_run_id={task_run_id}",
                        flush=True,
                    )
                    proc.kill()
                break
            line = proc.stdout.readline()
            if line:
                print(line, end="", flush=True)
                continue
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        return_code = proc.wait()
    finally:
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass

    if cancel_event is not None and cancel_event.is_set():
        return {
            "success": False,
            "status": "cancelled",
            "isolated_subprocess": True,
            "return_code": int(return_code),
        }
    if return_code != 0:
        raise RuntimeError(
            f"Isolated SLOT planning subprocess failed with return_code={return_code}. "
            f"result_json={result_json}"
        )
    if not os.path.exists(result_json):
        raise RuntimeError(f"Isolated SLOT planning subprocess did not write result_json={result_json}")
    with open(result_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    last_result = payload.get("last_result") or {}
    result = {
        "success": bool(payload.get("success")),
        "isolated_subprocess": True,
        "result_json": result_json,
        "num_env_sessions": 0,
    }
    if isinstance(last_result, dict):
        result.update(last_result)
    result["isolated_all_results"] = payload.get("results", [])
    return result


def _env_worker_key(params):
    fixed_camera_presets = params.get("fixed_camera_presets")
    return (
        int(params.get("task_id", 0)),
        int(params.get("perturbation_id", 0)),
        str(params.get("scene_profile", "realm_droid10")),
        str(params.get("task_variant", "default")),
        tuple(fixed_camera_presets or ()),
    )


def _stream_worker_stdout(proc, prefix):
    def _worker():
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(f"{prefix}{line}", end="", flush=True)
        except Exception as exc:
            print(f"[slot_sim_server] worker stdout stream ended with error: {exc}", flush=True)

    thread = threading.Thread(target=_worker, name="slot-sim-worker-stdout", daemon=True)
    thread.start()
    return thread


def _worker_call(endpoint, *, data=None, host="127.0.0.1", port=0, timeout_ms=10000):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
    socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
    try:
        socket.connect(f"tcp://{host}:{int(port)}")
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        socket.send(MsgSerializer.to_bytes(request))
        response = MsgSerializer.from_bytes(socket.recv())
        if "error" in response:
            raise RuntimeError(f"worker endpoint {endpoint} error: {response['error']}")
        return response
    finally:
        socket.close(0)
        context.term()


def _stop_env_worker(worker_state, reason):
    proc = worker_state.get("proc") if isinstance(worker_state, dict) else None
    if proc is None:
        return
    port = worker_state.get("port")
    if proc.poll() is None and port is not None:
        try:
            print(
                f"[slot_sim_server] stopping env worker reason={reason} "
                f"pid={proc.pid} port={port} key={worker_state.get('env_key')}",
                flush=True,
            )
            _worker_call("shutdown", port=port, timeout_ms=5000)
        except Exception as exc:
            print(f"[slot_sim_server] worker graceful shutdown failed: {exc}", flush=True)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print(f"[slot_sim_server] killing env worker pid={proc.pid}", flush=True)
            proc.kill()
            proc.wait(timeout=10)
    worker_state.clear()


def _start_env_worker(args, worker_state, env_key):
    port = int(args.worker_port)
    try:
        existing = _worker_call("ping", port=port, timeout_ms=500)
    except Exception:
        existing = None
    if existing is not None:
        print(
            "[slot_sim_server] found stale env worker on requested port; shutting it down "
            f"port={port} response_status={existing.get('status')} task={existing.get('task')}",
            flush=True,
        )
        try:
            _worker_call("shutdown", port=port, timeout_ms=2000)
        except Exception as exc:
            print(f"[slot_sim_server] stale worker shutdown request failed: {exc}", flush=True)
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                _worker_call("ping", port=port, timeout_ms=300)
            except Exception:
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"stale env worker on port={port} did not shut down; "
                "choose a different --worker-port or kill the stale process"
            )

    cmd = [
        sys.executable,
        SLOT_SIM_SERVER_PATH,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--in-process-env-cache",
    ]
    if args.disable_object_states:
        cmd.append("--disable-object-states")
    if args.disable_hq_rendering:
        cmd.append("--disable-hq-rendering")

    env = os.environ.copy()
    env.setdefault("OMNIGIBSON_HEADLESS", "1")
    print(
        "[slot_sim_server] starting env worker "
        f"env_key={env_key} port={port} cmd={' '.join(cmd)}",
        flush=True,
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=REPO_ROOT,
        env=env,
    )
    _stream_worker_stdout(proc, prefix="[slot_env_worker] ")
    deadline = time.time() + float(args.worker_start_timeout_s)
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"env worker exited early with return_code={proc.returncode}")
        try:
            _worker_call("ping", port=port, timeout_ms=1000)
            worker_state.clear()
            worker_state.update({"proc": proc, "port": port, "env_key": env_key})
            print(
                f"[slot_sim_server] env worker ready pid={proc.pid} port={port} env_key={env_key}",
                flush=True,
            )
            return worker_state
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    _stop_env_worker({"proc": proc, "port": port, "env_key": env_key}, reason="worker_start_timeout")
    raise RuntimeError(f"env worker did not become ready on port={port}: {last_error}")


def _run_slot_planning_reused_env_worker(args, worker_state, params):
    env_key = _env_worker_key(params)
    proc = worker_state.get("proc")
    same_env = (
        proc is not None
        and proc.poll() is None
        and worker_state.get("env_key") == env_key
    )
    if same_env:
        print(
            "[slot_sim_server] reusing existing env worker "
            f"pid={proc.pid} port={worker_state.get('port')} env_key={env_key}",
            flush=True,
        )
    else:
        if proc is not None:
            print(
                "[slot_sim_server] env key changed or worker died; restarting worker "
                f"old_key={worker_state.get('env_key')} new_key={env_key} "
                f"old_alive={proc.poll() is None}",
                flush=True,
            )
            _stop_env_worker(worker_state, reason="env_key_changed")
        _start_env_worker(args, worker_state, env_key)

    worker_params = dict(params)
    worker_params["max_env_sessions"] = 1
    worker_params["_cancel_event"] = None
    worker_params.pop("_cancel_event", None)
    task_run_id = int(params.get("_task_run_id", 0))
    start_response = _worker_call(
        "run_slot_planning",
        data=worker_params,
        port=worker_state["port"],
        timeout_ms=int(args.worker_request_timeout_ms),
    )
    worker_task_run_id = start_response.get("task_run_id", task_run_id)
    if worker_task_run_id is None:
        worker_task_run_id = task_run_id
    worker_task_run_id = int(worker_task_run_id)
    cancel_event = params.get("_cancel_event")
    while True:
        if cancel_event is not None and cancel_event.is_set():
            try:
                _worker_call(
                    "cancel_current",
                    port=worker_state["port"],
                    timeout_ms=int(args.worker_request_timeout_ms),
                )
            except Exception as exc:
                print(f"[slot_sim_server] worker cancel request failed: {exc}", flush=True)
            if bool(args.kill_worker_on_cancel):
                _stop_env_worker(worker_state, reason="cancelled_parent_task")
            return {"success": False, "status": "cancelled", "worker_reused": bool(same_env)}
        time.sleep(max(0.2, float(args.worker_poll_interval_s)))
        status = _worker_call(
            "status",
            port=worker_state["port"],
            timeout_ms=int(args.worker_request_timeout_ms),
        )
        status_task_run_id = status.get("task_run_id")
        if status_task_run_id is None and status.get("status") == "idle":
            raise RuntimeError(
                "env worker became idle before reporting completion: "
                f"parent_task_run_id={task_run_id} expected_worker_task={worker_task_run_id} "
                f"status={status}"
            )
        if status_task_run_id is not None and int(status_task_run_id) != worker_task_run_id:
            raise RuntimeError(
                "env worker switched tasks unexpectedly: "
                f"parent_task_run_id={task_run_id} expected_worker_task={worker_task_run_id} "
                f"got={status_task_run_id}"
            )
        state = status.get("status")
        if state == "completed":
            result = status.get("result", {}) or {}
            result["worker_reused"] = bool(same_env)
            result["worker_env_key"] = env_key
            result["worker_pid"] = int(worker_state["proc"].pid)
            return result
        if state in {"failed", "cancelled"}:
            error = status.get("error")
            if state == "cancelled":
                return {"success": False, "status": "cancelled", "worker_reused": bool(same_env)}
            raise RuntimeError(f"env worker planning failed: {error}")


def _task_status_payload(task):
    if task is None:
        return {"status": "idle", "has_task": False}
    running = task.get("status") == "running"
    payload = {
        "status": task.get("status", "unknown"),
        "has_task": True,
        "task_run_id": task.get("task_run_id"),
        "running": running,
        "cancel_requested": bool(task.get("cancel_event") is not None and task["cancel_event"].is_set()),
        "started_at": task.get("started_at"),
        "ended_at": task.get("ended_at"),
        "duration_s": None
        if task.get("started_at") is None
        else float((task.get("ended_at") or time.time()) - task["started_at"]),
        "params": task.get("params_summary"),
    }
    if task.get("error") is not None:
        payload["error"] = task["error"]
    if task.get("result") is not None:
        payload["result"] = task["result"]
    return payload


def _make_planning_task(params, task_run_id):
    cancel_event = threading.Event()
    return {
        "task_run_id": int(task_run_id),
        "cancel_event": cancel_event,
        "params": dict(params),
        "status": "pending",
        "started_at": None,
        "ended_at": None,
        "error": None,
        "result": None,
        "params_summary": {
            "task_id": params.get("task_id"),
            "perturbation_id": params.get("perturbation_id"),
            "scene_profile": params.get("scene_profile"),
            "task_variant": params.get("task_variant"),
            "max_steps": params.get("max_steps"),
            "model_port": params.get("port"),
        },
    }


def _control_server_loop(args, state, state_lock, shutdown_event, task_queue):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.host}:{args.port}")
    print(f"SLOT sim server listening on tcp://{args.host}:{args.port}", flush=True)

    try:
        while not shutdown_event.is_set():
            try:
                request = MsgSerializer.from_bytes(socket.recv())
                endpoint = request.get("endpoint")
                if endpoint == "ping":
                    with state_lock:
                        task_payload = _task_status_payload(state.get("current_task"))
                    socket.send(
                        MsgSerializer.to_bytes(
                            {
                                "status": "success",
                                "server": "slot_sim",
                                "task": task_payload,
                                "planning_module_path": VJEPA_PLANNING_PATH,
                                "requires_sim_server_restart_for_code_changes": True,
                            }
                        )
                    )
                elif endpoint == "status":
                    with state_lock:
                        status_payload = _task_status_payload(state.get("current_task"))
                    socket.send(MsgSerializer.to_bytes(status_payload))
                elif endpoint == "cancel_current":
                    with state_lock:
                        task = state.get("current_task")
                        if task is not None and task.get("status") in {"pending", "running"}:
                            print(
                                "[slot_sim_server] cancellation requested "
                                f"task_run_id={task.get('task_run_id')}",
                                flush=True,
                            )
                            task["cancel_event"].set()
                            if task.get("status") == "pending":
                                task["status"] = "cancelled"
                                task["ended_at"] = time.time()
                        status_payload = _task_status_payload(task)
                    socket.send(MsgSerializer.to_bytes(status_payload))
                elif endpoint == "shutdown":
                    shutdown_event.set()
                    with state_lock:
                        task = state.get("current_task")
                        if task is not None and task.get("status") in {"pending", "running"}:
                            task["cancel_event"].set()
                    socket.send(MsgSerializer.to_bytes({"status": "success"}))
                elif endpoint == "run_slot_planning":
                    params = request.get("data", {}) or {}
                    with state_lock:
                        old_task = state.get("current_task")
                        cancelled_previous = bool(
                            old_task is not None and old_task.get("status") in {"pending", "running"}
                        )
                        if cancelled_previous:
                            print(
                                "[slot_sim_server] cancelling active/pending task before queueing new run "
                                f"old_task_run_id={old_task.get('task_run_id')}",
                                flush=True,
                            )
                            old_task["cancel_event"].set()
                            if old_task.get("status") == "pending":
                                old_task["status"] = "cancelled"
                                old_task["ended_at"] = time.time()
                        task = _make_planning_task(params, state["next_task_run_id"])
                        state["next_task_run_id"] += 1
                        state["current_task"] = task
                        before_sessions = int(state.get("num_env_sessions", 0))
                    print(
                        "[slot_sim_server] queued run_slot_planning "
                        f"task_run_id={task['task_run_id']} "
                        f"task_id={params.get('task_id')} perturbation_id={params.get('perturbation_id')} "
                        f"max_steps={params.get('max_steps')} model_port={params.get('port')} "
                        f"cached_sessions={before_sessions}",
                        flush=True,
                    )
                    task_queue.put(task)
                    socket.send(
                        MsgSerializer.to_bytes(
                            {
                                "status": "started",
                                "task_run_id": task["task_run_id"],
                                "cancelled_previous": bool(cancelled_previous),
                                "num_env_sessions": before_sessions,
                                "reused_env_session": bool(before_sessions > 0),
                            }
                        )
                    )
                else:
                    socket.send(MsgSerializer.to_bytes({"error": f"Unknown endpoint: {endpoint}"}))
            except Exception as exc:
                traceback.print_exc()
                try:
                    socket.send(MsgSerializer.to_bytes({"error": str(exc)}))
                except Exception:
                    pass
    finally:
        socket.close(0)
        context.term()


def _drain_cancelled_or_superseded_tasks(task_queue, task):
    latest = task
    while True:
        try:
            next_task = task_queue.get_nowait()
        except queue.Empty:
            return latest
        if latest is not None and latest.get("status") == "pending":
            latest["cancel_event"].set()
            latest["status"] = "cancelled"
            latest["ended_at"] = time.time()
        latest = next_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument(
        "--disable-object-states",
        action="store_true",
        help=(
            "Disable full OmniGibson object-state initialization. Default matches successful-case collection "
            "and keeps object states enabled for more faithful materials / scene behavior."
        ),
    )
    parser.add_argument(
        "--enable-object-states",
        action="store_true",
        help=(
            "Compatibility flag; object states are enabled by default unless --disable-object-states is set."
        ),
    )
    parser.add_argument(
        "--enable-hq-rendering",
        action="store_true",
        help=(
            "Compatibility flag; high-quality RTX rendering is enabled by default unless --disable-hq-rendering is set."
        ),
    )
    parser.add_argument(
        "--disable-hq-rendering",
        action="store_true",
        help="Deprecated compatibility flag; HQ rendering is disabled by default.",
    )
    parser.add_argument(
        "--in-process-env-cache",
        action="store_true",
        help=(
            "Use the old fast path that keeps OmniGibson/Kit and env sessions in this server process. "
            "Default is isolated subprocess mode because scene switching in one Kit process is unstable."
        ),
    )
    parser.add_argument(
        "--one-shot-subprocess",
        action="store_true",
        help=(
            "Launch a fresh local-sim subprocess for every planning request. "
            "Default reuses one worker subprocess when the requested env key is unchanged."
        ),
    )
    parser.add_argument(
        "--worker-port",
        type=int,
        default=0,
        help=(
            "Port for the supervised env worker. 0 chooses parent --port + 10000, "
            "which avoids collisions when multiple sim servers run on different ports."
        ),
    )
    parser.add_argument("--worker-start-timeout-s", type=float, default=1200.0)
    parser.add_argument("--worker-request-timeout-ms", type=int, default=7200000)
    parser.add_argument("--worker-poll-interval-s", type=float, default=5.0)
    parser.add_argument("--kill-worker-on-cancel", action="store_true")
    args, unknown_args = parser.parse_known_args()
    if int(args.worker_port) <= 0:
        args.worker_port = int(args.port) + 10000
    allowed_kit_args = {
        "--ext-folder",
        os.environ.get("REALM_KIT_EXTS_DIR", ""),
        "--no-window",
    }
    unknown_user_args = [
        arg for arg in unknown_args
        if not arg.startswith("--/") and arg not in allowed_kit_args
    ]
    if unknown_user_args:
        parser.error(f"unrecognized arguments: {' '.join(unknown_user_args)}")

    if args.in_process_env_cache:
        _set_omnigibson_macros_once(
            enable_object_states=bool(not args.disable_object_states),
            enable_hq_rendering=bool(not args.disable_hq_rendering),
        )
    else:
        print(
            "[slot_sim_server] using supervised env-worker mode by default; "
            "same env key reuses the current worker/env, different env key restarts the worker. "
            "Use --one-shot-subprocess for one subprocess per request or --in-process-env-cache "
            "for the old resident-env behavior.",
            flush=True,
        )

    session_cache = {}
    state_lock = threading.Lock()
    shutdown_event = threading.Event()
    task_queue = queue.Queue()
    worker_state = {}
    state = {
        "current_task": None,
        "next_task_run_id": 1,
        "num_env_sessions": 0,
    }

    control_thread = threading.Thread(
        target=_control_server_loop,
        args=(args, state, state_lock, shutdown_event, task_queue),
        name="slot-sim-control",
        daemon=True,
    )
    control_thread.start()

    shared = _load_vjepa_planning_module() if args.in_process_env_cache else None
    try:
        while not shutdown_event.is_set():
            try:
                task = task_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            task = _drain_cancelled_or_superseded_tasks(task_queue, task)
            if task is None or task["cancel_event"].is_set():
                continue

            with state_lock:
                task["status"] = "running"
                task["started_at"] = time.time()
                task["ended_at"] = None
                state["current_task"] = task

            worker_params = dict(task["params"])
            worker_params["_task_run_id"] = int(task["task_run_id"])
            worker_params["_cancel_event"] = task["cancel_event"]
            started = time.perf_counter()
            print(
                "[slot_sim_server] running planning on main thread "
                f"task_run_id={task['task_run_id']}",
                flush=True,
            )
            try:
                if args.in_process_env_cache:
                    planning_result = _run_slot_planning(shared, session_cache, worker_params)
                elif args.one_shot_subprocess:
                    planning_result = _run_slot_planning_isolated(worker_params)
                else:
                    planning_result = _run_slot_planning_reused_env_worker(args, worker_state, worker_params)
                status = "cancelled" if task["cancel_event"].is_set() else "completed"
                result = {
                    "duration_s": float(time.perf_counter() - started),
                    "num_env_sessions": int(len(session_cache)),
                }
                if isinstance(planning_result, dict):
                    result.update(planning_result)
                handed_off = handoff_result_outputs(result, OUTPUT_ROOT)
                if handed_off:
                    print(
                        "[slot_sim_server] transferred output ownership to host user: "
                        + ", ".join(handed_off),
                        flush=True,
                    )
                error = None
            except Exception as exc:
                error_traceback = traceback.format_exc()
                traceback.print_exc()
                print(error_traceback, flush=True)
                status = "failed"
                result = None
                error = error_traceback

            if status == "cancelled":
                print(
                    "[slot_sim_server] skipping simulator stop after cancellation; "
                    "the next env reset/reload will handle simulator state as needed.",
                    flush=True,
                )
            elif args.in_process_env_cache and shared is not None:
                try:
                    _ensure_sim_stopped(shared, "finishing the current planning run", timeout_s=5.0)
                except Exception as exc:
                    print(f"[slot_sim_server] warning: failed to stop simulator after planning: {exc}", flush=True)

            with state_lock:
                task["status"] = status
                task["result"] = result
                task["error"] = error
                task["ended_at"] = time.time()
                if args.in_process_env_cache:
                    state["num_env_sessions"] = int(len(session_cache))
                else:
                    proc = worker_state.get("proc")
                    state["num_env_sessions"] = int(proc is not None and proc.poll() is None)

            print(
                "[slot_sim_server] planning finished "
                f"task_run_id={task['task_run_id']} status={status} "
                f"duration_s={float(time.perf_counter() - started):.2f}",
                flush=True,
            )
    finally:
        shutdown_event.set()
        if not args.in_process_env_cache:
            _stop_env_worker(worker_state, reason="parent_shutdown")
        if shared is not None:
            shared_og = getattr(shared, "og", None)
            if shared_og is not None and hasattr(shared_og, "shutdown"):
                shared_og.shutdown()
        control_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
