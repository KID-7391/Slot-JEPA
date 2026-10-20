import argparse
import io
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

import msgpack
import numpy as np
import zmq

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from _output_permissions import configure_output_umask, handoff_output_ownership


configure_output_umask()

VJEPA_PLANNING_PATH = os.path.join(THIS_DIR, "vjepa_planning.py")
DEFAULT_SUCCESS_CASE_DIR = None
SUPPORTED_SCENE_PROFILES = ["realm_droid10", "impact"]
DEFAULT_VJEPA_MAX_SETTLE_STEPS = 4
DEFAULT_VJEPA_SETTLE_POS_TOL = 0.01
DEFAULT_VJEPA_SETTLE_ROT_TOL = 0.05
DEFAULT_ABLATION_MAX_STEPS = 40
DEFAULT_A2_A4_MAX_STEPS = 100
DEFAULT_FULL_ATTEMPTS = 3
DEFAULT_ABLATION_ATTEMPTS = 1
DEFAULT_A2_A4_ATTEMPTS = 3
DEFAULT_ROTATE_DETERMINISTIC_STEP_RAD = 0.2
DEFAULT_ROTATE_DETERMINISTIC_AXIS = "yaw"
DEFAULT_ROTATE_DETERMINISTIC_OVERSHOOT_RAD = 0.25
SUPPORTED_TASKS = [
    # Reach: place targets (0-5).
    "reach_put_green_block_into_bowl",
    "reach_put_banana_into_box",
    "reach_put_orange_into_bowl",
    "reach_put_lemon_into_tray",
    "reach_put_can_into_bowl",
    "reach_put_marker_into_bowl",
    # Reach: pick targets (6-11).
    "reach_pick_spoon",
    "reach_pick_bowl",
    "reach_pick_lemon",
    "reach_pick_can_of_soda",
    "reach_pick_marker",
    "reach_pick_banana",
    # Reach: rotate targets (12-17).
    "reach_rotate_marker",
    "reach_rotate_mug",
    "reach_rotate_water_bottle",
    "reach_rotate_can_of_soda",
    "reach_rotate_banana",
    "reach_rotate_toy_dice",
    # Place (18-23), pick (24-29), and rotate (30-35) use the
    # same within-family object order as their reach counterparts above.
    "put_green_block_into_bowl",
    "put_banana_into_box",
    "put_orange_into_bowl",
    "put_lemon_into_tray",
    "put_can_into_bowl",
    "put_marker_into_bowl",
    "pick_spoon",
    "pick_bowl",
    "pick_lemon",
    "pick_can_of_soda",
    "pick_marker",
    "pick_banana",
    "rotate_marker",
    "rotate_mug",
    "rotate_water_bottle",
    "rotate_can_of_soda",
    "rotate_banana",
    "rotate_toy_dice",
]
SUPPORTED_PERTURBATIONS = [
    "Default",
    "V-AUG",
    "V-VIEW",
    "V-SC",
    "V-LIGHT",
    "S-PROP",
    "S-LANG",
    "S-MO",
    "S-AFF",
    "S-INT",
    "B-HOBJ",
    "SB-NOUN",
    "SB-VRB",
    "VB-POSE",
    "VB-MOBJ",
    "VSB-NOBJ",
]
DEFAULT_TASK_VARIANT = "default"
ALL_TASK_VARIANTS = "all"
REALM_OUTPUT_ROOT = Path(REPO_ROOT) / "realm" / "outputs"
ABLATION_VARIANTS = (
    "full",
    "a1_sequential_predictor",
    "a2_no_uot",
    "a3_recent_context",
    "a4_in_trajectory_random",
)
A2_A4_VARIANTS = {
    "a2_no_uot",
    "a3_recent_context",
    "a4_in_trajectory_random",
}


def _slot_output_namespace(ablation_variant):
    variant = str(ablation_variant or "full")
    return "slot" if variant == "full" else f"slot-{variant.replace('_', '-')}"


def _available_task_variants(task_id, scene_profile):
    if scene_profile != "realm_droid10":
        return [DEFAULT_TASK_VARIANT]
    task_name = SUPPORTED_TASKS[int(task_id)]
    task_dir = Path(REPO_ROOT) / "realm" / "config" / "tasks" / "REALM_DROID10" / task_name
    if not task_dir.exists():
        raise FileNotFoundError(f"No task config directory found for task_id={task_id}: {task_dir}")
    variants = []
    for yaml_path in sorted(task_dir.glob("*.yaml")):
        variant = yaml_path.stem
        if " " in variant:
            continue
        variants.append(variant)
    if not variants:
        raise FileNotFoundError(f"No task variants found under {task_dir}")
    if DEFAULT_TASK_VARIANT in variants:
        variants.remove(DEFAULT_TASK_VARIANT)
        variants.insert(0, DEFAULT_TASK_VARIANT)
    return variants


def _resolve_run_specs(args):
    task_id_text = str(args.task_id).strip()
    if not task_id_text:
        raise ValueError("--task_id cannot be empty.")
    if "," in task_id_text:
        task_ids = []
        seen = set()
        for item in task_id_text.split(","):
            item = item.strip()
            if not item:
                continue
            task_id = int(item)
            if task_id == -1:
                raise ValueError("--task_id=-1 cannot be mixed with an explicit comma-separated task list.")
            if not (0 <= task_id < len(SUPPORTED_TASKS)):
                raise ValueError(f"Invalid task_id={task_id}; expected [0, {len(SUPPORTED_TASKS) - 1}].")
            if task_id not in seen:
                task_ids.append(task_id)
                seen.add(task_id)
        if not task_ids:
            raise ValueError("--task_id comma-separated list did not contain any valid ids.")
    elif int(task_id_text) == -1:
        task_ids = list(range(len(SUPPORTED_TASKS)))
    else:
        if not (0 <= int(task_id_text) < len(SUPPORTED_TASKS)):
            raise ValueError(f"Invalid task_id={args.task_id}; expected -1 or [0, {len(SUPPORTED_TASKS) - 1}].")
        task_ids = [int(task_id_text)]

    specs = []
    for task_id in task_ids:
        if args.task_variant in {ALL_TASK_VARIANTS, "all_variants"}:
            variants = _available_task_variants(task_id, args.scene_profile)
        else:
            variants = [args.task_variant or DEFAULT_TASK_VARIANT]
        for variant in variants:
            specs.append(
                {
                    "task_id": int(task_id),
                    "task_name": SUPPORTED_TASKS[int(task_id)],
                    "task_variant": variant,
                }
            )
    return specs


def _try_plot_view_losses(output_dir):
    if not output_dir:
        return
    step_trace_path = os.path.join(output_dir, "step_traces.pt")
    if not os.path.exists(step_trace_path):
        return
    try:
        import torch
        from scripts.plot_view_losses import plot_view_losses_from_step_traces

        step_traces = torch.load(step_trace_path, map_location="cpu")
        save_path = os.path.join(output_dir, "view_losses_over_steps.png")
        if plot_view_losses_from_step_traces(step_traces, save_path):
            print(f"[slot_planning] saved per-view loss plot: {save_path}", flush=True)
    except Exception as exc:
        print(f"[slot_planning] warning: failed to plot per-view losses: {exc}", flush=True)


def _make_case_dir_name(task, perturbation, state_frame, scene_profile, task_variant=DEFAULT_TASK_VARIANT):
    suffix = "" if scene_profile == "realm_droid10" else f"_{scene_profile}"
    # The shared planner and task YAMLs call the default variant `default`.
    # Keep the old wrapper spelling as a no-suffix alias for existing outputs.
    variant_suffix = (
        ""
        if task_variant in (None, DEFAULT_TASK_VARIANT, "all_variants")
        else f"_{task_variant}"
    )
    return f"{task}{variant_suffix}_{perturbation}_{state_frame}_frame{suffix}"


def _get_slot_output_base_dir(
    task, perturbation, state_frame, scene_profile, goal_mode, task_variant, ablation_variant="full"
):
    case_name = _make_case_dir_name(
        task=task,
        perturbation=perturbation,
        state_frame=state_frame,
        scene_profile=scene_profile,
        task_variant=task_variant,
    )
    return REALM_OUTPUT_ROOT / _slot_output_namespace(ablation_variant) / str(task) / f"{case_name}_{goal_mode}"


def _attempt_number_from_name(name):
    match = re.search(r"_attempt(\d+)$", str(name))
    if match:
        return int(match.group(1))
    return None


def _attempt_dirs(output_base_dir):
    output_base_dir = Path(output_base_dir)
    if not output_base_dir.exists():
        return []
    dirs = []
    for child in output_base_dir.iterdir():
        if not child.is_dir():
            continue
        attempt_num = _attempt_number_from_name(child.name)
        if attempt_num is not None:
            dirs.append((child.name, attempt_num, child))
    return sorted(dirs, key=lambda item: item[0])


def _load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _attempt_dir_status(attempt_dir):
    """Return completion / success status for one attempt output directory."""
    attempt_dir = Path(attempt_dir)
    summary = _load_json_if_exists(attempt_dir / "step_trace_summary.json")
    if not isinstance(summary, dict):
        return {"complete": False, "success": False, "reason": "missing_step_trace_summary"}
    if bool(summary.get("cancelled", False)):
        return {"complete": False, "success": False, "reason": "cancelled"}
    if int(summary.get("num_steps") or 0) <= 0:
        return {"complete": False, "success": False, "reason": "empty_step_trace"}

    # A successful final result is authoritative; optional visualization files
    # must not cause an already successful case to be scheduled again.
    result = _load_json_if_exists(attempt_dir / "planning_result.json")
    if isinstance(result, dict) and (
        bool(result.get("success", False)) or result.get("status") == "success"
    ):
        return {"complete": True, "success": True, "reason": "complete_success"}

    # These are written near the end of a normal run. Their absence usually
    # means the process was interrupted after trace flush but before final plots.
    for required_name in ("trajectory_pca.png", "view_losses_over_steps.png", "goal_manifest.json"):
        if not (attempt_dir / required_name).exists():
            return {"complete": False, "success": False, "reason": f"missing_{required_name}"}

    if isinstance(result, dict):
        if result.get("status") in {"cancelled", "failed"}:
            return {"complete": False, "success": False, "reason": f"attempt_status_{result.get('status')}"}
        return {"complete": True, "success": False, "reason": "complete_failure"}

    # Legacy output directories predate planning_result.json. If they reached
    # all final artifacts and were not cancelled, treat them as complete failed
    # attempts; successful runs normally stop retrying before attempt03 exists.
    return {"complete": True, "success": False, "reason": "legacy_complete_failure"}


def _filter_specs_by_existing_attempts(args, run_specs):
    if args.disable_skip_completed_attempts:
        return run_specs, []
    if not (0 <= int(args.perturbation_id) < len(SUPPORTED_PERTURBATIONS)):
        raise ValueError(
            f"Invalid perturbation_id={args.perturbation_id}; "
            f"expected [0, {len(SUPPORTED_PERTURBATIONS) - 1}]."
        )
    perturbation = SUPPORTED_PERTURBATIONS[int(args.perturbation_id)]
    required_attempts = max(1, int(args.skip_completed_attempts_threshold))
    active_specs = []
    skipped = []
    for spec in run_specs:
        output_base_dir = _get_slot_output_base_dir(
            task=spec["task_name"],
            perturbation=perturbation,
            state_frame=args.state_frame,
            scene_profile=args.scene_profile,
            goal_mode=args.goal_mode,
            task_variant=spec["task_variant"],
            ablation_variant=args.ablation_variant,
        )
        attempt_dirs = _attempt_dirs(output_base_dir)
        latest = attempt_dirs[-required_attempts:]
        latest_attempt_nums = [int(item[1]) for item in latest]
        latest_dirs = [item[2] for item in latest]
        expected_attempt_nums = list(range(1, required_attempts + 1))
        completion_reasons = []
        skip_reason = None
        if attempt_dirs:
            newest_dir = attempt_dirs[-1][2]
            newest_status = _attempt_dir_status(newest_dir)
            latest_success = None
            latest_success_status = None
            for _, _, attempt_dir in reversed(attempt_dirs):
                status = _attempt_dir_status(attempt_dir)
                if bool(status["complete"]) and bool(status["success"]):
                    latest_success = attempt_dir
                    latest_success_status = status
                    break
            if latest_success is not None:
                completion_reasons.append(
                    {
                        "dir": str(latest_success),
                        "complete": True,
                        "success": True,
                        "reason": latest_success_status["reason"],
                        "role": "latest_success",
                    }
                )
                skip_reason = "existing_attempt_success"
            else:
                completion_reasons.append(
                    {
                        "dir": str(newest_dir),
                        "complete": bool(newest_status["complete"]),
                        "success": bool(newest_status["success"]),
                        "reason": newest_status["reason"],
                        "role": "newest",
                    }
                )

        complete_failed_retry_set = (
            skip_reason is None
            and len(latest) == required_attempts
            and latest_attempt_nums == expected_attempt_nums
        )
        if complete_failed_retry_set:
            completion_reasons = []
            for attempt_dir in latest_dirs:
                status = _attempt_dir_status(attempt_dir)
                completion_reasons.append(
                    {
                        "dir": str(attempt_dir),
                        "complete": bool(status["complete"]),
                        "success": bool(status["success"]),
                        "reason": status["reason"],
                    }
                )
                if (not bool(status["complete"])) or bool(status["success"]):
                    complete_failed_retry_set = False
                    break
            if complete_failed_retry_set:
                skip_reason = "latest_three_attempts_complete_failed"

        if skip_reason is not None:
            skipped.append(
                {
                    "task_id": int(spec["task_id"]),
                    "task": spec["task_name"],
                    "task_variant": spec["task_variant"],
                    "output_base_dir": str(output_base_dir),
                    "latest_attempts": latest_attempt_nums,
                    "latest_attempt_dirs": [str(path) for path in latest_dirs],
                    "completion_reasons": completion_reasons,
                    "required_attempts": int(required_attempts),
                    "skip_reason": skip_reason,
                }
            )
            print(
                "[slot_planning] skipping completed experiment before contacting sim server: "
                f"task={spec['task_name']} task_id={spec['task_id']} variant={spec['task_variant']} "
                f"reason={skip_reason} latest_attempts={latest_attempt_nums} required={expected_attempt_nums} "
                f"output_base_dir={output_base_dir}",
                flush=True,
            )
        else:
            if attempt_dirs:
                newest = attempt_dirs[-1]
                print(
                    "[slot_planning] will run experiment; latest attempts are not a complete failed retry set: "
                    f"task={spec['task_name']} variant={spec['task_variant']} "
                    f"newest={newest[0]} latest_attempts={latest_attempt_nums} "
                    f"required={expected_attempt_nums} output_base_dir={output_base_dir}",
                    flush=True,
                )
            active_specs.append(spec)
    return active_specs, skipped


class _MsgSerializer:
    @staticmethod
    def to_bytes(data):
        return msgpack.packb(data, default=_MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data):
        return msgpack.unpackb(data, object_hook=_MsgSerializer.decode_custom_classes)

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


class _SimServerClient:
    def __init__(self, host, port, timeout_ms):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.set_timeout(timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")

    def set_timeout(self, timeout_ms):
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))

    def call_endpoint(self, endpoint, data=None):
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.socket.send(_MsgSerializer.to_bytes(request))
            response = _MsgSerializer.from_bytes(self.socket.recv())
        except zmq.Again as exc:
            raise TimeoutError(f"Timed out waiting for SLOT sim server endpoint `{endpoint}`.") from exc
        if "error" in response:
            raise RuntimeError(f"Sim server error: {response['error']}")
        return response


def _load_vjepa_planning_module():
    spec = importlib.util.spec_from_file_location("vjepa_planning_shared", VJEPA_PLANNING_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-server-host", type=str, default="127.0.0.1")
    parser.add_argument("--sim-server-port", type=int, default=8023)
    parser.add_argument("--sim-server-connect-timeout-ms", type=int, default=10000)
    parser.add_argument("--sim-server-timeout-ms", type=int, default=7200000)
    parser.add_argument("--sim-server-poll-interval-s", type=float, default=5.0)
    parser.add_argument(
        "--max-env-sessions",
        type=int,
        default=1,
        help=(
            "Maximum env sessions cached by the SLOT sim server. Default 1 avoids keeping multiple "
            "Isaac/RTX scenes alive, which can corrupt robot gripper rendering."
        ),
    )
    parser.add_argument("--local-sim", action="store_true")
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help=(
            f"Maximum number of full planning attempts per case (dynamic default: "
            f"{DEFAULT_FULL_ATTEMPTS} for full/A0, {DEFAULT_ABLATION_ATTEMPTS} for A1, "
            f"and {DEFAULT_A2_A4_ATTEMPTS} for A2-A4)."
        ),
    )
    parser.add_argument(
        "--disable-skip-completed-attempts",
        action="store_true",
        help=(
            "Do not skip experiments that already have completed attempt output directories. "
            "By default, slot_planning skips before contacting the sim server."
        ),
    )
    parser.add_argument(
        "--skip-completed-attempts-threshold",
        type=int,
        default=None,
        help=(
            "Number of attemptXX output directories required to skip an experiment. "
            f"Dynamic default: {DEFAULT_FULL_ATTEMPTS} for full/A0, "
            f"{DEFAULT_ABLATION_ATTEMPTS} for A1, and {DEFAULT_A2_A4_ATTEMPTS} for A2-A4."
        ),
    )
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument(
        "--ablation-variant",
        choices=ABLATION_VARIANTS,
        default="full",
        help=(
            "Label the run and isolate its output/skip-resume namespace. The matching "
            "serve_slot.py ablation flag must be used on the model server."
        ),
    )
    parser.add_argument(
        "--task_id",
        type=str,
        default="0",
        help=(
            "Task id(s) to run. Use a single id like `0`, a comma-separated list like `0,1,2`, "
            "or `-1` to run all supported tasks."
        ),
    )
    parser.add_argument("--perturbation_id", type=int, default=0)
    parser.add_argument("--state-frame", type=str, choices=["camera"], default="camera")
    parser.add_argument("--success-case-dir", type=str, default=DEFAULT_SUCCESS_CASE_DIR)
    parser.add_argument("--goal-mode", type=str, choices=["first", "end", "all"], default="all")
    parser.add_argument("--scene-profile", type=str, choices=SUPPORTED_SCENE_PROFILES, default="realm_droid10")
    parser.add_argument(
        "--task-variant",
        type=str,
        default=DEFAULT_TASK_VARIANT,
        help="Task variant to run. Use `all` to run every yaml variant available for each selected task.",
    )
    parser.add_argument(
        "--fixed-camera-presets",
        nargs=2,
        default=None,
        metavar=("CAM1", "CAM2"),
        help="Override the two fixed external camera presets. They must match the released successful_case targets.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Maximum environment steps per attempt (dynamic default: "
            f"{DEFAULT_ABLATION_MAX_STEPS} for full/A0 and A1, "
            f"{DEFAULT_A2_A4_MAX_STEPS} for A2-A4)."
        ),
    )
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
        help="Extra radians added beyond the success-case rotate target.",
    )
    parser.add_argument(
        "--rotate-deterministic-axis",
        type=str,
        choices=["roll", "pitch", "yaw", "x", "y", "z"],
        default=DEFAULT_ROTATE_DETERMINISTIC_AXIS,
    )
    parser.add_argument("--enable-settle-debug", action="store_true")
    parser.add_argument("--controller-only-goal-delta", action="store_true")
    parser.add_argument("--slot-model-gripper-units", type=str, choices=["raw_width", "normalized"], default="normalized")
    parser.add_argument(
        "--result-json",
        type=str,
        default=None,
        help="Optional path for writing machine-readable local-sim results. Used by the resident sim broker.",
    )
    args = parser.parse_args()
    if args.ablation_variant in A2_A4_VARIANTS:
        default_attempts = DEFAULT_A2_A4_ATTEMPTS
        default_max_steps = DEFAULT_A2_A4_MAX_STEPS
    elif args.ablation_variant == "full":
        default_attempts = DEFAULT_FULL_ATTEMPTS
        default_max_steps = DEFAULT_ABLATION_MAX_STEPS
    else:
        default_attempts = DEFAULT_ABLATION_ATTEMPTS
        default_max_steps = DEFAULT_ABLATION_MAX_STEPS
    if args.retries is None:
        args.retries = default_attempts
    if args.skip_completed_attempts_threshold is None:
        args.skip_completed_attempts_threshold = default_attempts
    if args.max_steps is None:
        args.max_steps = default_max_steps
    run_specs = _resolve_run_specs(args)
    run_specs, skipped_specs = _filter_specs_by_existing_attempts(args, run_specs)
    if not run_specs:
        print(
            "[slot_planning] all resolved experiments already have completed attempts; "
            "nothing to submit to sim server.",
            flush=True,
        )
        if args.result_json:
            payload = {
                "status": "skipped",
                "success": False,
                "results": [],
                "skipped": skipped_specs,
                "last_result": None,
            }
            os.makedirs(os.path.dirname(os.path.abspath(args.result_json)), exist_ok=True)
            with open(args.result_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        return

    if not args.local_sim:
        client = _SimServerClient(
            host=args.sim_server_host,
            port=args.sim_server_port,
            timeout_ms=args.sim_server_connect_timeout_ms,
        )
        started = time.perf_counter()
        final_response = None
        try:
            ping_response = client.call_endpoint("ping")
            print(
                "[slot_planning] connected sim server "
                f"planning_module={ping_response.get('planning_module_path')} "
                f"restart_required_for_code_changes={ping_response.get('requires_sim_server_restart_for_code_changes')}",
                flush=True,
            )
            client.set_timeout(args.sim_server_timeout_ms)
            max_attempts = max(1, int(args.retries))
            all_results = []
            all_results.extend(
                {
                    "task_id": int(item["task_id"]),
                    "task": item["task"],
                    "task_variant": item["task_variant"],
                    "attempt": None,
                    "success": False,
                    "skipped": True,
                    "output_dir": item["output_base_dir"],
                    "latest_attempts": item["latest_attempts"],
                    "latest_attempt_dirs": item["latest_attempt_dirs"],
                    "completion_reasons": item["completion_reasons"],
                    "skip_reason": item["skip_reason"],
                }
                for item in skipped_specs
            )
            for run_idx, spec in enumerate(run_specs):
                for attempt_idx in range(max_attempts):
                    payload = vars(args).copy()
                    payload["task_id"] = int(spec["task_id"])
                    payload["task_variant"] = str(spec["task_variant"])
                    payload["retry_attempt"] = int(attempt_idx)
                    payload["retry_max_attempts"] = int(max_attempts)
                    print(
                        "[slot_planning] connected to SLOT sim server; submitting run_slot_planning "
                        f"run={run_idx + 1}/{len(run_specs)} "
                        f"task={spec['task_name']} task_id={spec['task_id']} "
                        f"variant={spec['task_variant']} perturbation_id={args.perturbation_id} "
                        f"attempt={attempt_idx + 1}/{max_attempts} max_steps={args.max_steps}. "
                        "Waiting for server acknowledgement...",
                        flush=True,
                    )
                    start_response = client.call_endpoint("run_slot_planning", data=payload)
                    task_run_id = start_response.get("task_run_id")
                    print(
                        "[slot_planning] server accepted task "
                        f"task_run_id={task_run_id} cancelled_previous={start_response.get('cancelled_previous')}. "
                        "Polling status...",
                        flush=True,
                    )
                    response = {}
                    while True:
                        time.sleep(max(0.1, float(args.sim_server_poll_interval_s)))
                        status = client.call_endpoint("status")
                        if status.get("task_run_id") != task_run_id:
                            raise RuntimeError(
                                "SLOT sim server switched to a different task while this client was waiting: "
                                f"expected={task_run_id}, got={status.get('task_run_id')}"
                            )
                        elapsed = float(status.get("duration_s") or 0.0)
                        state = status.get("status")
                        print(
                            f"[slot_planning] task_run_id={task_run_id} status={state} elapsed_s={elapsed:.1f}",
                            flush=True,
                        )
                        if state == "completed":
                            response = status.get("result", {}) or {}
                            break
                        if state in {"failed", "cancelled"}:
                            response = {"success": False, "error": status.get("error"), "status": state}
                            print(
                                f"[slot_planning] attempt {attempt_idx + 1} ended with {state}: {status.get('error')}",
                                flush=True,
                            )
                            break
                    final_response = response
                    output_dir = response.get("output_dir") if isinstance(response, dict) else None
                    _try_plot_view_losses(output_dir)
                    handoff_output_ownership(output_dir, REALM_OUTPUT_ROOT)
                    all_results.append(
                        {
                            "task_id": int(spec["task_id"]),
                            "task": spec["task_name"],
                            "task_variant": spec["task_variant"],
                            "attempt": int(attempt_idx),
                            "success": bool(response.get("success")) if isinstance(response, dict) else False,
                            "output_dir": output_dir,
                        }
                    )
                    print(
                        "[slot_planning] attempt completed "
                        f"run={run_idx + 1}/{len(run_specs)} "
                        f"attempt={attempt_idx + 1}/{max_attempts} "
                        f"success={bool(response.get('success')) if isinstance(response, dict) else None} "
                        f"output_dir={output_dir}",
                        flush=True,
                    )
                    if isinstance(response, dict) and response.get("success"):
                        break
        except Exception as exc:
            raise RuntimeError(
                "Failed to contact SLOT sim server. Start it first with:\n"
                f"  OMNIGIBSON_HEADLESS=1 python runtime/slot_sim_server.py --port {args.sim_server_port}\n"
                "Or run local Isaac/Omnigibson in this process with --local-sim."
            ) from exc
        print(
            "[slot_planning] sim-server run completed "
            f"success={bool(final_response.get('success')) if isinstance(final_response, dict) else None} "
            f"num_runs={len(run_specs)} "
            f"duration_s={time.perf_counter() - started:.2f}",
            flush=True,
        )
        return

    shared = _load_vjepa_planning_module()
    local_results = []

    if args.controller_only_goal_delta:
        if len(run_specs) != 1:
            raise ValueError("--controller-only-goal-delta only supports one resolved task/variant at a time.")
        spec = run_specs[0]
        result = shared.run_controller_goal_delta_diagnostic(
            task_id=int(spec["task_id"]),
            perturbation_id=args.perturbation_id,
            state_frame=args.state_frame,
            success_case_dir=args.success_case_dir,
            goal_mode=args.goal_mode,
            scene_profile=args.scene_profile,
            task_variant=str(spec["task_variant"]),
            max_settle_steps=args.max_settle_steps,
            settle_pos_tol=args.settle_pos_tol,
            settle_rot_tol=args.settle_rot_tol,
            enable_settle_debug=args.enable_settle_debug,
        )
        local_results.append(
            {
                "task_id": int(spec["task_id"]),
                "task": spec["task_name"],
                "task_variant": str(spec["task_variant"]),
                "attempt": 0,
                "success": bool(result.get("success")) if isinstance(result, dict) else False,
                "output_dir": result.get("output_dir") if isinstance(result, dict) else None,
                "result": result,
            }
        )
    else:
        max_attempts = max(1, int(args.retries))
        for run_idx, spec in enumerate(run_specs):
            for attempt_idx in range(max_attempts):
                result = shared.run_vjepa_planning(
                    port=args.port,
                    task_id=int(spec["task_id"]),
                    perturbation_id=args.perturbation_id,
                    state_frame=args.state_frame,
                    success_case_dir=args.success_case_dir,
                    goal_mode=args.goal_mode,
                    scene_profile=args.scene_profile,
                    task_variant=str(spec["task_variant"]),
                    max_steps=args.max_steps,
                    action_repeat=args.action_repeat,
                    replay_mode=args.replay_mode,
                    max_settle_steps=args.max_settle_steps,
                    settle_pos_tol=args.settle_pos_tol,
                    settle_rot_tol=args.settle_rot_tol,
                    enable_settle_debug=args.enable_settle_debug,
                    model_type="slot",
                    slot_model_gripper_units=args.slot_model_gripper_units,
                    output_run_suffix=f"attempt{attempt_idx + 1:02d}",
                    fixed_camera_presets=args.fixed_camera_presets,
                    rotate_deterministic_after_grasp=not args.disable_rotate_deterministic_after_grasp,
                    rotate_deterministic_step_rad=args.rotate_deterministic_step_rad,
                    rotate_deterministic_axis=args.rotate_deterministic_axis,
                    rotate_deterministic_overshoot_rad=args.rotate_deterministic_overshoot_rad,
                    output_namespace=_slot_output_namespace(args.ablation_variant),
                )
                if isinstance(result, dict):
                    handoff_output_ownership(result.get("output_dir"), REALM_OUTPUT_ROOT)
                print(
                    f"[slot_planning] local run {run_idx + 1}/{len(run_specs)} "
                    f"task={spec['task_name']} variant={spec['task_variant']} "
                    f"attempt {attempt_idx + 1}/{max_attempts} "
                    f"success={None if result is None else result.get('success')} "
                    f"output_dir={None if result is None else result.get('output_dir')}",
                    flush=True,
                )
                local_results.append(
                    {
                        "task_id": int(spec["task_id"]),
                        "task": spec["task_name"],
                        "task_variant": str(spec["task_variant"]),
                        "attempt": int(attempt_idx),
                        "success": bool(result.get("success")) if isinstance(result, dict) else False,
                        "output_dir": result.get("output_dir") if isinstance(result, dict) else None,
                        "result": result,
                    }
                )
                if isinstance(result, dict) and result.get("success"):
                    break

    shared.og.shutdown()
    if args.result_json:
        payload = {
            "status": "success",
            "success": any(bool(item.get("success")) for item in local_results),
            "results": local_results,
            "last_result": local_results[-1].get("result") if local_results else None,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.result_json)), exist_ok=True)
        with open(args.result_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    sys.exit(0)


if __name__ == "__main__":
    main()
