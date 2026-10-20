import argparse
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path

import msgpack
import numpy as np
import zmq


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

VJEPA_PLANNING_PATH = os.path.join(THIS_DIR, "vjepa_planning.py")
DEFAULT_TASK_VARIANT = "default"
ALL_TASK_VARIANTS = "all"
REALM_OUTPUT_ROOT = Path(REPO_ROOT) / "realm" / "outputs"


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
        self.socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self.socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self.socket.connect(f"tcp://{host}:{port}")

    def close(self):
        self.socket.close(0)
        self.context.term()

    def call_endpoint(self, endpoint, data=None):
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.socket.send(_MsgSerializer.to_bytes(request))
            response = _MsgSerializer.from_bytes(self.socket.recv())
        except zmq.Again as exc:
            raise TimeoutError(f"Timed out waiting for sim server endpoint `{endpoint}`.") from exc
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(f"Sim server error: {response['error']}")
        return response


def _load_vjepa_planning_module():
    spec = importlib.util.spec_from_file_location("vjepa_planning_shared", VJEPA_PLANNING_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parse_task_ids(task_id_value, shared=None):
    raw = str(task_id_value).strip()
    if raw == "-1":
        if shared is None:
            shared = _load_vjepa_planning_module()
        return list(range(len(shared.SUPPORTED_TASKS)))
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _task_name(shared, task_id):
    try:
        return shared.SUPPORTED_TASKS[int(task_id)]
    except Exception:
        return f"task_{int(task_id)}"


def _available_task_variants(shared, task_id, scene_profile):
    if str(scene_profile).lower() != "realm_droid10":
        return [DEFAULT_TASK_VARIANT]
    task_name = _task_name(shared, task_id)
    task_dir = os.path.join(REPO_ROOT, "realm", "config", "tasks", "REALM_DROID10", task_name)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"No task config directory found for task_id={task_id}: {task_dir}")
    variants = []
    for entry in sorted(os.listdir(task_dir)):
        if not entry.endswith(".yaml"):
            continue
        variant = os.path.splitext(entry)[0]
        if " " in variant:
            continue
        variants.append(variant)
    if not variants:
        raise FileNotFoundError(f"No task variants found under {task_dir}")
    if DEFAULT_TASK_VARIANT in variants:
        variants.remove(DEFAULT_TASK_VARIANT)
        variants.insert(0, DEFAULT_TASK_VARIANT)
    return variants


def _resolve_run_specs(args, shared):
    specs = []
    for task_id in _parse_task_ids(args.task_id, shared=shared):
        if args.task_variant in {ALL_TASK_VARIANTS, "all_variants"}:
            variants = _available_task_variants(shared, task_id, args.scene_profile)
        else:
            variants = [args.task_variant or DEFAULT_TASK_VARIANT]
        for variant in variants:
            specs.append(
                {
                    "task_id": int(task_id),
                    "task_name": _task_name(shared, task_id),
                    "task_variant": str(variant),
                }
            )
    return specs


def _write_result_json(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)


def _attempt_number_from_name(name):
    digits = ""
    text = str(name)
    marker = "_attempt"
    if marker in text:
        tail = text.split(marker)[-1]
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
    return int(digits) if digits else None


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
        with open(path, "r", encoding="utf-8") as fin:
            return json.load(fin)
    except Exception:
        return None


def _attempt_dir_status(attempt_dir):
    attempt_dir = Path(attempt_dir)
    summary = _load_json_if_exists(attempt_dir / "step_trace_summary.json")
    if not isinstance(summary, dict):
        return {"complete": False, "success": False, "reason": "missing_step_trace_summary"}
    if bool(summary.get("cancelled", False)):
        return {"complete": False, "success": False, "reason": "cancelled"}
    if int(summary.get("num_steps") or 0) <= 0:
        return {"complete": False, "success": False, "reason": "empty_step_trace"}
    for required_name in ("trajectory_pca.png", "view_losses_over_steps.png", "goal_manifest.json"):
        if not (attempt_dir / required_name).exists():
            return {"complete": False, "success": False, "reason": f"missing_{required_name}"}
    result = _load_json_if_exists(attempt_dir / "planning_result.json")
    if isinstance(result, dict):
        if bool(result.get("success", False)) or result.get("status") == "success":
            return {"complete": True, "success": True, "reason": "complete_success"}
        if result.get("status") in {"cancelled", "failed"}:
            return {"complete": False, "success": False, "reason": f"attempt_status_{result.get('status')}"}
        return {"complete": True, "success": False, "reason": "complete_failure"}
    return {"complete": True, "success": False, "reason": "legacy_complete_failure"}


def _get_output_base_dir(args, spec):
    perturbation = str(_load_vjepa_planning_module().SUPPORTED_PERTURBATIONS[int(args.perturbation_id)])
    variant_suffix = "" if spec["task_variant"] in (None, DEFAULT_TASK_VARIANT) else f"_{spec['task_variant']}"
    scene_suffix = "" if args.scene_profile == "realm_droid10" else f"_{args.scene_profile}"
    dir_name = f"{spec['task_name']}{variant_suffix}_{perturbation}_{args.state_frame}_frame{scene_suffix}_{args.goal_mode}"
    return REALM_OUTPUT_ROOT / str(args.model_type).strip().lower() / str(spec["task_name"]) / dir_name


def _filter_specs_by_existing_attempts(args, run_specs):
    if args.disable_skip_completed_attempts:
        return run_specs, []
    shared = _load_vjepa_planning_module()
    if not (0 <= int(args.perturbation_id) < len(shared.SUPPORTED_PERTURBATIONS)):
        raise ValueError(
            f"Invalid perturbation_id={args.perturbation_id}; expected [0, {len(shared.SUPPORTED_PERTURBATIONS) - 1}]."
        )
    required_attempts = max(1, int(args.skip_completed_attempts_threshold))
    active_specs = []
    skipped = []
    for spec in run_specs:
        output_base_dir = _get_output_base_dir(args, spec)
        attempt_dirs = _attempt_dirs(output_base_dir)
        latest = attempt_dirs[-required_attempts:] if len(attempt_dirs) >= required_attempts else []
        latest_attempt_nums = [int(item[1]) for item in latest]
        latest_dirs = [item[2] for item in latest]
        expected_attempt_nums = list(range(1, required_attempts + 1))
        completion_reasons = []
        skip_reason = None
        if attempt_dirs:
            newest_dir = attempt_dirs[-1][2]
            newest_status = _attempt_dir_status(newest_dir)
            completion_reasons.append(
                {
                    "dir": str(newest_dir),
                    "complete": bool(newest_status["complete"]),
                    "success": bool(newest_status["success"]),
                    "reason": newest_status["reason"],
                    "role": "newest",
                }
            )
            if bool(newest_status["complete"]) and bool(newest_status["success"]):
                skip_reason = "latest_attempt_success"
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
                f"[{args.model_type}_planning] skipping completed experiment before contacting sim server: "
                f"task={spec['task_name']} task_id={spec['task_id']} variant={spec['task_variant']} "
                f"reason={skip_reason} latest_attempts={latest_attempt_nums} required={expected_attempt_nums} "
                f"output_base_dir={output_base_dir}",
                flush=True,
            )
        else:
            active_specs.append(spec)
    return active_specs, skipped


def build_parser(model_type, default_model_port):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default=model_type)
    parser.add_argument("--sim-server-host", type=str, default="127.0.0.1")
    parser.add_argument("--sim-server-port", type=int, default=8023)
    parser.add_argument("--sim-server-timeout-ms", type=int, default=7200000)
    parser.add_argument("--sim-server-poll-interval-s", type=float, default=5.0)
    parser.add_argument("--local-sim", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--disable-skip-completed-attempts", action="store_true")
    parser.add_argument("--skip-completed-attempts-threshold", type=int, default=3)
    parser.add_argument("--port", type=int, default=default_model_port)
    parser.add_argument("--task_id", type=str, default="0")
    parser.add_argument("--perturbation_id", type=int, default=0)
    parser.add_argument("--state-frame", type=str, choices=["camera"], default="camera")
    parser.add_argument("--success-case-dir", type=str, default=None)
    parser.add_argument("--goal-mode", type=str, choices=["first", "end", "all"], default="all")
    parser.add_argument("--scene-profile", type=str, default="realm_droid10")
    parser.add_argument("--task-variant", type=str, default="default")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument(
        "--replay-mode",
        type=str,
        choices=["robot_absolute", "camera_delta", "camera_delta_closed_loop"],
        default="robot_absolute",
    )
    parser.add_argument("--max-settle-steps", type=int, default=4)
    parser.add_argument("--settle-pos-tol", type=float, default=0.01)
    parser.add_argument("--settle-rot-tol", type=float, default=0.05)
    parser.add_argument("--disable-rotate-deterministic-after-grasp", action="store_true")
    parser.add_argument("--rotate-deterministic-step-rad", type=float, default=0.2)
    parser.add_argument("--rotate-deterministic-overshoot-rad", type=float, default=0.25)
    parser.add_argument("--rotate-deterministic-axis", type=str, choices=["roll", "pitch", "yaw", "x", "y", "z"], default="yaw")
    parser.add_argument("--enable-settle-debug", action="store_true")
    parser.add_argument("--controller-only-goal-delta", action="store_true")
    parser.add_argument("--fixed-camera-presets", nargs=2, default=None, metavar=("CAM1", "CAM2"))
    parser.add_argument("--result-json", type=str, default=None)
    return parser


def _payload_from_args(args, task_id, attempt_idx):
    payload = vars(args).copy()
    payload["task_id"] = int(task_id)
    payload["retry_attempt"] = int(attempt_idx)
    payload["retry_max_attempts"] = int(max(1, args.retries))
    payload["model_type"] = str(args.model_type)
    if payload.get("success_case_dir") is None:
        payload.pop("success_case_dir", None)
    return payload


def run_feature_planning_cli(model_type, default_model_port):
    parser = build_parser(model_type=model_type, default_model_port=default_model_port)
    args, unknown_args = parser.parse_known_args()
    unknown_user_args = [arg for arg in unknown_args if not arg.startswith("--/")]
    if unknown_user_args:
        parser.error(f"unrecognized arguments: {' '.join(unknown_user_args)}")

    shared = _load_vjepa_planning_module()
    run_specs = _resolve_run_specs(args, shared)
    run_specs, skipped_specs = _filter_specs_by_existing_attempts(args, run_specs)
    all_results = []
    final_response = None
    if not run_specs:
        payload = {
            "model_type": args.model_type,
            "status": "skipped",
            "success": False,
            "results": [],
            "skipped": skipped_specs,
            "last_result": None,
        }
        print(
            f"[{args.model_type}_planning] all resolved experiments already have completed attempts; nothing to run.",
            flush=True,
        )
        _write_result_json(args.result_json, payload)
        return payload
    if not args.local_sim:
        client = _SimServerClient(args.sim_server_host, args.sim_server_port, args.sim_server_timeout_ms)
        try:
            all_results.extend(
                {
                    "task_id": int(item["task_id"]),
                    "task_name": item["task"],
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
            for spec in run_specs:
                for attempt_idx in range(max(1, int(args.retries))):
                    payload = _payload_from_args(args, spec["task_id"], attempt_idx)
                    payload["task_variant"] = str(spec["task_variant"])
                    print(
                        f"[{args.model_type}_planning] submitting run_slot_planning "
                        f"task_id={spec['task_id']} variant={spec['task_variant']} "
                        f"attempt={attempt_idx + 1}/{args.retries} "
                        f"model_port={args.port}",
                        flush=True,
                    )
                    start_response = client.call_endpoint("run_slot_planning", data=payload)
                    task_run_id = start_response.get("task_run_id")
                    while True:
                        time.sleep(max(0.1, float(args.sim_server_poll_interval_s)))
                        status = client.call_endpoint("status")
                        if status.get("task_run_id") != task_run_id:
                            raise RuntimeError(
                                "Sim server switched to another task: "
                                f"expected={task_run_id}, got={status.get('task_run_id')}"
                            )
                        state = status.get("status")
                        print(
                            f"[{args.model_type}_planning] task_run_id={task_run_id} "
                            f"status={state} elapsed_s={float(status.get('duration_s') or 0.0):.1f}",
                            flush=True,
                        )
                        if state == "completed":
                            final_response = status.get("result", {}) or {}
                            break
                        if state in {"failed", "cancelled"}:
                            final_response = {"success": False, "status": state, "error": status.get("error")}
                            break
                    all_results.append(
                        {
                            "task_id": int(spec["task_id"]),
                            "task_name": spec["task_name"],
                            "task_variant": spec["task_variant"],
                            "attempt": int(attempt_idx),
                            "result": final_response,
                        }
                    )
                    if isinstance(final_response, dict) and final_response.get("success"):
                        break
        finally:
            client.close()
        payload = {"model_type": args.model_type, "results": all_results, "last_result": final_response}
        _write_result_json(args.result_json, payload)
        return payload

    try:
        all_results.extend(
            {
                "task_id": int(item["task_id"]),
                "task_name": item["task"],
                "task_variant": item["task_variant"],
                "attempt": None,
                "success": False,
                "skipped": True,
                "output_dir": item["output_base_dir"],
                "latest_attempts": item["latest_attempts"],
                "latest_attempt_dirs": item["latest_attempt_dirs"],
                "completion_reasons": item["completion_reasons"],
                "skip_reason": item["skip_reason"],
                "result": None,
            }
            for item in skipped_specs
        )
        for spec in run_specs:
            for attempt_idx in range(max(1, int(args.retries))):
                kwargs = dict(
                    port=args.port,
                    task_id=int(spec["task_id"]),
                    perturbation_id=args.perturbation_id,
                    state_frame=args.state_frame,
                    goal_mode=args.goal_mode,
                    scene_profile=args.scene_profile,
                    task_variant=spec["task_variant"],
                    max_steps=args.max_steps,
                    action_repeat=args.action_repeat,
                    replay_mode=args.replay_mode,
                    max_settle_steps=args.max_settle_steps,
                    settle_pos_tol=args.settle_pos_tol,
                    settle_rot_tol=args.settle_rot_tol,
                    enable_settle_debug=args.enable_settle_debug,
                    model_type=args.model_type,
                    output_run_suffix=f"attempt{attempt_idx + 1:02d}",
                    fixed_camera_presets=args.fixed_camera_presets,
                    rotate_deterministic_after_grasp=not args.disable_rotate_deterministic_after_grasp,
                    rotate_deterministic_step_rad=args.rotate_deterministic_step_rad,
                    rotate_deterministic_axis=args.rotate_deterministic_axis,
                    rotate_deterministic_overshoot_rad=args.rotate_deterministic_overshoot_rad,
                )
                if args.success_case_dir is not None:
                    kwargs["success_case_dir"] = args.success_case_dir
                if args.controller_only_goal_delta:
                    result = shared.run_controller_goal_delta_diagnostic(
                        task_id=int(spec["task_id"]),
                        perturbation_id=args.perturbation_id,
                        state_frame=args.state_frame,
                        success_case_dir=args.success_case_dir or shared.DEFAULT_SUCCESS_CASE_DIR,
                        goal_mode=args.goal_mode,
                        scene_profile=args.scene_profile,
                        task_variant=spec["task_variant"],
                        max_settle_steps=args.max_settle_steps,
                        settle_pos_tol=args.settle_pos_tol,
                        settle_rot_tol=args.settle_rot_tol,
                        enable_settle_debug=args.enable_settle_debug,
                        fixed_camera_presets=args.fixed_camera_presets,
                    )
                else:
                    result = shared.run_vjepa_planning(**kwargs)
                print(
                    f"[{args.model_type}_planning] local task={spec['task_name']} "
                    f"variant={spec['task_variant']} "
                    f"attempt={attempt_idx + 1}/{args.retries} "
                    f"success={None if result is None else result.get('success')} "
                    f"output_dir={None if result is None else result.get('output_dir')}",
                    flush=True,
                )
                all_results.append(
                    {
                        "task_id": int(spec["task_id"]),
                        "task_name": spec["task_name"],
                        "task_variant": spec["task_variant"],
                        "attempt": int(attempt_idx),
                        "result": result,
                    }
                )
                final_response = result
                if isinstance(result, dict) and result.get("success"):
                    break
    finally:
        if hasattr(shared, "og"):
            shared.og.shutdown()

    payload = {"model_type": args.model_type, "results": all_results, "last_result": final_response}
    _write_result_json(args.result_json, payload)
    return payload
