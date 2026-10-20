"""Run resumable, case-addressed planning jobs across multiple GPUs."""

import argparse
import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import multiprocessing
import os
import queue
import re
import subprocess
import sys
import traceback
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf

SCRIPTS_ROOT = Path(__file__).resolve().parent
PLANAR_ROOT = SCRIPTS_ROOT.parent
for path in (SCRIPTS_ROOT, PLANAR_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Import registers the replace_slash resolver used by planning configs.
import custom_resolvers  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--devices", required=True, help="Comma-separated physical GPU ids")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Stable case-results directory. Existing completed case_XXXX/result.json "
            "files are reused. The default is stable for the composed config."
        ),
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable videos, slot/PCA plots, and other visualization artifacts.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Hydra overrides forwarded to every plan.py process",
    )
    return parser.parse_args()


def json_compatible(value):
    """Recursively convert common numeric/container types to JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    # Do not import torch in the launcher before workers select their GPUs.
    # A tensor can still be recognized safely once plan.py has imported torch.
    if type(value).__module__.startswith("torch") and hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as file:
        json.dump(json_compatible(payload), file, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def atomic_write_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as file:
        file.write(content)
    os.replace(temporary, path)


def requested_case_ids(cfg):
    total = int(cfg.n_evals)
    configured = cfg.get("eval_case_indices")
    if configured is None:
        return list(range(total))
    case_ids = [int(case_id) for case_id in configured]
    if total > len(case_ids):
        raise ValueError(
            f"n_evals={total}, but eval_case_indices contains only {len(case_ids)} cases"
        )
    case_ids = case_ids[:total]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("eval_case_indices must not contain duplicate case ids")
    return case_ids


def config_fingerprint(config_name, cfg):
    identity = OmegaConf.to_container(cfg, resolve=False)
    for key in (
        "n_evals",
        "eval_case_indices",
        "eval_case_offset",
        "hydra",
        "save_video",
        "visualization",
        "n_plot_samples",
        "slot_pca",
        "wall_trajectory_visualization",
    ):
        identity.pop(key, None)
    checkpoint = (
        PLANAR_ROOT
        / str(cfg.get("ckpt_base_path", "."))
        / "outputs"
        / str(cfg.get("model_name", ""))
        / "checkpoints"
        / f"model_{cfg.get('model_epoch', 'latest')}.pth"
    ).resolve()
    checkpoint_identity = None
    if checkpoint.exists():
        stat = checkpoint.stat()
        checkpoint_identity = {
            "path": str(checkpoint),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    payload = {
        "config_name": config_name,
        "config": identity,
        "checkpoint": checkpoint_identity,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _read_final_metrics(attempt_dir):
    """Return the last complete final-eval record from an append-only log."""
    logs_path = attempt_dir / "logs.json"
    if not logs_path.exists():
        return None
    final_metrics = None
    try:
        with logs_path.open() as file:
            for line in file:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict) and isinstance(
                    entry.get("final_eval/success_rate"), (int, float)
                ):
                    final_metrics = entry
    except OSError:
        return None
    return final_metrics


def _recover_completed_result(case_dir, case_id, fingerprint):
    """Recover runs that completed planning but failed while serializing metrics."""
    for attempt_dir in reversed(sorted(case_dir.glob("attempt_*"))):
        attempt_result_path = attempt_dir / "result.json"
        if attempt_result_path.exists():
            try:
                with attempt_result_path.open() as file:
                    attempt_result = json.load(file)
            except (OSError, json.JSONDecodeError):
                attempt_result = None
            if (
                isinstance(attempt_result, dict)
                and attempt_result.get("config_fingerprint") not in (None, fingerprint)
            ):
                continue
        metrics = _read_final_metrics(attempt_dir)
        if metrics is None:
            continue
        result = {
            "status": planning_outcome(metrics),
            "case_id": int(case_id),
            "config_fingerprint": fingerprint,
            "metrics": metrics,
            "artifacts_dir": attempt_dir.name,
            "recovered_from": "logs.json",
        }
        atomic_write_json(attempt_result_path, result)
        atomic_write_json(case_dir / "result.json", result)
        return result
    return None


def load_completed_result(case_dir, case_id, fingerprint):
    result_path = case_dir / "result.json"
    if not result_path.exists():
        return _recover_completed_result(case_dir, case_id, fingerprint)
    try:
        with result_path.open() as file:
            result = json.load(file)
    except (OSError, json.JSONDecodeError):
        return _recover_completed_result(case_dir, case_id, fingerprint)
    if (
        result.get("status") not in {"success", "failed", "complete"}
        or int(result.get("case_id", -1)) != int(case_id)
        or result.get("config_fingerprint") != fingerprint
        or not isinstance(result.get("metrics"), dict)
    ):
        return _recover_completed_result(case_dir, case_id, fingerprint)
    return result


def planning_outcome(metrics):
    """Classify a fully evaluated single case without conflating it with errors."""
    success_rate = metrics.get("final_eval/success_rate")
    if not isinstance(success_rate, (int, float)):
        raise ValueError("Single-case result is missing numeric final_eval/success_rate")
    return "success" if float(success_rate) >= 0.5 else "failed"


def terminal_status(result):
    status = result["status"]
    if status == "complete":
        # Backward compatibility with results written before outcome tagging.
        return planning_outcome(result["metrics"])
    return status


def next_attempt_dir(case_dir):
    attempt = 1
    while (case_dir / f"attempt_{attempt:03d}").exists():
        attempt += 1
    attempt_dir = case_dir / f"attempt_{attempt:03d}"
    attempt_dir.mkdir(parents=True)
    return attempt_dir


def aggregate_results(results):
    sums = {}
    counts = {}
    for result in results:
        for key, value in result["metrics"].items():
            if isinstance(value, (int, float)):
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


def _replace_hydra_runtime_cwd(value):
    if isinstance(value, dict):
        return {
            key: _replace_hydra_runtime_cwd(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_hydra_runtime_cwd(item) for item in value]
    if isinstance(value, str):
        return value.replace("${hydra:runtime.cwd}", str(PLANAR_ROOT))
    return value


def compose_runtime_config(config_name, forwarded):
    """Compose outside Hydra's launcher while preserving its original-cwd paths."""
    with initialize_config_dir(version_base=None, config_dir=str(PLANAR_ROOT / "conf")):
        cfg = compose(config_name=config_name, overrides=forwarded)
    raw = OmegaConf.to_container(cfg, resolve=False)
    raw.pop("hydra", None)
    raw = _replace_hydra_runtime_cwd(raw)
    resolved = OmegaConf.to_container(OmegaConf.create(raw), resolve=True)
    resolved["saved_folder"] = str(PLANAR_ROOT)
    resolved["wandb_logging"] = False
    return resolved


def save_shared_test_config(output_dir, config_name, forwarded):
    """Save the complete resolved config once for a case-addressed test run."""
    config = compose_runtime_config(config_name, forwarded)
    # Workers replace this with each case's attempt directory. It is dispatch
    # metadata rather than part of the shared experiment configuration.
    config.pop("saved_folder", None)
    content = (
        "# Complete shared test configuration. Per-case dispatch only overrides "
        "n_evals, eval_case_indices, eval_case_offset, saved_folder, and "
        "wandb_logging.\n"
        + OmegaConf.to_yaml(OmegaConf.create(config), resolve=True, sort_keys=False)
    )
    path = Path(output_dir) / "config.yaml"
    if path.exists():
        existing = path.read_text()
        if existing != content:
            raise RuntimeError(
                f"Existing shared test config differs from this launch: {path}"
            )
        return path
    atomic_write_text(path, content)
    return path


def persistent_gpu_worker(
    device,
    config_name,
    forwarded,
    output_dir,
    fingerprint,
    task_queue,
    result_queue,
):
    """Own one GPU, load its model once, then consume case ids until shutdown."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    os.environ["HYDRA_FULL_ERROR"] = "1"
    worker_log_path = Path(output_dir) / "_workers" / f"gpu_{device}.log"
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with worker_log_path.open("a") as worker_log, redirect_stdout(
            worker_log
        ), redirect_stderr(worker_log):
            from plan import build_planning_runtime, run_planning_with_runtime

            base_cfg = compose_runtime_config(config_name, forwarded)
            runtime = build_planning_runtime(base_cfg)
            print(f"GPU {device}: resident model ready", flush=True)
        result_queue.put({"type": "ready", "device": device})
    except Exception:
        result_queue.put(
            {
                "type": "fatal",
                "device": device,
                "error": traceback.format_exc(),
            }
        )
        return

    while True:
        case_id = task_queue.get()
        if case_id is None:
            break
        case_dir = Path(output_dir) / f"case_{case_id:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir = next_attempt_dir(case_dir)
        case_cfg = copy.deepcopy(base_cfg)
        case_cfg.update(
            {
                "n_evals": 1,
                "eval_case_indices": [int(case_id)],
                "eval_case_offset": 0,
                "saved_folder": str(attempt_dir),
                "wandb_logging": False,
            }
        )
        previous_cwd = Path.cwd()
        try:
            os.chdir(attempt_dir)
            with (attempt_dir / "plan.log").open("w") as case_log, redirect_stdout(
                case_log
            ), redirect_stderr(case_log):
                metrics = json_compatible(
                    run_planning_with_runtime(case_cfg, runtime)
                )
            outcome = planning_outcome(metrics)
            result = {
                "status": outcome,
                "case_id": int(case_id),
                "config_fingerprint": fingerprint,
                "metrics": metrics,
                "artifacts_dir": attempt_dir.name,
            }
            atomic_write_json(attempt_dir / "result.json", result)
            atomic_write_json(case_dir / "result.json", result)
            result_queue.put(
                {
                    "type": "complete",
                    "device": device,
                    "case_id": int(case_id),
                    "result": result,
                }
            )
        except Exception:
            error = traceback.format_exc()
            with (attempt_dir / "error.log").open("w") as error_log:
                error_log.write(error)
            atomic_write_json(
                attempt_dir / "result.json",
                {
                    "status": "error",
                    "case_id": int(case_id),
                    "config_fingerprint": fingerprint,
                    "error": error.splitlines()[-1],
                },
            )
            result_queue.put(
                {
                    "type": "failed",
                    "device": device,
                    "case_id": int(case_id),
                    "attempt_dir": str(attempt_dir),
                    "error": error,
                }
            )
        finally:
            os.chdir(previous_cwd)


def main():
    args = parse_args()
    repo = PLANAR_ROOT
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one GPU id")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicate GPU ids")

    forwarded = list(args.overrides)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if args.no_visualization:
        forwarded.append("++visualization=false")

    # Build a missing frame-slot cache/retrieval bank once with all selected
    # GPUs before resident case workers load the model. The builder is a no-op
    # when either a packaged or local bank already exists.
    cache_command = [
        "accelerate",
        "launch",
        "--num_processes",
        str(len(devices)),
    ]
    if len(devices) > 1:
        cache_command.append("--multi_gpu")
    cache_command.extend(
        [
            str(SCRIPTS_ROOT / "build_retrieval_cache.py"),
            "--config-name",
            args.config_name,
            *forwarded,
        ]
    )
    cache_environment = os.environ.copy()
    cache_environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    cache_environment.setdefault("WANDB_MODE", "disabled")
    print(
        f"Preparing retrieval cache on {len(devices)} GPU(s): "
        f"{','.join(devices)}",
        flush=True,
    )
    subprocess.run(
        cache_command,
        cwd=PLANAR_ROOT,
        env=cache_environment,
        check=True,
    )
    # This launcher is not itself managed by @hydra.main, so a raw composed
    # config has no HydraConfig available for ${hydra:runtime.cwd}. Materialize
    # those paths through the same helper used by resident workers before the
    # parent process inspects or fingerprints the config.
    cfg = OmegaConf.create(compose_runtime_config(args.config_name, forwarded))
    case_ids = requested_case_ids(cfg)
    fingerprint = config_fingerprint(args.config_name, cfg)

    if args.output_dir is None:
        model_name = str(cfg.get("model_name", "model") or "model")
        model_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
        output_dir = (
            repo
            / "plan_outputs"
            / f"{args.config_name}_{model_tag}_{fingerprint}_cases"
        )
    else:
        output_dir = args.output_dir.expanduser()
        if not output_dir.is_absolute():
            output_dir = (repo / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as file:
            previous_manifest = json.load(file)
        previous_fingerprint = previous_manifest.get("config_fingerprint")
        if previous_fingerprint != fingerprint:
            raise RuntimeError(
                f"Output directory {output_dir} belongs to config fingerprint "
                f"{previous_fingerprint}, not {fingerprint}. Choose another --output-dir."
            )

    config_path = save_shared_test_config(
        output_dir, args.config_name, forwarded
    )

    completed = {}
    for case_id in case_ids:
        case_dir = output_dir / f"case_{case_id:04d}"
        result = load_completed_result(case_dir, case_id, fingerprint)
        if result is not None:
            completed[case_id] = result
    pending = [case_id for case_id in case_ids if case_id not in completed]
    manifest = {
        "config_name": args.config_name,
        "config_fingerprint": fingerprint,
        "devices": devices,
        "requested_case_ids": case_ids,
        "completed_before_launch": sorted(completed),
        "pending_before_launch": pending,
        "overrides": forwarded,
    }
    atomic_write_json(manifest_path, manifest)
    print(
        f"Cases: requested={len(case_ids)} completed={len(completed)} "
        f"pending={len(pending)} output={output_dir} config={config_path}",
        flush=True,
    )

    failures = []
    if pending:
        context = multiprocessing.get_context("spawn")
        task_queue = context.Queue()
        result_queue = context.Queue()
        workers = [
            context.Process(
                target=persistent_gpu_worker,
                args=(
                    device,
                    args.config_name,
                    forwarded,
                    str(output_dir),
                    fingerprint,
                    task_queue,
                    result_queue,
                ),
                name=f"planning-gpu-{device}",
            )
            for device in devices[: len(pending)]
        ]
        for case_id in pending:
            task_queue.put(case_id)
        for _ in workers:
            task_queue.put(None)
        for worker in workers:
            worker.start()

        expected = len(pending)
        finished = 0
        fatal_errors = []
        try:
            while finished < expected:
                try:
                    message = result_queue.get(timeout=1.0)
                except queue.Empty:
                    if not any(worker.is_alive() for worker in workers):
                        break
                    continue
                message_type = message["type"]
                if message_type == "ready":
                    print(
                        f"GPU {message['device']}: resident model loaded once",
                        flush=True,
                    )
                elif message_type == "complete":
                    case_id = message["case_id"]
                    completed[case_id] = message["result"]
                    finished += 1
                    print(
                        f"GPU {message['device']}: evaluated case {case_id} "
                        f"as {message['result']['status']}",
                        flush=True,
                    )
                elif message_type == "failed":
                    finished += 1
                    failures.append(
                        (
                            message["case_id"],
                            message["device"],
                            message["error"].splitlines()[-1],
                            message["attempt_dir"],
                        )
                    )
                elif message_type == "fatal":
                    fatal_errors.append(message)
            for worker in workers:
                worker.join()
        except KeyboardInterrupt:
            print("Interrupted; terminating resident GPU workers...", flush=True)
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            for worker in workers:
                worker.join()
            raise SystemExit(130)

        incomplete = [
            case_id
            for case_id in pending
            if case_id not in completed
            and not any(failure[0] == case_id for failure in failures)
        ]
        fatal_reason = (
            fatal_errors[0]["error"].splitlines()[-1]
            if fatal_errors
            else "resident worker exited before reporting a result"
        )
        failures.extend(
            (case_id, "unknown", fatal_reason, output_dir / f"case_{case_id:04d}")
            for case_id in incomplete
        )

    ordered_results = [completed[case_id] for case_id in case_ids if case_id in completed]
    successful_case_ids = [
        result["case_id"]
        for result in ordered_results
        if terminal_status(result) == "success"
    ]
    failed_case_ids = [
        result["case_id"]
        for result in ordered_results
        if terminal_status(result) == "failed"
    ]
    aggregate = aggregate_results(ordered_results)
    summary = {
        "config_fingerprint": fingerprint,
        "requested_case_ids": case_ids,
        "attempted_case_ids": [result["case_id"] for result in ordered_results],
        "completed_case_ids": [result["case_id"] for result in ordered_results],
        "successful_case_ids": successful_case_ids,
        "failed_case_ids": failed_case_ids,
        "error_case_ids": [failure[0] for failure in failures],
        "aggregate": aggregate,
        "cases": ordered_results,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    with (output_dir / "logs.json").open("w") as file:
        file.write(json.dumps(aggregate) + "\n")

    if failures:
        details = ", ".join(
            f"case {case_id} on GPU {device}: {reason}, artifacts={attempt_dir}"
            for case_id, device, reason, attempt_dir in failures
        )
        raise RuntimeError(f"Planning failed for {len(failures)} case(s): {details}")
    print(json.dumps(aggregate, indent=2))
    print(f"Case-addressed planning outputs: {output_dir}")


if __name__ == "__main__":
    main()
