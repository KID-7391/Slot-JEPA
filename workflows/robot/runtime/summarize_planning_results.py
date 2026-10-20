"""Summarize Slot-JEPA robot planning results."""

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_OUTPUTS_ROOT = Path("realm/outputs")
ABLATION_VARIANTS = (
    {
        "ablation_id": "A0",
        "ablation_variant": "full",
        "paper_name": "Slot-JEPA (Full)",
        "output_namespace": "slot",
    },
    {
        "ablation_id": "A1",
        "ablation_variant": "a1_sequential_predictor",
        "paper_name": "Slot-JEPA w/ Sequential Predictor",
        "output_namespace": "slot-a1-sequential-predictor",
    },
    {
        "ablation_id": "A2",
        "ablation_variant": "a2_no_uot",
        "paper_name": "Slot-JEPA w/o UOT",
        "output_namespace": "slot-a2-no-uot",
    },
    {
        "ablation_id": "A3",
        "ablation_variant": "a3_recent_context",
        "paper_name": "Slot-JEPA w/ Recent Context",
        "output_namespace": "slot-a3-recent-context",
    },
    {
        "ablation_id": "A4",
        "ablation_variant": "a4_in_trajectory_random",
        "paper_name": "Slot-JEPA w/ Random Memory",
        "output_namespace": "slot-a4-in-trajectory-random",
    },
)
ABLATION_BY_NAMESPACE = {
    row["output_namespace"]: row for row in ABLATION_VARIANTS
}
DEFAULT_MODELS = tuple(ABLATION_BY_NAMESPACE) + (
    "vjepa-droid",
    "vjepa-query-predictor",
)
DEFAULT_TASK_TYPES = ("reach", "put", "pick", "rotate")


def _attempt_number_from_name(name: str):
    lower = str(name).lower()
    marker = "attempt"
    idx = lower.rfind(marker)
    if idx < 0:
        return None
    digits = ""
    for ch in lower[idx + len(marker) :]:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


def _attempt_dirs(run_dir: Path):
    dirs = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        attempt_num = _attempt_number_from_name(child.name)
        if attempt_num is not None:
            dirs.append((child.name, attempt_num, child))
    return sorted(dirs, key=lambda item: item[0])


def _load_json_if_exists(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fin:
            return json.load(fin)
    except Exception:
        return None


def _attempt_dir_status(attempt_dir: Path):
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


def _summarize_run_dir(run_dir: Path):
    attempt_infos = []
    for _, attempt_num, attempt_dir in _attempt_dirs(run_dir):
        status = _attempt_dir_status(attempt_dir)
        attempt_infos.append(
            {
                "attempt_num": int(attempt_num),
                "attempt_dir": str(attempt_dir),
                "complete": bool(status["complete"]),
                "success": bool(status["success"]),
                "reason": str(status["reason"]),
            }
        )

    if not attempt_infos:
        return {
            "task_name": run_dir.parent.name,
            "run_name": run_dir.name,
            "num_attempts": 0,
            "success": False,
            "complete": False,
            "latest_reason": "no_attempt_dirs",
            "attempts": [],
        }

    latest = attempt_infos[-1]
    any_success = any(item["success"] for item in attempt_infos)
    any_complete = any(item["complete"] for item in attempt_infos)
    return {
        "task_name": run_dir.parent.name,
        "run_name": run_dir.name,
        "num_attempts": len(attempt_infos),
        "success": bool(any_success),
        "complete": bool(any_complete),
        "latest_reason": str(latest["reason"]),
        "latest_complete": bool(latest["complete"]),
        "latest_success": bool(latest["success"]),
        "attempts": attempt_infos,
    }


def _collect_model_runs(model_root: Path):
    runs = []
    if not model_root.exists():
        return runs
    for task_dir in sorted(child for child in model_root.iterdir() if child.is_dir()):
        for run_dir in sorted(child for child in task_dir.iterdir() if child.is_dir()):
            runs.append(_summarize_run_dir(run_dir))
    return runs


def _success_rate(num: int, den: int):
    return 0.0 if den <= 0 else float(num) / float(den)


def _task_type_from_name(task_name: str):
    task_name = str(task_name).strip().lower()
    if task_name.startswith("reach_"):
        return "reach"
    if task_name.startswith("put_"):
        return "put"
    if task_name.startswith("pick_"):
        return "pick"
    if task_name.startswith("rotate_"):
        return "rotate"
    return "other"


def _build_model_summary(model_name: str, model_root: Path):
    runs = _collect_model_runs(model_root)
    task_type_counter = Counter()
    task_type_success_counter = Counter()
    reason_counter = Counter()
    for run in runs:
        task_type = _task_type_from_name(run["task_name"])
        run["task_type"] = task_type
        task_type_counter[task_type] += 1
        if run["success"]:
            task_type_success_counter[task_type] += 1
        reason_counter[run["latest_reason"]] += 1

    total_runs = len(runs)
    successful_runs = sum(1 for run in runs if run["success"])
    complete_runs = sum(1 for run in runs if run["complete"])
    incomplete_runs = total_runs - complete_runs
    total_attempts = sum(int(run["num_attempts"]) for run in runs)

    per_task_type = []
    ordered_task_types = [task_type for task_type in DEFAULT_TASK_TYPES if task_type in task_type_counter]
    for task_type in sorted(task_type_counter.keys()):
        if task_type not in ordered_task_types:
            ordered_task_types.append(task_type)
    for task_type in ordered_task_types:
        total = int(task_type_counter[task_type])
        success = int(task_type_success_counter[task_type])
        per_task_type.append(
            {
                "task_type": task_type,
                "num_runs": total,
                "num_success": success,
                "success_rate": _success_rate(success, total),
            }
        )

    ablation = ABLATION_BY_NAMESPACE.get(model_name)
    return {
        "model_name": model_name,
        "experiment_group": "slot_ablation" if ablation is not None else "baseline",
        "ablation_id": None if ablation is None else ablation["ablation_id"],
        "ablation_variant": None if ablation is None else ablation["ablation_variant"],
        "paper_name": model_name if ablation is None else ablation["paper_name"],
        "model_root": str(model_root),
        "exists": bool(model_root.exists()),
        "num_runs": total_runs,
        "num_attempts": total_attempts,
        "num_complete_runs": complete_runs,
        "num_incomplete_runs": incomplete_runs,
        "num_successful_runs": successful_runs,
        "success_rate": _success_rate(successful_runs, total_runs),
        "latest_reason_counts": dict(sorted(reason_counter.items())),
        "per_task_type": per_task_type,
        "runs": runs,
    }


def _print_model_summary(summary, show_task_types: bool):
    model_name = summary["model_name"]
    ablation_id = summary.get("ablation_id")
    title = summary.get("paper_name") or model_name
    label = f"{ablation_id} — {title} [{model_name}]" if ablation_id else model_name
    print(f"\n=== {label} ===")
    if not summary["exists"]:
        print(f"missing: {summary['model_root']}")
        return
    print(
        "runs={runs} success={success} ({rate:.1%}) complete={complete} incomplete={incomplete} attempts={attempts}".format(
            runs=summary["num_runs"],
            success=summary["num_successful_runs"],
            rate=summary["success_rate"],
            complete=summary["num_complete_runs"],
            incomplete=summary["num_incomplete_runs"],
            attempts=summary["num_attempts"],
        )
    )
    if summary["latest_reason_counts"]:
        print("latest_reasons:")
        for reason, count in summary["latest_reason_counts"].items():
            print(f"  {reason}: {count}")
    if show_task_types and summary["per_task_type"]:
        print("per_task_type:")
        for row in summary["per_task_type"]:
            print(
                "  {task_type}: {success}/{total} ({rate:.1%})".format(
                    task_type=row["task_type"],
                    success=row["num_success"],
                    total=row["num_runs"],
                    rate=row["success_rate"],
                )
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=list(DEFAULT_MODELS),
        help=(
            "Model output subdirectories under outputs-root. By default this includes "
            "all Slot-JEPA ablations A0-A4 plus the V-JEPA baselines."
        ),
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional path to save full summary JSON")
    parser.add_argument("--show-task-types", default=True, help="Print per-task-type success stats")
    return parser.parse_args()


def main():
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    model_names = [str(name).strip() for name in args.models if str(name).strip()]

    summaries = []
    for model_name in model_names:
        summary = _build_model_summary(model_name, outputs_root / model_name)
        summaries.append(summary)
        _print_model_summary(summary, show_task_types=bool(args.show_task_types))

    payload = {
        "outputs_root": str(outputs_root),
        "models": summaries,
        "ablation_variants": [
            summary
            for summary in summaries
            if summary.get("experiment_group") == "slot_ablation"
        ],
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fout:
            json.dump(payload, fout, ensure_ascii=False, indent=2)
        print(f"\nsaved_json={args.json}")


if __name__ == "__main__":
    main()
