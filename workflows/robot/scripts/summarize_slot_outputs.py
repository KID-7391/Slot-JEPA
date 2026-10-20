#!/usr/bin/env python3
"""Summarize SLOT planning output completion and success rates.

This intentionally mirrors the skip/completion logic in runtime/slot_planning.py:
- latest complete successful attempt => done_success
- latest attempt01/02/03 all complete failed => done_failed
- otherwise => incomplete
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PUT_TASKS = {
    "put_green_block_into_bowl",
    "put_banana_into_box",
    "put_orange_into_bowl",
    "put_lemon_into_tray",
    "put_can_into_bowl",
    "put_marker_into_bowl",
}

PICK_TASKS = {
    "pick_spoon",
    "pick_bowl",
    "pick_lemon",
    "pick_can_of_soda",
    "pick_marker",
    "pick_banana",
}

ROTATE_TASKS = {
    "rotate_marker",
    "rotate_mug",
    "rotate_water_bottle",
    "rotate_can_of_soda",
    "rotate_banana",
    "rotate_toy_dice",
}

REACH_TASKS = {
    "reach_put_green_block_into_bowl",
    "reach_put_banana_into_box",
    "reach_rotate_marker",
    "reach_rotate_mug",
    "reach_pick_spoon",
    "reach_pick_bowl",
    "reach_put_orange_into_bowl",
    "reach_put_lemon_into_tray",
    "reach_put_can_into_bowl",
    "reach_put_marker_into_bowl",
    "reach_pick_lemon",
    "reach_pick_can_of_soda",
    "reach_pick_marker",
    "reach_pick_banana",
    "reach_rotate_water_bottle",
    "reach_rotate_can_of_soda",
    "reach_rotate_banana",
    "reach_rotate_toy_dice",
}

CATEGORY_PRINT_ORDER = ("put", "pick", "rotate", "reach", "other")

ABLATION_VARIANTS = (
    ("A0", "full", "Slot-JEPA (Full)", "slot"),
    (
        "A1",
        "a1_sequential_predictor",
        "Slot-JEPA w/ Sequential Predictor",
        "slot-a1-sequential-predictor",
    ),
    ("A2", "a2_no_uot", "Slot-JEPA w/o UOT", "slot-a2-no-uot"),
    (
        "A3",
        "a3_recent_context",
        "Slot-JEPA w/ Recent Context",
        "slot-a3-recent-context",
    ),
    (
        "A4",
        "a4_in_trajectory_random",
        "Slot-JEPA w/ Random Memory",
        "slot-a4-in-trajectory-random",
    ),
)
ABLATION_BY_NAMESPACE = {
    namespace: {
        "ablation_id": ablation_id,
        "ablation_variant": variant,
        "paper_name": paper_name,
        "output_namespace": namespace,
    }
    for ablation_id, variant, paper_name, namespace in ABLATION_VARIANTS
}

REQUIRED_FINAL_ARTIFACTS = (
    "trajectory_pca.png",
    "view_losses_over_steps.png",
    "goal_manifest.json",
)


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def attempt_number_from_name(name: str) -> int | None:
    match = re.search(r"_attempt(\d+)$", str(name))
    return int(match.group(1)) if match else None


def attempt_dirs(output_base_dir: Path) -> list[tuple[str, int, Path]]:
    if not output_base_dir.exists():
        return []
    dirs = []
    for child in output_base_dir.iterdir():
        if not child.is_dir():
            continue
        attempt_num = attempt_number_from_name(child.name)
        if attempt_num is not None:
            dirs.append((child.name, attempt_num, child))
    return sorted(dirs, key=lambda item: item[0])


def attempt_dir_status(attempt_dir: Path) -> dict[str, Any]:
    summary = load_json_if_exists(attempt_dir / "step_trace_summary.json")
    if not isinstance(summary, dict):
        return {
            "complete": False,
            "success": False,
            "reason": "missing_step_trace_summary",
            "num_steps": None,
        }
    if bool(summary.get("cancelled", False)):
        return {
            "complete": False,
            "success": False,
            "reason": "cancelled",
            "num_steps": summary.get("num_steps"),
        }
    if int(summary.get("num_steps") or 0) <= 0:
        return {
            "complete": False,
            "success": False,
            "reason": "empty_step_trace",
            "num_steps": summary.get("num_steps"),
        }
    for required_name in REQUIRED_FINAL_ARTIFACTS:
        if not (attempt_dir / required_name).exists():
            return {
                "complete": False,
                "success": False,
                "reason": f"missing_{required_name}",
                "num_steps": summary.get("num_steps"),
            }

    result = load_json_if_exists(attempt_dir / "planning_result.json")
    if isinstance(result, dict):
        if bool(result.get("success", False)) or result.get("status") == "success":
            return {
                "complete": True,
                "success": True,
                "reason": "complete_success",
                "num_steps": summary.get("num_steps"),
            }
        if result.get("status") in {"cancelled", "failed"}:
            return {
                "complete": False,
                "success": False,
                "reason": f"attempt_status_{result.get('status')}",
                "num_steps": summary.get("num_steps"),
            }
        return {
            "complete": True,
            "success": False,
            "reason": "complete_failure",
            "num_steps": summary.get("num_steps"),
        }

    return {
        "complete": True,
        "success": False,
        "reason": "legacy_complete_failure",
        "num_steps": summary.get("num_steps"),
    }


def task_category(task: str) -> str:
    if task in REACH_TASKS or task.startswith("reach_"):
        return "reach"
    if task in PUT_TASKS or task.startswith("put_"):
        return "put"
    if task in PICK_TASKS or task.startswith("pick_"):
        return "pick"
    if task in ROTATE_TASKS or task.startswith("rotate_"):
        return "rotate"
    return "other"


def classify_case(output_base_dir: Path, required_attempts: int = 3) -> dict[str, Any]:
    attempts = attempt_dirs(output_base_dir)
    if not attempts:
        return {
            "status": "incomplete",
            "reason": "no_attempt_dirs",
            "attempts": [],
        }

    newest_name, _, newest_path = attempts[-1]
    newest_status = attempt_dir_status(newest_path)
    if newest_status["complete"] and newest_status["success"]:
        return {
            "status": "done_success",
            "reason": "latest_attempt_success",
            "attempts": [{"name": newest_name, "path": str(newest_path), **newest_status}],
        }

    latest = attempts[-required_attempts:] if len(attempts) >= required_attempts else []
    latest_nums = [int(item[1]) for item in latest]
    expected_nums = list(range(1, required_attempts + 1))
    if len(latest) == required_attempts and latest_nums == expected_nums:
        latest_statuses = []
        all_complete_failed = True
        for attempt_name, _, attempt_path in latest:
            status = attempt_dir_status(attempt_path)
            latest_statuses.append({"name": attempt_name, "path": str(attempt_path), **status})
            if (not status["complete"]) or status["success"]:
                all_complete_failed = False
        if all_complete_failed:
            return {
                "status": "done_failed",
                "reason": "latest_three_attempts_complete_failed",
                "attempts": latest_statuses,
            }

    shown_attempts = []
    for attempt_name, _, attempt_path in attempts[-required_attempts:]:
        status = attempt_dir_status(attempt_path)
        shown_attempts.append({"name": attempt_name, "path": str(attempt_path), **status})
    return {
        "status": "incomplete",
        "reason": newest_status["reason"],
        "attempts": shown_attempts,
    }


def scan_slot_outputs(
    root: Path,
    required_attempts: int = 3,
    ablation: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    if not root.exists():
        return rows
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for case_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            classification = classify_case(case_dir, required_attempts=required_attempts)
            rows.append(
                {
                    "ablation_id": None if ablation is None else ablation["ablation_id"],
                    "ablation_variant": None if ablation is None else ablation["ablation_variant"],
                    "paper_name": None if ablation is None else ablation["paper_name"],
                    "output_namespace": root.name,
                    "task": task_dir.name,
                    "category": task_category(task_dir.name),
                    "case": case_dir.name,
                    "output_base_dir": str(case_dir),
                    **classification,
                }
            )
    return rows


def summarize(
    rows: list[dict[str, Any]],
    variants: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    statuses = ("done_success", "done_failed", "incomplete")
    summary: dict[str, Any] = {
        "overall": {status: 0 for status in statuses},
        "by_category": {},
        "by_variant": {},
        "total_cases_with_dirs": len(rows),
    }
    for variant in variants or []:
        summary["by_variant"][variant["ablation_variant"]] = {
            **variant,
            "exists": bool(variant.get("exists", False)),
            "done_success": 0,
            "done_failed": 0,
            "incomplete": 0,
            "total": 0,
            "completed": 0,
            "success_rate_completed": None,
            "success_rate_all": None,
        }
    for row in rows:
        status = row["status"]
        category = row["category"]
        summary["overall"].setdefault(status, 0)
        summary["overall"][status] += 1
        bucket = summary["by_category"].setdefault(
            category,
            {
                "done_success": 0,
                "done_failed": 0,
                "incomplete": 0,
                "total": 0,
                "completed": 0,
                "success_rate_completed": None,
                "success_rate_all": None,
            },
        )
        bucket.setdefault(status, 0)
        bucket[status] += 1
        bucket["total"] += 1

        variant_name = row.get("ablation_variant")
        if variant_name is not None:
            variant_bucket = summary["by_variant"].setdefault(
                variant_name,
                {
                    "ablation_id": row.get("ablation_id"),
                    "ablation_variant": variant_name,
                    "paper_name": row.get("paper_name"),
                    "output_namespace": row.get("output_namespace"),
                    "exists": True,
                    "done_success": 0,
                    "done_failed": 0,
                    "incomplete": 0,
                    "total": 0,
                    "completed": 0,
                    "success_rate_completed": None,
                    "success_rate_all": None,
                },
            )
            variant_bucket[status] += 1
            variant_bucket["total"] += 1

    for bucket in summary["by_category"].values():
        completed = int(bucket.get("done_success", 0)) + int(bucket.get("done_failed", 0))
        total = int(bucket.get("total", 0))
        success = int(bucket.get("done_success", 0))
        bucket["completed"] = completed
        bucket["success_rate_completed"] = (success / completed) if completed else None
        bucket["success_rate_all"] = (success / total) if total else None
    for bucket in summary["by_variant"].values():
        completed = int(bucket["done_success"]) + int(bucket["done_failed"])
        total = int(bucket["total"])
        success = int(bucket["done_success"])
        bucket["completed"] = completed
        bucket["success_rate_completed"] = success / completed if completed else None
        bucket["success_rate_all"] = success / total if total else None
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "ablation_id",
        "ablation_variant",
        "paper_name",
        "output_namespace",
        "category",
        "task",
        "case",
        "status",
        "reason",
        "output_base_dir",
        "attempts",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "ablation_id": row.get("ablation_id"),
                    "ablation_variant": row.get("ablation_variant"),
                    "paper_name": row.get("paper_name"),
                    "output_namespace": row.get("output_namespace"),
                    "category": row.get("category"),
                    "task": row.get("task"),
                    "case": row.get("case"),
                    "status": row.get("status"),
                    "reason": row.get("reason"),
                    "output_base_dir": row.get("output_base_dir"),
                    "attempts": json.dumps(row.get("attempts", []), ensure_ascii=False),
                }
            )


def print_report(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print("SUMMARY")
    overall = summary["overall"]
    for key in ("done_success", "done_failed", "incomplete"):
        print(f"{key}: {overall.get(key, 0)}")
    print(f"total_cases_with_dirs: {summary['total_cases_with_dirs']}")
    print()

    if summary.get("by_variant"):
        print("SUCCESS RATES BY ABLATION")
        for bucket in summary["by_variant"].values():
            completed_rate = bucket["success_rate_completed"]
            all_rate = bucket["success_rate_all"]
            completed_rate_s = "n/a" if completed_rate is None else f"{completed_rate:.3f}"
            all_rate_s = "n/a" if all_rate is None else f"{all_rate:.3f}"
            print(
                f"{bucket['ablation_id']} {bucket['ablation_variant']}: "
                f"exists={bucket['exists']} success={bucket['done_success']} "
                f"failed={bucket['done_failed']} incomplete={bucket['incomplete']} "
                f"completed={bucket['completed']} total={bucket['total']} "
                f"success_rate_completed={completed_rate_s} success_rate_all={all_rate_s}"
            )
        print()

    print("SUCCESS RATES BY CATEGORY")
    for category in CATEGORY_PRINT_ORDER:
        bucket = summary["by_category"].get(category)
        if not bucket:
            continue
        completed_rate = bucket["success_rate_completed"]
        all_rate = bucket["success_rate_all"]
        completed_rate_s = "n/a" if completed_rate is None else f"{completed_rate:.3f}"
        all_rate_s = "n/a" if all_rate is None else f"{all_rate:.3f}"
        print(
            f"{category}: success={bucket['done_success']} failed={bucket['done_failed']} "
            f"incomplete={bucket['incomplete']} completed={bucket['completed']} total={bucket['total']} "
            f"success_rate_completed={completed_rate_s} success_rate_all={all_rate_s}"
        )
    print()

    for status, title in (
        ("done_success", "DONE_SUCCESS"),
        ("done_failed", "DONE_FAILED"),
        ("incomplete", "INCOMPLETE"),
    ):
        print(title)
        for row in rows:
            if row["status"] != status:
                continue
            attempts = "; ".join(
                f"{item['name']}:{item['reason']}:steps={item.get('num_steps')}"
                for item in row.get("attempts", [])
            )
            print(f"- {row['task']}/{row['case']} | {row['reason']} | {attempts}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Scan one SLOT output root. If omitted, scan and report every "
            "registered ablation namespace A0-A4."
        ),
    )
    parser.add_argument(
        "--required-attempts",
        type=int,
        default=3,
        help="Number of complete failed attempts required to classify a case as done_failed.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("slot_output_status_report.json"),
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("slot_output_status_report.csv"),
        help="Path to write CSV case report.",
    )
    args = parser.parse_args()

    required_attempts = max(1, int(args.required_attempts))
    if args.root is None:
        output_parent = Path("realm/outputs")
        variants = []
        rows = []
        for ablation in ABLATION_BY_NAMESPACE.values():
            root = output_parent / ablation["output_namespace"]
            variant = {**ablation, "root": str(root), "exists": root.exists()}
            variants.append(variant)
            rows.extend(
                scan_slot_outputs(
                    root,
                    required_attempts=required_attempts,
                    ablation=ablation,
                )
            )
    else:
        root = Path(args.root)
        ablation = ABLATION_BY_NAMESPACE.get(root.name)
        variants = [] if ablation is None else [
            {**ablation, "root": str(root), "exists": root.exists()}
        ]
        rows = scan_slot_outputs(
            root,
            required_attempts=required_attempts,
            ablation=ablation,
        )
    summary = summarize(rows, variants=variants)
    report = {"summary": summary, "variants": variants, "cases": rows}

    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(rows, args.csv_out)
    print_report(rows, summary)
    print(f"wrote {args.json_out}")
    print(f"wrote {args.csv_out}")


if __name__ == "__main__":
    main()
