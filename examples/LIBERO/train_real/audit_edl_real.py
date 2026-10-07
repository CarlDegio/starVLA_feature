#!/usr/bin/env python3
"""Preview leading-idle cuts and audit normalization without changing any dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from convert_edl_real import (
    ACTION_FILES, STATE_FILES, PROJECT, TASKS, add_trim_arguments,
    floor_action_quantiles, leading_motion_start, load_vectors, plan_episode, statistics, trim_settings, write_json,
)

NAMES = [*[f"left_joint_{i}" for i in range(6)], *[f"right_joint_{i}" for i in range(6)],
         "left_gripper", "right_gripper"]


def dimension_rows(task, scope, column, values, args, episode=""):
    stats = statistics(values)
    bounded = floor_action_quantiles(stats, args.min_action_quantile_span)[0] if column == "action" else stats
    rows = []
    for i, name in enumerate(NAMES):
        span = stats["q99"][i] - stats["q01"][i]
        reasons = []
        if span <= args.small_quantile_span:
            reasons.append("small_q99_minus_q01")
        if stats["std"][i] <= args.small_std:
            reasons.append("small_std")
        rows.append({"task": task, "scope": scope, "episode": episode, "column": column,
                     "dimension": i, "name": name, "frames": len(values),
                     **{k: stats[k][i] for k in ("q01", "q99", "std", "min", "max", "mean")},
                     "q99_minus_q01": span, "flags": ";".join(reasons),
                     "normalization_q01": bounded["q01"][i], "normalization_q99": bounded["q99"][i],
                     "range_expanded": column == "action" and span < args.min_action_quantile_span})
    return rows


def write_csv(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        raise ValueError("CSV requires a header even when there are no findings.")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT / "playground/Datasets")
    parser.add_argument("--dataset-root", type=Path, default=PROJECT / "playground/Datasets/edl_real")
    parser.add_argument("--report-dir", type=Path, default=PROJECT / "playground/Datasets/edl_real_audit")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--small-quantile-span", type=float, default=0.01)
    parser.add_argument("--small-std", type=float, default=0.001)
    parser.add_argument("--fps", type=float, default=30.0)
    add_trim_arguments(parser)
    args = parser.parse_args()
    settings = trim_settings(args)
    if not all(np.isfinite(x) and x > 0 for x in (args.small_quantile_span, args.small_std)):
        parser.error("Audit thresholds must be finite and positive.")
    report = {"leading_idle_trim": settings,
              "max_idle_seconds": args.max_idle_seconds,
              "idle_filter_scope": "middle_only; trim_prefix_and_stationary_suffix; end_grace_by_pause_start",
              "end_idle_grace_seconds": args.end_idle_grace_seconds,
              "min_action_quantile_span": args.min_action_quantile_span,
              "audit_thresholds": {"q99_minus_q01_lte": args.small_quantile_span, "std_lte": args.small_std},
              "units": "Raw recorded units; audit only. Proposed action bounds use the minimum span, never drop low-variance dimensions.",
              "tasks": []}
    global_rows, episode_flags, previews = [], [], []
    for task in args.tasks:
        before = {"action": [], "observation.state": []}
        after = {"action": [], "observation.state": []}
        filtered = {"action": [], "observation.state": []}
        task_cuts, kept_cuts, tail_cuts, unresolved, excluded = [], [], [], [], []
        sources = sorted(p for p in (args.source_root / task).iterdir() if p.is_dir())
        for source in sources:
            plan = plan_episode(source, args)
            if plan["status"] == "excluded":
                excluded.append(plan)
            if plan["reason_code"] == "invalid_recording":
                unresolved.append(plan)
                previews.append({"task": task, "episode": source.name, "source_frames": plan["source_frames"],
                                 "source_start_frame": None, "kept_frames": 0, "frames_after_prefix": None, "removed_seconds": None,
                                 "needs_review": True, "excluded": True, "exclusion_reason": plan["reason_code"]})
                continue
            metadata = json.loads((source / "metadata.json").read_text())
            n = int(metadata["num_frames"])
            action = load_vectors(source, ACTION_FILES, n)
            state = load_vectors(source, STATE_FILES, n)
            try:
                start = leading_motion_start(action, settings)
            except ValueError as error:
                unresolved.append({"episode": source.name, "error": str(error)})
                start = None
            previews.append({"task": task, "episode": source.name, "source_frames": n,
                             "source_start_frame": start, "kept_frames": plan["source_end_frame"] - start if plan["status"] == "keep" else 0,
                             "frames_after_prefix": n - start if start is not None else None,
                             "removed_seconds": start / metadata["control_hz"] if start is not None else None,
                             "needs_review": start is None or start > n * 0.4,
                             "excluded": plan["status"] == "excluded", "exclusion_reason": plan["reason_code"]})
            if start is not None:
                task_cuts.append(start)
                if plan["status"] == "keep":
                    kept_cuts.append(start)
                    tail_cuts.append(n - plan["source_end_frame"])
            for column, values in (("action", action), ("observation.state", state)):
                before[column].append(values)
                if start is not None:
                    after[column].append(values[start:])
                    if plan["status"] == "keep":
                        filtered[column].append(values[start:plan["source_end_frame"]])
                for scope, subset in (("before_trim", values), ("after_trim", values[start:] if start is not None else None),
                                      ("after_filter", values[start:plan["source_end_frame"]] if plan["status"] == "keep" else None)):
                    if subset is not None:
                        episode_flags.extend(row for row in dimension_rows(task, scope, column, subset, args, source.name)
                                             if row["flags"])
        task_rows = []
        for scope, data in (("before_trim", before), ("after_trim", after), ("after_filter", filtered)):
            # An unresolved trajectory makes a whole-task post-trim result incomplete.
            if scope == "after_trim" and unresolved:
                continue
            for column, pieces in data.items():
                if pieces:
                    task_rows.extend(dimension_rows(task, scope, column, np.concatenate(pieces), args))
        global_rows.extend(task_rows)
        summary = {"task": task, "episodes": len(sources), "source_frames": sum(map(len, before["action"])),
                   "removed_frames": sum(task_cuts), "remaining_frames": sum(map(len, after["action"])),
                   "excluded_episodes": excluded, "kept_episodes": len(sources) - len(excluded),
                   "excluded_known_frames": sum(p["source_frames"] or 0 for p in excluded),
                   "trimmed_frames_in_kept_episodes": sum(kept_cuts),
                   "trimmed_trailing_frames_in_kept_episodes": sum(tail_cuts),
                   "final_kept_frames": sum(map(len, filtered["action"])),
                   "normalization_floor_filtered_frames": 0,
                   "cut_frame_quantiles": dict(zip(("min", "median", "p90", "max"),
                                                    np.quantile(task_cuts, [0, .5, .9, 1]).tolist())) if task_cuts else {},
                   "unresolved_episodes": unresolved, "global_flagged_dimensions": [r for r in task_rows if r["flags"]],
                   "minima": []}
        saved_path = args.dataset_root / task / "meta/stats.json"
        if saved_path.is_file():
            saved = json.loads(saved_path.read_text())
            summary["existing_saved_stats"] = {key: saved[key] for key in before}
            # Existing metadata is reported separately; it may describe an older conversion.
            summary["existing_saved_stats_path"] = str(saved_path)
        for scope in ("before_trim", "after_trim", "after_filter"):
            for column in before:
                rows = [r for r in task_rows if r["scope"] == scope and r["column"] == column]
                if rows:
                    low_span = min(rows, key=lambda r: r["q99_minus_q01"])
                    low_std = min(rows, key=lambda r: r["std"])
                    summary["minima"].append({"scope": scope, "column": column,
                                              "span_name": low_span["name"], "min_span": low_span["q99_minus_q01"],
                                              "std_name": low_std["name"], "min_std": low_std["std"]})
        summary["episode_flag_counts"] = {scope: {
            column: {"episodes": len({r["episode"] for r in episode_flags if r["task"] == task and r["scope"] == scope and r["column"] == column}),
                     "dimension_findings": sum(r["task"] == task and r["scope"] == scope and r["column"] == column for r in episode_flags)}
            for column in before} for scope in ("before_trim", "after_trim", "after_filter")}
        summary["quantile_floor_impact"] = {scope: {
            "global_action_dimensions": sum(r["scope"] == scope and r["column"] == "action" and r["range_expanded"] for r in task_rows),
            "episode_action_dimensions": sum(r["task"] == task and r["scope"] == scope and r["column"] == "action" and r["range_expanded"] for r in episode_flags),
            "affected_episodes": len({r["episode"] for r in episode_flags if r["task"] == task and r["scope"] == scope and r["column"] == "action" and r["range_expanded"]})}
            for scope in ("before_trim", "after_trim", "after_filter")}
        report["tasks"].append(summary)
        print(json.dumps({k: v for k, v in summary.items() if not k.startswith("existing_saved")}), flush=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.report_dir / "normalization_report.json", report)
    write_csv(args.report_dir / "global_dimensions.csv", global_rows)
    write_csv(args.report_dir / "episode_flags.csv", episode_flags, list(global_rows[0]))
    write_csv(args.report_dir / "trimming_preview.csv", previews)
    print(f"Audit written to {args.report_dir}; datasets were not modified.", flush=True)


if __name__ == "__main__":
    main()
