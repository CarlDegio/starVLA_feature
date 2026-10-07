#!/usr/bin/env python3
"""Convert recorded dual-arm NPY/MP4 episodes to local LeRobot v2.1 datasets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[3]
TASKS = {
    "classification_the_blocks": "Collect the blocks on the table by color: place the gray blocks in the left basket and the pink blocks in the right basket.",
    "insert_the_two_tubes_into_the_rack_one_by_one": "Insert the test tube into the test tube rack.",
    "place_the_slippers_on_the_shoe_rack": "Place the slippers on the shoe rack.",
}
CAMERAS = ("top", "left", "right")
SLICES = {"left_joints": (0, 6), "right_joints": (6, 12),
          "left_gripper": (12, 13), "right_gripper": (13, 14)}
ACTION_FILES = ("action-left-joint", "action-left-gripper",
                "action-right-joint", "action-right-gripper")
STATE_FILES = ("left-joint_pos", "left-gripper_pos", "right-joint_pos", "right-gripper_pos")
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNK_SIZE = 1000


def add_trim_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-trim-leading-idle", action="store_true",
                        help="Keep the complete recording instead of trimming its idle prefix.")
    parser.add_argument("--idle-joint-threshold", type=float, default=0.005,
                        help="Joint displacement tolerance from the first action, in recorded units.")
    parser.add_argument("--idle-gripper-threshold", type=float, default=0.01,
                        help="Gripper displacement tolerance from the first action, in recorded units.")
    parser.add_argument("--motion-confirm-frames", type=int, default=3,
                        help="Consecutive frames exceeding a tolerance; retain the first of these frames.")
    parser.add_argument("--max-idle-seconds", type=float, default=1.0,
                        help="Exclude an episode with a middle all-dimension idle interval this long; ignore prefix/tail; 0 disables.")
    parser.add_argument("--end-idle-grace-seconds", type=float, default=3.0,
                        help="Pauses starting within the final seconds of the source recording do not exclude the episode.")
    parser.add_argument("--min-action-quantile-span", type=float, default=0.01,
                        help="Expand smaller action q01/q99 ranges symmetrically around their mean; 0 disables.")


def trim_settings(args) -> dict:
    thresholds = (args.idle_joint_threshold, args.idle_gripper_threshold)
    if not all(np.isfinite(v) and v > 0 for v in thresholds) or args.motion_confirm_frames < 1:
        raise ValueError("Idle tolerances must be finite and positive; motion confirmation must be >= 1.")
    if not all(np.isfinite(v) and v >= 0 for v in (args.max_idle_seconds, args.min_action_quantile_span,
                                                 args.end_idle_grace_seconds)):
        raise ValueError("Maximum idle duration and minimum quantile span must be finite and nonnegative.")
    return {"enabled": not args.no_trim_leading_idle, "method": "displacement_from_first_action_v1",
            "joint_threshold": args.idle_joint_threshold, "gripper_threshold": args.idle_gripper_threshold,
            "motion_confirm_frames": args.motion_confirm_frames}


def leading_motion_start(action: np.ndarray, settings: dict) -> int:
    """Find sustained motion in ANY of all 14 dimensions, without filtering later pauses.

    Compare with the first action, not just adjacent frames: slow cumulative motion
    must not be discarded. Confirmation rejects isolated noise; return its FIRST
    frame, not the last confirmation frame. The caller excludes all-idle episodes.
    """
    if action.ndim != 2 or action.shape[1] != 14 or len(action) == 0 or not np.isfinite(action).all():
        raise ValueError("Motion detection requires a nonempty, finite (frames, 14) action array.")
    if not settings["enabled"]:
        return 0
    tolerance = np.array([settings["joint_threshold"]] * 12 + [settings["gripper_threshold"]] * 2)
    moving = np.any(np.abs(action.astype(np.float64) - action[0]) > tolerance, axis=1)
    confirmation = settings["motion_confirm_frames"]
    if len(action) >= confirmation:
        starts = np.flatnonzero(np.convolve(moving.astype(np.int64), np.ones(confirmation, dtype=np.int64),
                                            mode="valid") == confirmation)
        if len(starts):
            return int(starts[0])
    raise ValueError("No sustained motion detected with the configured idle tolerances.")


def long_idle_intervals(action: np.ndarray, fps: float, settings: dict, seconds: float) -> list[dict]:
    """Locate windows where EVERY action dimension stays inside its noise tolerance.

    Use each window's full min/max range, not adjacent-frame differences, so slow
    accumulated motion is not mistaken for a stop. Intervals use source frame indices.
    """
    if seconds <= 0:
        return []
    window = int(np.ceil(seconds * fps)) + 1
    if len(action) < window:
        return []
    tolerance = np.array([settings["joint_threshold"]] * 12 + [settings["gripper_threshold"]] * 2)
    windows = np.lib.stride_tricks.sliding_window_view(action, window, axis=0)
    starts = np.flatnonzero(np.all(np.ptp(windows, axis=-1) <= tolerance, axis=1))
    groups = np.split(starts, np.flatnonzero(np.diff(starts) > 1) + 1)
    return [{"start_frame": int(g[0]), "end_frame_exclusive": int(g[-1] + window),
             "duration_seconds": float((g[-1] + window - 1 - g[0]) / fps)} for g in groups if len(g)]


def trailing_motion_end(action: np.ndarray, settings: dict) -> int:
    """Drop the stationary suffix, retaining its first frame as the final target.

    A suffix is stationary only if its full per-dimension range stays within the
    tolerances for at least motion_confirm_frames. Motion after a late pause remains.
    """
    tolerance = np.array([settings["joint_threshold"]] * 12 + [settings["gripper_threshold"]] * 2)
    low, high = action[-1].copy(), action[-1].copy()
    suffix_start = len(action) - 1
    for i in range(len(action) - 2, -1, -1):
        low, high = np.minimum(low, action[i]), np.maximum(high, action[i])
        if np.any(high - low > tolerance):
            break
        suffix_start = i
    if len(action) - suffix_start >= settings["motion_confirm_frames"]:
        return suffix_start + 1
    return len(action)


def floor_action_quantiles(raw: dict, minimum_span: float) -> tuple[dict, list[dict]]:
    """Return normalization bounds; preserve raw descriptive statistics separately."""
    if not np.isfinite(minimum_span) or minimum_span < 0:
        raise ValueError("Minimum action quantile span must be finite and nonnegative.")
    result = {**raw, "q01": list(raw["q01"]), "q99": list(raw["q99"])}
    adjustments = []
    for i, (lo, hi, mean) in enumerate(zip(raw["q01"], raw["q99"], raw["mean"])):
        if hi - lo < minimum_span:
            # Outward rounding avoids an interval microscopically narrower than the floor.
            lower = float(np.nextafter(mean - minimum_span / 2, -np.inf))
            upper = float(np.nextafter(mean + minimum_span / 2, np.inf))
            result["q01"][i], result["q99"][i] = lower, upper
            adjustments.append({"dimension": i, "raw_q01": lo, "raw_q99": hi,
                                "mean": mean, "q01": lower, "q99": upper})
    return result, adjustments


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def update_task_instruction(task: str, args) -> None:
    """Relabel a completed conversion without modifying recordings or frame data."""
    output = args.output_root / task
    meta = output / "meta"
    manifest_path = meta / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads((meta / "conversion_complete.json").read_text())
    info = json.loads((meta / "info.json").read_text())
    instruction = args.instructions.get(task, TASKS[task])
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Instruction must be a nonempty string: {task}")
    if manifest["task"] != task or info["total_tasks"] != 1:
        raise ValueError(f"Expected a single-task edl_real conversion: {output}")
    tasks = [json.loads(line) for line in (meta / "tasks.jsonl").read_text().splitlines()]
    episodes = [json.loads(line) for line in (meta / "episodes.jsonl").read_text().splitlines()]
    allowed = {manifest["instruction"], instruction}
    if len(tasks) != 1 or tasks[0]["task_index"] != 0 or tasks[0]["task"] not in allowed:
        raise ValueError(f"Unexpected task annotations: {output}")
    if len(episodes) != complete["episodes"] or len(episodes) != info["total_episodes"]:
        raise ValueError(f"Incomplete episode metadata: {output}")
    markers = []
    for i, episode in enumerate(episodes):
        marker = meta / "conversion_episodes" / f"episode_{i:06d}.json"
        value = json.loads(marker.read_text())
        if (episode["episode_index"] != i or value["episode"]["episode_index"] != i
                or episode["tasks"] not in ([text] for text in allowed)
                or value["episode"]["tasks"] not in ([text] for text in allowed)):
            raise ValueError(f"Unexpected episode annotations: {marker}")
        markers.append((marker, value))
    # Each write is atomic. Keep the old manifest until last so interrupted relabeling
    # can be resumed with the same requested instruction (old/new annotations accepted).
    for marker, value in markers:
        if value["episode"]["tasks"] != [instruction]:
            value["episode"]["tasks"] = [instruction]
            write_json(marker, value)
    for episode in episodes:
        episode["tasks"] = [instruction]
    write_jsonl(meta / "episodes.jsonl", episodes)
    write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": instruction}])
    manifest["instruction"] = instruction
    write_json(manifest_path, manifest)
    print(f"[{task}] Updated instruction for {len(episodes)} episodes: {instruction}", flush=True)


def statistics(values: np.ndarray) -> dict:
    # Avoid float32 accumulation error falsely reporting variance for constant dimensions.
    values = np.asarray(values, dtype=np.float64)
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "q01": np.quantile(values, .01, axis=0).tolist(),
            "q99": np.quantile(values, .99, axis=0).tolist(), "count": [len(values)]}


def video_info(path: Path) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = stream.frames or sum(1 for _ in container.decode(video=0))
        return {"frames": frames, "width": stream.width, "height": stream.height,
                "fps": float(stream.average_rate), "codec": stream.codec_context.name}


def load_vectors(source: Path, names: tuple[str, ...], frames: int) -> np.ndarray:
    vectors = []
    for name, dimension in zip(names, (6, 1, 6, 1)):
        value = np.load(source / f"{name}.npy", allow_pickle=False)
        if value.shape != (frames, dimension) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {source / name}: expected {(frames, dimension)}, got {value.shape}")
        vectors.append(value)
    # StarVLA exports normalization stats with non-gripper groups before grippers.
    return np.concatenate([vectors[0], vectors[2], vectors[1], vectors[3]], axis=1).astype(np.float32)


def plan_episode(source: Path, args) -> dict:
    """Exclude unusable recordings before numbering episodes or writing any frames.

    A constant subset of dimensions is valid (e.g. single-arm manipulation).
    Missing/malformed/misaligned data, no motion, or long middle pauses are excluded.
    Unexpected operational failures, including permission errors, still abort.
    """
    plan = {"source_episode": source.name, "source_frames": None, "source_start_frame": None,
            "status": "excluded", "reason_code": None, "reason": None}
    try:
        if not (source / "write_complete.flag").is_file():
            raise ValueError("Missing write_complete.flag: incomplete recording")
        metadata = json.loads((source / "metadata.json").read_text())
        n = int(metadata["num_frames"])
        plan["source_frames"] = n
        fps = float(metadata["control_hz"])
        if n < 1 or int(metadata["num_arm_joints"]) != 6 or fps != args.fps:
            raise ValueError("Unexpected frame count, joint count or control rate")
        action = load_vectors(source, ACTION_FILES, n)
        load_vectors(source, STATE_FILES, n)
        for camera in CAMERAS:
            times = np.load(source / f"{camera}-timestamp.npy", allow_pickle=False)
            if times.shape != (n,) or not np.isfinite(times).all():
                raise ValueError(f"Invalid {camera} camera timestamps")
            try:
                info = video_info(source / f"{camera}-images-rgb.mp4")
            except av.error.FFmpegError as error:
                if isinstance(error, PermissionError):
                    raise
                raise ValueError(f"Unreadable {camera} video: {error}") from error
            if info["frames"] != n or not np.isclose(info["fps"], fps):
                raise ValueError(f"Video/array alignment mismatch: {camera}, {info}, array frames={n}")
    except (ValueError, FileNotFoundError, EOFError, KeyError, TypeError) as error:
        plan.update(reason_code="invalid_recording", reason=str(error))
        return plan
    try:
        start = leading_motion_start(action, trim_settings(args))
    except ValueError as error:
        plan.update(reason_code="no_sustained_motion", reason=str(error))
        return plan
    plan["source_start_frame"] = start
    plan["source_end_frame"] = max(start + 1, trailing_motion_end(action, trim_settings(args)))
    idle = long_idle_intervals(action, fps, trim_settings(args), args.max_idle_seconds)
    grace_boundary = (n - 1) - args.end_idle_grace_seconds * fps
    middle_idle = [interval for interval in idle
                   if start < interval["start_frame"] < grace_boundary and interval["end_frame_exclusive"] < n]
    plan["ignored_end_idle_intervals"] = [i for i in idle if i["start_frame"] >= grace_boundary]
    if middle_idle:
        plan.update(reason_code="middle_idle", reason=f"Middle all-dimension idle interval >= {args.max_idle_seconds}s",
                    middle_idle_intervals=middle_idle)
        return plan
    plan.update(status="keep", source_start_frame=start)
    return plan


def source_signature(source: Path, args, episode_index: int, global_index: int, source_start: int, source_end: int) -> dict:
    return {"source": str(source.resolve()), "episode_index": episode_index,
            "global_index": global_index, "width": args.width, "height": args.height,
            "crf": args.crf, "preset": args.preset, "format_version": 7,
            "source_end_frame": source_end, "end_idle_grace_seconds": args.end_idle_grace_seconds,
            "min_action_quantile_span": args.min_action_quantile_span, "max_idle_seconds": args.max_idle_seconds,
            "leading_idle_trim": trim_settings(args), "source_start_frame": source_start,
            "files": {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                      for p in sorted(source.iterdir()) if p.is_file()}}


def convert_episode(source: Path, output: Path, episode_index: int, global_index: int,
                    source_start: int, source_end: int, args) -> dict:
    signature = source_signature(source, args, episode_index, global_index, source_start, source_end)
    marker = output / "meta/conversion_episodes" / f"episode_{episode_index:06d}.json"
    parquet = output / DATA_PATH.format(episode_chunk=episode_index // CHUNK_SIZE,
                                        episode_index=episode_index)
    videos = {cam: output / VIDEO_PATH.format(episode_chunk=episode_index // CHUNK_SIZE,
              video_key=f"observation.images.{cam}", episode_index=episode_index) for cam in CAMERAS}
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["signature"] != signature:
            raise ValueError(f"Conversion settings or source changed for {source}; use a new output directory.")
        files = [parquet, *videos.values()]
        if all(p.is_file() and p.stat().st_size == previous["output_sizes"][str(p.relative_to(output))]
               for p in files):
            return previous["episode"]

    if not (source / "write_complete.flag").is_file():
        raise ValueError(f"Recording is not complete: {source}")
    metadata = json.loads((source / "metadata.json").read_text())
    source_frames = int(metadata["num_frames"])
    frames = source_end - source_start
    fps = float(metadata["control_hz"])
    if frames < 1 or int(metadata["num_arm_joints"]) != 6 or fps != args.fps:
        raise ValueError(f"Unexpected frame count, joint count or control rate: {source}")
    action = load_vectors(source, ACTION_FILES, source_frames)[source_start:source_end]
    state = load_vectors(source, STATE_FILES, source_frames)[source_start:source_end]
    columns = {
        "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 14)),
        "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 14)),
        # MP4 PTS describe an evenly spaced frame timeline, not the epoch camera timestamps.
        "timestamp": pa.array(np.arange(frames, dtype=np.float32) / fps),
        "frame_index": pa.array(np.arange(frames, dtype=np.int64)),
        "source_frame_index": pa.array(np.arange(source_start, source_end, dtype=np.int64)),
        "episode_index": pa.array(np.full(frames, episode_index, dtype=np.int64)),
        "index": pa.array(np.arange(global_index, global_index + frames, dtype=np.int64)),
        "task_index": pa.array(np.zeros(frames, dtype=np.int64)),
    }
    timestamp_summary = {}
    for camera in CAMERAS:
        times = np.load(source / f"{camera}-timestamp.npy", allow_pickle=False)
        if times.shape != (source_frames,) or not np.isfinite(times).all():
            raise ValueError(f"Invalid camera timestamps: {source}/{camera}")
        times = times[source_start:source_end]
        # Keep acquisition timestamps verbatim for auditing; they are milliseconds since epoch.
        columns[f"observation.camera_timestamp_ms.{camera}"] = pa.array(times, type=pa.float64())
        timestamp_summary[camera] = {"first_ms": float(times[0]), "last_ms": float(times[-1]),
                                     "non_monotonic_steps": int(np.sum(np.diff(times) < 0))}
        input_video = source / f"{camera}-images-rgb.mp4"
        source_video = video_info(input_video)
        if source_video["frames"] != source_frames or not np.isclose(source_video["fps"], fps):
            raise ValueError(f"Video/array alignment mismatch: {input_video}: {source_video}, frames={source_frames}")
        target = videos[camera]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.stem + ".partial.mp4")
        command = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-i", str(input_video), "-map", "0:v:0", "-an",
                   "-vf", f"trim=start_frame={source_start}:end_frame={source_end},setpts=PTS-STARTPTS,scale={args.width}:{args.height}:flags=area",
                   "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
                   "-pix_fmt", "yuv420p", "-threads", str(args.ffmpeg_threads),
                   "-fps_mode", "passthrough", "-movflags", "+faststart", str(temporary)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        actual = video_info(temporary)
        if (actual["frames"] != frames or actual["width"] != args.width
                or actual["height"] != args.height or not np.isclose(actual["fps"], fps)):
            raise ValueError(f"Converted video failed validation: {temporary}: {actual}")
        temporary.replace(target)

    parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary = parquet.with_suffix(".partial.parquet")
    pq.write_table(pa.table(columns), temporary, compression="zstd")
    temporary.replace(parquet)
    raw_stats = {"action": statistics(action), "observation.state": statistics(state)}
    normalized_action_stats, _ = floor_action_quantiles(raw_stats["action"], args.min_action_quantile_span)
    episode = {"episode_index": episode_index, "length": frames, "tasks": [args.instruction],
               "source_start_frame": source_start, "source_num_frames": source_frames,
               "source_end_frame": source_end, "trimmed_trailing_frames": source_frames - source_end,
               "trimmed_leading_frames": source_start,
               "source_episode": source.name, "source_task_name": metadata.get("task_name"),
               "camera_timestamps": timestamp_summary,
               "raw_stats": raw_stats,
               "stats": {"action": normalized_action_stats, "observation.state": raw_stats["observation.state"]}}
    write_json(marker, {"signature": signature, "episode": episode,
                       "output_sizes": {str(p.relative_to(output)): p.stat().st_size
                                        for p in [parquet, *videos.values()]}})
    return episode


def convert_task(task: str, args) -> dict:
    source = args.source_root / task
    output = args.output_root / task
    sources = sorted(p for p in source.iterdir() if p.is_dir())
    if args.limit_episodes:
        sources = sources[:args.limit_episodes]
    if not sources:
        raise ValueError(f"No episodes found: {source}")
    candidate_names = [p.name for p in sources]
    plans = [plan_episode(p, args) for p in sources]
    excluded = [p for p in plans if p["status"] == "excluded"]
    for plan in excluded:
        print(f"[{task}] EXCLUDED {plan['source_episode']}: {plan['reason_code']}: {plan['reason']}", flush=True)
    valid_plans = [p for p in plans if p["status"] == "keep"]
    if not valid_plans:
        raise ValueError(f"No usable episodes in {source}; exclusions: {json.dumps(excluded)}")
    sources = [source / p["source_episode"] for p in valid_plans]
    options = {"task": task, "instruction": args.instructions.get(task, TASKS[task]),
               "format_version": 7, "leading_idle_trim": trim_settings(args),
               "trailing_idle_trim": "full_range_within_tolerance_keep_first_stationary_frame",
               "end_idle_grace_seconds": args.end_idle_grace_seconds,
               "max_idle_seconds": args.max_idle_seconds,
               "normalization": {"min_action_quantile_span": args.min_action_quantile_span,
                                 "small_span_policy": "mean_plus_minus_half_minimum_span"},
               "invalid_episode_policy": "exclude_incomplete_misaligned_no_motion_or_middle_idle_with_end_grace_v4",
               "excluded_episodes": excluded,
               "width": args.width, "height": args.height, "fps": args.fps,
               "source_episodes": candidate_names, "action_semantics": "recorded_absolute_joint_targets"}
    manifest = output / "meta/conversion_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != options:
        raise ValueError(f"Existing conversion differs from requested options: {output}; use a new output directory.")
    # Plan all cuts before writing output, so global row indices remain contiguous.
    source_counts = [p["source_frames"] for p in valid_plans]
    starts = [p["source_start_frame"] for p in valid_plans]
    ends = [p["source_end_frame"] for p in valid_plans]
    counts = [end - start for end, start in zip(ends, starts)]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "meta/info.json").exists() and not manifest.exists():
        raise ValueError(f"Refusing to overwrite an unrelated dataset: {output}")
    write_json(manifest, options)
    write_jsonl(output / "meta/excluded_episodes.jsonl", excluded)
    write_jsonl(output / "meta/conversion_plan.jsonl", plans)
    args.instruction = options["instruction"]
    offsets = np.cumsum([0, *counts[:-1]]).tolist()
    episodes = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(convert_episode, p, output, i, offsets[i], starts[i], ends[i], args): p
                for i, p in enumerate(sources)}
        for future in as_completed(jobs):
            try:
                episode = future.result()
            except Exception as error:
                if isinstance(error, subprocess.CalledProcessError):
                    print(error.stderr, flush=True)
                raise RuntimeError(f"Conversion failed for {jobs[future]}") from error
            episodes.append(episode)
            print(f"[{task}] {len(episodes)}/{len(sources)} {episode['source_episode']} "
                  f"({episode['length']} frames, trimmed {episode['trimmed_leading_frames']})", flush=True)
    episodes.sort(key=lambda e: e["episode_index"])

    meta = output / "meta"
    write_jsonl(meta / "episodes.jsonl", [{k: v for k, v in e.items() if k not in ("stats", "raw_stats")} for e in episodes])
    write_jsonl(meta / "episodes_stats.jsonl", [{"episode_index": e["episode_index"], "stats": e["stats"]} for e in episodes])
    write_jsonl(meta / "episodes_stats.raw.jsonl", [{"episode_index": e["episode_index"], "stats": e["raw_stats"]} for e in episodes])
    write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": args.instruction}])
    names = [*[f"left_joint_{i}" for i in range(6)],
             *[f"right_joint_{i}" for i in range(6)], "left_gripper", "right_gripper"]
    features = {"observation.state": {"dtype": "float32", "shape": [14], "names": names},
                "action": {"dtype": "float32", "shape": [14], "names": names}}
    features.update({k: {"dtype": "float32" if k == "timestamp" else "int64", "shape": [1], "names": None}
                     for k in ("timestamp", "frame_index", "source_frame_index", "episode_index", "index", "task_index")})
    for camera in CAMERAS:
        features[f"observation.camera_timestamp_ms.{camera}"] = {"dtype": "float64", "shape": [1], "names": None}
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": [args.height, args.width, 3], "names": ["height", "width", "channels"],
            "info": {"video.fps": args.fps, "video.height": args.height, "video.width": args.width,
                     "video.channels": 3, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "has_audio": False}}
    modality = {key: {name: {"start": a, "end": b, "original_key": column, "absolute": True,
                            "dtype": "float32"} for name, (a, b) in SLICES.items()}
                for key, column in (("state", "observation.state"), ("action", "action"))}
    modality["video"] = {camera: {"original_key": f"observation.images.{camera}"} for camera in CAMERAS}
    modality["annotation"] = {"human.action.task_description": {"original_key": "task_index"}}
    write_json(meta / "modality.json", modality)
    # Exact global quantiles, rather than averages of episode quantiles.
    tables = [pq.read_table(output / DATA_PATH.format(episode_chunk=i // CHUNK_SIZE, episode_index=i),
                           columns=["action", "observation.state"]) for i in range(len(episodes))]
    raw_stats = {column: statistics(np.asarray([row for table in tables
                 for row in table[column].to_pylist()], dtype=np.float32)) for column in ("action", "observation.state")}
    action_stats, changes = floor_action_quantiles(raw_stats["action"], args.min_action_quantile_span)
    write_json(meta / "stats.raw.json", raw_stats)
    effective_stats = {"action": action_stats, "observation.state": raw_stats["observation.state"]}
    write_json(meta / "stats.json", effective_stats)
    # StarVLA consumes this cache, not LeRobot's stats.json. Its vector statistics
    # omit scalar `count`, which cannot be sliced into the 14 action dimensions.
    write_json(meta / "stats_gr00t.json", {"__format_version": 2, "__cache_config": {"mode": "abs"},
               "statistics": {column: {key: value for key, value in stat.items() if key != "count"}
                              for column, stat in effective_stats.items()}})
    write_json(meta / "normalization_adjustments.json", {"minimum_action_span": args.min_action_quantile_span,
               "global_action_adjustments": changes, "episodes_with_adjustments": sum(
                   bool(floor_action_quantiles(e["raw_stats"]["action"], args.min_action_quantile_span)[1]) for e in episodes)})
    summary = {"task": task, "episodes": len(episodes), "frames": sum(counts),
               "source_episodes": len(candidate_names), "excluded_episodes": len(excluded),
               "excluded_known_frames": sum(e["source_frames"] or 0 for e in excluded),
               "source_frames": sum(source_counts), "trimmed_leading_frames": sum(starts),
               "trimmed_trailing_frames": sum(n - end for n, end in zip(source_counts, ends)),
               "videos": len(episodes) * len(CAMERAS), "video_size": [args.width, args.height],
               "action_dim": 14, "output": str(output.resolve())}
    write_json(meta / "info.json", {"codebase_version": "v2.1", "robot_type": "edl_real_dual_arm",
               "total_episodes": len(episodes), "total_frames": sum(counts), "total_tasks": 1,
               "total_videos": len(episodes) * len(CAMERAS), "total_chunks": (len(episodes) + CHUNK_SIZE - 1) // CHUNK_SIZE,
               "chunks_size": CHUNK_SIZE, "fps": args.fps, "splits": {"train": f"0:{len(episodes)}"},
               "data_path": DATA_PATH, "video_path": VIDEO_PATH, "features": features})
    write_json(meta / "conversion_complete.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT / "playground/Datasets")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "playground/Datasets/edl_real")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--instruction-json", type=Path, help="Optional mapping from task directory names to instructions.")
    parser.add_argument("--update-instructions-only", action="store_true",
                        help="Update completed dataset annotations and resume markers; preserve videos, parquet and stats.")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ffmpeg-threads", type=int, default=2)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--limit-episodes", type=int, default=0, help="For a small conversion smoke run; use a separate output root.")
    add_trim_arguments(parser)
    args = parser.parse_args()
    args.instructions = json.loads(args.instruction_json.read_text()) if args.instruction_json else {}
    if args.update_instructions_only:
        for task in args.tasks:
            update_task_instruction(task, args)
        return
    if not args.ffmpeg:
        parser.error("ffmpeg is required; activate an environment containing ffmpeg or pass --ffmpeg.")
    if min(args.width, args.height, args.workers, args.ffmpeg_threads) < 1 or args.width % 2 or args.height % 2:
        parser.error("Video dimensions must be positive and even; worker counts must be positive.")
    trim_settings(args)
    summaries = [convert_task(task, args) for task in args.tasks]
    write_json(args.output_root / "conversion_summary.json", summaries)


if __name__ == "__main__":
    main()
