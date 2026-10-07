#!/usr/bin/env python3
"""Validate converted real data against source recordings and optionally sample the trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.LIBERO.train_real.convert_edl_real import (
    ACTION_FILES, CAMERAS, PROJECT, TASKS, floor_action_quantiles, leading_motion_start,
    load_vectors, statistics, trailing_motion_end, video_info, write_json,
)


def validate_task(task: str, args) -> dict:
    root = args.output_root / task
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    manifest = json.loads((root / "meta/conversion_manifest.json").read_text())
    instruction = manifest["instruction"]
    assert [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines()] == [
        {"task_index": 0, "task": instruction}]
    assert info["total_episodes"] == len(episodes)
    assert info["codebase_version"] == "v2.1"
    if "excluded_episodes" in manifest:
        excluded = [json.loads(line) for line in (root / "meta/excluded_episodes.jsonl").read_text().splitlines()]
        assert excluded == manifest["excluded_episodes"]
        kept_names = {e["source_episode"] for e in episodes}
        excluded_names = {e["source_episode"] for e in excluded}
        assert kept_names.isdisjoint(excluded_names)
        assert kept_names | excluded_names == set(manifest["source_episodes"])
    action_names = info["features"]["action"]["names"]
    assert action_names == [*[f"left_joint_{i}" for i in range(6)],
                            *[f"right_joint_{i}" for i in range(6)], "left_gripper", "right_gripper"]
    total = 0
    actions = []
    for i, episode in enumerate(episodes):
        assert episode["episode_index"] == i
        assert episode["tasks"] == [instruction]
        marker = json.loads((root / "meta/conversion_episodes" / f"episode_{i:06d}.json").read_text())
        assert marker["episode"]["tasks"] == [instruction]
        source = args.source_root / task / episode["source_episode"]
        source_n = int(json.loads((source / "metadata.json").read_text())["num_frames"])
        source_start = episode.get("source_start_frame", 0)
        source_end = episode.get("source_end_frame", source_n)
        if "trailing_idle_trim" in manifest:
            assert source_end == max(source_start + 1, trailing_motion_end(
                load_vectors(source, ACTION_FILES, source_n), manifest["leading_idle_trim"]))
            assert episode["trimmed_trailing_frames"] == source_n - source_end
        if "leading_idle_trim" in manifest:
            assert leading_motion_start(load_vectors(source, ACTION_FILES, source_n),
                                        manifest["leading_idle_trim"]) == source_start
            assert episode["source_num_frames"] == source_n
            assert episode["trimmed_leading_frames"] == source_start
        path = root / info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)
        table = pq.read_table(path)
        n = episode["length"]
        assert n == source_end - source_start
        assert table.num_rows == n
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        actions.append(action)
        assert state.shape == action.shape == (n, 14)
        for column, values, sources in (
                ("action", action, (("action-left-joint.npy", 0, 6), ("action-right-joint.npy", 6, 12),
                                    ("action-left-gripper.npy", 12, 13), ("action-right-gripper.npy", 13, 14))),
                ("state", state, (("left-joint_pos.npy", 0, 6), ("right-joint_pos.npy", 6, 12),
                                  ("left-gripper_pos.npy", 12, 13), ("right-gripper_pos.npy", 13, 14)))):
            assert np.isfinite(values).all(), (task, i, column)
            for filename, start, end in sources:
                expected = np.load(source / filename, allow_pickle=False).astype(np.float32)[source_start:source_end]
                np.testing.assert_array_equal(values[:, start:end], expected)
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(n))
        if "source_frame_index" in table.column_names:
            np.testing.assert_array_equal(table["source_frame_index"].to_numpy(), np.arange(source_start, source_end))
        np.testing.assert_array_equal(table["index"].to_numpy(), np.arange(total, total + n))
        np.testing.assert_array_equal(table["episode_index"].to_numpy(), np.full(n, i))
        np.testing.assert_array_equal(table["task_index"].to_numpy(), np.zeros(n, dtype=np.int64))
        np.testing.assert_allclose(table["timestamp"].to_numpy(), np.arange(n) / info["fps"], rtol=1e-6, atol=1e-6)
        for camera in CAMERAS:
            np.testing.assert_array_equal(table[f"observation.camera_timestamp_ms.{camera}"].to_numpy(),
                                          np.load(source / f"{camera}-timestamp.npy", allow_pickle=False)[source_start:source_end])
            key = f"observation.images.{camera}"
            assert info["features"][key]["shape"] == [240, 320, 3]
            video = root / info["video_path"].format(episode_chunk=i // info["chunks_size"],
                                                     episode_index=i, video_key=key)
            actual = video_info(video)
            assert actual["frames"] == n, (video, actual)
            assert (actual["width"], actual["height"]) == (320, 240), (video, actual)
            assert np.isclose(actual["fps"], info["fps"]), (video, actual)
        total += n
    assert total == info["total_frames"]
    assert info["total_videos"] == len(episodes) * 3
    if "normalization" in manifest:
        raw_action_stats = statistics(np.concatenate(actions))
        bounded, _ = floor_action_quantiles(raw_action_stats, manifest["normalization"]["min_action_quantile_span"])
        saved = json.loads((root / "meta/stats.json").read_text())["action"]
        saved_raw = json.loads((root / "meta/stats.raw.json").read_text())["action"]
        cached = json.loads((root / "meta/stats_gr00t.json").read_text())
        assert cached["__format_version"] == 2 and cached["__cache_config"] == {"mode": "abs"}
        for key, value in cached["statistics"]["action"].items():
            np.testing.assert_allclose(value, bounded[key], rtol=1e-10, atol=1e-12)
        for key in raw_action_stats:
            np.testing.assert_allclose(saved_raw[key], raw_action_stats[key], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(saved[key], bounded[key], rtol=1e-10, atol=1e-12)
    assert not list(root.glob("**/*.partial.*"))
    report = {"task": task, "episodes": len(episodes), "frames": total, "videos": len(episodes) * 3,
              "source_values_match": True, "stored_video_size": [320, 240], "instruction": instruction}
    if args.sample_training:
        from examples.LIBERO.train_real.data_config import ACTION_HORIZON
        from omegaconf import OmegaConf
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset
        from transformers import AutoProcessor

        cfg = OmegaConf.load(PROJECT / f"examples/LIBERO/train_real/configs/{task}.yaml")
        cfg.datasets.vla_data.data_root_dir = str(args.output_root.resolve())
        mixture = get_vla_dataset(cfg.datasets.vla_data)
        assert len(mixture.datasets) == 1 and mixture.datasets[0].dataset_name == task
        single = mixture.datasets[0]
        processor = AutoProcessor.from_pretrained(str(PROJECT / "playground/Pretrained_models/fast"),
                                                  trust_remote_code=True, local_files_only=True)
        assert cfg.framework.action_model.action_horizon == ACTION_HORIZON
        processor.action_dim, processor.time_horizon = 14, ACTION_HORIZON
        # Check normal samples and action padding near the final episode boundary.
        sample_positions = [(0, 0), (len(episodes) // 2, episodes[len(episodes) // 2]["length"] // 2),
                            (len(episodes) - 1, episodes[-1]["length"] - 1)]
        token_lengths = []
        for ep, frame in sample_positions:
            raw = single.get_step_data(ep, frame)
            transformed = single.transforms(raw)
            sample = single._pack_sample(transformed)
            assert sample["action"].shape == (ACTION_HORIZON, 14)
            assert np.isfinite(sample["action"]).all()
            assert [image.size for image in sample["image"]] == [(224, 224)] * 3
            assert sample["lang"] == episodes[ep]["tasks"][0]
            if ep == len(episodes) - 1 and frame == episodes[-1]["length"] - 1:
                np.testing.assert_array_equal(sample["action"], np.repeat(sample["action"][:1], ACTION_HORIZON, axis=0))
            tokens = processor(sample["action"])
            decoded = processor.decode(tokens)
            assert decoded.shape == (1, ACTION_HORIZON, 14) and np.isfinite(decoded).all()
            token_lengths.append(len(tokens[0]))
        # Verify saved inference statistics use the same 14D order as the training labels.
        stat_path = root / "meta/validation_dataset_statistics.json"
        mixture.save_dataset_statistics(stat_path)
        export = json.loads(stat_path.read_text())[single.tag]["action"]
        source_stats = json.loads((root / "meta/stats.json").read_text())["action"]
        np.testing.assert_allclose(export["q01"], source_stats["q01"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(export["q99"], source_stats["q99"], rtol=1e-6, atol=1e-6)
        from starVLA.model.tools import FrameworkTools
        # Verify the opt-in continuous deployment path; keep the shared default unchanged.
        normalized = np.tile(np.linspace(-0.8, 0.8, 14), (ACTION_HORIZON, 1))
        before = normalized.copy()
        expected = .5 * (normalized + 1) * (np.array(export["q99"]) - np.array(export["q01"])) + export["q01"]
        expected = np.where(export.get("mask", np.ones(14, dtype=bool)), expected, normalized)
        np.testing.assert_allclose(FrameworkTools.unnormalize_actions(normalized, export, gripper_channel_idx=-1), expected)
        np.testing.assert_array_equal(normalized, before)
        report.update({"training_samples_checked": len(sample_positions), "training_image_size": [224, 224],
                       "action_chunk_shape": [ACTION_HORIZON, 14], "fast_token_lengths": token_lengths,
                       "inference_statistics_order_matches": True, "continuous_unnormalization_checked": True})
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT / "playground/Datasets")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "playground/Datasets/edl_real")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--sample-training", action="store_true", help="Requires the repository's data-loading and FAST dependencies.")
    args = parser.parse_args()
    reports = [validate_task(task, args) for task in args.tasks]
    write_json(args.output_root / "validation_report.json", reports)


if __name__ == "__main__":
    main()
