"""Boundary cases for action-based leading-idle trimming (no GPU required)."""

import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
import json

import numpy as np

from convert_edl_real import (
    ACTION_FILES, STATE_FILES, CAMERAS, floor_action_quantiles, leading_motion_start,
    long_idle_intervals, plan_episode, statistics, trailing_motion_end,
)


class LeadingIdleTests(unittest.TestCase):
    settings = {"enabled": True, "joint_threshold": 0.005,
                "gripper_threshold": 0.01, "motion_confirm_frames": 3}

    def test_any_one_of_fourteen_dimensions_can_start_motion(self):
        for dim in range(14):
            with self.subTest(dim=dim):
                action = np.zeros((12, 14))
                action[5:, dim] = 0.02
                self.assertEqual(leading_motion_start(action, self.settings), 5)

    def test_idle_jitter_and_an_isolated_spike_do_not_start_motion(self):
        action = np.zeros((20, 14))
        action[1::2, 12:] = 0.0021914
        action[3, 0] = 0.05
        action[11:, 4] = 0.02
        self.assertEqual(leading_motion_start(action, self.settings), 11)

    def test_slow_cumulative_motion_is_detected(self):
        action = np.zeros((20, 14))
        action[:, 0] = np.arange(20) * 0.001
        self.assertEqual(leading_motion_start(action, self.settings), 6)

    def test_later_pauses_do_not_change_the_start(self):
        action = np.zeros((20, 14))
        action[1:5, 6] = 0.02
        self.assertEqual(leading_motion_start(action, self.settings), 1)

    def test_all_idle_or_unconfirmed_tail_requires_review(self):
        for n in (1, 2, 10):
            with self.assertRaisesRegex(ValueError, "No sustained motion"):
                leading_motion_start(np.zeros((n, 14)), self.settings)
        action = np.zeros((10, 14))
        action[-2:, 0] = 0.1
        with self.assertRaises(ValueError):
            leading_motion_start(action, self.settings)

    def test_disabled_trimming_preserves_all_frames(self):
        self.assertEqual(leading_motion_start(np.zeros((10, 14)), {**self.settings, "enabled": False}), 0)

    def test_constant_float32_data_has_zero_std(self):
        self.assertEqual(statistics(np.full((10000, 1), .123456, dtype=np.float32))["std"], [0.0])

    def test_quantile_floor_is_mean_centered_and_preserves_other_statistics(self):
        values = np.stack([np.full(100, .25), np.linspace(-1, 1, 100), np.linspace(.1, .102, 100)], axis=1)
        raw = statistics(values)
        bounded, changes = floor_action_quantiles(raw, .01)
        self.assertEqual([r["dimension"] for r in changes], [0, 2])
        for dim in (0, 2):
            self.assertGreaterEqual(bounded["q99"][dim] - bounded["q01"][dim], .01)
            self.assertAlmostEqual((bounded["q01"][dim] + bounded["q99"][dim]) / 2, raw["mean"][dim])
        self.assertEqual(bounded["q01"][1], raw["q01"][1])
        self.assertEqual(bounded["std"], raw["std"])
        self.assertEqual(raw["q01"][0], .25)  # Input must remain raw.
        self.assertEqual(floor_action_quantiles(raw, 0)[0], raw)

    def test_long_idle_requires_all_dimensions_and_checks_accumulated_motion(self):
        action = np.zeros((400, 14))
        action[:, 13] = np.arange(400) * .001  # Tiny adjacent changes but sustained motion.
        self.assertEqual(long_idle_intervals(action, 30, self.settings, 5), [])
        action[100:280, 13] = .1
        intervals = long_idle_intervals(action, 30, self.settings, 5)
        self.assertTrue(any(i["start_frame"] <= 100 and i["end_frame_exclusive"] >= 280 for i in intervals))
        self.assertEqual(long_idle_intervals(action, 30, self.settings, 0), [])

    def test_plan_excludes_invalid_recordings_but_keeps_single_arm_motion(self):
        args = SimpleNamespace(fps=30, no_trim_leading_idle=False, idle_joint_threshold=.005,
                               idle_gripper_threshold=.01, motion_confirm_frames=3,
                               max_idle_seconds=5, min_action_quantile_span=.01, end_idle_grace_seconds=3)
        with TemporaryDirectory() as directory:
            source = Path(directory)
            self.assertEqual(plan_episode(source, args)["reason_code"], "invalid_recording")
            (source / "write_complete.flag").touch()
            (source / "metadata.json").write_text(json.dumps({"num_frames": 60, "control_hz": 30, "num_arm_joints": 6}))
            for names in (ACTION_FILES, STATE_FILES):
                for name, dim in zip(names, (6, 1, 6, 1)):
                    np.save(source / f"{name}.npy", np.zeros((60, dim)))
            for camera in CAMERAS:
                np.save(source / f"{camera}-timestamp.npy", np.arange(60))
            with patch("convert_edl_real.video_info", return_value={"frames": 60, "fps": 30}):
                self.assertEqual(plan_episode(source, args)["reason_code"], "no_sustained_motion")
                values = np.zeros((60, 6))
                values[10:, 0] = np.arange(50) * .01
                np.save(source / "action-left-joint.npy", values)
                plan = plan_episode(source, args)
                self.assertEqual(plan["status"], "keep")
                self.assertEqual(plan["source_start_frame"], 11)
            with patch("convert_edl_real.video_info", return_value={"frames": 59, "fps": 30}):
                self.assertEqual(plan_episode(source, args)["reason_code"], "invalid_recording")

    def test_prefix_and_tail_stops_are_kept_but_middle_one_second_is_excluded(self):
        args = SimpleNamespace(fps=30, no_trim_leading_idle=False, idle_joint_threshold=.005,
                               idle_gripper_threshold=.01, motion_confirm_frames=3,
                               max_idle_seconds=1, min_action_quantile_span=.01, end_idle_grace_seconds=3)
        with TemporaryDirectory() as directory:
            source = Path(directory)
            n = 210
            (source / "write_complete.flag").touch()
            (source / "metadata.json").write_text(json.dumps({"num_frames": n, "control_hz": 30, "num_arm_joints": 6}))
            for names in (ACTION_FILES, STATE_FILES):
                for name, dim in zip(names, (6, 1, 6, 1)):
                    np.save(source / f"{name}.npy", np.zeros((n, dim)))
            for camera in CAMERAS:
                np.save(source / f"{camera}-timestamp.npy", np.arange(n))
            action = np.zeros((n, 6))
            action[61:150, 0] = np.arange(1, 90) * .01
            action[150:, 0] = .89
            np.save(source / "action-left-joint.npy", action)
            with patch("convert_edl_real.video_info", return_value={"frames": n, "fps": 30}):
                plan = plan_episode(source, args)
                self.assertEqual(plan["status"], "keep")
                self.assertEqual(plan["source_start_frame"], 61)
                self.assertEqual(plan["source_end_frame"], 150)
                late_action = action.copy()
                late_action[140:180, 0] = late_action[140, 0]
                np.save(source / "action-left-joint.npy", late_action)
                self.assertEqual(plan_episode(source, args)["status"], "keep")
                action[85:126, 0] = action[85, 0]
                np.save(source / "action-left-joint.npy", action)
                self.assertEqual(plan_episode(source, args)["reason_code"], "middle_idle")

    def test_trailing_trim_preserves_target_and_late_resumed_motion(self):
        action = np.zeros((100, 14))
        action[:, 0] = np.arange(100) * .01
        self.assertEqual(trailing_motion_end(action, self.settings), 100)
        action[50:, 0] = .5
        self.assertEqual(trailing_motion_end(action, self.settings), 51)
        action[-4:, 0] = [.51, .52, .53, .54]
        self.assertEqual(trailing_motion_end(action, self.settings), 100)


if __name__ == "__main__":
    unittest.main()
