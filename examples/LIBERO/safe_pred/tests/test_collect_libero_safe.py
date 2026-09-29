from __future__ import annotations

import pathlib
import unittest

from examples.LIBERO.safe_pred.collect_libero_safe import Args, max_steps_for_suite, validate_args


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


class CollectLiberoSafeTest(unittest.TestCase):
    def test_defaults_collect_one_hundred_trajectories_per_ten_task_suite(self) -> None:
        args = Args(pretrained_path="/tmp/model.pt", dataset_output_path="/tmp/safe.hdf5")
        self.assertEqual(args.num_trials_per_task, 10)
        validate_args(args)

    def test_validation_requires_provenance_and_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "pretrained_path"):
            validate_args(Args(dataset_output_path="/tmp/safe.hdf5"))
        with self.assertRaisesRegex(ValueError, "dataset_output_path"):
            validate_args(Args(pretrained_path="/tmp/model.pt"))

    def test_suite_horizons_match_standard_libero_evaluation(self) -> None:
        self.assertEqual(max_steps_for_suite("libero_spatial"), 220)
        self.assertEqual(max_steps_for_suite("libero_10"), 520)
        with self.assertRaisesRegex(ValueError, "Unknown task suite"):
            max_steps_for_suite("invalid")

    def test_shell_requests_both_diagnostics_and_has_no_video_output(self) -> None:
        script = (REPO_ROOT / "examples/LIBERO/safe_pred/collect_libero_safe.sh").read_text()
        self.assertIn('NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"', script)
        self.assertIn("collect_libero_safe.py", script)
        self.assertNotIn("video_out_path", script)
        self.assertNotIn("mimwrite", script)

    def test_root_metric_collector_defines_disjoint_train_and_test_splits(self) -> None:
        script = (REPO_ROOT / "collect_safe_metric.sh").read_text()
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            self.assertIn(suite, script)
        self.assertIn('NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"', script)
        self.assertIn('DATA_SPLIT="${DATA_SPLIT:-train}"', script)
        self.assertIn('EPISODE_START_INDEX="${EPISODE_START_INDEX:-0}"', script)
        self.assertIn('EPISODE_START_INDEX="${EPISODE_START_INDEX:-10}"', script)
        self.assertIn('SEED="${SEED:-7}"', script)
        self.assertIn('SEED="${SEED:-17}"', script)
        self.assertIn("examples/LIBERO/safe_pred/collect_libero_safe.sh", script)


if __name__ == "__main__":
    unittest.main()
