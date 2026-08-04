from __future__ import annotations

import unittest

import numpy as np

from examples.LIBERO.edl_pred.rejection import (
    CalibrationRecords,
    PolicyThresholds,
    apply_rejection,
    calibrate_policy,
    predictive_entropy,
)


class RejectionPolicyTest(unittest.TestCase):
    def test_predictive_entropy_is_normalized_and_finite_at_boundaries(self) -> None:
        values = predictive_entropy(np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(values, [0.0, 1.0, 0.0], atol=1e-12)

    def test_dual_policy_rejects_when_either_threshold_is_exceeded(self) -> None:
        result = apply_rejection(
            success_probability=np.array([0.8, 0.2, 0.7, 0.1]),
            au=np.array([0.2, 0.9, 0.2, 0.9]),
            eu=np.array([0.1, 0.1, 0.8, 0.8]),
            policy=PolicyThresholds("au_or_eu", tau_au=0.5, tau_eu=0.5),
        )

        np.testing.assert_array_equal(result.accepted, [True, False, False, False])
        np.testing.assert_array_equal(
            result.reason,
            ["accepted", "high_au", "high_eu", "high_au_and_eu"],
        )
        np.testing.assert_array_equal(
            result.decision,
            ["SUCCESS", "UNDETERMINED", "UNDETERMINED", "UNDETERMINED"],
        )

    def test_threshold_boundary_is_accepted(self) -> None:
        result = apply_rejection(
            success_probability=np.array([0.7]),
            au=np.array([0.5]),
            eu=None,
            policy=PolicyThresholds("au", tau_au=0.5),
        )
        self.assertTrue(result.accepted[0])

    def test_thresholds_round_trip_json_boundary_sentinels(self) -> None:
        original = PolicyThresholds("au_or_eu", tau_au=np.inf, tau_eu=0.5)
        restored = PolicyThresholds.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_default_calibration_maximizes_coverage_under_risk_constraint(self) -> None:
        episode_ids = np.repeat([f"episode_{index}" for index in range(10)], 10)
        episode_labels = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
        episode_probability = np.array([0.9, 0.8, 0.1, 0.2, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1])
        records = CalibrationRecords(
            episode_id=episode_ids,
            labels=np.repeat(episode_labels, 10),
            success_probability=np.repeat(episode_probability, 10),
            absolute_chunk=np.tile(np.arange(1, 11), 10),
            au=np.repeat(np.arange(10, dtype=float) / 10.0, 10),
            eu=np.zeros(100),
        )

        result = calibrate_policy(
            records,
            kind="au",
            max_selective_error=0.2,
            min_coverage=0.1,
            min_accepted_episodes=1,
            target_coverages=(0.5,),
        )

        self.assertTrue(result.default.available)
        self.assertEqual(result.default.accepted_count, 50)
        self.assertAlmostEqual(result.default.coverage, 0.5)
        self.assertAlmostEqual(result.default.selective_error, 0.2)
        self.assertAlmostEqual(result.default.thresholds.tau_au, 0.4)
        self.assertAlmostEqual(result.target_operating_points[0.5].coverage, 0.5)

    def test_calibration_marks_default_unavailable_without_enough_episodes(self) -> None:
        records = CalibrationRecords(
            episode_id=np.repeat(["episode_a", "episode_b"], 10),
            labels=np.repeat([0, 1], 10),
            success_probability=np.repeat([0.1, 0.9], 10),
            absolute_chunk=np.tile(np.arange(1, 11), 2),
            au=np.tile(np.linspace(0.1, 0.9, 10), 2),
            eu=np.zeros(20),
        )
        result = calibrate_policy(records, kind="au", min_accepted_episodes=3)
        self.assertFalse(result.default.available)
        self.assertIn("min_accepted_episodes", result.default.unavailable_reason)

    def test_dual_calibration_uses_exact_two_dimensional_thresholds(self) -> None:
        records = CalibrationRecords(
            episode_id=np.repeat([f"e{index}" for index in range(4)], 10),
            labels=np.repeat([1, 0, 1, 0], 10),
            success_probability=np.repeat([0.9, 0.1, 0.1, 0.9], 10),
            absolute_chunk=np.tile(np.arange(1, 11), 4),
            au=np.repeat([0.1, 0.2, 0.9, 0.1], 10),
            eu=np.repeat([0.1, 0.2, 0.1, 0.9], 10),
        )
        result = calibrate_policy(
            records,
            kind="au_or_eu",
            max_selective_error=0.0,
            min_coverage=0.25,
            min_accepted_episodes=1,
            target_coverages=(),
        )
        self.assertEqual(result.default.accepted_count, 20)
        self.assertAlmostEqual(result.default.thresholds.tau_au, 0.2)
        self.assertAlmostEqual(result.default.thresholds.tau_eu, 0.2)


if __name__ == "__main__":
    unittest.main()
