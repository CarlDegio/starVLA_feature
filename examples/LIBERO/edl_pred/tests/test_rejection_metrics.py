from __future__ import annotations

import unittest

import numpy as np

from examples.LIBERO.edl_pred.rejection_metrics import (
    EpisodeMetricRecord,
    aurc,
    error_detection_metrics,
    paired_episode_bootstrap,
    risk_coverage_curve,
    selective_metrics,
)


class RejectionMetricsTest(unittest.TestCase):
    def test_selective_metrics_keep_abstentions_out_of_accuracy_but_in_coverage(self) -> None:
        metrics = selective_metrics(
            labels=np.array([1, 0, 1, 0]),
            success_probability=np.array([0.8, 0.2, 0.1, 0.9]),
            accepted=np.array([True, True, False, False]),
        )
        self.assertAlmostEqual(metrics["coverage"], 0.5)
        self.assertEqual(metrics["accepted_count"], 2)
        self.assertAlmostEqual(metrics["selective_accuracy"], 1.0)
        self.assertEqual(metrics["rejected_count"], 2)
        self.assertAlmostEqual(metrics["rejected_error_rate"], 1.0)

    def test_error_detection_auc_treats_high_uncertainty_as_error(self) -> None:
        metrics = error_detection_metrics(
            labels=np.array([1, 0, 1, 0]),
            success_probability=np.array([0.9, 0.1, 0.1, 0.9]),
            uncertainty=np.array([0.1, 0.2, 0.8, 0.9]),
        )
        self.assertAlmostEqual(metrics["error_roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["error_pr_auc"], 1.0)
        self.assertAlmostEqual(metrics["error_rate_baseline"], 0.5)

    def test_risk_coverage_and_aurc_use_low_uncertainty_first(self) -> None:
        curve = risk_coverage_curve(
            labels=np.array([1, 0, 1, 0]),
            success_probability=np.array([0.9, 0.1, 0.1, 0.9]),
            uncertainty=np.array([0.1, 0.2, 0.8, 0.9]),
        )
        np.testing.assert_allclose(curve["risk"], [0.0, 0.0, 1 / 3, 0.5])
        self.assertAlmostEqual(aurc(curve), (0.0 + 0.0 + 1 / 3 + 0.5) / 4)

    def test_episode_bootstrap_resamples_complete_episode_records(self) -> None:
        records = [
            EpisodeMetricRecord("episode_a", 1, 1.0),
            EpisodeMetricRecord("episode_a", 2, 1.0),
            EpisodeMetricRecord("episode_b", 1, 3.0),
        ]
        result = paired_episode_bootstrap(
            records,
            statistic=lambda sample: float(np.mean([record.value for record in sample])),
            seed=7,
            replicates=100,
        )
        self.assertEqual(result.replicates, 100)
        self.assertEqual(result.valid_replicates, 100)
        self.assertLessEqual(result.lower, result.estimate)
        self.assertGreaterEqual(result.upper, result.estimate)


if __name__ == "__main__":
    unittest.main()
