from __future__ import annotations

import unittest

import numpy as np

from examples.LIBERO.edl_pred.rejection_metrics import (
    EpisodeMetricRecord,
    aurc,
    dual_threshold_frontier,
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

    def test_prefix_grid_dual_frontier_matches_exact_threshold_enumeration(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 1, 0])
        probability = np.asarray([0.2, 0.4, 0.8, 0.7, 0.3, 0.1])
        au = np.asarray([0.1, 0.2, 0.2, 0.4, 0.3, 0.4])
        eu = np.asarray([0.4, 0.1, 0.3, 0.2, 0.2, 0.4])

        actual = dual_threshold_frontier(labels, probability, au, eu)
        expected = self._naive_dual_frontier(labels, probability, au, eu)

        for field in ("accepted_count", "coverage", "risk", "tau_au", "tau_eu"):
            np.testing.assert_allclose(actual[field], expected[field])

    @staticmethod
    def _naive_dual_frontier(labels, probability, au, eu):
        errors = (probability >= 0.5) != labels
        best = {}
        for tau_au in (-np.inf, *np.unique(au), np.inf):
            for tau_eu in (-np.inf, *np.unique(eu), np.inf):
                accepted = (au <= tau_au) & (eu <= tau_eu)
                count = int(accepted.sum())
                if not count:
                    continue
                candidate = (float(errors[accepted].mean()), float(tau_au), float(tau_eu))
                key = lambda value: (value[0], -value[1], -value[2], value[1], value[2])
                if count not in best or key(candidate) < key(best[count]):
                    best[count] = candidate
        counts = np.asarray(sorted(best))
        return {
            "accepted_count": counts,
            "coverage": counts / labels.size,
            "risk": np.asarray([best[count][0] for count in counts]),
            "tau_au": np.asarray([best[count][1] for count in counts]),
            "tau_eu": np.asarray([best[count][2] for count in counts]),
        }


if __name__ == "__main__":
    unittest.main()
