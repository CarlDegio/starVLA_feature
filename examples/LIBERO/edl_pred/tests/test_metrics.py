"""Numerical contracts for verifier binary and sequence metrics."""

from __future__ import annotations

import unittest

import numpy as np

from examples.LIBERO.edl_pred.metrics import binary_metrics, sequence_metrics


class MetricTest(unittest.TestCase):
    def test_perfect_classifier_metrics(self) -> None:
        result = binary_metrics(
            labels=np.array([0, 0, 1, 1]),
            success_probability=np.array([0.1, 0.2, 0.8, 0.9]),
        )

        self.assertEqual(result["roc_auc"], 1.0)
        self.assertEqual(result["pr_auc"], 1.0)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["balanced_accuracy"], 1.0)
        self.assertEqual(result["f1"], 1.0)
        self.assertAlmostEqual(result["brier"], 0.025)
        self.assertAlmostEqual(result["ece"], 0.15)

    def test_tied_scores_use_average_ranks_and_grouped_pr_thresholds(self) -> None:
        # Treating the tied positive/negative pair as an arbitrary ordering
        # would yield ROC-AUC 0 or 1 instead of the expected 0.5.
        result = binary_metrics(
            labels=np.array([0, 1, 0, 1]),
            success_probability=np.array([0.2, 0.2, 0.8, 0.8]),
        )

        self.assertEqual(result["roc_auc"], 0.5)
        self.assertEqual(result["pr_auc"], 0.5)

    def test_threshold_metrics_use_positive_at_exactly_point_five(self) -> None:
        result = binary_metrics(
            labels=np.array([0, 0, 1, 1]),
            success_probability=np.array([0.49, 0.50, 0.50, 0.51]),
        )

        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["f1"], 0.8)

    def test_ece_uses_ten_equal_width_bins_with_one_in_final_bin(self) -> None:
        result = binary_metrics(
            labels=np.array([0, 1, 1, 0]),
            success_probability=np.array([0.01, 0.09, 0.10, 1.0]),
        )

        # Bin [0.0, 0.1) contributes 0.45 * 2/4; bins [0.1, 0.2) and
        # [0.9, 1.0] contribute 0.9/4 and 1.0/4 respectively.
        self.assertAlmostEqual(result["ece"], 0.7)

    def test_one_class_auc_is_none_but_threshold_metrics_remain_available(self) -> None:
        result = binary_metrics(np.zeros(3), np.array([0.1, 0.2, 0.9]))

        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["pr_auc"])
        self.assertEqual(result["accuracy"], 2.0 / 3.0)
        self.assertEqual(result["balanced_accuracy"], 2.0 / 3.0)
        self.assertEqual(result["f1"], 0.0)

    def test_sequence_metrics_excludes_padded_chunks_and_uses_last_valid_chunk(self) -> None:
        labels = np.array([0, 1])
        probabilities = np.array(
            [
                [0.1, 0.3, 0.99],
                [0.2, 0.9, 0.01],
            ]
        )
        chunk_mask = np.array(
            [
                [True, True, False],
                [True, True, False],
            ]
        )

        result = sequence_metrics(labels, probabilities, chunk_mask)

        self.assertEqual(result["chunk"]["roc_auc"], 0.75)
        self.assertEqual(result["final_chunk"]["roc_auc"], 1.0)
        self.assertEqual(result["final_chunk"]["accuracy"], 1.0)

    def test_metrics_reject_nonbinary_labels_invalid_probabilities_and_empty_sequences(self) -> None:
        invalid_binary_cases = (
            (np.array([0, 2]), np.array([0.1, 0.2])),
            (np.array([0, 1]), np.array([0.1, np.nan])),
            (np.array([0, 1]), np.array([0.1, 1.1])),
            (np.array([], dtype=np.int64), np.array([], dtype=np.float64)),
        )
        for labels, probabilities in invalid_binary_cases:
            with self.subTest(labels=labels, probabilities=probabilities):
                with self.assertRaises(ValueError):
                    binary_metrics(labels, probabilities)

        with self.assertRaisesRegex(ValueError, "valid chunk"):
            sequence_metrics(
                np.array([0, 1]),
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                np.array([[True, False], [False, False]]),
            )


if __name__ == "__main__":
    unittest.main()
