from __future__ import annotations

import unittest

import numpy as np

from examples.LIBERO.safe_pred.metrics import binary_ranking_metrics, evaluate_token_uncertainty_records


def _record(
    task_id: int,
    success: bool,
    nll_chunks: list[list[float]],
    entropy_chunks=None,
    suite: str | None = None,
) -> dict:
    entropy_chunks = nll_chunks if entropy_chunks is None else entropy_chunks
    counts = np.asarray([len(chunk) for chunk in nll_chunks], dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    record = {
        "task_id": task_id,
        "success": success,
        "num_action_tokens": counts,
        "token_offsets": offsets,
        "action_token_nll": np.asarray([value for chunk in nll_chunks for value in chunk], dtype=np.float32),
        "action_token_entropy": np.asarray(
            [value for chunk in entropy_chunks for value in chunk], dtype=np.float32
        ),
    }
    if suite is not None:
        record["suite"] = suite
    return record


class TokenUncertaintyMetricsTest(unittest.TestCase):
    def test_uses_task_common_prefix_before_episode_maximum(self) -> None:
        records = [
            _record(0, True, [[0.1, 0.2], [0.2], [99.0]]),
            _record(0, False, [[0.5], [0.8, 0.7]]),
            _record(1, True, [[0.1]]),
            _record(1, False, [[0.9], [99.0]]),
        ]

        result = evaluate_token_uncertainty_records(records)

        self.assertEqual(result["task_prefix_chunks"], {"0": 2, "1": 1})
        self.assertEqual(result["num_episodes"], 4)
        self.assertEqual(result["num_failures"], 2)
        self.assertEqual(result["failure_prevalence"], 0.5)
        for score_name in ("max_nll", "mean_nll", "max_entropy", "mean_entropy"):
            self.assertEqual(result["scores"][score_name]["roc_auc"], 1.0)
            self.assertEqual(result["scores"][score_name]["pr_auc"], 1.0)
            self.assertIsNone(result["scores"][score_name]["brier"])

    def test_tied_scores_use_average_ranks_and_grouped_average_precision(self) -> None:
        metrics = binary_ranking_metrics(
            labels=np.asarray([0, 1, 0, 1]),
            scores=np.asarray([0.5, 0.5, 0.5, 0.5]),
        )
        self.assertEqual(metrics["roc_auc"], 0.5)
        self.assertEqual(metrics["pr_auc"], 0.5)

    def test_common_prefix_keeps_same_task_id_separate_across_suites(self) -> None:
        records = [
            _record(0, True, [[0.1], [0.2], [99.0]], suite="suite_a"),
            _record(0, False, [[0.5], [0.8]], suite="suite_a"),
            _record(0, True, [[0.1]], suite="suite_b"),
            _record(0, False, [[0.9], [99.0]], suite="suite_b"),
        ]

        result = evaluate_token_uncertainty_records(records)

        self.assertEqual(
            result["task_prefix_chunks"],
            {"suite_a:0": 2, "suite_b:0": 1},
        )

    def test_one_class_ranking_metrics_are_undefined(self) -> None:
        metrics = binary_ranking_metrics(np.asarray([1, 1]), np.asarray([0.1, 0.2]))
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["pr_auc"])


if __name__ == "__main__":
    unittest.main()
