from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from examples.LIBERO.edl_pred.evaluate_rejection import (
    evaluate_prediction_sets,
    publish_evaluation,
)
from examples.LIBERO.edl_pred.rejection_data import EpisodePrediction


class EvaluateRejectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.calibration = {
            "schema_version": "1.0",
            "analysis_seed": 7,
            "primary_chunks": list(range(1, 11)),
            "objective": {"max_selective_error": 0.2},
            "runs": {
                "edl": {
                    "head": "edl",
                    "chunk_encoder": "token_attention_pool",
                    "token_position_encoding": "sinusoidal",
                    "policies": {
                        "au": self._policy("au", {"kind": "au", "tau_au": 0.5}),
                        "au_or_eu": self._policy(
                            "au_or_eu",
                            {"kind": "au_or_eu", "tau_au": 0.5, "tau_eu": 0.5},
                        ),
                        "predictive_entropy": self._policy(
                            "predictive_entropy",
                            {"kind": "predictive_entropy", "tau_entropy": 0.8},
                        ),
                    },
                },
                "softmax": {
                    "head": "softmax",
                    "chunk_encoder": "token_attention_pool",
                    "token_position_encoding": "sinusoidal",
                    "policies": {
                        "predictive_entropy": self._policy(
                            "predictive_entropy",
                            {"kind": "predictive_entropy", "tau_entropy": 0.8},
                        )
                    },
                },
            },
        }
        self.predictions = {
            "edl": self._predictions(edl=True),
            "softmax": self._predictions(edl=False),
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _policy(kind: str, thresholds: dict[str, object]) -> dict[str, object]:
        point = {
            "available": True,
            "target_coverage": None,
            "thresholds": thresholds,
            "coverage": 0.5,
            "abstention_rate": 0.5,
            "selective_error": 0.1,
            "selective_accuracy": 0.9,
            "accepted_count": 20,
            "accepted_episode_count": 4,
            "rejected_count": 20,
            "unavailable_reason": None,
        }
        target_point = dict(point)
        target_point["target_coverage"] = 0.5
        return {"kind": kind, "default": point, "target_operating_points": {"0.5": target_point}}

    @staticmethod
    def _predictions(*, edl: bool) -> tuple[EpisodePrediction, ...]:
        records = []
        for suite_index, suite in enumerate(("libero_goal", "libero_spatial")):
            for index in range(2):
                label = index
                probability = np.linspace(0.1, 0.9, 12)
                if label == 0:
                    probability = 1.0 - probability
                kwargs = {}
                if edl:
                    kwargs = {
                        "class_evidence": np.ones((12, 2)),
                        "verifier_au": np.linspace(0.2, 0.9, 12),
                        "verifier_eu": np.linspace(0.1, 0.3, 12),
                        "verifier_total_evidence": np.ones(12) * 8,
                    }
                records.append(
                    EpisodePrediction(
                        suite=suite,
                        episode_key=f"episode_{suite_index}_{index}",
                        label=label,
                        success_probability=probability,
                        failure_probability=1.0 - probability,
                        token_lengths=None,
                        **kwargs,
                    )
                )
        return tuple(records)

    def test_evaluation_applies_frozen_thresholds_and_publishes_complete_artifacts(self) -> None:
        evaluation = evaluate_prediction_sets(
            self.calibration,
            self.predictions,
            bootstrap_replicates=20,
        )
        output = publish_evaluation(self.root / "analysis", evaluation, self.predictions)

        for name in (
            "rejection_test_predictions.hdf5",
            "rejection_metrics.json",
            "rejection_metrics.csv",
            "risk_coverage.png",
            "metrics_by_chunk.png",
            "au_eu_quadrants.png",
            "edl_vs_softmax.png",
        ):
            self.assertGreater((output / name).stat().st_size, 0, name)
        self.assertEqual(evaluation["runs"]["edl"]["policies"]["au"]["thresholds"]["tau_au"], 0.5)
        self.assertEqual(
            evaluation["runs"]["edl"]["policies"]["au"]["target_operating_points"]["0.5"]["thresholds"]["tau_au"],
            0.5,
        )
        comparison = evaluation["matched_comparisons"][0]
        self.assertEqual(comparison["edl_run"], "edl")
        self.assertEqual(comparison["softmax_run"], "softmax")
        self.assertIn("coverage_difference", comparison)
        self.assertIn("aurc_difference", comparison)
        rows = list(csv.DictReader((output / "rejection_metrics.csv").open(encoding="utf-8")))
        self.assertTrue(any(row["absolute_chunk"] == "10" for row in rows))
        self.assertTrue(any(row["absolute_chunk"] == "12" and row["incomplete_support"] == "True" for row in rows))
        with h5py.File(output / "rejection_test_predictions.hdf5", "r") as handle:
            group = handle["runs/edl/episodes/libero_goal/episode_0_0/policies/au"]
            self.assertIn("accepted", group)
            self.assertIn("decision", group)
            self.assertIn("target_operating_points/0.5/accepted", group)

    def test_evaluation_rejects_prediction_identity_mismatch(self) -> None:
        mismatched = dict(self.predictions)
        mismatched["softmax"] = mismatched["softmax"][:-1]
        with self.assertRaisesRegex(ValueError, "identities"):
            evaluate_prediction_sets(self.calibration, mismatched, bootstrap_replicates=5)

    def test_unavailable_default_retains_targets_and_matched_aurc(self) -> None:
        calibration = dict(self.calibration)
        calibration["runs"] = {name: dict(run) for name, run in self.calibration["runs"].items()}
        softmax = calibration["runs"]["softmax"]
        softmax["policies"] = dict(softmax["policies"])
        entropy = dict(softmax["policies"]["predictive_entropy"])
        entropy["default"] = {
            "available": False,
            "thresholds": None,
            "unavailable_reason": "no feasible default",
        }
        softmax["policies"]["predictive_entropy"] = entropy

        evaluation = evaluate_prediction_sets(calibration, self.predictions, bootstrap_replicates=5)

        policy = evaluation["runs"]["softmax"]["policies"]["predictive_entropy"]
        self.assertFalse(policy["available"])
        self.assertTrue(policy["target_operating_points"]["0.5"]["available"])
        comparison = evaluation["matched_comparisons"][0]
        self.assertIsNone(comparison["coverage_difference"]["estimate"])
        self.assertIsNotNone(comparison["aurc_difference"]["estimate"])


if __name__ == "__main__":
    unittest.main()
