from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.LIBERO.edl_pred.artifacts import PredictionRecord, write_validation_predictions
from examples.LIBERO.edl_pred.dataset import EpisodeRef
from examples.LIBERO.edl_pred.rejection_data import (
    build_calibration_records,
    read_prediction_file,
    sha256_file,
    validate_matched_predictions,
)


class RejectionDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _record(self, key: str, label: int, *, chunks: int = 10, edl: bool = True) -> PredictionRecord:
        success = np.linspace(0.1, 0.9, chunks, dtype=np.float32)
        probabilities = np.stack([1.0 - success, success], axis=1)
        kwargs = {}
        if edl:
            kwargs = {
                "class_evidence": np.ones((chunks, 2), dtype=np.float32),
                "verifier_au": np.linspace(0.2, 0.8, chunks, dtype=np.float32),
                "verifier_eu": np.linspace(0.1, 0.3, chunks, dtype=np.float32),
                "verifier_total_evidence": np.ones(chunks, dtype=np.float32) * 2.0,
            }
        return PredictionRecord(
            ref=EpisodeRef("libero_goal", self.root / "source.hdf5", key, label),
            class_probabilities=probabilities,
            **kwargs,
        )

    def test_reader_and_calibration_preserve_absolute_chunks_and_episode_identity(self) -> None:
        path = self.root / "predictions.hdf5"
        write_validation_predictions(path, [self._record("episode_a", 0), self._record("episode_b", 1)])

        predictions = read_prediction_file(path, expected_head="edl")
        calibration = build_calibration_records(predictions, primary_chunks=tuple(range(1, 11)))

        self.assertEqual(len(predictions), 2)
        self.assertEqual(calibration.labels.shape, (20,))
        np.testing.assert_array_equal(calibration.absolute_chunk[:10], np.arange(1, 11))
        self.assertEqual(np.unique(calibration.episode_id).size, 2)
        self.assertIsNotNone(calibration.au)
        self.assertEqual(len(sha256_file(path)), 64)

    def test_reader_rejects_missing_edl_fields_for_edl_head(self) -> None:
        path = self.root / "softmax.hdf5"
        write_validation_predictions(path, [self._record("episode_a", 0, edl=False)])
        with self.assertRaisesRegex(ValueError, "EDL"):
            read_prediction_file(path, expected_head="edl")

    def test_calibration_rejects_episode_shorter_than_primary_horizon(self) -> None:
        path = self.root / "short.hdf5"
        write_validation_predictions(path, [self._record("episode_a", 0, chunks=9)])
        with self.assertRaisesRegex(ValueError, "chunk 10"):
            build_calibration_records(read_prediction_file(path, expected_head="edl"))

    def test_matched_predictions_reject_label_or_chunk_mismatch(self) -> None:
        left_path = self.root / "left.hdf5"
        right_path = self.root / "right.hdf5"
        write_validation_predictions(left_path, [self._record("episode_a", 0)])
        write_validation_predictions(right_path, [self._record("episode_a", 1)])
        with self.assertRaisesRegex(ValueError, "identities"):
            validate_matched_predictions(
                read_prediction_file(left_path, expected_head="edl"),
                read_prediction_file(right_path, expected_head="edl"),
            )


if __name__ == "__main__":
    unittest.main()
