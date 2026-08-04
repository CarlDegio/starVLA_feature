"""Round-trip and atomic-write tests for verifier research artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

from examples.LIBERO.edl_pred.artifacts import (
    PredictionRecord,
    append_metrics,
    plot_training_curves,
    prediction_records_from_output,
    save_checkpoint,
    write_split_manifest,
    write_validation_predictions,
)
from examples.LIBERO.edl_pred.dataset import EpisodeRef, SplitManifest
from examples.LIBERO.edl_pred.model import VerifierOutput


class ArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.ref = EpisodeRef(
            "libero_spatial",
            self.root / "source.hdf5",
            "task_000_episode_0000",
            1,
        )
        self.edl_record = PredictionRecord(
            ref=self.ref,
            class_probabilities=np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]], dtype=np.float32),
            class_evidence=np.array([[2.0, 1.0], [1.0, 3.0], [0.5, 5.0]], dtype=np.float32),
            verifier_au=np.array([0.4, 0.3, 0.2], dtype=np.float32),
            verifier_eu=np.array([0.5, 0.4, 0.3], dtype=np.float32),
            verifier_total_evidence=np.array([3.0, 4.0, 5.5], dtype=np.float32),
            pooling_weights=np.array(
                [[0.1, 0.2, 0.3, 0.15, 0.25], [0.2, 0.2, 0.2, 0.2, 0.2], [0.3, 0.2, 0.1, 0.2, 0.2]],
                dtype=np.float32,
            ),
            token_mask=np.ones((3, 5), dtype=bool),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prediction_hdf5_preserves_episode_chunk_outputs(self) -> None:
        path = self.root / "validation_predictions.hdf5"
        write_validation_predictions(path, [self.edl_record])

        with h5py.File(path, "r") as handle:
            episode = handle["episodes/libero_spatial/task_000_episode_0000"]
            self.assertEqual(episode.attrs["suite"], "libero_spatial")
            self.assertEqual(episode.attrs["episode_key"], "task_000_episode_0000")
            self.assertEqual(episode.attrs["label"], 1)
            np.testing.assert_allclose(episode["failure_probability"][:], [0.8, 0.4, 0.1])
            np.testing.assert_allclose(episode["success_probability"][:], [0.2, 0.6, 0.9])
            self.assertEqual(episode["class_evidence"].shape, (3, 2))
            self.assertEqual(episode["pooling_weights"].shape, (3, 5))
            np.testing.assert_array_equal(episode["token_lengths"][:], [5, 5, 5])

    def test_prediction_hdf5_drops_padded_chunks_and_token_values(self) -> None:
        record = PredictionRecord(
            ref=EpisodeRef("libero_goal", self.root / "source.hdf5", "episode_001", 0),
            class_probabilities=np.array([[0.9, 0.1], [0.6, 0.4]], dtype=np.float32),
            pooling_weights=np.array([[0.7, 0.3, 99.0], [1.0, 88.0, 77.0]], dtype=np.float32),
            token_mask=np.array([[True, True, False], [True, False, False]]),
        )
        path = self.root / "compact_predictions.hdf5"
        write_validation_predictions(path, [record])

        with h5py.File(path, "r") as handle:
            episode = handle["episodes/libero_goal/episode_001"]
            self.assertEqual(episode["success_probability"].shape, (2,))
            self.assertEqual(episode["pooling_weights"].shape, (2, 2))
            np.testing.assert_allclose(episode["pooling_weights"][:], [[0.7, 0.3], [1.0, 0.0]])
            np.testing.assert_array_equal(episode["token_lengths"][:], [2, 1])

    def test_prediction_hdf5_compacts_arbitrary_valid_token_positions_leftward(self) -> None:
        record = PredictionRecord(
            ref=EpisodeRef("libero_goal", self.root / "source.hdf5", "episode_nonprefix", 0),
            class_probabilities=np.array([[0.9, 0.1], [0.6, 0.4]], dtype=np.float32),
            pooling_weights=np.array([[99.0, 0.3, 88.0, 0.7], [1.0, 77.0, 66.0, 55.0]], dtype=np.float32),
            token_mask=np.array([[False, True, False, True], [True, False, False, False]]),
        )
        path = self.root / "nonprefix_predictions.hdf5"
        write_validation_predictions(path, [record])

        with h5py.File(path, "r") as handle:
            episode = handle["episodes/libero_goal/episode_nonprefix"]
            np.testing.assert_allclose(episode["pooling_weights"][:], [[0.3, 0.7], [1.0, 0.0]])
            np.testing.assert_array_equal(episode["token_lengths"][:], [2, 1])

    def test_prediction_records_extract_only_valid_chunks_from_verifier_output(self) -> None:
        refs = (
            self.ref,
            EpisodeRef("libero_object", self.root / "source.hdf5", "episode_002", 0),
        )
        chunk_mask = torch.tensor([[True, True, False], [True, False, False]])
        token_mask = torch.tensor(
            [
                [[True, True], [True, False], [False, False]],
                [[True, True], [False, False], [False, False]],
            ]
        )
        output = VerifierOutput(
            logits=torch.zeros(2, 3, 2),
            probabilities=torch.tensor(
                [
                    [[0.8, 0.2], [0.3, 0.7], [0.0, 0.0]],
                    [[0.9, 0.1], [0.0, 0.0], [0.0, 0.0]],
                ]
            ),
            evidence=torch.ones(2, 3, 2),
            alpha=torch.ones(2, 3, 2) * 2.0,
            verifier_au=torch.ones(2, 3),
            verifier_eu=torch.ones(2, 3) * 0.5,
            verifier_total_evidence=torch.ones(2, 3) * 2.0,
            pooling_weights=torch.tensor(
                [
                    [[0.5, 0.5], [1.0, 0.0], [0.0, 0.0]],
                    [[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
                ]
            ),
        )

        records = prediction_records_from_output(refs, output, chunk_mask, token_mask)

        self.assertEqual([record.class_probabilities.shape[0] for record in records], [2, 1])
        self.assertEqual(records[0].pooling_weights.shape, (2, 2))
        self.assertEqual(records[1].token_mask.shape, (1, 2))
        np.testing.assert_allclose(records[0].class_probabilities[:, 1], [0.2, 0.7])

    def test_prediction_records_from_output_preserves_nonprefix_tokens_for_export(self) -> None:
        token_mask = torch.tensor([[[False, True, False, True], [True, False, False, False]]])
        output = VerifierOutput(
            logits=torch.zeros(1, 2, 2),
            probabilities=torch.tensor([[[0.8, 0.2], [0.3, 0.7]]]),
            evidence=None,
            alpha=None,
            verifier_au=None,
            verifier_eu=None,
            verifier_total_evidence=None,
            pooling_weights=torch.tensor([[[0.0, 0.25, 0.0, 0.75], [1.0, 0.0, 0.0, 0.0]]]),
        )
        records = prediction_records_from_output(
            (self.ref,),
            output,
            torch.tensor([[True, True]]),
            token_mask,
        )
        path = self.root / "batch_nonprefix_predictions.hdf5"
        write_validation_predictions(path, records)

        with h5py.File(path, "r") as handle:
            episode = handle["episodes/libero_spatial/task_000_episode_0000"]
            np.testing.assert_allclose(episode["pooling_weights"][:], [[0.25, 0.75], [1.0, 0.0]])
            np.testing.assert_array_equal(episode["token_lengths"][:], [2, 1])

    def test_atomic_checkpoint_round_trip(self) -> None:
        path = self.root / "last.pt"
        save_checkpoint(path, {"epoch": 3, "model": {"x": torch.ones(1)}})

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["epoch"], 3)
        self.assertEqual(self._temporary_files(), [])

    def test_checkpoint_destination_ending_tmp_is_atomically_replaceable(self) -> None:
        path = self.root / "checkpoint.tmp"
        save_checkpoint(path, {"epoch": 1})
        save_checkpoint(path, {"epoch": 2})

        self.assertEqual(torch.load(path, map_location="cpu", weights_only=False)["epoch"], 2)
        self.assertEqual(self._temporary_files(), [])

    def test_same_stem_different_extensions_use_independent_concurrent_temp_paths(self) -> None:
        paths = (self.root / "metrics.jsonl", self.root / "metrics.h5")
        barrier = Barrier(2)
        original_save = torch.save

        def synchronized_save(*args: object, **kwargs: object) -> None:
            original_save(*args, **kwargs)
            barrier.wait(timeout=5)

        with mock.patch("examples.LIBERO.edl_pred.artifacts.torch.save", side_effect=synchronized_save):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(save_checkpoint, path, {"epoch": index}) for index, path in enumerate(paths)]
                for future in futures:
                    future.result()

        self.assertEqual([torch.load(path, map_location="cpu", weights_only=False)["epoch"] for path in paths], [0, 1])
        self.assertEqual(self._temporary_files(), [])

    def test_checkpoint_replace_failure_preserves_previous_file_and_removes_temp(self) -> None:
        path = self.root / "last.pt"
        save_checkpoint(path, {"epoch": 1})
        previous = path.read_bytes()
        with mock.patch.object(Path, "replace", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                save_checkpoint(path, {"epoch": 2})

        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(self._temporary_files(), [])

    def test_jsonl_append_is_atomic_and_requires_strictly_increasing_epochs(self) -> None:
        path = self.root / "metrics.jsonl"
        append_metrics(path, {"epoch": 1, "validation_total_loss": 0.8, "roc_auc": None})
        append_metrics(path, {"epoch": 2, "validation_total_loss": 0.4})

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["epoch"] for row in rows], [1, 2])
        previous = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            append_metrics(path, {"epoch": 2, "validation_total_loss": 0.3})
        self.assertEqual(path.read_bytes(), previous)

    def test_split_manifest_is_portable_and_refuses_unexpected_overwrite(self) -> None:
        path = self.root / "split_manifest.json"
        manifest = SplitManifest(
            train=(EpisodeRef("libero_goal", self.root / "train.hdf5", "episode_a", 0),),
            validation=(self.ref,),
            max_action_tokens=5,
        )
        write_split_manifest(path, manifest)

        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["max_action_tokens"], 5)
        self.assertEqual(payload["train"], [{"suite": "libero_goal", "episode_key": "episode_a", "label": 0}])
        self.assertEqual(payload["validation"], [{"suite": "libero_spatial", "episode_key": "task_000_episode_0000", "label": 1}])
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            write_split_manifest(path, manifest)

    def test_predictions_and_curve_refuse_overwrite_and_clean_up_after_atomic_failure(self) -> None:
        prediction_path = self.root / "validation_predictions.hdf5"
        with mock.patch.object(Path, "replace", side_effect=OSError("rename failure")):
            with self.assertRaisesRegex(OSError, "rename failure"):
                write_validation_predictions(prediction_path, [self.edl_record])
        self.assertFalse(prediction_path.exists())
        self.assertEqual(self._temporary_files(), [])

        curve_path = self.root / "training_curves.png"
        history = [
            {"epoch": 1, "train_total_loss": 0.9, "validation_total_loss": 0.8, "validation_chunk_roc_auc": 0.6},
            {"epoch": 2, "train_total_loss": 0.5, "validation_total_loss": 0.4, "validation_chunk_pr_auc": 0.7},
        ]
        plot_training_curves(curve_path, history)
        self.assertGreater(curve_path.stat().st_size, 0)
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            plot_training_curves(curve_path, history)

    def test_artifact_inputs_reject_invalid_records_and_missing_destination_directory(self) -> None:
        malformed = PredictionRecord(
            ref=self.ref,
            class_probabilities=np.array([[0.1, 0.1]], dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "sum"):
            write_validation_predictions(self.root / "invalid.hdf5", [malformed])
        with self.assertRaisesRegex(ValueError, "parent directory"):
            save_checkpoint(self.root / "missing" / "last.pt", {"epoch": 1})

    def _temporary_files(self) -> list[Path]:
        return sorted(path for path in self.root.iterdir() if path.name.startswith("."))


if __name__ == "__main__":
    unittest.main()
