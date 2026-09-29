from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import numpy as np

from examples.LIBERO.safe_pred.config import TrainConfig
from examples.LIBERO.safe_pred.evaluate import evaluate_checkpoint
from examples.LIBERO.safe_pred.tests.helpers import write_safe_dataset
from examples.LIBERO.safe_pred.train import train


class SafeTrainingSmokeTest(unittest.TestCase):
    def test_trains_and_reloads_both_safe_backbones(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            dataset_path = write_safe_dataset(
                root / "goal.hdf5",
                "libero_goal",
                [True, False, True, False, True, False, True, False],
            )
            for backbone in ("lstm", "mlp"):
                config = TrainConfig(
                    dataset_paths=[str(dataset_path)],
                    output_dir=str(root / "runs"),
                    run_name=f"smoke_{backbone}",
                    backbone=backbone,
                    feature_aggregation="last",
                    hidden_dim=8,
                    epochs=2,
                    batch_size=4,
                    learning_rate=1e-3,
                    val_fraction=0.25,
                    seed=3,
                    device="cpu",
                )

                summary = train(config)
                run_dir = pathlib.Path(summary["run_dir"])

                self.assertTrue((run_dir / "best.pt").is_file())
                self.assertTrue((run_dir / "final.pt").is_file())
                self.assertTrue((run_dir / "split_manifest.json").is_file())
                self.assertEqual(len((run_dir / "epochs.jsonl").read_text().splitlines()), 2)
                self.assertEqual(json.loads((run_dir / "summary.json").read_text())["backbone"], backbone)
                evaluation = evaluate_checkpoint(run_dir / "best.pt", [dataset_path], device="cpu")
                self.assertTrue(np.isfinite(evaluation["failure_scores"]).all())
                self.assertEqual(len(evaluation["failure_scores"]), 8)
                if backbone == "lstm":
                    self.assertIsNotNone(evaluation["metrics"]["brier"])
                else:
                    self.assertIsNone(evaluation["metrics"]["brier"])

    def test_refuses_to_overwrite_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            dataset_path = write_safe_dataset(root / "goal.hdf5", "libero_goal", [True, False, True, False])
            config = TrainConfig(
                dataset_paths=[str(dataset_path)],
                output_dir=str(root),
                run_name="existing",
                backbone="lstm",
                hidden_dim=4,
                epochs=1,
                batch_size=2,
                device="cpu",
            )
            train(config)
            with self.assertRaisesRegex(FileExistsError, "run directory"):
                train(config)


if __name__ == "__main__":
    unittest.main()
