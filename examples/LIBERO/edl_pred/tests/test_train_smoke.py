"""End-to-end CPU coverage for the verifier training entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import yaml

from examples.LIBERO.edl_pred.config import load_config
from examples.LIBERO.edl_pred.tests.helpers import write_uncertainty_hdf5
from examples.LIBERO.edl_pred.train import train


def write_training_config(
    root: Path,
    *,
    dataset: Path,
    run_name: str,
    overwrite: bool,
    head: str,
    class_balance: str = "none",
) -> Path:
    """Write a minimal complete verifier config backed by a synthetic suite."""
    config = {
        "run": {
            "name": run_name,
            "output_root": "outputs",
            "seed": 17,
            "overwrite": overwrite,
        },
        "data": {
            "datasets": {"libero_spatial": str(dataset)},
            "selected_suites": ["libero_spatial"],
            "validation_ratio": 0.25,
            "split_seed": 9,
            "max_action_tokens": "auto",
        },
        "model": {
            "chunk_encoder": "token_attention_pool",
            "token_position_encoding": "sinusoidal",
            "token_embed_dim": 8,
            "chunk_embed_dim": 8,
            "attention_heads": 2,
            "self_attention_layers": 1,
            "encoder_dropout": 0.0,
            "lstm_hidden_dim": 8,
            "lstm_layers": 1,
            "lstm_dropout": 0.0,
            "head": head,
        },
        "edl": {
            "evidence_activation": "softplus",
            "kl_weight": 0.01,
            "kl_anneal_epochs": 1,
        },
        "loss": {"class_balance": class_balance},
        "training": {
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 1,
            "num_workers": 0,
        },
    }
    path = root / f"{run_name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


class TrainingSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = write_uncertainty_hdf5(
            self.root / "source.hdf5",
            [
                ("episode_00", 0, [2, 1]),
                ("episode_01", 0, [1, 2]),
                ("episode_02", 0, [2, 2]),
                ("episode_03", 0, [1, 1]),
                ("episode_04", 1, [2, 1]),
                ("episode_05", 1, [1, 2]),
                ("episode_06", 1, [2, 2]),
                ("episode_07", 1, [1, 1]),
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_cpu_edl_training_writes_complete_best_model_artifacts(self) -> None:
        """Would fail if training skipped best restore, artifacts, or EDL metrics."""
        config_path = write_training_config(
            self.root,
            dataset=self.source,
            run_name="edl_cpu",
            overwrite=False,
            head="edl",
            class_balance="inverse_frequency",
        )
        source_before = self.source.read_bytes()

        output_dir = train(load_config(config_path))

        for relative in (
            "resolved_config.yaml",
            "split_manifest.json",
            "metrics.jsonl",
            "training_curves.png",
            "checkpoints/best.pt",
            "checkpoints/last.pt",
            "validation_predictions.hdf5",
        ):
            self.assertTrue((output_dir / relative).is_file(), relative)
        self.assertEqual(self.source.read_bytes(), source_before)
        resolved = yaml.safe_load((output_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(resolved["metadata"]["max_action_tokens"], 2)
        self.assertEqual(resolved["metadata"]["class_weights"], [1.0, 1.0])
        metrics = [json.loads(line) for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(metrics), 1)
        self.assertIn("validation_chunk_roc_auc", metrics[0])
        self.assertIn("validation_final_chunk_roc_auc", metrics[0])
        self.assertIn("validation_verifier_eu_label_0", metrics[0])
        with h5py.File(output_dir / "validation_predictions.hdf5", "r") as handle:
            groups: list[h5py.Group] = []
            handle["episodes"].visititems(
                lambda _name, value: groups.append(value) if isinstance(value, h5py.Group) and "success_probability" in value else None
            )
            self.assertGreater(len(groups), 0)
            self.assertTrue(all("class_evidence" in group for group in groups))

    def test_overwrite_is_scoped_to_the_named_run_directory(self) -> None:
        """Would fail if a collision were accepted or overwrite removed sibling runs."""
        output_root = self.root / "outputs"
        run_dir = output_root / "collision"
        sibling = output_root / "unrelated"
        run_dir.mkdir(parents=True)
        sibling.mkdir()
        (run_dir / "old.txt").write_text("old", encoding="utf-8")
        (sibling / "keep.txt").write_text("keep", encoding="utf-8")
        source_before = self.source.read_bytes()

        reject_config = load_config(
            write_training_config(
                self.root,
                dataset=self.source,
                run_name="collision",
                overwrite=False,
                head="softmax",
            )
        )
        with self.assertRaisesRegex(FileExistsError, "collision"):
            train(reject_config)
        self.assertEqual((run_dir / "old.txt").read_text(encoding="utf-8"), "old")

        replace_config = load_config(
            write_training_config(
                self.root,
                dataset=self.source,
                run_name="collision",
                overwrite=True,
                head="softmax",
            )
        )
        output_dir = train(replace_config)
        self.assertEqual(output_dir, run_dir)
        self.assertFalse((run_dir / "old.txt").exists())
        self.assertTrue((run_dir / "checkpoints" / "best.pt").is_file())
        self.assertEqual((sibling / "keep.txt").read_text(encoding="utf-8"), "keep")
        self.assertEqual(self.source.read_bytes(), source_before)

    def test_overwrite_rejects_source_hdf5_inside_named_run_before_deleting(self) -> None:
        """Would fail if overwrite removed an input HDF5 before source preflight."""
        run_dir = self.root / "outputs" / "source_run"
        source = write_uncertainty_hdf5(
            run_dir / "source.hdf5",
            [
                ("episode_00", 0, [1]),
                ("episode_01", 0, [1]),
                ("episode_02", 1, [1]),
                ("episode_03", 1, [1]),
            ],
        )
        source_before = source.read_bytes()
        config = load_config(
            write_training_config(
                self.root,
                dataset=source,
                run_name="source_run",
                overwrite=True,
                head="softmax",
            )
        )

        with self.assertRaisesRegex(ValueError, "inside run destination"):
            train(config)

        self.assertTrue(source.is_file())
        self.assertEqual(source.read_bytes(), source_before)
        self.assertFalse((run_dir / "checkpoints").exists())
        self.assertEqual(sorted(path.name for path in run_dir.iterdir()), ["source.hdf5"])

    def test_overwrite_rejects_destination_symlink_without_touching_target(self) -> None:
        """Would fail if resolving the run destination allowed target deletion."""
        output_root = self.root / "outputs"
        target = self.root / "unrelated_target"
        sentinel = target / "keep.txt"
        output_root.mkdir()
        target.mkdir()
        sentinel.write_text("keep", encoding="utf-8")
        destination_link = output_root / "alias"
        destination_link.symlink_to(target, target_is_directory=True)
        config = load_config(
            write_training_config(
                self.root,
                dataset=self.source,
                run_name="alias",
                overwrite=True,
                head="softmax",
            )
        )

        with self.assertRaisesRegex(ValueError, "symlink"):
            train(config)

        self.assertTrue(destination_link.is_symlink())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        self.assertFalse((target / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
