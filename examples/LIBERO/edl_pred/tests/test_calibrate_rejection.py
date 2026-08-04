from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from examples.LIBERO.edl_pred.artifacts import PredictionRecord, write_validation_predictions
from examples.LIBERO.edl_pred.calibrate_rejection import calibrate_sweep
from examples.LIBERO.edl_pred.dataset import EpisodeRef


class CalibrateRejectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.sweep = self.root / "sweep"
        self.runs = self.sweep / "runs"
        self.runs.mkdir(parents=True)
        manifest_runs = []
        for head in ("edl", "softmax"):
            run_name = f"attention_{head}"
            run_dir = self.runs / run_name
            (run_dir / "checkpoints").mkdir(parents=True)
            (run_dir / "checkpoints" / "best.pt").write_bytes(f"{head}-checkpoint".encode())
            (run_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "model": {
                            "head": head,
                            "chunk_encoder": "token_attention_pool",
                            "token_position_encoding": "sinusoidal",
                        }
                    }
                ),
                encoding="utf-8",
            )
            records = []
            for index in range(10):
                label = index % 2
                success = np.full(10, 0.8 if label else 0.2, dtype=np.float32)
                if index >= 8:
                    success[:] = 1.0 - success
                probabilities = np.stack([1.0 - success, success], axis=1)
                kwargs = {}
                if head == "edl":
                    kwargs = {
                        "class_evidence": np.ones((10, 2), dtype=np.float32),
                        "verifier_au": np.full(10, 0.1 + index / 10, dtype=np.float32),
                        "verifier_eu": np.full(10, 0.2, dtype=np.float32),
                        "verifier_total_evidence": np.full(10, 8.0, dtype=np.float32),
                    }
                records.append(
                    PredictionRecord(
                        EpisodeRef("libero_goal", self.root / "source.hdf5", f"episode_{index}", label),
                        probabilities,
                        **kwargs,
                    )
                )
            write_validation_predictions(run_dir / "validation_predictions.hdf5", records)
            manifest_runs.append(
                {
                    "run_name": run_name,
                    "output_path": str(run_dir),
                    "config_path": str(self.root / f"{run_name}.yaml"),
                    "log_path": str(self.root / f"{run_name}.log"),
                    "gpu_index": len(manifest_runs),
                }
            )
        (self.sweep / "sweep_manifest.json").write_text(
            json.dumps({"sweep_id": "synthetic", "started_at": "now", "runs": manifest_runs}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_calibration_writes_frozen_deterministic_operating_points(self) -> None:
        first = self.root / "first.json"
        second = self.root / "second.json"
        calibrate_sweep(self.sweep, first, min_accepted_episodes=1)
        calibrate_sweep(self.sweep, second, min_accepted_episodes=1)

        self.assertEqual(first.read_bytes(), second.read_bytes())
        payload = json.loads(first.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["primary_chunks"], list(range(1, 11)))
        self.assertEqual(set(payload["runs"]["attention_edl"]["policies"]), {"au", "au_or_eu", "predictive_entropy"})
        self.assertEqual(set(payload["runs"]["attention_softmax"]["policies"]), {"predictive_entropy"})
        self.assertTrue(payload["runs"]["attention_edl"]["policies"]["au"]["default"]["available"])

    def test_calibration_refuses_overwrite(self) -> None:
        output = self.root / "calibration.json"
        calibrate_sweep(self.sweep, output, min_accepted_episodes=1)
        with self.assertRaises(FileExistsError):
            calibrate_sweep(self.sweep, output, min_accepted_episodes=1)


if __name__ == "__main__":
    unittest.main()
