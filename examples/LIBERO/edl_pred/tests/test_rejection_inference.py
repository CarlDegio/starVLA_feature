from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from examples.LIBERO.edl_pred.config import EDLConfig, ModelConfig
from examples.LIBERO.edl_pred.dataset import TrajectoryDataset, index_suite_episodes
from examples.LIBERO.edl_pred.model import build_verifier
from examples.LIBERO.edl_pred.rejection_inference import load_frozen_verifier, predict_frozen_datasets
from examples.LIBERO.edl_pred.tests.helpers import write_uncertainty_hdf5


class RejectionInferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_path = write_uncertainty_hdf5(
            self.root / "test.hdf5",
            [("episode_a", 0, [2, 1]), ("episode_b", 1, [1, 2])],
        )
        self.model_config = ModelConfig(
            chunk_encoder="token_attention_pool",
            token_position_encoding="sinusoidal",
            token_embed_dim=8,
            chunk_embed_dim=8,
            attention_heads=2,
            self_attention_layers=1,
            encoder_dropout=0.0,
            lstm_hidden_dim=8,
            lstm_layers=1,
            lstm_dropout=0.0,
            head="edl",
        )
        self.edl_config = EDLConfig("softplus", 0.001, 20)
        torch.manual_seed(3)
        self.model = build_verifier(self.model_config, self.edl_config, max_action_tokens=2)
        self.checkpoint = self.root / "best.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "resolved_config": {
                    "model": asdict(self.model_config),
                    "edl": asdict(self.edl_config),
                },
                "max_action_tokens": 2,
                "split_manifest_identity": "synthetic",
            },
            self.checkpoint,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_frozen_inference_matches_direct_model_output(self) -> None:
        frozen = load_frozen_verifier(self.checkpoint, device="cpu")
        records = predict_frozen_datasets(
            frozen,
            {"libero_goal": self.dataset_path},
            batch_size=2,
        )
        refs, _ = index_suite_episodes("libero_goal", self.dataset_path)
        dataset = TrajectoryDataset(refs, max_action_tokens=2)
        try:
            sample = dataset[0]
            with torch.inference_mode():
                direct = self.model(
                    torch.from_numpy(sample.features).unsqueeze(0),
                    torch.from_numpy(sample.token_mask).unsqueeze(0),
                    torch.tensor([sample.features.shape[0]]),
                )
        finally:
            dataset.close()

        self.assertEqual(len(records), 2)
        np.testing.assert_allclose(
            records[0].class_probabilities,
            direct.probabilities[0].numpy(),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertIsNotNone(records[0].verifier_au)

    def test_test_dataset_token_count_must_fit_checkpoint(self) -> None:
        too_wide = write_uncertainty_hdf5(
            self.root / "wide.hdf5",
            [("episode_a", 0, [3])],
        )
        frozen = load_frozen_verifier(self.checkpoint, device="cpu")
        with self.assertRaisesRegex(ValueError, "max_action_tokens"):
            predict_frozen_datasets(frozen, {"libero_goal": too_wide}, batch_size=1)


if __name__ == "__main__":
    unittest.main()
