from __future__ import annotations

import unittest

import numpy as np
import torch

from examples.LIBERO.safe_pred.model import SafeFailureDetector


class SafeFailureDetectorTest(unittest.TestCase):
    def test_lstm_outputs_each_chunk_and_uses_history(self) -> None:
        torch.manual_seed(3)
        model = SafeFailureDetector("lstm", input_dim=2, hidden_dim=8)
        features = torch.tensor([[[0.0, 0.0], [1.0, 1.0]], [[5.0, 5.0], [1.0, 1.0]]])
        mask = torch.ones((2, 2), dtype=torch.bool)

        output = model(features, mask)

        self.assertEqual(tuple(output.logits.shape), (2, 2))
        self.assertIsNotNone(output.failure_probability)
        self.assertFalse(torch.allclose(output.failure_score[0, 1], output.failure_score[1, 1]))

    def test_mlp_scores_are_cumulative_and_padding_is_masked(self) -> None:
        torch.manual_seed(4)
        model = SafeFailureDetector("mlp", input_dim=2, hidden_dim=8)
        features = torch.ones((1, 3, 2))
        mask = torch.tensor([[True, True, False]])

        output = model(features, mask)

        self.assertIsNone(output.failure_probability)
        self.assertGreater(float(output.failure_score[0, 1]), float(output.failure_score[0, 0]))
        self.assertEqual(float(output.failure_score[0, 2]), 0.0)
        np.testing.assert_array_equal(output.chunk_mask.numpy(), mask.numpy())

    def test_rejects_non_prefix_masks(self) -> None:
        model = SafeFailureDetector("lstm", input_dim=2, hidden_dim=8)
        with self.assertRaisesRegex(ValueError, "prefix"):
            model(torch.ones((1, 3, 2)), torch.tensor([[True, False, True]]))


if __name__ == "__main__":
    unittest.main()
