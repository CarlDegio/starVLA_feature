"""Optional alpha export matches existing QwenEDL AU/EU, including ragged batches."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from starVLA.model.framework.VLM4A.QwenEDL import Qwenvl_EDL as QwenEDL
from examples.LIBERO.edl_pred_real.export import recompute_uncertainty


class DetailsTests(unittest.TestCase):
    def test_alpha_reconstructs_existing_uncertainty_and_token_order(self):
        policy = QwenEDL.__new__(QwenEDL)
        torch.nn.Module.__init__(policy)
        policy.edl_topk = 3
        policy.edl_evidence_fn = "softplus"
        policy.qwen_vl_interface = SimpleNamespace(_ACTION_TOKEN_MIN=2, _ACTION_TOKEN_MAX=5)
        generator = torch.Generator().manual_seed(17)
        generated = SimpleNamespace(sequences=torch.tensor([[0, 2, 7, 3], [0, 7, 5, 7]]),
            scores=tuple(torch.randn(2, 8, generator=generator) for _ in range(3)))
        expected = policy._compute_generation_uncertainty(generated)
        details = policy._generation_edl_details(generated)
        self.assertEqual(details["num_action_tokens"].tolist(), [2, 1])
        self.assertEqual(details["action_token_ids"].tolist(), [[2, 3], [5, -1]])
        self.assertEqual(details["action_token_mask"].tolist(), [[True, True], [True, False]])
        for row, count in enumerate((2, 1)):
            alpha = details["action_token_topk_alpha"][row, :count]
            au, eu = recompute_uncertainty(alpha)
            np.testing.assert_allclose(au, expected[6][row, :count], rtol=1e-4, atol=2e-6)
            np.testing.assert_allclose(eu, expected[7][row, :count], rtol=1e-4, atol=2e-6)
        self.assertTrue(np.isnan(details["action_token_topk_alpha"][1, 1]).all())
        self.assertTrue(all(2 <= value <= 5 for value in details["action_token_topk_ids"][0].flat))


if __name__ == "__main__":
    unittest.main()
