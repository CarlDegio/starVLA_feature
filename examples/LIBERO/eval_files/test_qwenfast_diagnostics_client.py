from __future__ import annotations

import unittest

import numpy as np

from examples.LIBERO.eval_files.model2libero_interface import ModelClient


class QwenFastDiagnosticsClientTest(unittest.TestCase):
    def test_extracts_first_batch_and_trims_padding(self) -> None:
        data = {
            "num_action_tokens": [2],
            "action_token_ids": [[100, 101, -1]],
            "action_token_nll": [[0.2, 0.4, 0.0]],
            "action_token_entropy": [[1.2, 1.4, 0.0]],
            "action_token_embedding_first": [[1.0, 2.0, 3.0]],
            "action_token_embedding_last": [[4.0, 5.0, 6.0]],
            "action_token_embedding_mean": [[2.5, 3.5, 4.5]],
        }

        actual = ModelClient._extract_qwenfast_chunk_diagnostics(data, chunk_idx=3)

        self.assertEqual(actual["chunk_idx"], 3)
        self.assertEqual(actual["num_action_tokens"], 2)
        self.assertEqual(actual["action_token_ids"], [100, 101])
        np.testing.assert_allclose(actual["action_token_nll"], [0.2, 0.4])
        np.testing.assert_allclose(actual["action_token_entropy"], [1.2, 1.4])
        self.assertEqual(actual["action_token_embedding_last"], [4.0, 5.0, 6.0])

    def test_returns_none_when_no_qwenfast_fields_are_present(self) -> None:
        self.assertIsNone(ModelClient._extract_qwenfast_chunk_diagnostics({}, chunk_idx=0))


if __name__ == "__main__":
    unittest.main()
