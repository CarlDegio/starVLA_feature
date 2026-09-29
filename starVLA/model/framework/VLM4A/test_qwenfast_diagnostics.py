from __future__ import annotations

import types
import unittest

import numpy as np
import torch

from starVLA.model.framework.VLM4A.qwenfast_diagnostics import compute_generation_diagnostics


class QwenFastDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def _generation() -> types.SimpleNamespace:
        scores = (
            torch.tensor([[0.2, -0.1, 0.4, 1.5, 0.3], [0.4, 0.2, 1.1, -0.2, 0.0]]),
            torch.tensor([[0.7, 1.2, -0.3, 0.1, 0.0], [1.4, 0.2, -0.1, 0.3, 0.5]]),
            torch.tensor([[-0.2, 0.1, 0.0, 0.3, 1.7], [0.1, 0.5, 1.3, -0.4, 0.2]]),
        )
        final_hidden = (
            torch.tensor([[[10.0, 11.0, 12.0]], [[20.0, 21.0, 22.0]]]),
            torch.tensor([[[13.0, 14.0, 15.0]], [[23.0, 24.0, 25.0]]]),
            torch.tensor([[[16.0, 17.0, 18.0]], [[26.0, 27.0, 28.0]]]),
        )
        hidden_states = tuple((torch.zeros_like(hidden), hidden) for hidden in final_hidden)
        return types.SimpleNamespace(
            sequences=torch.tensor([[9, 9, 3, 1, 4], [9, 9, 0, 2, 1]]),
            scores=scores,
            hidden_states=hidden_states,
        )

    def test_computes_full_vocabulary_statistics_for_action_tokens(self) -> None:
        generated = self._generation()

        actual = compute_generation_diagnostics(
            generated,
            action_token_min=3,
            action_token_max=4,
            include_token_uncertainty=True,
            include_latent_features=False,
        )

        selected_ids = torch.tensor([3, 4])
        selected_logits = torch.stack((generated.scores[0][0], generated.scores[2][0]))
        log_probs = torch.log_softmax(selected_logits, dim=-1)
        expected_nll = -log_probs.gather(1, selected_ids[:, None]).squeeze(1)
        expected_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

        np.testing.assert_array_equal(actual["num_action_tokens"], [2, 0])
        np.testing.assert_array_equal(actual["action_token_mask"], [[True, True], [False, False]])
        np.testing.assert_array_equal(actual["action_token_ids"], [[3, 4], [-1, -1]])
        np.testing.assert_allclose(actual["action_token_nll"][0], expected_nll.numpy(), rtol=1e-6)
        np.testing.assert_allclose(actual["action_token_entropy"][0], expected_entropy.numpy(), rtol=1e-6)
        np.testing.assert_array_equal(actual["action_token_nll"][1], [0.0, 0.0])
        np.testing.assert_array_equal(actual["action_token_entropy"][1], [0.0, 0.0])

    def test_aggregates_final_hidden_states_at_action_positions(self) -> None:
        actual = compute_generation_diagnostics(
            self._generation(),
            action_token_min=3,
            action_token_max=4,
            include_token_uncertainty=False,
            include_latent_features=True,
        )

        np.testing.assert_array_equal(actual["action_token_embedding_first"][0], [10.0, 11.0, 12.0])
        np.testing.assert_array_equal(actual["action_token_embedding_last"][0], [16.0, 17.0, 18.0])
        np.testing.assert_array_equal(actual["action_token_embedding_mean"][0], [13.0, 14.0, 15.0])
        np.testing.assert_array_equal(actual["action_token_embedding_first"][1], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(actual["action_token_embedding_last"][1], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(actual["action_token_embedding_mean"][1], [0.0, 0.0, 0.0])

    def test_requires_requested_generation_fields(self) -> None:
        generated = self._generation()
        generated.hidden_states = None

        with self.assertRaisesRegex(ValueError, "hidden_states"):
            compute_generation_diagnostics(
                generated,
                action_token_min=3,
                action_token_max=4,
                include_token_uncertainty=False,
                include_latent_features=True,
            )

    def test_rejects_no_requested_diagnostics(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one diagnostic"):
            compute_generation_diagnostics(
                self._generation(),
                action_token_min=3,
                action_token_max=4,
                include_token_uncertainty=False,
                include_latent_features=False,
            )


if __name__ == "__main__":
    unittest.main()
