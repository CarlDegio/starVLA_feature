"""Behavioral tests for the causal LIBERO EDL trajectory verifier."""

from __future__ import annotations

import math
import unittest

import torch
from torch.nn import functional as F

from examples.LIBERO.edl_pred.config import EDLConfig, ModelConfig
from examples.LIBERO.edl_pred.model import TrajectoryVerifier, build_verifier


def build_test_model(
    *,
    head: str,
    evidence_activation: str = "softplus",
    dropout: float = 0.0,
    chunk_encoder: str = "token_attention_pool",
) -> TrajectoryVerifier:
    model_config = ModelConfig(
        chunk_encoder=chunk_encoder,
        token_position_encoding="sinusoidal",
        token_embed_dim=8,
        chunk_embed_dim=16,
        attention_heads=2,
        self_attention_layers=1,
        encoder_dropout=dropout,
        lstm_hidden_dim=12,
        lstm_layers=1,
        lstm_dropout=dropout,
        head=head,
    )
    return build_verifier(
        model_config,
        EDLConfig(evidence_activation=evidence_activation, kl_weight=0.0, kl_anneal_epochs=1),
        max_action_tokens=5,
    )


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)
        self.features = torch.arange(2 * 4 * 5 * 2, dtype=torch.float32).reshape(2, 4, 5, 2) / 10.0
        self.token_mask = torch.tensor(
            [
                [
                    [True, True, True, False, False],
                    [True, True, True, True, True],
                    [False, False, False, False, False],
                    [False, False, False, False, False],
                ],
                [
                    [True, True, False, False, False],
                    [True, True, True, True, False],
                    [True, True, True, False, False],
                    [True, False, False, False, False],
                ],
            ]
        )
        self.chunk_lengths = torch.tensor([2, 4])
        self.chunk_mask = self.token_mask.any(dim=-1)

    def test_softmax_and_edl_heads_emit_every_valid_chunk(self) -> None:
        for head in ("softmax", "edl"):
            with self.subTest(head=head):
                output = build_test_model(head=head).eval()(
                    self.features,
                    self.token_mask,
                    self.chunk_lengths,
                )
                self.assertEqual(tuple(output.logits.shape), (2, 4, 2))
                self.assertEqual(tuple(output.probabilities.shape), (2, 4, 2))
                torch.testing.assert_close(
                    output.probabilities.sum(dim=-1)[self.chunk_mask],
                    torch.ones(self.chunk_mask.sum()),
                )
                self._assert_padded_fields_are_zero(output)

                if head == "edl":
                    self.assertEqual(tuple(output.evidence.shape), (2, 4, 2))
                    self.assertEqual(tuple(output.alpha.shape), (2, 4, 2))
                    self.assertTrue(torch.all(output.evidence >= 0))
                    self.assertIsNotNone(output.verifier_au)
                    self.assertIsNotNone(output.verifier_eu)
                    self.assertIsNotNone(output.verifier_total_evidence)
                else:
                    self.assertIsNone(output.evidence)
                    self.assertIsNone(output.alpha)
                    self.assertIsNone(output.verifier_au)
                    self.assertIsNone(output.verifier_eu)
                    self.assertIsNone(output.verifier_total_evidence)

    def test_future_chunks_cannot_change_earlier_predictions(self) -> None:
        model = build_test_model(head="softmax", dropout=0.0, chunk_encoder="mlp_flat").eval()
        changed = self.features.clone()
        changed[:, 2:] = 10000.0

        first = model(self.features, self.token_mask, self.chunk_lengths)
        second = model(changed, self.token_mask, self.chunk_lengths)

        torch.testing.assert_close(first.probabilities[:, :2], second.probabilities[:, :2])
        self.assertFalse(torch.allclose(first.probabilities[1, 2:], second.probabilities[1, 2:]))

    def test_padding_values_cannot_change_valid_predictions_or_pooling_weights(self) -> None:
        model = build_test_model(head="edl", dropout=0.0).eval()
        changed = self.features.clone()
        changed[~self.token_mask] = -10000.0

        first = model(self.features, self.token_mask, self.chunk_lengths)
        second = model(changed, self.token_mask, self.chunk_lengths)

        for name in (
            "logits",
            "probabilities",
            "evidence",
            "alpha",
            "verifier_au",
            "verifier_eu",
            "verifier_total_evidence",
            "pooling_weights",
        ):
            torch.testing.assert_close(getattr(first, name), getattr(second, name))

    def test_edl_activations_clamp_and_compute_uncertainty_diagnostics(self) -> None:
        expected_by_activation = {
            "softplus": F.softplus(torch.tensor([-100.0, 100.0])),
            "relu": torch.tensor([0.0, 100.0]),
            "exp": torch.exp(torch.tensor([-10.0, 10.0])),
        }
        for activation, expected_evidence in expected_by_activation.items():
            with self.subTest(activation=activation):
                model = build_test_model(
                    head="edl",
                    evidence_activation=activation,
                    chunk_encoder="mlp_flat",
                ).eval()
                with torch.no_grad():
                    model.classifier.weight.zero_()
                    model.classifier.bias.copy_(torch.tensor([-100.0, 100.0]))

                output = model(self.features, self.token_mask, self.chunk_lengths)
                valid_count = int(self.chunk_mask.sum())
                expected_alpha = expected_evidence + 1.0
                expected_strength = expected_alpha.sum()
                expected_probabilities = expected_alpha / expected_strength
                expected_au = (
                    expected_probabilities
                    * (torch.digamma(expected_strength + 1.0) - torch.digamma(expected_alpha + 1.0))
                ).sum() / math.log(2.0)

                torch.testing.assert_close(output.evidence[self.chunk_mask], expected_evidence.expand(valid_count, -1))
                torch.testing.assert_close(output.alpha[self.chunk_mask], expected_alpha.expand(valid_count, -1))
                torch.testing.assert_close(
                    output.probabilities[self.chunk_mask],
                    expected_probabilities.expand(valid_count, -1),
                )
                torch.testing.assert_close(
                    output.verifier_au[self.chunk_mask],
                    expected_au.expand(valid_count),
                )
                torch.testing.assert_close(
                    output.verifier_eu[self.chunk_mask],
                    (2.0 / expected_strength).expand(valid_count),
                )
                torch.testing.assert_close(
                    output.verifier_total_evidence[self.chunk_mask],
                    expected_evidence.sum().expand(valid_count),
                )

    def test_rejects_chunk_lengths_inconsistent_with_token_mask(self) -> None:
        model = build_test_model(head="softmax").eval()
        invalid_lengths = {
            "wrong_shape": self.chunk_lengths[:, None],
            "wrong_dtype": self.chunk_lengths.to(torch.float32),
            "zero": torch.tensor([0, 4]),
            "too_large": torch.tensor([2, 5]),
            "does_not_match_valid_chunk_prefix": torch.tensor([3, 4]),
        }
        for name, lengths in invalid_lengths.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "chunk_lengths"):
                    model(self.features, self.token_mask, lengths)

    def _assert_padded_fields_are_zero(self, output: object) -> None:
        for name in ("logits", "probabilities"):
            self.assertTrue(torch.all(getattr(output, name)[~self.chunk_mask] == 0))
        for name in ("evidence", "alpha", "verifier_au", "verifier_eu", "verifier_total_evidence"):
            value = getattr(output, name)
            if value is not None:
                self.assertTrue(torch.all(value[~self.chunk_mask] == 0))
        self.assertIsNotNone(output.pooling_weights)
        self.assertTrue(torch.all(output.pooling_weights[~self.token_mask] == 0))


if __name__ == "__main__":
    unittest.main()
