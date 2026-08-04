"""Behavioral tests for masked EDL action-chunk encoders."""

from __future__ import annotations

import unittest

import torch

from examples.LIBERO.edl_pred.config import ModelConfig
from examples.LIBERO.edl_pred.encoders import build_chunk_encoder, build_token_position


ENCODER_NAMES = ("mlp_flat", "token_attention_pool", "token_self_attention")
POSITION_NAMES = ("none", "sinusoidal", "learned")


def build_test_encoder(
    name: str,
    *,
    position: str = "sinusoidal",
    dropout: float = 0.0,
) -> torch.nn.Module:
    config = ModelConfig(
        chunk_encoder=name,
        token_position_encoding=position,
        token_embed_dim=8,
        chunk_embed_dim=16,
        attention_heads=2,
        self_attention_layers=1,
        encoder_dropout=dropout,
        lstm_hidden_dim=16,
        lstm_layers=1,
        lstm_dropout=0.0,
        head="edl",
    )
    return build_chunk_encoder(config, max_action_tokens=5)


class EncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.features = torch.arange(2 * 3 * 5 * 2, dtype=torch.float32).reshape(2, 3, 5, 2)
        self.token_mask = torch.tensor(
            [
                [[True, True, True, False, False], [True, True, True, True, True], [True, False, False, False, False]],
                [[True, True, False, False, False], [True, True, True, True, False], [True, True, True, False, False]],
            ]
        )

    def test_all_encoders_return_one_embedding_per_chunk(self) -> None:
        for name in ENCODER_NAMES:
            encoder = build_test_encoder(name, position="sinusoidal")
            output = encoder(self.features, self.token_mask)
            self.assertEqual(tuple(output.embedding.shape), (2, 3, 16))
            if name == "mlp_flat":
                self.assertIsNone(output.pooling_weights)
            else:
                self.assertEqual(tuple(output.pooling_weights.shape), (2, 3, 5))

    def test_padding_values_cannot_change_embedding(self) -> None:
        changed = self.features.clone()
        changed[~self.token_mask] = 9999.0
        for name in ENCODER_NAMES:
            encoder = build_test_encoder(name, dropout=0.0).eval()
            torch.testing.assert_close(
                encoder(self.features, self.token_mask).embedding,
                encoder(changed, self.token_mask).embedding,
            )

    def test_pooling_weights_are_zero_on_padding_and_sum_on_valid_tokens(self) -> None:
        output = build_test_encoder("token_attention_pool")(self.features, self.token_mask)
        self.assertTrue(torch.all(output.pooling_weights[~self.token_mask] == 0))
        torch.testing.assert_close(output.pooling_weights.sum(-1), torch.ones(2, 3))

    def test_pooling_weights_remain_normalized_when_attention_dropout_zeros_a_row(self) -> None:
        torch.manual_seed(19)
        output = build_test_encoder("token_attention_pool", dropout=0.99)(
            self.features,
            self.token_mask,
        )
        self.assertTrue(torch.isfinite(output.pooling_weights).all())
        torch.testing.assert_close(output.pooling_weights.sum(-1), torch.ones(2, 3))

    def test_all_masked_episode_padding_chunk_stays_finite_and_zero(self) -> None:
        token_mask = self.token_mask.clone()
        token_mask[0, 2] = False
        for name in ENCODER_NAMES:
            output = build_test_encoder(name)(self.features, token_mask)
            self.assertTrue(torch.isfinite(output.embedding).all())
            self.assertTrue(torch.all(output.embedding[0, 2] == 0))
            if output.pooling_weights is not None:
                self.assertTrue(torch.all(output.pooling_weights[0, 2] == 0))

    def test_all_false_batch_returns_exact_zeros_without_attention_calls(self) -> None:
        token_mask = torch.zeros_like(self.token_mask)
        for name in ENCODER_NAMES:
            encoder = build_test_encoder(name)
            calls: list[str] = []
            handles = []
            if name != "mlp_flat":
                handles.append(
                    encoder.pool.attention.register_forward_hook(
                        lambda *_: calls.append("pool"),
                    )
                )
            if name == "token_self_attention":
                handles.append(
                    encoder.self_attention.register_forward_hook(
                        lambda *_: calls.append("self_attention"),
                    )
                )
            try:
                output = encoder(self.features, token_mask)
            finally:
                for handle in handles:
                    handle.remove()

            self.assertTrue(torch.all(output.embedding == 0))
            if output.pooling_weights is not None:
                self.assertTrue(torch.all(output.pooling_weights == 0))
            self.assertEqual(calls, [])

    def test_all_position_modes_are_accepted_by_attention_encoder(self) -> None:
        for position in POSITION_NAMES:
            output = build_test_encoder("token_attention_pool", position=position)(
                self.features,
                self.token_mask,
            )
            self.assertEqual(tuple(output.embedding.shape), (2, 3, 16))

    def test_enabled_positions_make_token_order_observable(self) -> None:
        features = self.features.clone()
        features[:, :, :3] = features[:, :, torch.tensor([2, 1, 0])]
        full_mask = torch.ones_like(self.token_mask)
        for name in ("token_attention_pool", "token_self_attention"):
            for position in ("sinusoidal", "learned"):
                torch.manual_seed(11)
                encoder = build_test_encoder(name, position=position).eval()
                self.assertFalse(
                    torch.allclose(
                        encoder(self.features, full_mask).embedding,
                        encoder(features, full_mask).embedding,
                    )
                )

    def test_mlp_flat_accepts_but_ignores_position_setting(self) -> None:
        torch.manual_seed(13)
        baseline = build_test_encoder("mlp_flat", position="none").eval()
        expected = baseline(self.features, self.token_mask).embedding
        for position in ("sinusoidal", "learned"):
            torch.manual_seed(13)
            actual = build_test_encoder("mlp_flat", position=position).eval()(
                self.features,
                self.token_mask,
            ).embedding
            torch.testing.assert_close(actual, expected)

    def test_none_position_is_identity_and_other_positions_change_tokens(self) -> None:
        tokens = torch.zeros(2, 5, 8)
        self.assertIs(build_token_position("none", 5, 8)(tokens), tokens)
        self.assertFalse(torch.equal(build_token_position("sinusoidal", 5, 8)(tokens), tokens))
        torch.manual_seed(17)
        self.assertFalse(torch.equal(build_token_position("learned", 5, 8)(tokens), tokens))

    def test_position_modules_preserve_low_precision_dtype_with_odd_dimension(self) -> None:
        for kind in ("sinusoidal", "learned"):
            for dtype in (torch.float16, torch.bfloat16):
                with self.subTest(kind=kind, dtype=dtype):
                    position = build_token_position(kind, max_tokens=5, dim=7).to("cpu")
                    tokens = torch.zeros(2, 5, 7, dtype=dtype)
                    output = position(tokens)
                    self.assertEqual(position.max_tokens, 5)
                    self.assertEqual(output.shape, tokens.shape)
                    self.assertEqual(output.dtype, dtype)
                    self.assertEqual(output.device, tokens.device)

    def test_position_modules_reject_token_sequences_longer_than_maximum(self) -> None:
        for kind in ("sinusoidal", "learned"):
            with self.subTest(kind=kind):
                position = build_token_position(kind, max_tokens=5, dim=7)
                with self.assertRaisesRegex(ValueError, "max_tokens=5"):
                    position(torch.zeros(2, 6, 7))


if __name__ == "__main__":
    unittest.main()
