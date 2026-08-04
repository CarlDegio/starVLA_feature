"""Numerical and validation tests for masked verifier objectives."""

from __future__ import annotations

from dataclasses import replace
import math
import unittest

import torch

from examples.LIBERO.edl_pred.config import EDLConfig
from examples.LIBERO.edl_pred.losses import compute_verifier_loss
from examples.LIBERO.edl_pred.model import VerifierOutput


def make_edl_config(*, kl_weight: float = 0.0, kl_anneal_epochs: int = 1) -> EDLConfig:
    return EDLConfig(evidence_activation="softplus", kl_weight=kl_weight, kl_anneal_epochs=kl_anneal_epochs)


def make_softmax_output(logits: torch.Tensor) -> VerifierOutput:
    return VerifierOutput(
        logits=logits,
        probabilities=torch.softmax(logits, dim=-1),
        evidence=None,
        alpha=None,
        verifier_au=None,
        verifier_eu=None,
        verifier_total_evidence=None,
        pooling_weights=None,
    )


def make_edl_output(alpha: torch.Tensor) -> VerifierOutput:
    strength = alpha.sum(dim=-1, keepdim=True)
    valid_alpha = alpha > 0.0
    return VerifierOutput(
        logits=torch.where(valid_alpha, torch.log(alpha), torch.zeros_like(alpha)),
        probabilities=torch.where(strength > 0.0, alpha / strength, torch.zeros_like(alpha)),
        evidence=alpha - 1.0,
        alpha=alpha,
        verifier_au=None,
        verifier_eu=None,
        verifier_total_evidence=None,
        pooling_weights=None,
    )


def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """Independently compute KL(Dirichlet(alpha) || Dirichlet(1, 1))."""
    strength = alpha.sum(dim=-1)
    return (
        torch.lgamma(strength)
        - torch.lgamma(alpha).sum(dim=-1)
        - torch.lgamma(torch.tensor(2.0, dtype=alpha.dtype, device=alpha.device))
        + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(strength).unsqueeze(-1))).sum(dim=-1)
    )


class LossTest(unittest.TestCase):
    def test_softmax_loss_means_chunks_inside_each_episode_first(self) -> None:
        # Removing per-episode reduction would overweight the second episode.
        logits = torch.log(torch.tensor([[[0.8, 0.2], [0.2, 0.8]], [[0.4, 0.6], [0.1, 0.9]]]))
        result = compute_verifier_loss(
            make_softmax_output(logits),
            labels=torch.tensor([0, 1]),
            chunk_mask=torch.tensor([[True, False], [True, True]]),
            head="softmax",
            edl_config=make_edl_config(),
            epoch=0,
        )

        expected_ep0 = -math.log(0.8)
        expected_ep1 = (-math.log(0.6) - math.log(0.9)) / 2.0
        expected = torch.tensor((expected_ep0 + expected_ep1) / 2.0)
        torch.testing.assert_close(result.total, expected)
        torch.testing.assert_close(result.data, expected)
        torch.testing.assert_close(result.kl, torch.zeros_like(expected))
        torch.testing.assert_close(result.weighted_kl, torch.zeros_like(expected))

    def test_optional_inverse_frequency_weights_apply_after_episode_means(self) -> None:
        # Applying weights per chunk or before episode averaging changes this value.
        logits = torch.log(
            torch.tensor(
                [
                    [[0.5, 0.5], [0.5, 0.5]],
                    [[0.25, 0.75], [0.25, 0.75]],
                    [[0.1, 0.9], [0.1, 0.9]],
                ]
            )
        )
        inverse_frequency = torch.tensor([1.5, 0.75])  # Three episodes: one class 0 and two class 1.
        result = compute_verifier_loss(
            make_softmax_output(logits),
            labels=torch.tensor([0, 1, 1]),
            chunk_mask=torch.ones(3, 2, dtype=torch.bool),
            head="softmax",
            edl_config=make_edl_config(),
            epoch=0,
            class_weights=inverse_frequency,
        )

        expected = torch.tensor((1.5 * -math.log(0.5) + 0.75 * -math.log(0.75) + 0.75 * -math.log(0.9)) / 3.0)
        torch.testing.assert_close(result.total, expected)

    def test_integer_labels_are_normalized_for_cross_entropy(self) -> None:
        # Passing an accepted int32 label tensor directly to cross entropy errors.
        result = compute_verifier_loss(
            make_softmax_output(torch.log(torch.tensor([[[0.8, 0.2]]]))),
            labels=torch.tensor([0], dtype=torch.int32),
            chunk_mask=torch.tensor([[True]]),
            head="softmax",
            edl_config=make_edl_config(),
            epoch=0,
        )
        torch.testing.assert_close(result.total, torch.tensor(-math.log(0.8)))

    def test_padded_values_do_not_affect_softmax_loss(self) -> None:
        # Including masked chunk values in cross entropy must change this loss.
        logits = torch.log(torch.tensor([[[0.8, 0.2], [0.4, 0.6]], [[0.1, 0.9], [0.7, 0.3]]]))
        mask = torch.tensor([[True, False], [True, True]])
        kwargs = dict(
            labels=torch.tensor([0, 1]),
            chunk_mask=mask,
            head="softmax",
            edl_config=make_edl_config(),
            epoch=0,
        )
        first = compute_verifier_loss(make_softmax_output(logits), **kwargs)
        changed = logits.clone()
        changed[~mask] = torch.tensor([[-1_000_000.0, 1_000_000.0]])
        second = compute_verifier_loss(make_softmax_output(changed), **kwargs)
        torch.testing.assert_close(first.total, second.total)

    def test_padded_values_do_not_affect_edl_loss(self) -> None:
        # Evaluating padded alpha values would spuriously change data and KL.
        alpha = torch.tensor([[[2.0, 4.0], [0.0, 0.0]], [[3.0, 2.0], [2.0, 3.0]]])
        mask = torch.tensor([[True, False], [True, True]])
        kwargs = dict(
            labels=torch.tensor([1, 0]),
            chunk_mask=mask,
            head="edl",
            edl_config=make_edl_config(kl_weight=0.3, kl_anneal_epochs=1),
            epoch=1,
        )
        first = compute_verifier_loss(make_edl_output(alpha), **kwargs)
        changed = alpha.clone()
        changed[~mask] = torch.tensor([[1_000_000.0, 2_000_000.0]])
        second = compute_verifier_loss(make_edl_output(changed), **kwargs)
        torch.testing.assert_close(first.total, second.total)
        torch.testing.assert_close(first.data, second.data)
        torch.testing.assert_close(first.kl, second.kl)

    def test_edl_data_loss_matches_digamma_expected_bce(self) -> None:
        alpha = torch.tensor([[[2.0, 5.0]]])
        result = compute_verifier_loss(
            make_edl_output(alpha),
            labels=torch.tensor([1]),
            chunk_mask=torch.tensor([[True]]),
            head="edl",
            edl_config=make_edl_config(),
            epoch=0,
        )
        expected = torch.digamma(torch.tensor(7.0)) - torch.digamma(torch.tensor(5.0))
        torch.testing.assert_close(result.data, expected)
        torch.testing.assert_close(result.total, expected)

    def test_edl_kl_uses_adjusted_alpha_for_target_class(self) -> None:
        # A KL over the original alpha would penalize the target class evidence.
        alpha = torch.tensor([[[3.0, 11.0]]])
        result = compute_verifier_loss(
            make_edl_output(alpha),
            labels=torch.tensor([1]),
            chunk_mask=torch.tensor([[True]]),
            head="edl",
            edl_config=make_edl_config(kl_weight=1.0, kl_anneal_epochs=0),
            epoch=0,
        )
        expected_kl = dirichlet_kl_to_uniform(torch.tensor([[3.0, 1.0]])).squeeze(0)
        torch.testing.assert_close(result.kl, expected_kl)
        torch.testing.assert_close(result.weighted_kl, expected_kl)
        torch.testing.assert_close(result.total, result.data + expected_kl)

    def test_zero_kl_weight_excludes_regularizer_but_reports_raw_kl(self) -> None:
        alpha = torch.tensor([[[4.0, 2.0]]])
        result = compute_verifier_loss(
            make_edl_output(alpha),
            labels=torch.tensor([0]),
            chunk_mask=torch.tensor([[True]]),
            head="edl",
            edl_config=make_edl_config(kl_weight=0.0),
            epoch=0,
        )
        self.assertGreater(result.kl.item(), 0.0)
        torch.testing.assert_close(result.weighted_kl, torch.zeros_like(result.kl))
        torch.testing.assert_close(result.total, result.data)

    def test_edl_kl_weight_linearly_anneals_by_epoch(self) -> None:
        alpha = torch.tensor([[[4.0, 2.0]]])
        result = compute_verifier_loss(
            make_edl_output(alpha),
            labels=torch.tensor([0]),
            chunk_mask=torch.tensor([[True]]),
            head="edl",
            edl_config=make_edl_config(kl_weight=0.4, kl_anneal_epochs=4),
            epoch=2,
        )
        self.assertEqual(result.annealing_coefficient, 0.5)
        torch.testing.assert_close(result.weighted_kl, result.kl * 0.2)
        torch.testing.assert_close(result.total, result.data + result.kl * 0.2)

    def test_nonpositive_kl_anneal_duration_applies_full_weight_immediately(self) -> None:
        alpha = torch.tensor([[[4.0, 2.0]]])
        result = compute_verifier_loss(
            make_edl_output(alpha),
            labels=torch.tensor([0]),
            chunk_mask=torch.tensor([[True]]),
            head="edl",
            edl_config=make_edl_config(kl_weight=0.4, kl_anneal_epochs=-1),
            epoch=0,
        )
        self.assertEqual(result.annealing_coefficient, 1.0)
        torch.testing.assert_close(result.weighted_kl, result.kl * 0.4)

    def test_edl_loss_has_finite_gradients(self) -> None:
        alpha = torch.tensor([[[2.0, 4.0], [3.0, 2.0]]], requires_grad=True)
        result = compute_verifier_loss(
            make_edl_output(alpha),
            labels=torch.tensor([1]),
            chunk_mask=torch.tensor([[True, True]]),
            head="edl",
            edl_config=make_edl_config(kl_weight=0.2, kl_anneal_epochs=1),
            epoch=1,
        )
        result.total.backward()
        self.assertIsNotNone(alpha.grad)
        self.assertTrue(torch.isfinite(alpha.grad).all())
        self.assertGreater(alpha.grad.abs().sum().item(), 0.0)

    def test_rejects_invalid_shapes_labels_masks_and_empty_episodes(self) -> None:
        output = make_softmax_output(torch.zeros(2, 2, 2))
        cases = {
            "labels_shape": dict(labels=torch.tensor([[0, 1]]), chunk_mask=torch.ones(2, 2, dtype=torch.bool)),
            "labels_values": dict(labels=torch.tensor([0, 2]), chunk_mask=torch.ones(2, 2, dtype=torch.bool)),
            "mask_shape": dict(labels=torch.tensor([0, 1]), chunk_mask=torch.ones(2, 3, dtype=torch.bool)),
            "mask_dtype": dict(labels=torch.tensor([0, 1]), chunk_mask=torch.ones(2, 2)),
            "empty_episode": dict(labels=torch.tensor([0, 1]), chunk_mask=torch.tensor([[True, True], [False, False]])),
        }
        for name, kwargs in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    compute_verifier_loss(
                        output,
                        head="softmax",
                        edl_config=make_edl_config(),
                        epoch=0,
                        **kwargs,
                    )

    def test_rejects_an_empty_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch"):
            compute_verifier_loss(
                make_softmax_output(torch.empty(0, 1, 2)),
                labels=torch.empty(0, dtype=torch.long),
                chunk_mask=torch.empty(0, 1, dtype=torch.bool),
                head="softmax",
                edl_config=make_edl_config(),
                epoch=0,
            )

    def test_rejects_missing_head_specific_fields_and_mismatched_output_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha"):
            compute_verifier_loss(
                make_softmax_output(torch.zeros(1, 1, 2)),
                labels=torch.tensor([0]),
                chunk_mask=torch.tensor([[True]]),
                head="edl",
                edl_config=make_edl_config(),
                epoch=0,
            )
        malformed = make_softmax_output(torch.zeros(1, 1, 3))
        with self.assertRaisesRegex(ValueError, "logits"):
            compute_verifier_loss(
                malformed,
                labels=torch.tensor([0]),
                chunk_mask=torch.tensor([[True]]),
                head="softmax",
                edl_config=make_edl_config(),
                epoch=0,
            )

    def test_rejects_nonfinite_inputs(self) -> None:
        logits = torch.zeros(1, 1, 2)
        logits[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_verifier_loss(
                make_softmax_output(logits),
                labels=torch.tensor([0]),
                chunk_mask=torch.tensor([[True]]),
                head="softmax",
                edl_config=make_edl_config(),
                epoch=0,
            )

    def test_rejects_cpu_probabilities_with_a_different_dtype_than_logits(self) -> None:
        output = make_softmax_output(torch.zeros(1, 1, 2, dtype=torch.float32))
        output = replace(output, probabilities=output.probabilities.to(torch.float64))
        with self.assertRaisesRegex(ValueError, "probabilities.*dtype"):
            compute_verifier_loss(
                output,
                labels=torch.tensor([0]),
                chunk_mask=torch.tensor([[True]]),
                head="softmax",
                edl_config=make_edl_config(),
                epoch=0,
            )

    def test_rejects_cpu_edl_alpha_with_a_different_dtype_than_logits(self) -> None:
        output = make_edl_output(torch.tensor([[[2.0, 3.0]]], dtype=torch.float32))
        output = replace(output, alpha=output.alpha.to(torch.float64))
        with self.assertRaisesRegex(ValueError, "alpha.*dtype"):
            compute_verifier_loss(
                output,
                labels=torch.tensor([1]),
                chunk_mask=torch.tensor([[True]]),
                head="edl",
                edl_config=make_edl_config(),
                epoch=0,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_rejects_output_tensors_on_a_different_device_than_logits(self) -> None:
        softmax_output = make_softmax_output(torch.zeros(1, 1, 2))
        with self.subTest(field="probabilities"):
            with self.assertRaisesRegex(ValueError, "probabilities.*device"):
                compute_verifier_loss(
                    replace(softmax_output, probabilities=softmax_output.probabilities.cuda()),
                    labels=torch.tensor([0]),
                    chunk_mask=torch.tensor([[True]]),
                    head="softmax",
                    edl_config=make_edl_config(),
                    epoch=0,
                )

        edl_output = make_edl_output(torch.tensor([[[2.0, 3.0]]]))
        with self.subTest(field="alpha"):
            with self.assertRaisesRegex(ValueError, "alpha.*device"):
                compute_verifier_loss(
                    replace(edl_output, alpha=edl_output.alpha.cuda()),
                    labels=torch.tensor([1]),
                    chunk_mask=torch.tensor([[True]]),
                    head="edl",
                    edl_config=make_edl_config(),
                    epoch=0,
                )

    def test_rejects_invalid_epoch_and_class_weights(self) -> None:
        output = make_softmax_output(torch.zeros(1, 1, 2))
        kwargs = dict(
            output=output,
            labels=torch.tensor([0]),
            chunk_mask=torch.tensor([[True]]),
            head="softmax",
            edl_config=make_edl_config(),
        )
        for name, epoch, weights in (
            ("negative_epoch", -1, None),
            ("weight_shape", 0, torch.ones(1)),
            ("negative_weight", 0, torch.tensor([1.0, -1.0])),
            ("nonfinite_weight", 0, torch.tensor([1.0, float("inf")])),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    compute_verifier_loss(**kwargs, epoch=epoch, class_weights=weights)


if __name__ == "__main__":
    unittest.main()
