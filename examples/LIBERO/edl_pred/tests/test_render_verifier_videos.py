from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from examples.LIBERO.edl_pred.render_verifier_videos import (
    OnlineEDLVerifier,
    format_overlay_lines,
    overlay_verifier_text,
    validate_server_checkpoint,
)


LAUNCHER = Path(__file__).resolve().parents[1] / "render_verifier_videos.sh"


class _FakeVerifierModel:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def __call__(
        self,
        features: torch.Tensor,
        token_mask: torch.Tensor,
        chunk_lengths: torch.Tensor,
    ) -> SimpleNamespace:
        self.calls.append((features.clone(), token_mask.clone(), chunk_lengths.clone()))
        chunk_count = features.shape[1]
        probabilities = torch.zeros((1, chunk_count, 2), dtype=torch.float32)
        probabilities[..., 0] = 0.65
        probabilities[..., 1] = 0.35
        verifier_au = torch.full((1, chunk_count), 0.42, dtype=torch.float32)
        return SimpleNamespace(probabilities=probabilities, verifier_au=verifier_au)


class _FakeFrozenVerifier:
    def __init__(self) -> None:
        self.model = _FakeVerifierModel()
        self.model_config = SimpleNamespace(head="edl")
        self.max_action_tokens = 4
        self.device = torch.device("cpu")


def _uncertainty(au: list[float], eu: list[float]) -> dict[str, list[float]]:
    return {
        "action_token_aleatoric_uncertainty": au,
        "action_token_epistemic_uncertainty": eu,
    }


class OnlineEDLVerifierTest(unittest.TestCase):
    def test_accumulates_chunks_and_returns_last_chunk_prediction(self) -> None:
        frozen = _FakeFrozenVerifier()
        online = OnlineEDLVerifier(frozen)

        first = online.append_chunk(_uncertainty([0.1, 0.2], [0.3, 0.4]))
        second = online.append_chunk(_uncertainty([0.5], [0.6]))

        self.assertAlmostEqual(first.au, 0.42)
        self.assertAlmostEqual(second.failure_probability, 0.65)
        self.assertAlmostEqual(second.success_probability, 0.35)
        features, mask, lengths = frozen.model.calls[-1]
        self.assertEqual(tuple(features.shape), (1, 2, 4, 2))
        self.assertEqual(mask.tolist(), [[[True, True, False, False], [True, False, False, False]]])
        self.assertEqual(lengths.tolist(), [2])
        torch.testing.assert_close(features[0, 0, :2, 0], torch.tensor([0.1, 0.2]))
        torch.testing.assert_close(features[0, 0, :2, 1], torch.tensor([0.3, 0.4]))

    def test_reset_discards_previous_episode_history(self) -> None:
        frozen = _FakeFrozenVerifier()
        online = OnlineEDLVerifier(frozen)
        online.append_chunk(_uncertainty([0.1], [0.2]))
        online.reset()
        online.append_chunk(_uncertainty([0.3], [0.4]))

        features, _, lengths = frozen.model.calls[-1]
        self.assertEqual(tuple(features.shape), (1, 1, 4, 2))
        self.assertEqual(lengths.tolist(), [1])
        torch.testing.assert_close(features[0, 0, 0], torch.tensor([0.3, 0.4]))

    def test_rejects_mismatched_or_too_long_token_sequences(self) -> None:
        online = OnlineEDLVerifier(_FakeFrozenVerifier())
        with self.assertRaisesRegex(ValueError, "same non-zero length"):
            online.append_chunk(_uncertainty([0.1], [0.2, 0.3]))
        with self.assertRaisesRegex(ValueError, "max_action_tokens"):
            online.append_chunk(_uncertainty([0.1] * 5, [0.2] * 5))


class VideoOverlayTest(unittest.TestCase):
    def test_formats_requested_three_values(self) -> None:
        display = SimpleNamespace(
            au=0.12345,
            failure_probability=0.67891,
            success_probability=0.32109,
        )
        self.assertEqual(
            format_overlay_lines(display),
            ("AU: 0.123", "Failure: 0.679", "Success: 0.321"),
        )

    def test_draws_black_text_in_top_right_without_mutating_input(self) -> None:
        frame = np.full((256, 256, 3), 255, dtype=np.uint8)
        original = frame.copy()
        display = SimpleNamespace(
            au=0.42,
            failure_probability=0.65,
            success_probability=0.35,
        )

        annotated = overlay_verifier_text(frame, display)

        np.testing.assert_array_equal(frame, original)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertEqual(annotated.dtype, np.uint8)
        self.assertTrue(np.any(annotated[:100, 100:] < 64))
        np.testing.assert_array_equal(annotated[:, :80], original[:, :80])


class CheckpointValidationTest(unittest.TestCase):
    def test_accepts_equivalent_resolved_paths(self) -> None:
        expected = Path("/tmp/checkpoints/../checkpoints/policy.pt")
        metadata = {"ckpt_path": "/tmp/checkpoints/policy.pt"}
        self.assertEqual(
            validate_server_checkpoint(metadata, expected),
            Path("/tmp/checkpoints/policy.pt"),
        )

    def test_rejects_missing_or_different_server_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not include ckpt_path"):
            validate_server_checkpoint({}, "/tmp/policy.pt")
        with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
            validate_server_checkpoint(
                {"ckpt_path": "/tmp/other.pt"},
                "/tmp/policy.pt",
            )


class VideoLauncherTest(unittest.TestCase):
    def test_launcher_defaults_to_requested_policy_verifier_and_five_videos(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("qwen3fast_libero_all_edl_1e-2", source)
        self.assertIn("all_mlp_flat_edl_seed7/checkpoints/best.pt", source)
        self.assertIn('TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_10}"', source)
        self.assertIn('NUM_VIDEOS="${NUM_VIDEOS:-5}"', source)
        self.assertIn("-m examples.LIBERO.edl_pred.render_verifier_videos", source)

    def test_launcher_validates_task_id_before_formatting_output_directory(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertLess(
            source.index('if [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]]'),
            source.index('OUTPUT_DIR="'),
        )

    def test_launcher_has_valid_bash_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


if __name__ == "__main__":
    unittest.main()
