from __future__ import annotations

import pathlib
import tempfile
import unittest

import h5py
import numpy as np

from examples.LIBERO.safe_pred.storage import SafeDiagnosticsDatasetWriter


class SafeDiagnosticsDatasetWriterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = pathlib.Path(self.temp_dir.name) / "safe_libero_goal.hdf5"
        self.metadata = {
            "checkpoint_path": "/checkpoints/qwenfast.pt",
            "server_checkpoint_path": "/checkpoints/qwenfast.pt",
            "task_suite": "libero_goal",
            "collection_id": "safe_seed7",
            "seed": 7,
            "max_steps": 300,
            "action_chunk_size": 8,
        }

    @staticmethod
    def _chunks() -> list[dict]:
        return [
            {
                "chunk_idx": 0,
                "policy_step": 0,
                "env_step": 10,
                "num_action_tokens": 2,
                "action_token_ids": [100, 101],
                "action_token_nll": [0.2, 0.4],
                "action_token_entropy": [1.2, 1.4],
                "action_token_embedding_first": [1.0, 2.0, 3.0],
                "action_token_embedding_last": [4.0, 5.0, 6.0],
                "action_token_embedding_mean": [2.5, 3.5, 4.5],
            },
            {
                "chunk_idx": 1,
                "policy_step": 8,
                "env_step": 18,
                "num_action_tokens": 1,
                "action_token_ids": [102],
                "action_token_nll": [0.6],
                "action_token_entropy": [1.6],
                "action_token_embedding_first": [7.0, 8.0, 9.0],
                "action_token_embedding_last": [10.0, 11.0, 12.0],
                "action_token_embedding_mean": [8.5, 9.5, 10.5],
            },
        ]

    def _append(self, writer: SafeDiagnosticsDatasetWriter) -> None:
        writer.append_episode(
            task_id=2,
            episode_idx=3,
            task_description="put the bowl away",
            success=False,
            executed_steps=300,
            termination_reason="max_steps",
            diagnostic_chunks=self._chunks(),
        )

    def test_round_trip_preserves_token_offsets_and_embeddings(self) -> None:
        with SafeDiagnosticsDatasetWriter(self.path, self.metadata) as writer:
            self._append(writer)

        with h5py.File(self.path, "r") as dataset:
            self.assertEqual(dataset.attrs["schema_version"], "1.0")
            episode = dataset["episodes/task_002_episode_0003"]
            self.assertEqual(int(episode.attrs["success"]), 0)
            self.assertEqual(int(episode.attrs["hidden_dim"]), 3)
            np.testing.assert_array_equal(episode["chunk_idx"][:], [0, 1])
            np.testing.assert_array_equal(episode["num_action_tokens"][:], [2, 1])
            np.testing.assert_array_equal(episode["token_offsets"][:], [0, 2, 3])
            np.testing.assert_array_equal(episode["action_token_ids"][:], [100, 101, 102])
            np.testing.assert_allclose(episode["action_token_nll"][:], [0.2, 0.4, 0.6])
            np.testing.assert_allclose(episode["action_token_entropy"][:], [1.2, 1.4, 1.6])
            np.testing.assert_allclose(episode["embedding_last"][:], [[4, 5, 6], [10, 11, 12]])

    def test_rejects_mismatched_token_lengths(self) -> None:
        chunks = self._chunks()
        chunks[0]["action_token_entropy"] = [1.2]
        with SafeDiagnosticsDatasetWriter(self.path, self.metadata) as writer:
            with self.assertRaisesRegex(ValueError, "token field lengths"):
                writer.append_episode(
                    task_id=0,
                    episode_idx=0,
                    task_description="task",
                    success=True,
                    executed_steps=20,
                    termination_reason="success",
                    diagnostic_chunks=chunks,
                )

    def test_rejects_inconsistent_hidden_dimensions(self) -> None:
        chunks = self._chunks()
        chunks[1]["action_token_embedding_mean"] = [1.0, 2.0]
        with SafeDiagnosticsDatasetWriter(self.path, self.metadata) as writer:
            with self.assertRaisesRegex(ValueError, "hidden dimension"):
                writer.append_episode(
                    task_id=0,
                    episode_idx=0,
                    task_description="task",
                    success=True,
                    executed_steps=20,
                    termination_reason="success",
                    diagnostic_chunks=chunks,
                )

    def test_resume_detects_existing_episode(self) -> None:
        with SafeDiagnosticsDatasetWriter(self.path, self.metadata) as writer:
            self._append(writer)
        with SafeDiagnosticsDatasetWriter(self.path, self.metadata, resume=True) as writer:
            self.assertTrue(writer.has_episode(2, 3))
            with self.assertRaisesRegex(ValueError, "episode already exists"):
                self._append(writer)


if __name__ == "__main__":
    unittest.main()
