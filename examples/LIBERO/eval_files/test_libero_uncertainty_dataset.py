import pathlib
import tempfile
import unittest

import h5py
import numpy as np

from examples.LIBERO.eval_files.libero_uncertainty_dataset import LiberoUncertaintyDatasetWriter


class LiberoUncertaintyDatasetWriterTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.dataset_path = pathlib.Path(self.temp_dir.name) / "libero_spatial.hdf5"
        self.metadata = {
            "checkpoint_path": "/checkpoints/model.pt",
            "server_checkpoint_path": "/checkpoints/model.pt",
            "task_suite": "libero_spatial",
            "seed": 7,
            "max_steps": 220,
            "action_chunk_size": 8,
        }

    @staticmethod
    def _chunks():
        return [
            {
                "chunk_idx": 0,
                "policy_step": 0,
                "env_step": 10,
                "num_tokens": 3,
                "num_action_tokens": 2,
                "action_token_evidence": [3.5, 8.0],
                "action_token_aleatoric_uncertainty": [0.7, 0.2],
                "action_token_epistemic_uncertainty": [0.3, 0.1],
                "action_token_confidence": [0.4, 0.8],
                "action_token_rank": [1.0, 2.0],
                "low_evidence_threshold": 4.0,
                "low_evidence_count": 1,
            },
            {
                "chunk_idx": 1,
                "policy_step": 8,
                "env_step": 18,
                "num_tokens": 2,
                "num_action_tokens": 1,
                "action_token_evidence": [5.0],
                "action_token_aleatoric_uncertainty": [0.5],
                "action_token_epistemic_uncertainty": [0.25],
                "action_token_confidence": [0.6],
                "action_token_rank": [1.0],
            },
        ]

    def _append_episode(self, writer):
        return writer.append_episode(
            task_id=2,
            episode_idx=3,
            task_description="pick up the bowl",
            success=True,
            executed_steps=9,
            termination_reason="success",
            uncertainty_chunks=self._chunks(),
        )

    def test_round_trip_flattens_variable_length_action_tokens(self):
        with LiberoUncertaintyDatasetWriter(self.dataset_path, self.metadata) as writer:
            self.assertTrue(self._append_episode(writer))

        with h5py.File(self.dataset_path, "r") as dataset:
            self.assertEqual(dataset.attrs["schema_version"], "1.0")
            self.assertEqual(dataset.attrs["task_suite"], "libero_spatial")
            episode = dataset["episodes/task_002_episode_0003"]
            self.assertEqual(int(episode.attrs["success"]), 1)
            self.assertEqual(episode.attrs["task_description"], "pick up the bowl")
            self.assertEqual(int(episode.attrs["num_chunks"]), 2)
            np.testing.assert_array_equal(episode["chunk_idx"][:], [0, 1])
            np.testing.assert_array_equal(episode["num_generated_tokens"][:], [3, 2])
            np.testing.assert_array_equal(episode["num_action_tokens"][:], [2, 1])
            np.testing.assert_array_equal(episode["token_offsets"][:], [0, 2, 3])
            np.testing.assert_allclose(episode["evidence"][:], [3.5, 8.0, 5.0])
            np.testing.assert_allclose(episode["aleatoric_uncertainty"][:], [0.7, 0.2, 0.5])
            np.testing.assert_allclose(episode["epistemic_uncertainty"][:], [0.3, 0.1, 0.25])
            np.testing.assert_allclose(episode["confidence"][:], [0.4, 0.8, 0.6])
            np.testing.assert_array_equal(episode["rank"][:], [1, 2, 1])
            self.assertNotIn("low_evidence_threshold", episode)
            self.assertNotIn("low_evidence_count", episode)

    def test_rejects_mismatched_action_token_fields(self):
        chunks = self._chunks()
        chunks[0]["action_token_epistemic_uncertainty"] = [0.3]

        with LiberoUncertaintyDatasetWriter(self.dataset_path, self.metadata) as writer:
            with self.assertRaisesRegex(ValueError, "action-token field lengths must match"):
                writer.append_episode(
                    task_id=0,
                    episode_idx=0,
                    task_description="task",
                    success=False,
                    executed_steps=220,
                    termination_reason="max_steps",
                    uncertainty_chunks=chunks,
                )

    def test_existing_file_requires_resume_or_overwrite(self):
        with LiberoUncertaintyDatasetWriter(self.dataset_path, self.metadata) as writer:
            self._append_episode(writer)

        with self.assertRaises(FileExistsError):
            LiberoUncertaintyDatasetWriter(self.dataset_path, self.metadata)

        with LiberoUncertaintyDatasetWriter(self.dataset_path, self.metadata, resume=True) as writer:
            self.assertTrue(writer.has_episode(task_id=2, episode_idx=3))
            with self.assertRaisesRegex(ValueError, "episode already exists"):
                self._append_episode(writer)


if __name__ == "__main__":
    unittest.main()
