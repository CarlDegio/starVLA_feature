import tempfile
from pathlib import Path
import unittest

import h5py
import numpy as np

from deployment.real.manual_controller import ManualChunkController
from examples.LIBERO.edl_pred_real.recording import EpisodeRecorder
from examples.LIBERO.edl_pred_real.test_recording import Policy, Backend
from examples.LIBERO.edl_pred_real.export import export_dataset
from examples.LIBERO.edl_pred.dataset import index_suite_episodes, TrajectoryDataset


class ExportTests(unittest.TestCase):
    def test_30_step_chunk_metadata_and_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            policy, backend = Policy(), Backend()
            policy.metadata = {**policy.metadata, "horizon": 30, "uses_state": True}
            recorder = EpisodeRecorder(folder, policy.metadata, collection_id="test", seed_namespace="mock", mock=True)
            controller = ManualChunkController(policy, backend, recorder, "test")
            controller.control_hz = 100000
            controller.execute_once()
            recorder.finish(1)
            output = Path(folder) / "data.hdf5"
            export_dataset(folder, output, task="blocks", collection_id="test", seed_namespace="mock", allow_mock=True)
            with h5py.File(output) as file:
                self.assertEqual(file.attrs["action_chunk_size"], 30)
                episode = file["episodes/task_000_episode_0000"]
                self.assertEqual(episode.attrs["executed_steps"], 30)
                self.assertEqual(episode["real_chunks/000000/actions"].shape, (30, 14))

    def test_roundtrip_into_existing_verifier_and_recompute(self):
        with tempfile.TemporaryDirectory() as folder:
            for label in (0, 1):
                policy, backend = Policy(), Backend()
                recorder = EpisodeRecorder(folder, policy.metadata, collection_id="test", seed_namespace="mock", mock=True)
                controller = ManualChunkController(policy, backend, recorder, "test")
                controller.control_hz = 100000
                controller.execute_once()
                controller.execute_once()
                recorder.finish(label)
            output = Path(folder) / "data.hdf5"
            with self.assertRaisesRegex(ValueError, "No closed"):
                export_dataset(folder, output, task="blocks", collection_id="test", seed_namespace="mock")
            export_dataset(folder, output, task="blocks", collection_id="test", seed_namespace="mock", allow_mock=True)
            refs, maximum = index_suite_episodes("real_blocks", output)
            self.assertEqual(len(refs), 2)
            self.assertEqual(maximum, 3)
            dataset = TrajectoryDataset(refs, maximum)
            try:
                sample = dataset[0]
                self.assertEqual(sample.features.shape, (2, 3, 2))
                self.assertTrue(np.isfinite(sample.features).all())
                self.assertEqual({ref.label for ref in refs}, {0, 1})
            finally:
                dataset.close()
            with h5py.File(output) as file:
                chunk = file["episodes/task_000_episode_0000/real_chunks/000000"]
                self.assertEqual(chunk["image_top"].shape, (24, 32, 3))
                self.assertEqual(chunk["model_image_top"].shape, (224, 224, 3))
                self.assertEqual(chunk["actions"].shape, (15, 14))
            with self.assertRaises(FileExistsError):
                export_dataset(folder, output, task="blocks", collection_id="test", seed_namespace="mock", allow_mock=True)


if __name__ == "__main__":
    unittest.main()
