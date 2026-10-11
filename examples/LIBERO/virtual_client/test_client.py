import json
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from examples.LIBERO.virtual_client.client import Episode, NAMES, STAR_NAMES, replay
from deployment.real.yam_policy import STAR_TO_YAM, STAR_ACTION_KEYS, YAMStarVLAPolicy


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "meta").mkdir()
        self.raw = np.arange(14, dtype=np.float32)[None] + np.arange(7, dtype=np.float32)[:, None] / 10
        info = {"codebase_version": "v2.1", "robot_type": "edl_real_dual_arm", "fps": 30,
                "chunks_size": 1000, "data_path": "episode_{episode_index:06d}.parquet",
                "video_path": "{video_key}.mp4",
                "features": {k: {"names": STAR_NAMES} for k in ("action", "observation.state")}}
        (self.root / "meta/info.json").write_text(json.dumps(info))
        (self.root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 7}))
        (self.root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "test"}))
        pq.write_table(pa.table({"observation.state": self.raw.tolist(), "action": (self.raw + .25).tolist(),
                                "frame_index": list(range(7)), "episode_index": [0] * 7,
                                "task_index": [0] * 7, "timestamp": np.arange(7) / 30}),
                       self.root / "episode_000000.parquet")
        for role in ("top", "left", "right"):
            with av.open(str(self.root / f"observation.images.{role}.mp4"), "w") as container:
                stream = container.add_stream("libx264", rate=30)
                stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
                for i in range(7):
                    pixels = np.zeros((24, 32, 3), np.uint8)
                    pixels[:, :, 0] = 80 + i * 20
                    for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
        self.episode = Episode(self.root, 0)
        raw = self.raw
        class Client:
            metadata = {"backend": "starvla", "task": "slippers", "horizon": 3,
                        "action_dim": 14, "state_dim": 14, "control_hz": 30,
                        "action_keys": ["action.left_joints", "action.left_gripper", "action.right_joints", "action.right_gripper"],
                        "action_space": "absolute_joint_radians", "action_stride": 1, "execution_speed": 1}
            def __init__(self):
                self.observations = []
            def infer(self, obs):
                self.observations.append(obs)
                frame = (len(self.observations) - 1) * 3
                selected = np.minimum(np.arange(frame, frame + 3), 6)
                return {"actions": (raw[selected] + .25)[:, STAR_TO_YAM]}
        self.client = Client()

    def tearDown(self):
        self.episode.close()
        self.tmp.cleanup()

    def test_frames_gt_order_rgb_tail_and_report(self):
        out = self.root / "result"
        summary = replay(self.client, self.episode, out, task="slippers")
        self.assertEqual(summary["compared_frames"], 7)
        self.assertEqual(summary["joint_mae_rad"], 0)
        self.assertTrue(summary["episode_complete"])
        self.assertFalse(summary["robot_commands_sent"])
        for i, obs in enumerate(self.client.observations):
            np.testing.assert_array_equal(obs["state"], self.raw[i * 3, STAR_TO_YAM])
            self.assertAlmostEqual(float(obs["images"]["top"][0, 0, 0]), 80 + i * 60, delta=4)
            self.assertLess(obs["images"]["top"][0, 0, 2], 5)  # RGB, not BGR
        with np.load(out / "comparison.npz") as result:
            np.testing.assert_array_equal(result["frame_indices"], np.arange(7))
            np.testing.assert_array_equal(result["action_names"], NAMES)
            np.testing.assert_array_equal(result["gt_actions"], (self.raw + .25)[:, STAR_TO_YAM])
        with np.load(out / "chunk_000002.npz") as result:
            self.assertEqual(result["predicted_actions"].shape, (3, 14))
            self.assertEqual(result["gt_actions"].shape, (1, 14))
        self.assertTrue((out / "comparison.png").is_file())
        self.assertIn("超出 episode，不评分", (out / "report.html").read_text())
        chunks = list(map(json.loads, (out / "chunks.jsonl").read_text().splitlines()))
        self.assertEqual([c["frame_start"] for c in chunks], [0, 3, 6])
        self.assertFalse(chunks[1]["same_as_previous_prediction"])

    def test_constant_output_is_detected_even_with_changed_inputs(self):
        self.client.infer = lambda obs: {"actions": np.full((3, 14), .5, np.float32)}
        summary = replay(self.client, self.episode, self.root / "constant", task="slippers", max_chunks=2, plots=False)
        self.assertEqual(summary["constant_chunks"], 2)
        self.assertEqual(summary["repeated_chunks"], 1)
        self.assertFalse(summary["episode_complete"])
        chunks = list(map(json.loads, (self.root / "constant/chunks.jsonl").read_text().splitlines()))
        self.assertTrue(chunks[1]["raw_state_changed"])
        self.assertTrue(all(chunks[1]["image_changed"].values()))

    def test_30_step_model_returns_15_and_replay_advances_15(self):
        class RecordedEpisode:
            root = Path("synthetic")
            index, fps, length = 0, 30, 32
            entry, info = {}, {}
            source_frames = np.arange(32)
            states = np.tile(np.arange(32, dtype=np.float32)[:, None], (1, 14))
            actions = states.copy()
            requested = []

            def observation(self, frame):
                self.requested.append(frame)
                return {"state": self.states[frame], "prompt": str(frame),
                        "images": {r: np.zeros((24, 32, 3), np.uint8) for r in ("top", "left", "right")}}

        class Wrapper:
            metadata = {"action_keys": STAR_ACTION_KEYS, "action_chunk_size": 30}

            def predict_action(self, examples, **kwargs):
                start = int(examples[0]["lang"])
                actions = np.tile(np.arange(start, start + 30, dtype=np.float32)[None, :, None], (1, 1, 14))
                return {"actions": actions}

        episode = RecordedEpisode()
        client = YAMStarVLAPolicy(Wrapper(), task="slippers", default_prompt="test", execution_horizon=15)
        client.metadata["edl_details_available"] = True
        out = self.root / "prefix"
        summary = replay(client, episode, out, task="slippers", plots=False)
        self.assertEqual(episode.requested, [0, 15, 30])
        self.assertEqual(summary["compared_frames"], 32)
        self.assertEqual(summary["joint_mae_rad"], 0)
        self.assertEqual(summary["frame_stride"], 15)
        self.assertEqual(summary["model_horizon"], 30)
        with np.load(out / "chunk_000002.npz") as result:
            self.assertEqual(result["predicted_actions"].shape, (15, 14))
            self.assertEqual(result["gt_actions"].shape, (2, 14))
            self.assertEqual(result["diagnostic__full_model_actions"].shape, (30, 14))

    def test_metadata_mismatch_and_existing_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "Policy task"):
            replay(self.client, self.episode, self.root / "bad", task="blocks", plots=False)
        self.assertEqual(len(self.client.observations), 0)
        with self.assertRaises(FileExistsError):
            replay(self.client, self.episode, self.root, task="slippers", plots=False)


if __name__ == "__main__":
    unittest.main()
