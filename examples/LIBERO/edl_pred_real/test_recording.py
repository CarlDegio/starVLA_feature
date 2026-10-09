import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np
from scipy.special import digamma

from deployment.real.manual_controller import ManualChunkController
from examples.LIBERO.edl_pred_real.recording import EpisodeRecorder


class Policy:
    metadata = {"backend": "starvla", "task": "blocks", "horizon": 15, "control_hz": 30,
                "action_stride": 1, "execution_speed": 1, "supports_rtc": False,
                "state_dim": 14, "action_dim": 14, "default_prompt": "test", "ckpt_path": "/mock/ckpt.pt",
                "edl_details_available": True}

    def __init__(self, *args):
        self.calls = 0
        self.closed = False

    def infer(self, observation):
        self.calls += 1
        assert observation["return_edl_details"]
        alpha = np.tile(np.array([2, 3], dtype=np.float32), (1, 3, 1))
        strength = alpha.sum(-1, keepdims=True)
        au = ((alpha / strength) * (digamma(strength + 1) - digamma(alpha + 1))).sum(-1) / np.log(2)
        return {"actions": np.full((15, 14), .02, dtype=np.float32),
                "model_images": {role: np.zeros((224, 224, 3), dtype=np.uint8) for role in ("top", "left", "right")},
                "diagnostics": {"action_token_topk_alpha": alpha,
                    "action_token_topk_ids": np.tile([2, 3], (1, 3, 1)),
                    "action_token_ids": np.full((1, 3), 3), "action_token_mask": np.ones((1, 3), dtype=bool),
                    "num_action_tokens": np.array([3]), "action_token_evidence": np.full((1, 3), 2, dtype=np.float32),
                    "action_token_aleatoric_uncertainty": au,
                    "action_token_epistemic_uncertainty": np.full((1, 3), .4, dtype=np.float32),
                    "action_token_confidence": np.full((1, 3), .6, dtype=np.float32),
                    "action_token_rank": np.ones((1, 3), dtype=np.float32),
                    "token_uncertainty": np.ones((1, 4), dtype=np.float32)}}

    def close(self):
        self.closed = True


class Backend:
    def __init__(self):
        self.rows = []

    def observe(self, prompt):
        return {"state": np.zeros(14, dtype=np.float32), "prompt": prompt,
                "images": {role: np.zeros((24, 32, 3), dtype=np.uint8) for role in ("top", "left", "right")}}

    def execute(self, row, index, recorder, chunk, cancel):
        assert (chunk / "prediction.npz").is_file()  # write-ahead recording
        self.rows.append(row.copy())
        recorder.event(chunk, "mock_command", row=index)
        return True


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.policy = Policy()
        self.backend = Backend()
        self.rec = EpisodeRecorder(self.tmp.name, self.policy.metadata,
                                   collection_id="test", seed_namespace="mock", mock=True)
        self.controller = ManualChunkController(self.policy, self.backend, self.rec, "test")
        self.controller.control_hz = 100000

    def tearDown(self):
        self.tmp.cleanup()

    def test_one_trigger_one_horizon_no_background_inference(self):
        self.assertEqual(self.policy.calls, 0)
        result = self.controller.execute_once()
        self.assertEqual(result["completed_rows"], 15)
        self.assertEqual(len(self.backend.rows), 15)
        self.assertEqual(self.policy.calls, 1)
        self.controller.execute_once()
        self.assertEqual(self.policy.calls, 2)
        self.assertEqual(len(self.backend.rows), 30)
        self.rec.finish(1)
        self.assertEqual(json.loads((self.rec.path / "episode.json").read_text())["success"], 1)

    def test_late_response_after_cancel_never_executes(self):
        entered, release = threading.Event(), threading.Event()
        original = self.policy.infer
        def blocked(obs):
            entered.set()
            release.wait(3)
            return original(obs)
        self.policy.infer = blocked
        thread = threading.Thread(target=self.controller.execute_once)
        thread.start()
        self.assertTrue(entered.wait(1))
        with self.assertRaisesRegex(RuntimeError, "already running"):
            self.controller.execute_once()
        self.controller.stop()
        release.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.backend.rows, [])
        result = json.loads((self.rec.path / "chunk_000000/result.json").read_text())
        self.assertEqual(result["status"], "cancelled")

    def test_bad_diagnostics_and_write_failure_send_no_actions(self):
        original = self.policy.infer
        def bad(obs):
            response = original(obs)
            del response["diagnostics"]["action_token_topk_alpha"]
            return response
        self.policy.infer = bad
        with self.assertRaises(KeyError):
            self.controller.execute_once()
        self.assertEqual(self.backend.rows, [])
        self.assertTrue(self.controller.cancel.is_set())

    def test_disk_failure_before_execution(self):
        def fail(*args):
            raise OSError("disk full")
        self.rec.save_prediction = fail
        with self.assertRaises(OSError):
            self.controller.execute_once()
        self.assertEqual(self.backend.rows, [])

    def test_drop_removes_only_current_episode(self):
        other = Path(self.tmp.name) / "keep"
        other.mkdir()
        self.controller.execute_once()
        self.rec.discard()
        self.assertFalse(self.rec.path.exists())
        self.assertTrue(other.is_dir())


if __name__ == "__main__":
    unittest.main()
