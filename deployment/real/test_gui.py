import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from yam_abc_reproduce.config import CameraConfig, RobotConfig, RobotUnitConfig, StationConfig
from yam_abc_reproduce.deploy.loop import DeployLoop
from yam_abc_reproduce.gui.server import create_app

from deployment.real.gui import install_manual_profile
from examples.LIBERO.edl_pred_real.test_recording import Policy


class GUITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = StationConfig(robot=RobotConfig(type="yam", num_arm_joints=6,
            robots=[RobotUnitConfig(type="yam_left"), RobotUnitConfig(type="yam_right")]),
            cameras=[CameraConfig(name=role, role=role, type="mock", mode="mono", width=32, height=24)
                     for role in ("top", "left", "right")],
            save_root=str(Path(self.tmp.name) / "ordinary"), control_hz=30)
        self.app = install_manual_profile(create_app(self.cfg, mock=True), lambda: self.cfg,
                                          Path(self.tmp.name) / "records")
        self.manager = self.app.state.manual_episode
        self.manager.client_factory = Policy
        self.client = TestClient(self.app)
        self.args = dict(host="127.0.0.1", port=8002, prompt="test", collection_id="test",
                         seed_namespace="mock", max_joint_speed=.3)
        self.home_patch = patch.object(DeployLoop, "move_to_home", self.fast_home)
        self.home_patch.start()

    @staticmethod
    def fast_home(loop, pose):
        loop._command([np.array(pose[:7]), np.array(pose[7:])])

    def tearDown(self):
        if self.manager._end_thread:
            self.manager._end_thread.join(5)
        self.home_patch.stop()
        if self.manager.policy:
            self.manager.policy.close()
        self.app.state.session._teardown_units()
        self.app.state.session._disconnect_cameras()
        self.tmp.cleanup()

    def finish(self):
        self.manager.end()
        self.manager._end_thread.join(5)
        self.assertFalse(self.manager._end_thread.is_alive())
        self.assertEqual(self.manager.phase, "await_label")

    def test_first_pose_preserved_two_chunks_return_and_yes_label(self):
        class ChangingPolicy(Policy):
            def infer(self, obs):
                response = super().infer(obs)
                response["actions"][-1] = np.arange(14) * .001 + self.calls * .01
                return response
        self.manager.client_factory = ChangingPolicy
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertFalse(self.app.state.session.live)
        self.assertIsNone(self.cfg.deploy_home_pose)  # No station home is needed.
        response = self.client.post("/api/real/step", json=self.args)
        self.assertEqual(response.status_code, 200, response.text)
        first_pose = list(self.manager.home)
        self.assertEqual(self.manager.policy.calls, 1)
        self.assertEqual(self.manager.phase, "paused")
        first = self.client.get("/api/real/status").json()["latest_chunk"]
        self.assertEqual((first["chunk_number"], first["status"], first["completed_rows"]),
                         (1, "completed", 15))
        np.testing.assert_allclose(first["last_action"], np.arange(14) * .001 + .01)
        self.assertAlmostEqual(first["max_model_speed"], .3)
        self.assertEqual(first["max_model_speed_joint"], "left_joint_0")
        self.assertEqual(first["max_model_speed_step"], 14)
        self.assertGreater(first["max_chunk_target_speed"], first["max_model_speed"])
        self.assertEqual(first["max_chunk_target_speed_from"], "measured_state")
        self.assertEqual(len(first["all_actions"]), 15)
        self.assertEqual(len(first["all_actions"][0]), 14)
        self.manager.step(**self.args)
        self.assertEqual(self.manager.policy.calls, 2)
        self.assertEqual(self.manager.home, first_pose)
        second = self.client.get("/api/real/status").json()["latest_chunk"]
        self.assertEqual((second["chunk_number"], second["status"], second["completed_rows"]),
                         (2, "completed", 15))
        np.testing.assert_allclose(second["last_action"], np.arange(14) * .001 + .02)
        self.assertEqual(self.client.post("/api/deploy/start", json={}).status_code, 409)
        self.finish()
        self.assertIsNone(self.manager.error)
        path = self.manager.recorder.path
        np.testing.assert_allclose(np.concatenate([u.robot.get_joint_pos() for u in self.app.state.session.units]), first_pose)
        with self.assertRaises(RuntimeError):
            self.manager.label("stale-episode-id", "y")
        self.manager.label(self.manager.recorder.metadata["episode_id"], "y")
        self.assertEqual(json.loads((path / "episode.json").read_text())["success"], 1)
        self.assertFalse(json.loads((path / "end_episode.json").read_text())["label_pending"])
        self.assertEqual(self.manager.phase, "idle")
        self.assertIsNone(self.manager.status()["latest_chunk"])

    def test_no_label_is_not_failure_and_drop_removes_data(self):
        self.manager.step(**self.args)
        self.finish()
        path = self.manager.recorder.path
        self.assertIsNone(json.loads((path / "episode.json").read_text())["success"])
        with self.assertRaises(ValueError):
            self.manager.label(self.manager.recorder.metadata["episode_id"], "cancel")
        self.assertTrue(path.exists())
        self.manager.label(self.manager.recorder.metadata["episode_id"], "drop")
        self.assertFalse(path.exists())

    def test_end_cancels_pending_response_before_return(self):
        entered, release = threading.Event(), threading.Event()
        class Slow(Policy):
            def infer(self, obs):
                entered.set()
                release.wait(5)
                return super().infer(obs)
        self.manager.client_factory = Slow
        result = []
        thread = threading.Thread(target=lambda: result.append(self.manager.step(**self.args)))
        thread.start()
        self.assertTrue(entered.wait(2))
        pending = self.manager.status()["latest_chunk"]
        self.assertEqual(pending["status"], "inferring")
        self.assertIsNone(pending["last_action"])
        self.manager.end()
        with self.assertRaises(RuntimeError):
            self.manager.step(**self.args)
        release.set()
        thread.join(5)
        self.manager._end_thread.join(5)
        self.assertEqual(result[0]["completed_rows"], 0)
        self.assertEqual(self.manager.phase, "await_label")
        cancelled = self.manager.status()["latest_chunk"]
        self.assertEqual((cancelled["status"], cancelled["completed_rows"]), ("cancelled", 0))
        self.assertEqual(len(cancelled["last_action"]), 14)
        events = (self.manager.recorder.path / "chunk_000000/events.jsonl").read_text()
        self.assertNotIn('"command_attempt"', events)

    def test_estop_during_return_does_not_resume_commands(self):
        self.manager.step(**self.args)
        sent = []
        def interrupted(loop, pose):
            self.app.state.session.estop()
            sent.append(loop._command([np.ones(7), np.ones(7)]))
        with patch.object(DeployLoop, "move_to_home", interrupted):
            self.finish()
        self.assertEqual(sent, [False])
        self.assertIsNotNone(self.manager.error)
        self.manager.label(self.manager.recorder.metadata["episode_id"], "n")
        self.assertTrue(self.app.state.session._estopped)


if __name__ == "__main__":
    unittest.main()
