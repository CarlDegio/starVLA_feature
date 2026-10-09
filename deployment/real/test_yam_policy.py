"""Contract regressions; all policies here are stubs and send no robot commands."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.request import ProxyHandler, build_opener

import numpy as np
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from deployment.model_server.tools import msgpack_numpy
from deployment.real.infer import evaluate
from deployment.real.websocket_server import YAMWebsocketServer
from deployment.real.yam_policy import STAR_ACTION_KEYS, YAMStarVLAPolicy


class StubWrapper:
    metadata = {"action_keys": STAR_ACTION_KEYS, "action_chunk_size": 15}

    def predict_action(self, examples, **kwargs):
        self.examples = examples
        return {"actions": np.tile(np.arange(14, dtype=np.float32), (1, 15, 1)),
                "uncertainty": np.array([0.25], dtype=np.float32)}


def observation():
    return {"state": np.zeros(14, dtype=np.float32), "prompt": "test task",
            "images": {role: np.full((24, 32, 3), value, dtype=np.uint8)
                       for role, value in (("right", 30), ("top", 10), ("left", 20))}}


def policy():
    return YAMStarVLAPolicy(StubWrapper(), task="blocks", default_prompt="test task")


class AdapterTests(unittest.TestCase):
    def test_camera_and_joint_order_and_diagnostics(self):
        adapter = policy()
        result = adapter.infer(observation())
        self.assertEqual(result["actions"].shape, (15, 14))
        np.testing.assert_array_equal(result["actions"][0],
                                      [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13])
        self.assertEqual(result["actions"].dtype, np.float32)
        self.assertEqual(result["diagnostics"]["uncertainty"][0], 0.25)
        example = adapter.wrapper.examples[0]
        self.assertEqual([image.size for image in example["image"]], [(224, 224)] * 3)
        self.assertEqual([np.asarray(image)[0, 0, 0] for image in example["image"]], [10, 20, 30])
        self.assertEqual(example["lang"], "test task")
        self.assertNotIn("state", example)
        self.assertEqual(adapter.metadata["control_hz"], 30)
        self.assertEqual(adapter.metadata["action_keys"][1], "action.left_gripper")
        self.assertEqual(adapter.metadata["starvla_action_keys"], STAR_ACTION_KEYS)

    def test_reject_incompatible_requests(self):
        adapter = policy()
        for updates in ({"state": np.zeros(13)}, {"state": np.full(14, np.nan)},
                        {"prompt": " "}, {"images": {}}, {"action_prefix": []},
                        {"prefix_length": 4}, {"execution_speed": 2}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                adapter.infer({**observation(), **updates})
        obs = observation()
        obs["images"]["top"] = np.zeros((2, 2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            adapter.infer(obs)

    def test_reject_wrong_checkpoint_and_bad_predictions(self):
        wrong = StubWrapper()
        wrong.metadata = {**wrong.metadata, "action_keys": list(reversed(STAR_ACTION_KEYS))}
        with self.assertRaises(ValueError):
            YAMStarVLAPolicy(wrong, task="blocks", default_prompt="test")
        adapter = policy()
        for actions in (np.zeros((15, 14)), np.zeros((1, 32, 14)), np.full((1, 15, 14), np.nan)):
            adapter.wrapper.predict_action = lambda examples, **kwargs: {"actions": actions}
            with self.assertRaises(ValueError):
                adapter.infer(observation())

    def test_recorded_32_step_ground_truth_uses_only_15_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.npz"
            obs = observation()
            np.savez(path, state=obs["state"], prompt=obs["prompt"],
                     gt_actions=np.zeros((32, 14)), **obs["images"])
            args = SimpleNamespace(output=Path(folder), repeat=1)
            evaluate(policy(), args, [path], "test task", None)
            import json
            report = json.loads((Path(folder) / "report.json").read_text())
            self.assertEqual(report["samples"][0]["compared_steps"], 15)
            self.assertFalse(report["robot_commands_sent"])


class WireTests(unittest.IsolatedAsyncioTestCase):
    async def test_handshake_flat_reply_health_and_error(self):
        server = YAMWebsocketServer(policy())
        async with serve(server.handler, "127.0.0.1", 0, compression=None,
                         process_request=server.health_check) as listener:
            port = listener.sockets[0].getsockname()[1]
            def health():
                with build_opener(ProxyHandler({})).open(f"http://127.0.0.1:{port}/healthz") as response:
                    return response.read()
            self.assertEqual(await asyncio.to_thread(health), b"OK\n")
            async with connect(f"ws://127.0.0.1:{port}", proxy=None) as client:
                metadata = msgpack_numpy.unpackb(await client.recv())
                self.assertEqual(metadata["horizon"], 15)
                self.assertFalse(metadata["supports_rtc"])
                await client.send(msgpack_numpy.packb(observation()))
                response = msgpack_numpy.unpackb(await client.recv())
                self.assertEqual(response["actions"].shape, (15, 14))
                self.assertNotIn("data", response)
                await client.send(msgpack_numpy.packb({**observation(), "prefix_length": 4}))
                self.assertIn("disable RTC", await client.recv())


if __name__ == "__main__":
    unittest.main()
