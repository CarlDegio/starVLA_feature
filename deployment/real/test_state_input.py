"""CPU regressions for training-equivalent state preprocessing; no robot/GPU."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.real.yam_policy import STAR_ACTION_KEYS, STAR_STATE_KEYS, STAR_TO_YAM, YAMStarVLAPolicy
from deployment.real.test_yam_policy import observation
from starVLA.model.framework.VLM4A.QwenEDL import Qwenvl_EDL


class StateInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "checkpoints").mkdir()
        self.ckpt = root / "checkpoints/model.pt"
        self.ckpt.touch()  # Only metadata/transforms are loaded in this test.
        self.cfg = {"framework": {"name": "QwenEDL", "action_model": {
            "action_dim": 14, "state_dim": 14, "action_horizon": 30},
            "state_input": {"enabled": True, "num_bins": 256}},
            "datasets": {"vla_data": {"data_mix": "edl_real_place_the_slippers_on_the_shoe_rack"}}}
        (root / "config.yaml").write_text(json.dumps(self.cfg))
        self.low = np.arange(14, dtype=np.float32) - 7
        self.high = self.low + np.arange(14, dtype=np.float32) / 10 + 1
        self.high[3] = self.low[3]  # Constant joint: normalized zero, never raw radians.
        stats = {"q01": self.low.tolist(), "q99": self.high.tolist(),
                 "min": self.low.tolist(), "max": self.high.tolist(),
                 "mean": ((self.low + self.high) / 2).tolist(), "std": [1.] * 14}
        action_stats = {**stats, "q01": [-10.] * 14, "q99": [10.] * 14}
        (root / "dataset_statistics.json").write_text(json.dumps({
            "new_embodiment": {"state": stats, "action": action_stats}}))
        self.processor = PolicyNormProcessor(str(self.ckpt))

    def tearDown(self):
        self.tmp.cleanup()

    def test_state_quantiles_clipping_and_constant_dimensions(self):
        span = self.high - self.low
        raw = np.stack([self.low - 2, self.low + .25 * span, self.high + 2])
        actual = self.processor.apply_state(raw)
        expected = np.stack([-np.ones(14), -.5 * np.ones(14), np.ones(14)])
        expected[:, 3] = 0
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        for bad in (np.zeros(14), np.zeros((1, 13)), np.full((1, 14), np.nan)):
            with self.assertRaises(ValueError):
                self.processor.apply_state(bad)

    def test_yam_reorder_matches_training_state_text_and_30_actions(self):
        class Wrapper:
            metadata = {"action_keys": STAR_ACTION_KEYS, "state_keys": STAR_STATE_KEYS,
                        "action_chunk_size": 30}
            def predict_action(self, examples, **kwargs):
                self.example = examples[0]
                return {"actions": np.zeros((1, 30, 14), dtype=np.float32)}
        wrapper = Wrapper()
        adapter = YAMStarVLAPolicy(wrapper, task="slippers", default_prompt="test",
                                  state_processor=self.processor)
        obs = observation()
        raw = self.low + np.linspace(.1, .9, 14, dtype=np.float32) * (self.high - self.low)
        obs["state"] = raw[STAR_TO_YAM]
        original = obs["state"].copy()
        out = adapter.infer(obs)
        expected = self.processor.apply_state(raw[None]).astype(np.float16)
        np.testing.assert_array_equal(wrapper.example["state"], expected)
        np.testing.assert_array_equal(obs["state"], original)
        self.assertTrue(adapter.metadata["uses_state"])
        self.assertEqual(out["actions"].shape, (30, 14))
        # Use the actual training and generation instruction builder, not a
        # separate approximation of binning (float16 rounding matters here).
        model = Qwenvl_EDL.__new__(Qwenvl_EDL)
        torch.nn.Module.__init__(model)
        model.config = OmegaConf.create(self.cfg)
        training_text = model._build_instructions([{"lang": obs["prompt"], "state": expected}])
        self.assertEqual(model._build_instructions([wrapper.example]), training_text)
        self.assertIn("[STATE]", training_text[0])
        self.assertIn("[ACTION]", training_text[0])
        obs["state"][0] += 1
        adapter.infer(obs)
        self.assertNotEqual(model._build_instructions([wrapper.example]), training_text)


if __name__ == "__main__":
    unittest.main()
