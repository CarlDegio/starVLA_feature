"""State normalization, packing, and opt-in QwenEDL conditioning contracts."""

import copy
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.policy_norm_processor import _build_dataset_metadata
from examples.LIBERO.train_real.data_config import ACTION_HORIZON, EDLRealDualArmDataConfig
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.model.framework.VLM4A.QwenEDL import Qwenvl_EDL


def statistics(low, high):
    low, high = np.asarray(low), np.asarray(high)
    return {"min": low.tolist(), "max": high.tolist(), "q01": low.tolist(),
            "q99": high.tolist(), "mean": ((low + high) / 2).tolist(), "std": np.ones(14).tolist()}


def instruction_model(enabled=True):
    model = Qwenvl_EDL.__new__(Qwenvl_EDL)
    torch.nn.Module.__init__(model)
    model.config = OmegaConf.create({"framework": {
        "state_input": {"enabled": enabled, "num_bins": 256}, "action_model": {"state_dim": 14}}})
    return model


class StateInputTests(unittest.TestCase):
    def test_measured_state_uses_own_stats_and_action_order(self):
        config = EDLRealDualArmDataConfig()
        low = np.arange(14, dtype=np.float32) * 10
        high = low + 2
        # Constant gripper and distinct per-dimension values expose ordering bugs.
        high[12] = low[12]
        expected = np.linspace(-1.5, 1.5, 14, dtype=np.float32)
        state = low + (expected + 1) / 2 * (high - low)
        state[12] = 123  # Constant-state fallback must be zero, not the raw value.
        meta = _build_dataset_metadata(
            {"state": statistics(low, high), "action": statistics(np.full(14, -10), np.full(14, 10))},
            config.embodiment_tag, config.action_keys, config.state_keys,
            config.action_key_dims, config.state_key_dims)
        transform = config.transform()
        transform.set_metadata(meta)
        raw = {}
        cursor = 0
        for key, dim in config.state_key_dims.items():
            raw[key] = state[None, cursor:cursor + dim].copy()
            raw[key.replace("state.", "action.")] = np.tile(np.arange(14, dtype=np.float32)[cursor:cursor + dim], (ACTION_HORIZON, 1))
            cursor += dim
        for key in config.video_keys:
            raw[key] = np.zeros((1, 240, 320, 3), dtype=np.uint8)
        raw[config.language_keys[0]] = ["classify blocks"]
        dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
        dataset._modality_keys = {k: v.modality_keys for k, v in config.modality_config().items()}
        dataset.data_cfg = {"include_state": True}
        dataset.tag = config.embodiment_tag.value
        sample = dataset._pack_sample(transform(copy.deepcopy(raw)))
        expected = np.clip(expected, -1, 1)
        expected[12] = 0
        self.assertEqual(sample["state"].shape, (1, 14))
        np.testing.assert_allclose(sample["state"][0], expected, atol=1e-3)
        self.assertEqual(sample["action"].shape, (ACTION_HORIZON, 14))
        np.testing.assert_allclose(sample["action"], np.tile(np.arange(14) / 10, (ACTION_HORIZON, 1)), atol=1e-3)
        dataset.data_cfg["include_state"] = False
        without_state = dataset._pack_sample(transform(copy.deepcopy(raw)))
        self.assertNotIn("state", without_state)
        np.testing.assert_array_equal(without_state["action"], sample["action"])

    def test_quantized_state_is_current_ordered_and_not_mutated(self):
        model = instruction_model()
        state = np.linspace(-1, 1, 14, dtype=np.float32)[None]
        before = state.copy()
        instruction = model._build_instructions([{"lang": "task", "state": state}])[0]
        bins = [int(value) for value in instruction.split("[STATE] ")[1].split(" [ACTION]")[0].split()]
        self.assertEqual(len(bins), 14)
        self.assertEqual(bins[0], 0)
        self.assertEqual(bins[-1], 255)
        self.assertEqual(bins, sorted(bins))
        np.testing.assert_array_equal(state, before)
        self.assertEqual(instruction, model._build_instructions([{"lang": "task", "state": torch.from_numpy(state)}])[0])

    def test_missing_invalid_or_raw_state_is_rejected(self):
        model = instruction_model()
        for example in ({"lang": "task"}, {"lang": "task", "state": np.zeros(14)},
                        {"lang": "task", "state": np.zeros((2, 14))},
                        {"lang": "task", "state": np.full((1, 14), np.nan)},
                        {"lang": "task", "state": np.full((1, 14), 2)}):
            with self.subTest(example=example), self.assertRaises(ValueError):
                model._build_instructions([example])

    def test_libero_instructions_unchanged_even_if_state_present(self):
        model = instruction_model(enabled=False)
        examples = [{"lang": "original instruction", "state": np.full((1, 7), np.nan)}, {"lang": "another"}]
        self.assertEqual(model._build_instructions(examples), [e["lang"] for e in examples])
        del model.config.framework.state_input
        self.assertEqual(model._build_instructions(examples), [e["lang"] for e in examples])

    def test_all_real_configs_enable_both_state_controls(self):
        for path in (Path(__file__).parent / "configs").glob("*.yaml"):
            cfg = OmegaConf.load(path)
            self.assertTrue(cfg.datasets.vla_data.include_state)
            self.assertTrue(cfg.framework.state_input.enabled)
            self.assertEqual(cfg.framework.action_model.state_dim, 14)
            self.assertEqual(cfg.framework.action_model.action_horizon, ACTION_HORIZON)
            self.assertEqual(cfg.framework.action_model.future_action_window_size, ACTION_HORIZON - 1)
            self.assertEqual(EDLRealDualArmDataConfig().modality_config()["action"].delta_indices,
                             list(range(ACTION_HORIZON)))
            self.assertTrue(cfg.run_id.endswith("_state"))


if __name__ == "__main__":
    unittest.main()
