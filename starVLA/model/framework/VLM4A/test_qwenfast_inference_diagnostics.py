from __future__ import annotations

import contextlib
import types
import unittest
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.model.framework.VLM4A.QwenFast import Qwenvl_Fast


class _FakeGenerateModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.sequences = torch.tensor([[9, 3, 4]])
        self.generated = types.SimpleNamespace(
            sequences=self.sequences,
            scores=(torch.tensor([[0.0, 0.1, 0.2, 1.1, 0.3]]), torch.tensor([[0.2, 0.0, 0.1, 0.4, 1.4]])),
            hidden_states=(
                (torch.tensor([[[1.0, 2.0]]]),),
                (torch.tensor([[[3.0, 4.0]]]),),
            ),
        )

    def generate(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.generated if kwargs.get("return_dict_in_generate") else self.sequences


class _FakeFastTokenizer:
    def decode(self, token_ids):
        self.last_token_ids = token_ids
        return np.asarray([[[0.25] * 7]], dtype=np.float32)


class QwenFastInferenceDiagnosticsTest(unittest.TestCase):
    def _framework(self) -> Qwenvl_Fast:
        framework = Qwenvl_Fast.__new__(Qwenvl_Fast)
        model = _FakeGenerateModel()
        framework.qwen_vl_interface = types.SimpleNamespace(
            model=model,
            _ACTION_TOKEN_MIN=3,
            _ACTION_TOKEN_MAX=4,
            build_qwenvl_inputs=lambda **kwargs: {"input_ids": torch.tensor([[9]])},
        )
        framework.action_model = types.SimpleNamespace(fast_tokenizer=_FakeFastTokenizer())
        framework.config = OmegaConf.create(
            {
                "trainer": {"eval_max_new_tokens": 2},
                "framework": {
                    "inference_diagnostics": {
                        "token_uncertainty": False,
                        "latent_features": False,
                    }
                },
            }
        )
        return framework

    @mock.patch(
        "starVLA.model.framework.VLM4A.QwenFast.to_pil_preserve",
        side_effect=lambda value: value,
    )
    @mock.patch(
        "starVLA.model.framework.VLM4A.QwenFast.torch.autocast",
        return_value=contextlib.nullcontext(),
    )
    def test_diagnostics_are_opt_in_and_do_not_change_actions(self, _autocast, _to_pil) -> None:
        framework = self._framework()
        examples = [{"image": [np.zeros((2, 2, 3), dtype=np.uint8)], "lang": "move"}]

        legacy = framework.predict_action(examples)
        diagnostic = framework.predict_action(
            examples,
            return_token_uncertainty=True,
            return_latent_features=True,
        )

        np.testing.assert_array_equal(legacy["normalized_actions"], diagnostic["normalized_actions"])
        self.assertNotIn("action_token_nll", legacy)
        self.assertIn("action_token_nll", diagnostic)
        self.assertIn("action_token_embedding_last", diagnostic)
        legacy_call, diagnostic_call = framework.qwen_vl_interface.model.calls
        self.assertNotIn("return_dict_in_generate", legacy_call)
        self.assertTrue(diagnostic_call["return_dict_in_generate"])
        self.assertTrue(diagnostic_call["output_scores"])
        self.assertTrue(diagnostic_call["output_hidden_states"])

    def test_policy_wrapper_forwards_qwenfast_diagnostics(self) -> None:
        wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
        wrapper._default_unnorm_key = "libero"
        wrapper._available_unnorm_keys = ["libero"]
        wrapper._framework = types.SimpleNamespace(
            predict_action=lambda **kwargs: {
                "normalized_actions": np.zeros((1, 1, 7), dtype=np.float32),
                "action_token_nll": np.asarray([[0.2, 0.4]], dtype=np.float32),
                "action_token_embedding_last": np.asarray([[1.0, 2.0]], dtype=np.float32),
                "num_action_tokens": np.asarray([2], dtype=np.int64),
            }
        )
        wrapper._get_processor = lambda key: types.SimpleNamespace(unapply_actions=lambda value: value)

        result = wrapper.predict_action(examples=[{"image": [], "lang": "move"}])

        self.assertIn("action_token_nll", result)
        self.assertIn("action_token_embedding_last", result)
        np.testing.assert_array_equal(result["num_action_tokens"], [2])


if __name__ == "__main__":
    unittest.main()
