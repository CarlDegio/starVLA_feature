"""CPU-only FAST recovery tests; no model weights or robot drivers."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.fft import idct

from deployment.real.fast_decode import decode_fast_chunk
from deployment.real.test_yam_policy import observation
from deployment.real.yam_policy import STAR_ACTION_KEYS, STAR_TO_YAM, YAMStarVLAPolicy


class Tokenizer:
    min_token, scale = -354, 10

    def __init__(self, length=420, raw=None):
        self.raw = raw if raw is not None else ''.join(chr(354 + i % 11) for i in range(length))
        self.bpe_tokenizer = SimpleNamespace(decode=lambda ids: self.raw)


class DecodeTests(unittest.TestCase):
    def decode(self, tok, **kwargs):
        return decode_fast_chunk(tok, [[1, 2]], model_horizon=30, execution_horizon=15, **kwargs)

    def test_valid_trajectory_matches_original_inverse_dct(self):
        tok = Tokenizer()
        result, audit = self.decode(tok)
        expected = idct((np.asarray(list(map(ord, tok.raw))) - 354).reshape(30, 14) / 10,
                        axis=0, norm='ortho')
        np.testing.assert_array_equal(result[0], expected)
        self.assertFalse(audit['fast_decode_recovered'][0])

    def test_incomplete_rows_are_truncated_not_padded(self):
        for count, steps in ((418, 29), (419, 29), (421, 30), (433, 30), (406, 29), (210, 15)):
            with self.subTest(count=count):
                tok = Tokenizer(count)
                result, audit = self.decode(tok)
                expected = idct((np.asarray(list(map(ord, tok.raw[:steps * 14]))) - 354)
                                .reshape(steps, 14) / 10, axis=0, norm='ortho')
                np.testing.assert_array_equal(result[0], expected)
                self.assertEqual(result.shape, (1, steps, 14))
                self.assertEqual(audit['fast_decode_dropped_coefficients'][0], count - steps * 14)
                self.assertTrue(audit['fast_decode_recovered'][0])

    def test_rejects_replacement_character_short_and_strict_bad_length(self):
        for raw in ('', 'a' * 30, 'a' * 209, 'a' * 25 + '\ufffd' + 'a' * 394):
            with self.subTest(length=len(raw)), self.assertRaisesRegex(ValueError, 'FAST decode rejected'):
                self.decode(Tokenizer(raw=raw))
        with self.assertRaisesRegex(ValueError, 'length mismatch'):
            self.decode(Tokenizer(418), mode='strict')

    def test_real_adapter_uses_callback_and_returns_recovered_prefix(self):
        class Wrapper:
            metadata = {'action_keys': STAR_ACTION_KEYS, 'action_chunk_size': 30}
            def predict_action(self, **kwargs):
                return {'actions': kwargs['fast_action_decoder'](Tokenizer(418), [[1, 2]])}
        adapter = YAMStarVLAPolicy(Wrapper(), task='slippers', default_prompt='test',
                                  execution_horizon=15, fast_decode_mode='truncate')
        result = adapter.infer({**observation(), 'return_edl_details': True})
        expected, _ = self.decode(Tokenizer(418))
        np.testing.assert_allclose(result['actions'], expected[0, :15][:, STAR_TO_YAM], atol=1e-7)
        self.assertEqual(result['diagnostics']['full_model_actions'].shape, (29, 14))
        self.assertEqual(result['diagnostics']['fast_decode_horizon'][0], 29)

    def test_framework_invokes_decoder_before_wrapper_unnormalization(self):
        import torch
        from omegaconf import OmegaConf
        from starVLA.model.framework.VLM4A.QwenEDL import Qwenvl_EDL
        model = Qwenvl_EDL.__new__(Qwenvl_EDL)
        torch.nn.Module.__init__(model)
        model.config = OmegaConf.create({'trainer': {'eval_max_new_tokens': 256}})
        model._build_instructions = lambda examples: ['test']
        model.qwen_vl_interface = SimpleNamespace(
            build_qwenvl_inputs=lambda **kwargs: {},
            model=SimpleNamespace(generate=lambda **kwargs: SimpleNamespace(sequences=torch.tensor([[151670]]))))
        model.action_model = SimpleNamespace(fast_tokenizer=Tokenizer(418))
        model._extract_action_token_ids = lambda generated: [[151670]]
        model._decode_action_tokens = lambda generated: [[1]]
        model._compute_generation_uncertainty = lambda generated: (np.zeros((1,)),) * 10
        callback = lambda tok, ids: decode_fast_chunk(tok, ids, model_horizon=30, execution_horizon=15)[0]
        with patch('torch.autocast', lambda *args, **kwargs: nullcontext()):
            result = model.predict_action([{'image': [np.zeros((24, 32, 3), np.uint8)], 'lang': 'test'}],
                                          fast_action_decoder=callback)
        self.assertEqual(result['normalized_actions'].shape, (1, 29, 14))


if __name__ == '__main__':
    unittest.main()
