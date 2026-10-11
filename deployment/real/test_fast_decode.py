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

    def test_coefficients_are_padded_or_truncated_with_fixed_dct_horizon(self):
        for count in (*range(415, 426), 433, 406, 210, 30, 1, 0):
            with self.subTest(count=count):
                tok = Tokenizer(count)
                result, audit = self.decode(tok)
                # Build the frequency matrix independently; absent entries are
                # zero AFTER offset restoration, not char/token ID zero.
                frequencies = np.zeros((30, 14))
                for index, char in enumerate(tok.raw[:420]):
                    frequencies[index // 14, index % 14] = (ord(char) - 354) / 10
                expected = idct(frequencies, axis=0, norm='ortho')
                np.testing.assert_array_equal(result[0], expected)
                self.assertEqual(result.shape, (1, 30, 14))
                self.assertEqual(audit['fast_decode_used_coefficients'][0], min(count, 420))
                self.assertEqual(audit['fast_decode_dropped_coefficients'][0], max(count - 420, 0))
                self.assertEqual(audit['fast_decode_padded_coefficients'][0], max(420 - count, 0))
                self.assertEqual(audit['fast_decode_horizon'][0], 30)
                self.assertEqual(audit['fast_decode_recovered'][0], count != 420)

    def test_rejects_replacement_characters_and_strict_bad_length(self):
        for mode in ('truncate', 'strict'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'U\\+FFFD'):
                self.decode(Tokenizer(raw='a' * 25 + '\ufffd' + 'a' * 394), mode=mode)
        for count in (0, 13, 418, 421):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'length mismatch'):
                self.decode(Tokenizer(count), mode='strict')

    def test_missing_high_frequency_coefficients_preserve_dc_scaling(self):
        # Only the first joint's DC coefficient is nonzero. A short sequence
        # must retain the 30-step scale, rather than become a 15-step IDCT.
        tok = Tokenizer(raw=chr(364) + chr(354) * 209)
        result, _ = self.decode(tok)
        np.testing.assert_allclose(result[0, :, 0], 1 / np.sqrt(30))
        np.testing.assert_array_equal(result[0, :, 1:], 0)

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
        self.assertEqual(result['diagnostics']['full_model_actions'].shape, (30, 14))
        self.assertEqual(result['diagnostics']['fast_decode_horizon'][0], 30)
        self.assertEqual(result['diagnostics']['fast_decode_padded_coefficients'][0], 2)

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
        self.assertEqual(result['normalized_actions'].shape, (1, 30, 14))


if __name__ == '__main__':
    unittest.main()
