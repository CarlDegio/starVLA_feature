"""Retry bounds, error selection, and the real WebSocket reconnect protocol."""
import asyncio
import unittest

import numpy as np
from websockets.asyncio.server import serve

from deployment.real.infer import RecordedClient
from deployment.real.inference_retry import infer_with_token_retries
from deployment.real.test_yam_policy import observation, policy
from deployment.real.websocket_server import YAMWebsocketServer

INVALID = "FAST decode rejected: U+FFFD at coefficient positions [2]; coefficients=421"


class RetryTests(unittest.TestCase):
    def test_fresh_inputs_and_success_on_fourth_attempt(self):
        seen = []
        def infer(obs):
            seen.append(obs)
            if len(seen) < 4:
                raise RuntimeError(INVALID)
            return {"actions": np.ones((15, 14))}
        result = infer_with_token_retries(infer, 0, retry_observation=lambda attempt, exc: attempt)
        self.assertEqual(seen, [0, 1, 2, 3])
        self.assertEqual(result['diagnostics']['fast_decode_inference_attempts'].tolist(), [4])
        self.assertEqual(result['diagnostics']['fast_decode_retry_count'].tolist(), [3])

    def test_exhaustion_is_four_requests_and_no_fallback(self):
        seen = []
        obs = object()
        def infer(value):
            seen.append(value)
            raise RuntimeError(INVALID)
        with self.assertRaisesRegex(RuntimeError, 'U\+FFFD'):
            infer_with_token_retries(infer, obs)
        self.assertEqual(seen, [obs] * 4)

    def test_unrelated_errors_are_not_retried(self):
        for error in ('FAST decode rejected: length mismatch',
                      'FAST decode rejected: nonfinite reconstructed actions', 'connection failed'):
            seen = []
            def infer(obs):
                seen.append(obs)
                raise RuntimeError(error)
            with self.subTest(error=error), self.assertRaises(RuntimeError):
                infer_with_token_retries(infer, 0)
            self.assertEqual(seen, [0])

    def test_zero_retries_and_bad_configuration(self):
        calls = []
        def infer(obs):
            calls.append(obs)
            raise ValueError(INVALID)
        with self.assertRaises(ValueError):
            infer_with_token_retries(infer, 0, max_retries=0)
        self.assertEqual(len(calls), 1)
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                infer_with_token_retries(infer, 0, max_retries=value)


class WireRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_reconnect_after_server_closes_invalid_chunk(self):
        adapter = policy()
        original = adapter.infer
        calls = []
        def infer(obs):
            calls.append(obs)
            if len(calls) <= 2:
                raise ValueError(INVALID)
            return original(obs)
        adapter.infer = infer
        server = YAMWebsocketServer(adapter)
        async with serve(server.handler, '127.0.0.1', 0, compression=None) as listener:
            port = listener.sockets[0].getsockname()[1]
            def request():
                client = RecordedClient('127.0.0.1', port, 10)
                try:
                    return client.infer(observation())
                finally:
                    client.close()
            result = await asyncio.to_thread(request)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['actions'].shape, (15, 14))
        self.assertEqual(result['diagnostics']['fast_decode_retry_count'].tolist(), [2])

    async def test_reconnect_rejects_changed_checkpoint(self):
        adapter = policy()
        calls = []
        def infer(obs):
            calls.append(obs)
            adapter.metadata = {**adapter.metadata, 'ckpt_path':'/changed/ckpt.pt'}
            raise ValueError(INVALID)
        adapter.infer = infer
        server = YAMWebsocketServer(adapter)
        async with serve(server.handler, '127.0.0.1', 0, compression=None) as listener:
            port = listener.sockets[0].getsockname()[1]
            def request():
                client = RecordedClient('127.0.0.1', port, 10)
                try:
                    with self.assertRaisesRegex(ValueError, 'metadata changed'):
                        client.infer(observation())
                finally:
                    client.close()
            await asyncio.to_thread(request)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
