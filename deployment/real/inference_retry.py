"""Bounded client retries for invalid FAST Unicode, with optional fresh inputs."""
import logging

import numpy as np


def infer_with_token_retries(infer_once, observation, *, max_retries=3, retry_observation=None):
    """One initial request and at most max_retries additional requests.

    Only the explicit invalid-Unicode FAST rejection is retryable. A live
    caller can capture and persist a new observation in retry_observation;
    recorded clients repeat the same input. Generation settings stay unchanged.
    """
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    for attempt in range(max_retries + 1):
        try:
            result = infer_once(observation)
        except (RuntimeError, ValueError) as exc:
            if "FAST decode rejected: U+FFFD at coefficient positions" not in str(exc):
                raise
            if attempt == max_retries:
                logging.warning("Invalid FAST tokens persisted after %d attempts", attempt + 1)
                raise
            logging.warning("Invalid FAST tokens: retry %d/%d", attempt + 1, max_retries)
            if retry_observation is not None:
                observation = retry_observation(attempt + 1, exc)
            continue
        diagnostics = result.setdefault("diagnostics", {})
        diagnostics["fast_decode_inference_attempts"] = np.asarray([attempt + 1], dtype=np.int32)
        diagnostics["fast_decode_retry_count"] = np.asarray([attempt], dtype=np.int32)
        return result
