"""Experimental FAST recovery for real inference; never pads coefficients."""
import logging

import numpy as np
from scipy.fft import idct


def decode_fast_chunk(tokenizer, tokens, *, model_horizon, execution_horizon, action_dim=14, mode="truncate"):
    """Decode one sample, retaining complete frequency rows on length errors.

    A shorter inverse DCT changes the time basis and amplitude scaling. This is
    an experimental reconstruction, NOT removal of bad time-domain tail steps.
    Invalid Unicode and insufficient coefficients are explicit errors, never
    zero-action fallbacks. Return the actual decoded horizon and audit fields.
    """
    if mode not in ("truncate", "strict"):
        raise ValueError(f"Unknown FAST decode mode: {mode}")
    if not 1 <= execution_horizon <= model_horizon or action_dim < 1:
        raise ValueError("Invalid FAST model/execution horizon or action dimension")
    if len(tokens) != 1:
        raise ValueError("Real FAST recovery supports a single observation per request")
    ids = [] if tokens[0] is None else list(tokens[0])
    raw = tokenizer.bpe_tokenizer.decode(ids)
    count, expected = len(raw), model_horizon * action_dim
    context = f"coefficients={count}, expected={expected}, FAST token IDs={ids}"
    if "\ufffd" in raw:
        positions = [i for i, char in enumerate(raw) if char == "\ufffd"]
        raise ValueError(f"FAST decode rejected: U+FFFD at coefficient positions {positions}; {context}")
    if count != expected and mode == "strict":
        raise ValueError(f"FAST decode rejected: length mismatch; {context}")
    steps = min(model_horizon, count // action_dim)
    if steps < execution_horizon:
        raise ValueError(f"FAST decode rejected: only {steps} complete frequency rows, "
                         f"need at least {execution_horizon}; {context}")
    used = steps * action_dim
    coefficients = np.asarray(list(map(ord, raw[:used])), dtype=np.float64) + tokenizer.min_token
    actions = idct(coefficients.reshape(steps, action_dim) / tokenizer.scale, axis=0, norm="ortho")
    if not np.isfinite(actions).all():
        raise ValueError(f"FAST decode rejected: nonfinite reconstructed actions; {context}")
    recovered = count != expected
    if recovered:
        logging.warning("FAST truncate recovery: %d -> %d coefficients, inverse DCT horizon=%d "
                        "(model=%d), return first %d steps; no zero padding",
                        count, used, steps, model_horizon, execution_horizon)
    diagnostics = {
        "fast_decode_recovered": np.asarray([recovered]),
        "fast_decode_coefficient_count": np.asarray([count], dtype=np.int32),
        "fast_decode_expected_coefficients": np.asarray([expected], dtype=np.int32),
        "fast_decode_used_coefficients": np.asarray([used], dtype=np.int32),
        "fast_decode_dropped_coefficients": np.asarray([count - used], dtype=np.int32),
        "fast_decode_horizon": np.asarray([steps], dtype=np.int32),
    }
    return actions[None], diagnostics
