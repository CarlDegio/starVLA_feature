"""Real FAST recovery with a fixed model horizon for inverse DCT."""
import logging

import numpy as np
from scipy.fft import idct


def decode_fast_chunk(tokenizer, tokens, *, model_horizon, execution_horizon, action_dim=14, mode="truncate"):
    """Decode one sample, padding/truncating coefficients to the model horizon.

    The legacy ``truncate`` mode name is retained for launcher compatibility.
    Padding happens after restoring ``min_token``, so missing coefficients are
    actual frequency-space zeros. Inverse DCT always uses the model horizon;
    the caller selects the execution prefix afterwards. ``strict`` rejects
    length mismatches. Invalid Unicode/nonfinite actions remain explicit errors.
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
    used = min(count, expected)
    padded, dropped = expected - used, count - used
    coefficients = np.zeros(expected, dtype=np.float64)
    coefficients[:used] = np.asarray(list(map(ord, raw[:used])), dtype=np.float64) + tokenizer.min_token
    actions = idct(coefficients.reshape(model_horizon, action_dim) / tokenizer.scale, axis=0, norm="ortho")
    if not np.isfinite(actions).all():
        raise ValueError(f"FAST decode rejected: nonfinite reconstructed actions; {context}")
    recovered = count != expected
    if recovered:
        logging.warning("FAST coefficient recovery: %d -> %d coefficients (zero-padded=%d, dropped=%d), "
                        "inverse DCT horizon=%d, return first %d steps",
                        count, expected, padded, dropped, model_horizon, execution_horizon)
    diagnostics = {
        "fast_decode_recovered": np.asarray([recovered]),
        "fast_decode_coefficient_count": np.asarray([count], dtype=np.int32),
        "fast_decode_expected_coefficients": np.asarray([expected], dtype=np.int32),
        "fast_decode_used_coefficients": np.asarray([used], dtype=np.int32),
        "fast_decode_dropped_coefficients": np.asarray([dropped], dtype=np.int32),
        "fast_decode_padded_coefficients": np.asarray([padded], dtype=np.int32),
        "fast_decode_horizon": np.asarray([model_horizon], dtype=np.int32),
    }
    return actions[None], diagnostics
