# AU-EU Chunk-Time Plot Design

## Goal

Extend the LIBERO uncertainty diagnostic figure from two rows by three columns
to two rows by four columns. The new bottom-right panel visualizes the same
action-token AU/EU points as the existing quadrant panel, colored by chunk
time.

## Layout

- Keep the existing six panels unchanged.
- Disable the top-right axis.
- Add the temporal AU/EU scatter plot to the bottom-right axis.
- Increase the figure width from 18 to 24 inches while retaining the existing
  height.

## Time Encoding

Every token produced by one action chunk receives that record's `chunk_idx`, so
all points from the same chunk have exactly the same color. Use Matplotlib's
`Blues` colormap: earlier chunks are light blue and later chunks are dark blue.

The color normalization must not depend on the observed episode length. The
fixed maximum chunk index is:

```text
max_chunk_idx = max(ceil(max_steps / action_chunk_size) - 1, 1)
```

Here, `max_steps` is the fixed LIBERO task-suite horizon and
`action_chunk_size` is the server handshake value already stored by
`ModelClient`. The colorbar is labeled `Chunk index`.

## Data Flow

Pass `max_steps` and `client_model.action_chunk_size` from `eval_libero()` to
`_save_uncertainty_artifacts()`. While collecting the existing AU/EU quadrant
arrays, also repeat each record's `chunk_idx` once per aligned action token.

Old or malformed records without a usable `chunk_idx` fall back to collection
order. Empty AU/EU input produces the same centered no-data message pattern as
the existing panels.

## Verification

- Run Python syntax compilation for the modified evaluation script.
- Generate a diagnostic PNG from synthetic chunks with an episode shorter than
  the configured maximum.
- Confirm that points in the same chunk share a color, later chunks are darker,
  the colorbar upper bound is the theoretical maximum chunk index, and the
  top-right axis is disabled.
