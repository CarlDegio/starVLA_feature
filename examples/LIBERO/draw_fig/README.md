# Evidence plots

Run from the repository root:

```bash
conda run -n starvla python examples/LIBERO/draw_fig/plot_evidence.py
```

By default, the script reads the two `rollout_*.jsonl` files beside it and
writes a combined 2-by-2 figure to `outputs/failure_success_evidence.png`
and a vector PDF with the same stem. The failure trajectory is the top row;
the success trajectory is the bottom row.
Optional positional paths select specific JSONL files; `--output-dir` changes
the destination.

- Left: total action-token count (gray), count with evidence strictly below
  4 (red), and their ratio (black, right axis), per chunk.
- Right: normalized selected action-token evidence, computed as
  `clip(evidence, 0, 8) / 4 - 1`. Raw values 0, 4, and 8 map to -1, 0,
  and 1; values above 8 saturate at 1. Negative points are red, other points
  green, and the red dashed threshold is 0. Axis limits are [-1.1, 1.1]
  to keep boundary points visible. Counts use raw values.

Both column legends are boxed and arranged in three rows below the bottom
panels only, shared across the failure and success rows.

The panels correspond to row 2 / column 1 and row 1 / column 3 of the original
diagnostic figure. The JSONL inputs are not modified.
