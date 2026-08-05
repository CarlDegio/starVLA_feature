# LIBERO EDL Trajectory Verifier

Train a causal verifier on per-action-token aleatoric uncertainty (AU) and
epistemic uncertainty (EU) exported by the LIBERO collector. Each training
example is a complete episode; the verifier predicts the final binary success
label after every valid action chunk.

## Run

```bash
conda activate starvla
python -m pip install -r examples/LIBERO/edl_pred/requirements.txt
python examples/LIBERO/edl_pred/train.py \
  --config examples/LIBERO/edl_pred/configs/default.yaml
```

The smaller CPU-oriented config is useful for an environment check:

```bash
python examples/LIBERO/edl_pred/train.py \
  --config examples/LIBERO/edl_pred/configs/smoke.yaml
```

The entrypoint also works as a package module:

```bash
python -m examples.LIBERO.edl_pred.train \
  --config examples/LIBERO/edl_pred/configs/smoke.yaml
```

## Configuration

`data.datasets` maps suite names to collector-format HDF5 files and
`data.selected_suites` selects the suites included in the deterministic,
per-suite stratified train/validation split. Set `data.max_action_tokens` to
`auto` to resolve the largest observed chunk token count, or use an explicit
safe maximum.

`model.chunk_encoder` is one of `mlp_flat`, `token_attention_pool`, or
`token_self_attention`. Attention encoders support `none`, `sinusoidal`, and
`learned` `model.token_position_encoding`; `mlp_flat` has position-specific
parameters and does not use that setting. `model.head` is either `softmax` or
`edl`.

The model receives only ordered AU/EU action-token pairs from the current and
preceding chunks. It receives no task, language, action, timestep, chunk
index, episode length, or progress feature. The recurrent model is therefore
causal and cannot use future chunks to score an earlier chunk.

With the EDL head, evidence is transformed into a two-class Dirichlet
distribution. `success_probability` is the success Dirichlet mean;
`class_evidence`, `verifier_au`, `verifier_eu`, and
`verifier_total_evidence` describe verifier uncertainty, not the source VLA
token AU/EU inputs.

`loss.class_balance: inverse_frequency` computes normalized inverse-frequency
weights from the training split only. This changes the effective class prior,
so probability calibration can be biased. The resolved weights and warning are
stored in `resolved_config.yaml`.

## Outputs

Each run writes `<run.output_root>/<run.name>/`:

- `resolved_config.yaml`: parsed configuration plus resolved device, token
  maximum, and class-balance metadata.
- `split_manifest.json`: portable train/validation episode identities.
- `metrics.jsonl`: one episode-weighted train/validation record per epoch,
  including chunk and final-chunk validation metrics.
- `training_curves.png`: loss and available ranking curves.
- `checkpoints/best.pt`: lowest validation episode-mean total loss.
- `checkpoints/last.pt`: final completed epoch checkpoint.
- `validation_predictions.hdf5`: predictions generated only after restoring
  `best.pt`; it contains one group per validation episode and per-valid-chunk
  probabilities plus available EDL and attention diagnostics.

Runs refuse a non-empty destination by default. Set `run.overwrite: true` only
to replace that exact named run directory; other output-root directories and
source HDF5 files are never modified.

## Eight-GPU Sweep

`configs/sweep/` fixes a single all-suite, seed-7 comparison matrix. Every
run uses batch size 64, 200 epochs, the default optimizer settings, and a
unique non-overwriting destination under
`outputs/sweeps/all_suites_seed7_v1/runs/`.

| GPU | Encoder | Head | Position encoding |
| --- | --- | --- | --- |
| 0 | `mlp_flat` | `softmax` | ignored |
| 1 | `mlp_flat` | `edl` | ignored |
| 2 | `token_attention_pool` | `softmax` | `sinusoidal` |
| 3 | `token_attention_pool` | `edl` | `sinusoidal` |
| 4 | `token_self_attention` | `softmax` | `sinusoidal` |
| 5 | `token_self_attention` | `edl` | `sinusoidal` |
| 6 | `token_attention_pool` | `edl` | `none` |
| 7 | `token_attention_pool` | `edl` | `learned` |

The launcher first validates all config/run mappings and destination/log
collisions, then requires GPUs 0 through 7 to each report at least 2048 MiB
free. It writes `sweep_manifest.json` before starting any training process,
preserves partial runs after a failure, and reports every child exit status.
The actual launch is controller-owned and must not be started until review has
approved the preflighted experiment:

```bash
bash examples/LIBERO/edl_pred/run_gpu_sweep.sh
```

After all eight runs complete successfully, produce deterministic summary
artifacts with:

```bash
conda run -n starvla python examples/LIBERO/edl_pred/summarize_sweep.py \
  --sweep-root examples/LIBERO/edl_pred/outputs/sweeps/all_suites_seed7_v1
```

The summarizer rejects incomplete or malformed manifest runs, chooses the
earliest epoch when validation-loss minima tie, preserves undefined metrics as
`null`, and writes `sweep_summary.json`, `sweep_summary.csv`, and
`sweep_comparison.png` beside the manifest. It refuses existing summary files
by default; after inspecting a completed sweep, use `--overwrite` only to
replace existing regular summary files. Directories and symlinks are never
accepted as summary destinations. A summary-directory reservation is held while
all three files are staged and published; a publication failure restores the
previous generation (or leaves all three absent for the default no-overwrite
mode).

## Frozen Selective Rejection

The rejection analysis does not retrain a verifier. It uses absolute chunks
1 through 10 from each run's existing `validation_predictions.hdf5` to freeze
one global threshold for each applicable policy:

- EDL AU: accept when `AU <= tau_au`;
- EDL AU-or-EU: accept when both `AU <= tau_au` and `EU <= tau_eu`;
- EDL and softmax predictive entropy: accept when entropy is below its frozen
  threshold.

The default point maximizes calibration coverage subject to selective error
at most 20%, at least 10% coverage, and at least ten accepted episodes. Points
targeting 25%, 50%, 75%, and 90% coverage are also frozen for diagnostics.

Calibrate the completed sweep once:

```bash
conda run -n starvla python -m examples.LIBERO.edl_pred.calibrate_rejection \
  --sweep-root examples/LIBERO/edl_pred/outputs/sweeps/all_suites_seed7_v1 \
  --output examples/LIBERO/edl_pred/outputs/sweeps/all_suites_seed7_v1/rejection/rejection_calibration.json
```

Collect a new rollout set with a collection identity and seed namespace that
were not used by the verifier source data. The dedicated script collects 100
episodes from each of the four suites by default (10 tasks times 10 trials),
uses one identity across the bundle, and disables videos and normal evaluation
artifacts. It starts from LIBERO init-state index 10, while the verifier source
data uses indices 0 through 9. With `policy_server.zsh` already running:

```bash
COLLECTION_ID=rejection_test_v1 \
SEED_NAMESPACE=heldout_rejection_test_v1 \
SEED=17 \
bash examples/LIBERO/edl_pred/collect_rejection_test.sh
```

Files are written under
`examples/LIBERO/edl_pred/datasets/<collection_id>/`. An interrupted bundle can
be continued with the same identity and `RESUME=1`; existing files are never
overwritten by default. `EPISODES_PER_SUITE` is configurable but must be
divisible by the ten LIBERO tasks.

The collector defaults to seed 17 and writes `collection_id`,
`seed_namespace`, suite, checkpoint, seed, action chunk size, and server
metadata into each HDF5 root. The evaluator rejects missing identities,
mismatched suite bundles, and any test file whose path, content hash,
collection ID, or seed namespace overlaps the calibration source data.

Run frozen inference and rejection analysis on the independent HDF5 files:

```bash
conda run -n starvla python -m examples.LIBERO.edl_pred.evaluate_rejection \
  --calibration examples/LIBERO/edl_pred/outputs/sweeps/all_suites_seed7_v1/rejection/rejection_calibration.json \
  --dataset libero_spatial=/path/to/test_libero_spatial.hdf5 \
  --dataset libero_object=/path/to/test_libero_object.hdf5 \
  --dataset libero_goal=/path/to/test_libero_goal.hdf5 \
  --dataset libero_10=/path/to/test_libero_10.hdf5 \
  --output-dir examples/LIBERO/edl_pred/outputs/rejection_test_v1
```

The output contains strict JSON and CSV metrics, raw and rejected per-chunk
HDF5 predictions, risk-coverage and fixed-chunk plots, suite-level support,
and paired episode-bootstrap comparisons between matched EDL AU and softmax
entropy. `UNDETERMINED` chunks count against coverage but are excluded from
conditional selective accuracy. Chunks after 10 are retained with explicit
incomplete-support counts and are diagnostic rather than primary.
