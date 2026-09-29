# QwenFast Token Uncertainty and SAFE Failure Prediction

This package evaluates failure prediction from a frozen, original QwenFast
policy. It supports:

- four training-free token uncertainty scores;
- SAFE-LSTM using final-layer QwenFast action-token features; and
- SAFE-MLP using the cumulative score formulation from SAFE.

The policy checkpoint is not modified or fine-tuned.

## Diagnostics

For every generated FAST action token, QwenFast can optionally return:

- `action_token_nll`: selected-token negative log likelihood from the full
  vocabulary softmax;
- `action_token_entropy`: entropy of the full vocabulary distribution; and
- `action_token_embedding_{first,last,mean}`: aggregations of final-layer
  action-token hidden states before the language-model vocabulary projection.

Diagnostics are requested by the collector. Normal QwenFast evaluation keeps
the legacy inference path and does not request generation scores or hidden
states.

## Collect A Suite

Start `policy_server.zsh` in the `starvla` environment with an original
QwenFast checkpoint, normally:

```text
playground/Checkpoints/qwen3fast_libero_all/checkpoints/<checkpoint>.pt
```

Then run the collector with the LIBERO environment:

```bash
LIBERO_HOME=/path/to/LIBERO \
LIBERO_PYTHON=/mnt/miniconda3/envs/libero/bin/python \
CKPT=/mnt/starVLA/playground/Checkpoints/qwen3fast_libero_all/checkpoints/steps_30000_pytorch_model.pt \
TASK_SUITE_NAME=libero_goal \
bash examples/LIBERO/safe_pred/collect_libero_safe.sh
```

The default is 10 episodes for each of the suite's 10 tasks, or 100
trajectories. Override `NUM_TRIALS_PER_TASK`, `EPISODE_START_INDEX`,
`MAX_TASKS`, `SEED`, `OVERWRITE`, or `RESUME` through environment variables.
The script writes only HDF5 data under `examples/LIBERO/safe_pred/datasets/`;
it does not generate videos, plots, or standard LIBERO result artifacts.

The collector verifies that its `CKPT` exactly matches the checkpoint reported
by the policy server before writing data.

## Token Uncertainty Baseline

For one action chunk with generated-token probabilities `p_i` and predictive
entropies `H_i`, the four scores are:

```text
max_nll      = max_i(-log p_i)
mean_nll     = mean_i(-log p_i)
max_entropy  = max_i(H_i)
mean_entropy = mean_i(H_i)
```

The episode score is the maximum chunk score within a task-specific common
prefix. The common prefix is the minimum collected chunk count among all
episodes of that task, preventing successful LIBERO rollouts from being
distinguished only because they terminate earlier.

```bash
conda run -n starvla python -m examples.LIBERO.safe_pred.evaluate_token_uncertainty \
  --args.dataset-paths examples/LIBERO/safe_pred/datasets/example.hdf5 \
  --args.output-path examples/LIBERO/safe_pred/results/token_uncertainty.json
```

The JSON reports failure-positive ROC-AUC, average precision (PR-AUC), failure
prevalence, and common prefix lengths. Brier is `null`: raw NLL and entropy are
ranking scores, not calibrated failure probabilities.

## Train SAFE

Edit `configs/default.yaml` or copy it to a run-specific config and set
`dataset_paths`. The paper-faithful OpenVLA-style primary feature is `last`.

```bash
conda run -n starvla python -m examples.LIBERO.safe_pred.train \
  --config-path examples/LIBERO/safe_pred/configs/default.yaml
```

Important defaults:

```text
backbone: lstm
feature_aggregation: last
hidden_dim: 256
num_layers: 1
batch_size: 64
val_fraction: 0.1 per suite, sampled at episode level
early_stopping_patience: 0 (disabled; fixed epochs)
```

The detector never receives task ID, chunk ID, or elapsed time as an input.
The LSTM outputs a failure probability at every chunk and is trained with
class-weighted BCE. The MLP produces local sigmoid scores accumulated over
time and uses the SAFE cumulative objective; its score is not treated as a
probability.

Each unique run directory contains:

```text
config.json
split_manifest.json
epochs.jsonl
best.pt
final.pt
summary.json
```

Evaluate a checkpoint with:

```bash
conda run -n starvla python -m examples.LIBERO.safe_pred.evaluate \
  --args.checkpoint-path examples/LIBERO/safe_pred/runs/<run>/best.pt \
  --args.dataset-paths examples/LIBERO/safe_pred/datasets/example.hdf5 \
  --args.output-path examples/LIBERO/safe_pred/runs/<run>/evaluation.json
```

SAFE-LSTM evaluation reports ROC-AUC, PR-AUC, and Brier. SAFE-MLP reports
ROC-AUC and PR-AUC, with Brier unavailable because its cumulative score is not
a probability.
