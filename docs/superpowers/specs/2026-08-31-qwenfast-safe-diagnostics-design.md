# QwenFast Token Uncertainty and SAFE Diagnostics Design

## Objective

Extend the original StarVLA QwenFast policy with optional, inference-only
diagnostics that support two failure-detection experiments:

1. training-free token uncertainty baselines based on generated action-token
   negative log likelihood and predictive entropy; and
2. SAFE-style learned failure detectors based on the final VLM layer's action
   token hidden states.

The QwenFast policy weights, action generation behavior, and checkpoint format
must remain unchanged. SAFE models are external consumers of frozen QwenFast
features, not new policy frameworks.

## Architectural Boundaries

### QwenFast policy

`starVLA/model/framework/VLM4A/QwenFast.py` remains responsible for image and
instruction preprocessing, autoregressive FAST-token generation, action-token
decoding, and returning normalized actions. It may request and return optional
diagnostic data, but it must not:

- own a SAFE classifier;
- load a SAFE checkpoint;
- maintain recurrent state across environment episodes;
- aggregate scores across chunks or trajectories; or
- compute evaluation metrics.

No `SAFEQwenFast` framework is introduced. QwenFast with diagnostics enabled
and QwenFast with diagnostics disabled use the same registry entry and policy
checkpoint.

### Generation diagnostics

Add a focused helper module:

`starVLA/model/framework/VLM4A/qwenfast_diagnostics.py`

It converts generation outputs into model-native observables. It owns:

- generated-token and action-token position alignment;
- selected-token negative log likelihood;
- full-vocabulary predictive entropy;
- selection of hidden states at generated action-token positions; and
- first, last, and mean aggregation of final-layer action-token features.

It does not know about LIBERO episodes, success labels, or detector models.

### SAFE research workflow

Add an independent package:

`examples/LIBERO/safe_pred/`

It owns rollout collection, HDF5 storage, SAFE datasets, MLP/LSTM detector
training, offline evaluation, and optional online verifier state. It consumes
diagnostics returned by QwenFast and does not modify QwenFast weights.

## QwenFast Inference Interface

Diagnostics are opt-in through a framework configuration and an optional
per-call override. Normal evaluation keeps diagnostics disabled by default.

The initial configuration surface is:

```yaml
framework:
  inference_diagnostics:
    token_uncertainty: false
    latent_features: false
```

When either diagnostic is enabled, generation must return enough information
to align each newly generated token with its generation-step logits. Latent
feature extraction must retain only the final layer needed by SAFE rather than
materializing every transformer layer after generation when the underlying
model API permits this.

The returned dictionary may contain:

```text
normalized_actions
action_token_nll
action_token_entropy
action_token_embedding_first
action_token_embedding_last
action_token_embedding_mean
action_token_ids
```

All token-level arrays are filtered to actual FAST action-token positions.
Selected-token NLL and predictive entropy are computed from the
full-vocabulary softmax distribution at those positions; they do not use
top-k truncation. Embeddings are taken from the final VLM representation fed
to the language-model head, before projection to vocabulary logits.

For batches with no generated action token, diagnostics return correctly
shaped empty token arrays and no invalid embedding. Existing action decoding
behavior remains authoritative for handling such generations.

## Token Uncertainty Baseline

For each generated action chunk at policy call `t`, compute four scalar scores
from the action-token arrays:

```text
max_nll(t)      = max_i action_token_nll[t, i]
mean_nll(t)     = mean_i action_token_nll[t, i]
max_entropy(t)  = max_i action_token_entropy[t, i]
mean_entropy(t) = mean_i action_token_entropy[t, i]
```

The collector stores raw per-token values and token counts. Aggregation is
performed offline so definitions can be audited and changed without rerunning
the policy.

For an episode, each baseline's failure score is the maximum chunk score over
a common evaluation prefix. To prevent LIBERO rollout-length leakage, all
episodes of a task use the same prefix length, defined as the minimum collected
chunk count for that task unless an explicit fixed prefix is configured.

The episode label is `1` for failure and `0` for success. Report ROC-AUC and
failure-positive average precision (PR-AUC). Raw NLL and entropy are ranking
scores rather than failure probabilities, so uncalibrated Brier score is not
reported as a comparable probabilistic metric.

## SAFE Features and Models

At each action-chunk generation call, final-layer hidden vectors at generated
action-token positions form a matrix of shape `[num_action_tokens, hidden_dim]`.
The collector stores first, last, and mean aggregations. The paper-faithful
OpenVLA-style primary feature is the last-token vector; first and mean are
available as declared ablations.

Each SAFE trajectory consists of:

```text
embedding_{first,last,mean}: [num_chunks, hidden_dim]
chunk_count: scalar
task_id: scalar or string metadata
success: scalar
```

The primary SAFE-LSTM has one recurrent layer with hidden size 256, followed by
a scalar linear projection and sigmoid. It predicts at every chunk and uses
the final episode label at every valid chunk. Padding is excluded with masks.
The primary loss is class-frequency-weighted binary cross entropy with L2
regularization.

The SAFE-MLP has two layers with hidden size 256 and processes one chunk feature
at a time. Its exact cumulative score and training objective follow the SAFE
paper and are kept separate from the LSTM implementation so their score
semantics are explicit.

Training and validation splits are made at episode level. Evaluation uses the
same common-prefix and episode-level maximum-score protocol as token
uncertainty, permitting direct ROC-AUC and PR-AUC comparisons.

## Collection and Deployment Flow

The policy server wrapper transparently forwards diagnostic arrays when they
are present. Existing clients remain compatible when they are absent.

The SAFE collector records one HDF5 dataset per LIBERO suite and produces no
videos or standard evaluation result artifacts. Each action chunk is written
once, with the final success label attached when the episode ends. Collection
uses an original QwenFast checkpoint for the softmax baseline and SAFE latent
features.

Offline training reads only the HDF5 files. Online SAFE-LSTM inference, when
enabled later, keeps recurrent state outside the global QwenFast framework and
resets it at every episode boundary. A client-local verifier is the default;
server-side recurrent state would require explicit session IDs before it is
safe for concurrent environments.

## Compatibility and Failure Handling

- Diagnostics disabled must produce the same generated token IDs and decoded
  actions as the current QwenFast path.
- No new trainable QwenFast parameters or state-dict keys are introduced.
- Diagnostic arrays must contain finite values and preserve batch boundaries.
- Empty and variable-length action-token sequences must be represented without
  conflating padding with valid values.
- Collector writes must be finalized even when LIBERO emits the known ignored
  EGL destructor exception during interpreter shutdown.
- Existing QwenEDL diagnostic field names retain their current semantics;
  softmax NLL/entropy use distinct names and are never stored as EDL AU/EU.

## Verification

Unit tests cover:

1. selected-token NLL and full-vocabulary entropy against direct PyTorch
   calculations;
2. action-token filtering and generated-step alignment for variable lengths;
3. first, last, and mean final-layer feature aggregation;
4. empty-generation behavior and batched ragged outputs;
5. identical QwenFast actions with diagnostics enabled and disabled on a
   deterministic mocked generation;
6. policy-wrapper forwarding of optional diagnostics;
7. HDF5 round trips for token and latent trajectories;
8. SAFE padding masks, per-chunk outputs, recurrent reset, and training smoke
   tests; and
9. episode score aggregation with task-specific common prefixes and metric
   direction (`failure=1`, larger score means more failure-like).

A final smoke test in the `starvla` environment loads the original QwenFast
checkpoint, performs one diagnostic inference, and confirms finite token
statistics and a final-layer embedding with the expected hidden dimension.
