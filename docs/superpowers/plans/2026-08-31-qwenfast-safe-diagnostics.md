# QwenFast Token Uncertainty and SAFE Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export faithful QwenFast softmax token uncertainty and final-layer action-token features, then provide LIBERO collection, token-baseline evaluation, and SAFE MLP/LSTM training under `examples/LIBERO/safe_pred/`.

**Architecture:** QwenFast remains a frozen policy and optionally emits model-native diagnostics through a small helper module. The policy wrapper and LIBERO client forward those arrays without trajectory logic. A separate `safe_pred` package owns HDF5 trajectories, common-prefix episode metrics, and learned SAFE detectors.

**Tech Stack:** Python 3.10, PyTorch, Hugging Face generation outputs, NumPy, h5py, Tyro, YAML, unittest.

**Spec:** `docs/superpowers/specs/2026-08-31-qwenfast-safe-diagnostics-design.md`

## Global Constraints

- Preserve QwenFast action generation, checkpoint parameters, and registry name.
- Do not introduce `SAFEQwenFast` or store recurrent state in the global policy.
- Compute selected-token NLL and entropy from the full vocabulary without top-k truncation.
- Filter token diagnostics and latent features to generated FAST action-token positions.
- Keep existing QwenEDL field names and HDF5 schema unchanged.
- Treat failure as the positive class for token-baseline ROC-AUC and PR-AUC.
- Do not report uncalibrated Brier score for raw NLL or entropy.
- Use task-specific common chunk prefixes to prevent LIBERO trajectory-length leakage.

---

### Task 1: Pure QwenFast Generation Diagnostics

**Files:**
- Create: `starVLA/model/framework/VLM4A/qwenfast_diagnostics.py`
- Create: `starVLA/model/framework/VLM4A/test_qwenfast_diagnostics.py`

**Interfaces:**
- Consumes: a Hugging Face generation result with `sequences`, `scores`, and optional `hidden_states`, plus inclusive action-token bounds.
- Produces: `compute_generation_diagnostics(generated, action_token_min, action_token_max, include_token_uncertainty, include_latent_features) -> dict[str, np.ndarray]`.

- [ ] **Step 1: Write failing numerical tests**

Create deterministic two-step logits and generated IDs. Assert:

```python
expected_log_probs = torch.log_softmax(logits, dim=-1)
expected_nll = -expected_log_probs.gather(-1, selected_ids[:, None]).squeeze(-1)
expected_entropy = -(expected_log_probs.exp() * expected_log_probs).sum(-1)
```

Only IDs in `[action_token_min, action_token_max]` are selected. Test first,
last, and mean aggregation from final-layer hidden vectors and a mixed batch
where one row has no action token.

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
conda run -n starvla python -m unittest starVLA.model.framework.VLM4A.test_qwenfast_diagnostics -v
```

Expected: import failure for `qwenfast_diagnostics`.

- [ ] **Step 3: Implement the pure helper**

Implement step alignment using:

```python
num_steps = len(generated.scores)
generated_ids = generated.sequences[:, -num_steps:]
step_hidden = generated.hidden_states[step_idx][-1][:, -1, :]
```

Return padded finite arrays plus `action_token_mask` and
`num_action_tokens`. Use `-1` for padded token IDs and zero for padded float
values. Raise clear `ValueError`s for missing requested generation fields or
shape disagreement.

- [ ] **Step 4: Run the focused test**

Run the command from Step 2. Expected: all tests pass.

### Task 2: Integrate Diagnostics Without Changing QwenFast Actions

**Files:**
- Modify: `starVLA/model/framework/VLM4A/QwenFast.py:47-224`
- Modify: `deployment/model_server/policy_wrapper.py:155-177`
- Create: `starVLA/model/framework/VLM4A/test_qwenfast_inference_diagnostics.py`
- Modify: `examples/LIBERO/eval_files/model2libero_interface.py:32-306`
- Create: `examples/LIBERO/eval_files/test_qwenfast_diagnostics_client.py`

**Interfaces:**
- Consumes: `return_token_uncertainty: bool` and `return_latent_features: bool` passed through `predict_action`.
- Produces: optional arrays named `action_token_nll`, `action_token_entropy`, `action_token_ids`, `action_token_mask`, `num_action_tokens`, and `action_token_embedding_{first,last,mean}`.

- [ ] **Step 1: Write failing integration tests**

Mock Qwen generation and FAST decoding. Verify the disabled call uses the
legacy tensor return path, while the enabled call uses
`return_dict_in_generate=True`, `output_scores=True`, and
`output_hidden_states=True` as needed. Assert both calls decode identical token
IDs and actions. Test policy-wrapper forwarding and client trimming by
`num_action_tokens`.

- [ ] **Step 2: Run integration tests and confirm failure**

```bash
conda run -n starvla python -m unittest \
  starVLA.model.framework.VLM4A.test_qwenfast_inference_diagnostics \
  examples.LIBERO.eval_files.test_qwenfast_diagnostics_client -v
```

- [ ] **Step 3: Add opt-in QwenFast configuration and inference output**

Add defaults:

```python
inference_diagnostics: dict = field(default_factory=lambda: {
    "token_uncertainty": False,
    "latent_features": False,
})
```

Per-call flags override configuration. When diagnostics are disabled, preserve
the current `generate()` call. When enabled, request only the needed scores or
hidden states and merge `compute_generation_diagnostics(...)` into the return
dictionary.

- [ ] **Step 4: Forward and normalize optional arrays**

Add all new diagnostic names to `PolicyWrapper.predict_action`. Extend
`ModelClient` with constructor flags, include them in the server payload, trim
token rows using `num_action_tokens`, and expose a new
`chunk_diagnostics` record without assigning SAFE quantities to existing EDL
AU/EU names.

- [ ] **Step 5: Run integration and existing uncertainty tests**

```bash
conda run -n starvla python -m unittest \
  starVLA.model.framework.VLM4A.test_qwenfast_diagnostics \
  starVLA.model.framework.VLM4A.test_qwenfast_inference_diagnostics \
  examples.LIBERO.eval_files.test_qwenfast_diagnostics_client \
  examples.LIBERO.eval_files.test_libero_uncertainty_dataset -v
```

### Task 3: SAFE HDF5 Schema and LIBERO Collector

**Files:**
- Create: `examples/LIBERO/safe_pred/__init__.py`
- Create: `examples/LIBERO/safe_pred/storage.py`
- Create: `examples/LIBERO/safe_pred/collect_libero_safe.py`
- Create: `examples/LIBERO/safe_pred/collect_libero_safe.sh`
- Create: `examples/LIBERO/safe_pred/tests/__init__.py`
- Create: `examples/LIBERO/safe_pred/tests/test_storage.py`
- Create: `examples/LIBERO/safe_pred/tests/test_collect_libero_safe.py`

**Interfaces:**
- Consumes: one `ModelClient.chunk_diagnostics` mapping per newly generated action chunk.
- Produces: one suite-level HDF5 file with atomic episode writes and flattened token arrays plus `[num_chunks, hidden_dim]` feature matrices.

- [ ] **Step 1: Write failing storage tests**

Test two variable-token chunks and assert datasets:

```text
chunk_idx, policy_step, env_step, num_action_tokens, token_offsets,
action_token_ids, action_token_nll, action_token_entropy,
embedding_first, embedding_last, embedding_mean
```

Assert finite values, matching token lengths, constant hidden dimension,
strictly increasing chunk IDs, duplicate rejection, resume behavior, and final
`success`/`task_id` attributes.

- [ ] **Step 2: Run storage tests and confirm failure**

```bash
conda run -n starvla python -m unittest examples.LIBERO.safe_pred.tests.test_storage -v
```

- [ ] **Step 3: Implement atomic HDF5 writer**

Use `_in_progress/<episode>` followed by `h5py.File.move` and flush after each
episode. Store schema version `1.0`, checkpoint path, suite, seed, collection
ID, action chunk size, and hidden dimension as metadata. Close through context
management and `atexit` so the EGL destructor warning cannot discard data.

- [ ] **Step 4: Implement no-video LIBERO collection**

Adapt the existing LIBERO loop but omit plots, JSON, and videos. Instantiate:

```python
ModelClient(
    return_token_uncertainty=True,
    return_latent_features=True,
    ...,
)
```

Append diagnostics only when `new_chunk` is true. Finalize the episode before
environment teardown and print the absolute dataset path after each flush and
at clean exit.

- [ ] **Step 5: Add the shell entrypoint and argument tests**

`collect_libero_safe.sh` follows `collect_libero_dataset.sh`, defaults to 10
trials per task (100 trajectories for a 10-task suite), writes beneath
`examples/LIBERO/safe_pred/datasets/`, supports
`OVERWRITE`/`RESUME`, and produces no videos. Test shell syntax and the Python
argument validation without importing MuJoCo.

- [ ] **Step 6: Run collector tests**

```bash
bash -n examples/LIBERO/safe_pred/collect_libero_safe.sh
conda run -n starvla python -m unittest \
  examples.LIBERO.safe_pred.tests.test_storage \
  examples.LIBERO.safe_pred.tests.test_collect_libero_safe -v
```

### Task 4: Training-Free Token Uncertainty Evaluation

**Files:**
- Create: `examples/LIBERO/safe_pred/metrics.py`
- Create: `examples/LIBERO/safe_pred/evaluate_token_uncertainty.py`
- Create: `examples/LIBERO/safe_pred/tests/test_metrics.py`
- Create: `examples/LIBERO/safe_pred/tests/test_evaluate_token_uncertainty.py`

**Interfaces:**
- Consumes: one or more SAFE HDF5 suite files.
- Produces: JSON containing task prefix lengths, failure prevalence, episode counts, ROC-AUC, and failure-positive average precision for four raw scores.

- [ ] **Step 1: Write failing metric and leakage-control tests**

Construct two tasks with unequal episode lengths. Assert each task uses its
minimum chunk count, then compute:

```python
chunk_score = reducer(token_values_for_chunk)
episode_score = max(chunk_score[:task_common_prefix])
```

Test max/mean NLL, max/mean entropy, tied-score ROC-AUC, grouped average
precision, one-class handling, and `failure = 1 - success`.

- [ ] **Step 2: Run tests and confirm failure**

```bash
conda run -n starvla python -m unittest \
  examples.LIBERO.safe_pred.tests.test_metrics \
  examples.LIBERO.safe_pred.tests.test_evaluate_token_uncertainty -v
```

- [ ] **Step 3: Implement ranking metrics and CLI**

Use dependency-free NumPy ranking implementations consistent with
`examples/LIBERO/edl_pred/metrics.py`, but accept unrestricted finite failure
scores. Save deterministic JSON and explicitly set `brier` to `null` with a
reason field stating that raw token uncertainty is not a failure probability.

- [ ] **Step 4: Run token evaluation tests**

Run the command from Step 2. Expected: all tests pass.

### Task 5: SAFE Dataset, MLP/LSTM, Training, and Evaluation

**Files:**
- Create: `examples/LIBERO/safe_pred/config.py`
- Create: `examples/LIBERO/safe_pred/dataset.py`
- Create: `examples/LIBERO/safe_pred/model.py`
- Create: `examples/LIBERO/safe_pred/train.py`
- Create: `examples/LIBERO/safe_pred/evaluate.py`
- Create: `examples/LIBERO/safe_pred/configs/default.yaml`
- Create: `examples/LIBERO/safe_pred/configs/smoke.yaml`
- Create: `examples/LIBERO/safe_pred/tests/test_dataset.py`
- Create: `examples/LIBERO/safe_pred/tests/test_model.py`
- Create: `examples/LIBERO/safe_pred/tests/test_train_smoke.py`

**Interfaces:**
- Consumes: SAFE HDF5 episodes and `feature_aggregation in {first,last,mean}`.
- Produces: per-chunk failure logits/probabilities shaped `[batch, chunks]`, masked BCE training artifacts, and episode-level validation metrics.

- [ ] **Step 1: Write failing dataset/model tests**

Assert episode-level random train/validation splitting within each suite,
padding masks, hidden-dimension checks, and deterministic seeds. For LSTM,
assert one output per valid chunk and that changing a historical feature can
change a later output. For MLP, assert chunkwise outputs and SAFE cumulative
score semantics.

- [ ] **Step 2: Run tests and confirm failure**

```bash
conda run -n starvla python -m unittest \
  examples.LIBERO.safe_pred.tests.test_dataset \
  examples.LIBERO.safe_pred.tests.test_model -v
```

- [ ] **Step 3: Implement configuration, datasets, and models**

The default LSTM is:

```python
nn.LSTM(input_size=feature_dim, hidden_size=256, num_layers=1, batch_first=True)
nn.Linear(256, 1)
```

Use logits for weighted BCE and sigmoid only for metrics. The MLP has two
hidden layers of width 256. Neither model receives task ID, chunk ID, or
elapsed time as an input feature.

- [ ] **Step 4: Implement training and evaluation artifacts**

Use Adam, batch size 64, inverse-frequency positive weighting, configurable L2
weight, fixed epochs, and optional validation early stopping disabled by
default. Save config, split manifest, best/final checkpoints, epoch JSONL, and
summary JSON in a unique run directory. Evaluate common-prefix episode ROC-AUC,
PR-AUC, and Brier from LSTM failure probabilities.

- [ ] **Step 5: Add and run CPU smoke training**

Generate a tiny temporary HDF5 dataset, train both backbones for two epochs,
reload each checkpoint, and verify finite per-chunk probabilities and metrics.

```bash
conda run -n starvla python -m unittest \
  examples.LIBERO.safe_pred.tests.test_dataset \
  examples.LIBERO.safe_pred.tests.test_model \
  examples.LIBERO.safe_pred.tests.test_train_smoke -v
```

### Task 6: Documentation and Full Verification

**Files:**
- Create: `examples/LIBERO/safe_pred/README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: all prior commands and public CLIs.
- Produces: reproducible collection, baseline evaluation, SAFE training, and SAFE evaluation commands.

- [ ] **Step 1: Document workflows and score definitions**

Document policy-server checkpoint requirements, collector environment split
(`starvla` server and `libero` client), all four token score formulas, common
prefix behavior, SAFE feature layer, config examples, output paths, and why raw
token Brier is unavailable.

- [ ] **Step 2: Run the complete non-GPU suite**

```bash
conda run -n starvla python -m unittest discover \
  -s examples/LIBERO/safe_pred/tests -p 'test_*.py' -v
conda run -n starvla python -m unittest \
  starVLA.model.framework.VLM4A.test_qwenfast_diagnostics \
  starVLA.model.framework.VLM4A.test_qwenfast_inference_diagnostics \
  examples.LIBERO.eval_files.test_qwenfast_diagnostics_client -v
bash -n examples/LIBERO/safe_pred/collect_libero_safe.sh
```

- [ ] **Step 3: Run one real QwenFast diagnostic smoke inference when GPU capacity permits**

Start the existing policy server with the original QwenFast checkpoint, run a
single collector episode, and verify the HDF5 episode contains finite NLL,
entropy, and `[num_chunks, hidden_dim]` embeddings. This smoke test may be
deferred if the required policy server is not running; all mocked and CPU tests
must still pass.
