# EDL Verifier Selective Rejection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add frozen-checkpoint AU/AU-or-EU rejection calibration and independent fixed-chunk evaluation that fairly compares EDL with matched softmax entropy.

**Architecture:** Keep training untouched. Pure NumPy modules define uncertainty scores, decisions, threshold search, risk-coverage metrics, and paired episode bootstrap; thin HDF5/checkpoint adapters load existing validation predictions and run frozen verifiers on independent collector datasets. Two CLIs publish immutable calibration and test-analysis artifacts.

**Tech Stack:** Python 3.10, NumPy, PyTorch, h5py, matplotlib, PyYAML, pytest.

---

## File Structure

- Create `examples/LIBERO/edl_pred/rejection.py`: policy definitions, uncertainty scores, decisions, threshold candidates, and exact calibration.
- Create `examples/LIBERO/edl_pred/rejection_metrics.py`: fixed-chunk metrics, risk-coverage, AURC, suite macro aggregation, and paired bootstrap.
- Create `examples/LIBERO/edl_pred/rejection_data.py`: strict readers for validation predictions, collector datasets, checkpoints, and identity hashes.
- Create `examples/LIBERO/edl_pred/rejection_inference.py`: frozen checkpoint reconstruction and test prediction.
- Create `examples/LIBERO/edl_pred/rejection_artifacts.py`: atomic JSON/CSV/HDF5 publication and plots.
- Create `examples/LIBERO/edl_pred/calibrate_rejection.py`: calibration CLI.
- Create `examples/LIBERO/edl_pred/evaluate_rejection.py`: independent test CLI.
- Modify `examples/LIBERO/edl_pred/dataset.py`: expose strict all-episode suite indexing for frozen inference.
- Modify `examples/LIBERO/eval_files/eval_libero.py`: include collection identity metadata.
- Modify `examples/LIBERO/eval_files/collect_libero_dataset.sh`: expose collection ID and seed namespace.
- Modify `examples/LIBERO/edl_pred/README.md`: document calibration, collection, and frozen evaluation commands.
- Add focused tests under `examples/LIBERO/edl_pred/tests/` and update collector tests under `examples/LIBERO/eval_files/`.

### Task 1: Pure Rejection Policies

**Files:**
- Create: `examples/LIBERO/edl_pred/rejection.py`
- Create: `examples/LIBERO/edl_pred/tests/test_rejection.py`

- [ ] **Step 1: Write failing score and decision tests**

Test normalized binary entropy, AU-only boundaries, AU-or-EU OR semantics, softmax entropy rejection, and exact rejection reasons:

```python
def test_dual_policy_rejects_when_either_threshold_is_exceeded():
    result = apply_rejection(
        success_probability=np.array([0.8, 0.2, 0.7, 0.1]),
        au=np.array([0.2, 0.9, 0.2, 0.9]),
        eu=np.array([0.1, 0.1, 0.8, 0.8]),
        policy=PolicyThresholds("au_or_eu", tau_au=0.5, tau_eu=0.5),
    )
    assert result.accepted.tolist() == [True, False, False, False]
    assert result.reason.tolist() == ["accepted", "high_au", "high_eu", "high_au_and_eu"]
    assert result.decision.tolist() == ["SUCCESS", "UNDETERMINED", "UNDETERMINED", "UNDETERMINED"]
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `/mnt/miniconda3/envs/libero/bin/python -m pytest -q examples/LIBERO/edl_pred/tests/test_rejection.py`

Expected: collection/import failure because `rejection.py` does not exist.

- [ ] **Step 3: Implement validated immutable policy types**

Provide:

```python
@dataclass(frozen=True)
class PolicyThresholds:
    kind: Literal["au", "au_or_eu", "predictive_entropy"]
    tau_au: float | None = None
    tau_eu: float | None = None
    tau_entropy: float | None = None

@dataclass(frozen=True)
class RejectionResult:
    accepted: np.ndarray
    predicted_label: np.ndarray
    decision: np.ndarray
    reason: np.ndarray

def predictive_entropy(success_probability: np.ndarray) -> np.ndarray: ...
def apply_rejection(..., policy: PolicyThresholds) -> RejectionResult: ...
```

Validate finite one-dimensional arrays, probability range, equal shapes, required EDL fields, threshold range, and inclusive acceptance at the threshold.

- [ ] **Step 4: Run focused and existing metrics tests**

Run: `/mnt/miniconda3/envs/libero/bin/python -m pytest -q examples/LIBERO/edl_pred/tests/test_rejection.py examples/LIBERO/edl_pred/tests/test_metrics.py`

Expected: PASS.

### Task 2: Deterministic Threshold Calibration

**Files:**
- Modify: `examples/LIBERO/edl_pred/rejection.py`
- Modify: `examples/LIBERO/edl_pred/tests/test_rejection.py`

- [ ] **Step 1: Write failing exact-search tests**

Cover maximum coverage under `max_selective_error=0.20`, minimum record/episode support, deterministic tie breaking, unavailable points, target coverage selection, and two-dimensional AU/EU candidates.

```python
def test_calibration_maximizes_coverage_under_error_constraint():
    point = calibrate_default_operating_point(records, PolicyKind.AU, max_selective_error=0.2)
    assert point.available
    assert point.coverage == pytest.approx(0.75)
    assert point.selective_error <= 0.2
```

- [ ] **Step 2: Run and verify RED**

Run the single new test and confirm the calibration API is missing.

- [ ] **Step 3: Implement exact candidate search**

Add `CalibrationRecords`, `OperatingPoint`, `CalibrationResult`, `candidate_thresholds`, `calibrate_policy`, and deterministic policy serialization. Search observed values plus `-inf/+inf` sentinels. AU-or-EU searches the exact Cartesian threshold product. Target coverage tie order is absolute coverage distance, lower risk, larger thresholds, then lexical order.

- [ ] **Step 4: Run all rejection tests**

Expected: PASS with exact threshold and unavailable-point assertions.

### Task 3: Risk-Coverage and Fixed-Chunk Metrics

**Files:**
- Create: `examples/LIBERO/edl_pred/rejection_metrics.py`
- Create: `examples/LIBERO/edl_pred/tests/test_rejection_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Test error ROC/PR, cumulative risk ordering, AURC, selective support, rejected error, balanced accuracy, per-suite coverage, one-class `null`, and fixed chunk support.

```python
def test_selective_metrics_keep_abstentions_out_of_accuracy_but_in_coverage():
    metrics = selective_metrics(labels, probabilities, accepted)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["accepted_count"] == 2
    assert metrics["selective_accuracy"] == pytest.approx(1.0)
    assert metrics["rejected_count"] == 2
```

- [ ] **Step 2: Verify RED**

Run the new test module and confirm import failure.

- [ ] **Step 3: Implement pure metric functions**

Reuse `binary_metrics` for standard binary metrics. Add `error_detection_metrics`, `risk_coverage_curve`, `aurc`, `dual_threshold_frontier`, `selective_metrics`, `metrics_by_absolute_chunk`, and suite-macro aggregation. Every row includes active support, class counts, survival fraction, and `incomplete_support` after the common support horizon.

- [ ] **Step 4: Add paired episode bootstrap**

Implement deterministic episode resampling with all chunks retained together:

```python
def paired_episode_bootstrap(
    episode_records: Sequence[EpisodeMetricRecord],
    statistic: Callable[[Sequence[EpisodeMetricRecord]], float | None],
    *, seed: int,
    replicates: int = 10_000,
) -> BootstrapInterval: ...
```

Skip undefined replicates, record valid replicate count, and return `null` when no valid replicate exists.

- [ ] **Step 5: Run metric tests**

Expected: PASS.

### Task 4: Strict Prediction and Checkpoint Readers

**Files:**
- Create: `examples/LIBERO/edl_pred/rejection_data.py`
- Modify: `examples/LIBERO/edl_pred/dataset.py`
- Create: `examples/LIBERO/edl_pred/tests/test_rejection_data.py`
- Modify: `examples/LIBERO/edl_pred/tests/test_dataset.py`

- [ ] **Step 1: Write failing reader/index tests**

Test public suite indexing, matched calibration identities, minimum ten chunks, EDL field requirements, softmax field exclusion, content hashes, checkpoint schema, and mismatched labels/chunk counts.

- [ ] **Step 2: Verify RED**

Run both focused modules and confirm missing APIs.

- [ ] **Step 3: Expose `index_suite_episodes`**

Wrap the existing strict `_index_suite` behavior without duplicating schema validation:

```python
def index_suite_episodes(suite: str, path: str | Path) -> tuple[tuple[EpisodeRef, ...], int]:
    refs, observed_max = _index_suite(suite, Path(path).expanduser().resolve())
    return tuple(refs), observed_max
```

- [ ] **Step 4: Implement immutable prediction/checkpoint records**

Read existing `validation_predictions.hdf5` into per-episode records containing suite, key, label, probabilities, optional AU/EU/evidence, and chunk count. Reconstruct `ModelConfig` and `EDLConfig` from the checkpoint `resolved_config`, validate `max_action_tokens`, `split_manifest_identity`, and state dict, and compute streaming SHA-256 file hashes.

- [ ] **Step 5: Run reader and dataset tests**

Expected: PASS.

### Task 5: Calibration Artifact and CLI

**Files:**
- Create: `examples/LIBERO/edl_pred/rejection_artifacts.py`
- Create: `examples/LIBERO/edl_pred/calibrate_rejection.py`
- Create: `examples/LIBERO/edl_pred/tests/test_rejection_artifacts.py`
- Create: `examples/LIBERO/edl_pred/tests/test_calibrate_rejection.py`

- [ ] **Step 1: Write failing artifact and CLI tests**

Use a temporary miniature sweep with matched EDL/softmax HDF5 predictions. Assert strict identity matching, chunks 1-10 only, default and target operating points, source hashes, atomic refusal to overwrite, and deterministic JSON bytes.

- [ ] **Step 2: Verify RED**

Run focused tests and confirm missing modules.

- [ ] **Step 3: Implement atomic calibration publication**

Add staged JSON writing with `allow_nan=False`, schema version `1.0`, canonical sorted keys, run/checkpoint/prediction identities, calibration support, policy definitions, thresholds, objectives, and unavailable reasons.

- [ ] **Step 4: Implement CLI**

```bash
python -m examples.LIBERO.edl_pred.calibrate_rejection \
  --sweep-root <completed-sweep> \
  --output <rejection_calibration.json> \
  --max-selective-error 0.20 \
  --primary-chunks 1-10 \
  --analysis-seed 20260805
```

Discover canonical run directories from `sweep_manifest.json`, calibrate all matched model policies, and reject overwrite unless `--overwrite` is supplied.

- [ ] **Step 5: Run calibration tests**

Expected: PASS and byte-identical repeated outputs in separate directories.

### Task 6: Frozen Checkpoint Inference

**Files:**
- Create: `examples/LIBERO/edl_pred/rejection_inference.py`
- Create: `examples/LIBERO/edl_pred/tests/test_rejection_inference.py`

- [ ] **Step 1: Write failing toy-checkpoint inference tests**

Create a toy collector HDF5 and checkpoint from the real `TrajectoryVerifier`. Assert complete episode order, no gradients, raw prediction equality with direct model execution, EDL diagnostics, softmax omission, and token-limit failure.

- [ ] **Step 2: Verify RED**

Run focused tests and confirm the inference API is missing.

- [ ] **Step 3: Implement frozen inference**

Build `TrajectoryDataset` from `index_suite_episodes`, use existing `collate_trajectories`, restore `best.pt`, call `model.eval()` under `torch.inference_mode()`, and convert batches through `prediction_records_from_output`. Device and batch size are explicit arguments; no optimizer or training config is created.

- [ ] **Step 4: Run inference tests**

Expected: PASS on CPU.

### Task 7: Independent Collection Identity

**Files:**
- Modify: `examples/LIBERO/eval_files/eval_libero.py`
- Modify: `examples/LIBERO/eval_files/collect_libero_dataset.sh`
- Modify: `examples/LIBERO/eval_files/test_libero_uncertainty_dataset.py`
- Modify: `examples/LIBERO/eval_files/test_libero_client_script.py`

- [ ] **Step 1: Write failing metadata tests**

Require non-empty `collection_id` and `seed_namespace` when writing a rejection test dataset, verify resume equality, and assert shell forwarding of `COLLECTION_ID`, `SEED_NAMESPACE`, and `SEED`.

- [ ] **Step 2: Verify RED**

Run focused collector tests and confirm missing arguments/metadata.

- [ ] **Step 3: Implement optional collection identity arguments**

Add `collection_id: str = ""` and `seed_namespace: str = ""` to evaluation args, include both in root metadata, and expose shell defaults that create a deterministic explicit namespace from run ID, suite, and seed. Existing non-test collection remains backward compatible.

- [ ] **Step 4: Run collector tests and shell syntax check**

Run focused pytest modules and `bash -n examples/LIBERO/eval_files/collect_libero_dataset.sh`.

Expected: PASS.

### Task 8: Frozen Test Evaluation and Artifacts

**Files:**
- Modify: `examples/LIBERO/edl_pred/rejection_artifacts.py`
- Create: `examples/LIBERO/edl_pred/evaluate_rejection.py`
- Create: `examples/LIBERO/edl_pred/tests/test_evaluate_rejection.py`

- [ ] **Step 1: Write failing end-to-end synthetic test**

Calibrate on synthetic validation predictions, infer frozen toy checkpoints on independent collector files, apply thresholds without modification, and assert HDF5/JSON/CSV outputs plus all four PNG files. Change test labels and confirm calibration bytes remain unchanged.

- [ ] **Step 2: Verify RED**

Run focused integration test and confirm evaluator is missing.

- [ ] **Step 3: Implement evaluation orchestration**

CLI contract:

```bash
python -m examples.LIBERO.edl_pred.evaluate_rejection \
  --calibration <rejection_calibration.json> \
  --dataset libero_spatial=<path> \
  --dataset libero_object=<path> \
  --dataset libero_goal=<path> \
  --dataset libero_10=<path> \
  --output-dir <new-directory> \
  --device auto --batch-size 64 --bootstrap-replicates 10000
```

Validate collection/checkpoint identity and non-overlap, run all frozen models, apply saved thresholds, calculate per-chunk/per-suite/macro metrics and paired intervals, and stage publication atomically.

- [ ] **Step 4: Implement artifacts and plots**

Write `rejection_test_predictions.hdf5`, `rejection_metrics.json`, flat CSV, risk-coverage, chunk-metric, AU/EU quadrant, and EDL-versus-softmax plots. Every plotted point carries support in JSON/CSV; post-chunk-10 points are styled as incomplete support.

- [ ] **Step 5: Run integration tests**

Expected: PASS with non-empty PNGs and strict schemas.

### Task 9: Documentation and Existing-Sweep Calibration Smoke Test

**Files:**
- Modify: `examples/LIBERO/edl_pred/README.md`
- Modify: `examples/LIBERO/edl_pred/tests/test_train_smoke.py` only if shared helpers require coverage

- [ ] **Step 1: Document exact workflow**

Document calibration, independent collection with a new seed namespace, frozen evaluation, artifact meanings, fixed chunk interpretation, 20% risk constraint, and the prohibition on test recalibration.

- [ ] **Step 2: Run full tests**

Run:

```bash
/mnt/miniconda3/envs/libero/bin/python -m pytest -q \
  examples/LIBERO/edl_pred/tests \
  examples/LIBERO/eval_files/test_libero_uncertainty_dataset.py \
  examples/LIBERO/eval_files/test_eval_libero_plot.py \
  examples/LIBERO/eval_files/test_libero_client_script.py
```

Expected: all tests pass with only the existing expected CUDA skip.

- [ ] **Step 3: Run static verification**

Run `python -m compileall -q examples/LIBERO/edl_pred`, shell syntax checks, and `git diff --check`.

- [ ] **Step 4: Calibrate the completed real sweep**

Run calibration against `outputs/sweeps/all_suites_seed7_v1` into a new rejection-analysis directory. Validate JSON with Python, inspect selected supports/risks, and ensure no source artifact was modified.

- [ ] **Step 5: Independent final review**

Review implementation against every design section, verify no test labels influence calibration, and report that independent test metrics remain pending until new rollout HDF5 files are collected.
