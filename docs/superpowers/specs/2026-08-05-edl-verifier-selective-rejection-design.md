# EDL Verifier Selective Rejection Design

## Objective

Extend the existing offline LIBERO trajectory-verifier evaluation with a
three-way decision policy:

- predict `SUCCESS` when a prediction is accepted and favors success;
- predict `FAILURE` when a prediction is accepted and favors failure;
- predict `UNDETERMINED` when classifier uncertainty triggers rejection.

The existing verifier checkpoints remain frozen. The current 40 validation
episodes are used only to calibrate rejection thresholds. A newly collected,
independent rollout set is used once for final evaluation without changing
models, thresholds, operating points, or metric definitions after inspection.

The primary research question is whether an EDL verifier provides a more
useful rejection ordering than a matched softmax verifier. In particular, the
evaluation must determine whether EDL AU, alone or combined with EU, achieves
greater coverage at a fixed selective error and lower area under the
risk-coverage curve than softmax predictive entropy.

## Scope

This phase adds:

- AU-only and AU-or-EU rejection policies for frozen EDL verifiers.
- Predictive-entropy rejection for both EDL and matched softmax verifiers.
- Calibration of deployable global thresholds from existing validation
  predictions.
- Frozen evaluation on newly collected rollout HDF5 files.
- Absolute-chunk metrics, risk-coverage analysis, per-suite reporting,
  bootstrap confidence intervals, and EDL-versus-softmax comparisons.
- Reproducible JSON, CSV, HDF5, and plot artifacts.

This phase does not:

- Retrain or fine-tune any trajectory verifier.
- Change the verifier architecture, loss, split, or best checkpoint.
- Add rejection decisions to `policy_server.zsh` or steer the VLA online.
- Fit a learned rejector or a third classification class.
- Treat final-chunk performance as the primary result.
- Recalibrate any threshold after examining the independent test results.

## Terminology

For an EDL binary classifier:

```text
alpha_k = evidence_k + 1
S = alpha_failure + alpha_success
probability_k = alpha_k / S
vacuity = EU = 2 / S
```

The existing `verifier_eu` field is the vacuity score and remains the canonical
field name for backward compatibility. Higher AU indicates greater expected
success/failure class ambiguity. Higher EU indicates lower total evidence.

This phase does not require ambiguity and insufficient evidence to produce
different operational actions: either can result in `UNDETERMINED`. The
rejection reason is nevertheless retained so their empirical behavior can be
analyzed separately.

## Compared Rejection Policies

### EDL AU-only

Accept a prediction only when:

```text
AU <= tau_au
```

This is the simplest test of the hypothesis that EDL aleatoric uncertainty
identifies prefixes whose eventual outcome remains ambiguous.

### EDL AU-or-EU

Accept a prediction only when:

```text
AU <= tau_au and EU <= tau_eu
```

Equivalently, the policy rejects when `AU > tau_au OR EU > tau_eu`. Every
rejection records one of `high_au`, `high_eu`, or `high_au_and_eu`.

### EDL Predictive Entropy

Compute normalized binary predictive entropy from the EDL mean probability:

```text
H_pred = -(p * log2(p) + (1 - p) * log2(1 - p))
```

This baseline determines whether AU provides information beyond probability
being close to 0.5.

### Softmax Predictive Entropy

Apply the same predictive-entropy rejection rule to the matched softmax
verifier. This is the primary non-EDL baseline.

No policy uses the ground-truth label when producing a test decision.

## Model Comparison Matrix

The primary comparison uses the strongest balanced EDL configuration and its
matched softmax encoder:

- `token_attention_pool + EDL + sinusoidal`;
- `token_attention_pool + softmax + sinusoidal`.

The same comparison is repeated with:

- `token_self_attention + EDL/softmax + sinusoidal`;
- `mlp_flat + EDL/softmax`.

The attention-pool learned-position and no-position EDL runs remain available
as diagnostics but are not primary EDL-versus-softmax comparisons because no
matched softmax run exists for those position settings.

## Calibration Data Contract

Calibration reads the existing `validation_predictions.hdf5` generated from
each run's restored `best.pt`. It must validate that compared runs contain the
same suite, episode key, label, and chunk count identities.

The calibration population is the Cartesian collection of the first ten
absolute chunks from every validation episode. These chunks are chosen because
all current validation episodes reach chunk 10. The implementation must verify
this invariant rather than assume it. If any calibration episode is shorter,
the calibration fails with an actionable error.

Each episode contributes exactly one record at each absolute chunk from 1 to
10. This prevents longer failed trajectories from receiving more calibration
weight than shorter successful trajectories.

## Global Threshold Calibration

Each policy receives one global threshold set that is applied unchanged at
every chunk. A threshold must not depend on suite, label, episode length,
relative progress, or absolute chunk index.

The default operating point maximizes calibration coverage subject to:

```text
selective_error <= 0.20
```

Candidate operating points must accept at least 10% of calibration records and
must include accepted records from at least ten distinct episodes. Among
feasible candidates, selection is deterministic with these tie breakers:

1. greater coverage;
2. lower selective error;
3. less aggressive rejection, represented by larger thresholds;
4. lexicographic threshold order for complete determinism.

For AU-only and entropy policies, candidates are observed calibration values
plus boundary sentinels. For AU-or-EU, candidates are the Cartesian product of
the observed AU and EU values plus boundary sentinels. The search space is
small enough for exact evaluation and must not use stochastic optimization.

Calibration also freezes operating points targeting 25%, 50%, 75%, and 90%
coverage. For these diagnostics, select the candidate whose coverage is
closest to the target, then prefer lower selective error and less aggressive
rejection. The 100% point is the unrejected classifier and requires no fitted
threshold.

If no candidate satisfies the default error constraint and minimum-acceptance
requirements, the default operating point is marked unavailable. The code must
not silently relax the 20% constraint.

## Independent Test Data

The final test set is collected after the policies and metrics are frozen.
Collection is configurable per suite, with 100 episodes per suite recommended
and 50 episodes per suite considered the minimum useful initial run.

Every test HDF5 file must identify:

- suite;
- collection ID;
- policy checkpoint identity;
- rollout seed or seed namespace;
- episode and task identity;
- collection configuration needed to reproduce the rollout.

Before inference, the evaluator verifies that test collection IDs and rollout
seed identities do not overlap the source data used by the verifier. It also
validates the uncertainty feature schema and rejects any action-token length
that exceeds the checkpoint's configured maximum instead of truncating it.

Test inference restores each existing `best.pt`, applies it to the new HDF5
episodes, and stores raw per-chunk predictions before applying rejection. The
calibration artifact is then applied without modification. Test labels are
used only for metrics after all decisions have been produced.

## Chunk Evaluation Protocol

Absolute chunk positions are primary because they do not require future
episode length. Results are reported separately at every chunk.

Chunks 1 through 10 form the primary comparison range. Every test episode must
contribute to a reported primary chunk; otherwise that chunk is explicitly
marked as having incomplete support.

Chunks after 10 remain valuable and are reported through the maximum observed
trajectory length. Each such row and plot point must include:

- number of active episodes;
- number of active failures and successes;
- fraction of all episodes still active;
- an incomplete-support marker.

Metrics after chunk 10 are diagnostic and are not averaged into the primary
score because success termination changes the surviving population. Relative
progress and final-chunk metrics may be retained as secondary descriptions but
must not be used to select or rank rejection policies.

The evaluator additionally publishes per-chunk oracle risk-coverage frontiers
as diagnostics. These frontiers do not define deployable thresholds and must
be visually and structurally separated from results produced by the frozen
global thresholds.

## Metrics

### Unrejected Classifier Metrics

At each chunk, suite, and suite-macro aggregation:

- accuracy and balanced accuracy at success probability 0.5;
- success ROC-AUC and PR-AUC;
- Brier score and ten-bin ECE;
- support and class counts.

### Uncertainty Error-Detection Metrics

Define prediction error after the frozen 0.5 class threshold. For every
continuous uncertainty score, report:

- error ROC-AUC;
- error PR-AUC and the empirical error-rate baseline;
- area under the risk-coverage curve (AURC).

The reported scores include EDL AU, EDL EU, EDL predictive entropy, and
softmax predictive entropy where available.

AU-only and entropy AURC are computed by sorting their continuous uncertainty
scores. AU-or-EU has no unique scalar ordering, so its risk-coverage curve is
the deterministic lower-risk envelope of all calibrated `(tau_au, tau_eu)`
candidate pairs at each attainable coverage. Its AURC is the area under that
envelope. This two-dimensional diagnostic frontier is distinct from the single
frozen AU-or-EU operating point used for test decisions.

### Selective Metrics

For every frozen operating point, report:

- coverage and abstention rate;
- accepted count and rejected count;
- selective error and selective accuracy;
- selective balanced accuracy;
- rejected-group error rate;
- accepted and rejected failure/success counts;
- per-suite coverage;
- rejection-reason counts for AU-or-EU.

`UNDETERMINED` records are excluded from the conditional selective accuracy
denominator but remain included in coverage, abstention, reason, and class
distribution statistics. A result must never report selective accuracy without
coverage and accepted support beside it.

### Confidence Intervals

Use paired episode-level bootstrap resampling with a fixed analysis seed. An
episode and all of its evaluated chunks are resampled together. Publish 95%
percentile intervals for primary metrics and paired EDL-minus-softmax metric
differences. Bootstrap replicate count is configurable and defaults to 10,000.

## Evidence for an EDL Advantage

The EDL claim is evaluated at matched encoder and checkpoint-selection
conditions. EDL is considered to provide useful additional rejection behavior
only when the independent test set supports all of the following:

1. The frozen EDL operating point preserves selective error no greater than
   20% and achieves higher coverage than a matched softmax operating point
   that also preserves the constraint. If either method violates the test
   constraint, report the violation and do not claim a constrained-coverage
   win.
2. EDL AU has lower AURC than matched softmax entropy. AU-or-EU additionally
   reports its two-dimensional frontier AURC.
3. The direction is consistent across most primary chunks and suites.
4. Paired episode bootstrap intervals support the aggregate difference.

EDL AU must also be compared with EDL predictive entropy. If their rankings
remain effectively identical, the result supports uncertainty-based rejection
but not a claim that the EDL evidence decomposition adds information beyond
the class probability.

AU-or-EU is considered better than AU-only only if EU increases test coverage
at the fixed risk constraint, lowers AURC, or captures a reproducible rejected
subset that AU-only misses. Calibration-only gains are not sufficient.

## Artifacts

Calibration writes `rejection_calibration.json` containing:

- schema version and analysis seed;
- calibration source paths and content hashes;
- verifier run and checkpoint identities;
- compared policy definitions;
- calibration chunks and support;
- threshold candidates, selected thresholds, objectives, and tie breakers;
- default and target-coverage operating points;
- unavailable-point reasons where applicable.

Frozen test evaluation writes:

- `rejection_test_predictions.hdf5` with raw probability, AU, EU, predictive
  entropy, accepted flag, three-way decision, and rejection reason per chunk;
- `rejection_metrics.json` with complete structured metrics and confidence
  intervals;
- `rejection_metrics.csv` with flat per-model, per-suite, per-chunk rows;
- `risk_coverage.png`;
- `metrics_by_chunk.png`;
- `au_eu_quadrants.png`;
- `edl_vs_softmax.png`.

All artifacts are written to a new analysis directory. Existing sweep runs,
checkpoints, predictions, summaries, and rollout datasets are read-only inputs
and are never overwritten.

## Failure Handling

Calibration and evaluation fail before writing final artifacts when:

- a required checkpoint, config, prediction field, or HDF5 dataset is missing;
- compared runs have different calibration episode identities or labels;
- a calibration episode has fewer than ten chunks;
- test and source collection identities overlap;
- test token counts exceed the checkpoint maximum;
- a threshold artifact does not match the evaluated checkpoint or policy.

Undefined one-class ranking metrics are represented as `null` with explicit
support counts, not coerced to zero. Artifact publication is staged and atomic
so a failed analysis cannot leave a mixture of old and new outputs.

## Testing Strategy

Unit tests cover:

- AU-only, AU-or-EU, and entropy accept/reject boundaries;
- rejection-reason assignment;
- exact deterministic threshold search and tie breaking;
- the 20% risk constraint and unavailable operating point;
- target-coverage operating points;
- risk-coverage and AURC calculations;
- episode-level paired bootstrap behavior;
- fixed-chunk support accounting;
- incomplete support after chunk 10;
- calibration/test identity and checkpoint validation;
- artifact schemas and atomic publication.

Integration tests use small synthetic HDF5 files and frozen toy checkpoints to
exercise calibration followed by test evaluation. A regression test verifies
that changing test labels or predictions cannot change saved calibration
thresholds. Another verifies that final-chunk or relative-progress metrics do
not enter the primary policy-ranking fields.

## Execution Order

1. Add pure rejection and selective-metric functions.
2. Add deterministic calibration and its artifact schema.
3. Add frozen-checkpoint inference on independent HDF5 data.
4. Add fixed-chunk, per-suite, bootstrap, and comparison metrics.
5. Add HDF5/JSON/CSV artifact publication and plots.
6. Extend the collection metadata needed to prove test independence.
7. Run calibration on the existing 40 validation episodes.
8. Collect the independent suite datasets.
9. Run the frozen test evaluation once and analyze the predefined results.
