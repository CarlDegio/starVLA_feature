"""Pure metrics for selective trajectory-verifier evaluation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from examples.LIBERO.edl_pred.metrics import binary_metrics


@dataclass(frozen=True)
class EpisodeMetricRecord:
    """A generic chunk record used by episode-grouped bootstrap tests/callers."""

    episode_id: str
    absolute_chunk: int
    value: float = 0.0


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float | None
    lower: float | None
    upper: float | None
    confidence_level: float
    replicates: int
    valid_replicates: int
    seed: int


def error_detection_metrics(
    labels: np.ndarray,
    success_probability: np.ndarray,
    uncertainty: np.ndarray,
) -> dict[str, float | int | None]:
    """Measure how well increasing uncertainty ranks prediction errors."""
    target, probability = _classification_arrays(labels, success_probability)
    score = _unit_scores(uncertainty, "uncertainty")
    if score.shape != target.shape:
        raise ValueError("uncertainty and labels must have the same shape")
    errors = ((probability >= 0.5).astype(np.int64) != target).astype(np.int64)
    ranking = binary_metrics(errors, score)
    return {
        "error_roc_auc": ranking["roc_auc"],
        "error_pr_auc": ranking["pr_auc"],
        "error_rate_baseline": float(np.mean(errors)),
        "error_count": int(errors.sum()),
        "correct_count": int(errors.size - errors.sum()),
        "support": int(errors.size),
    }


def risk_coverage_curve(
    labels: np.ndarray,
    success_probability: np.ndarray,
    uncertainty: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the exact threshold curve, grouping equal uncertainty scores."""
    target, probability = _classification_arrays(labels, success_probability)
    score = _finite_scores(uncertainty, "uncertainty")
    if score.shape != target.shape:
        raise ValueError("uncertainty and labels must have the same shape")
    errors = (probability >= 0.5) != target

    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_errors = errors[order]
    thresholds: list[float] = []
    accepted_counts: list[int] = []
    risks: list[float] = []
    cumulative_errors = 0
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        cumulative_errors += int(np.count_nonzero(sorted_errors[start:stop]))
        thresholds.append(float(sorted_score[start]))
        accepted_counts.append(stop)
        risks.append(cumulative_errors / stop)
        start = stop

    counts = np.asarray(accepted_counts, dtype=np.int64)
    return {
        "threshold": _read_only(np.asarray(thresholds, dtype=np.float64)),
        "accepted_count": _read_only(counts),
        "coverage": _read_only(counts.astype(np.float64) / target.size),
        "risk": _read_only(np.asarray(risks, dtype=np.float64)),
    }


def aurc(
    curve_or_coverage: Mapping[str, np.ndarray] | np.ndarray,
    risk: np.ndarray | None = None,
) -> float:
    """Integrate an empirical risk-coverage step curve from zero coverage."""
    if isinstance(curve_or_coverage, Mapping):
        if risk is not None:
            raise ValueError("risk must be omitted when passing a curve mapping")
        coverage = np.asarray(curve_or_coverage["coverage"], dtype=np.float64)
        risk_array = np.asarray(curve_or_coverage["risk"], dtype=np.float64)
    else:
        if risk is None:
            raise ValueError("risk is required when passing coverage directly")
        coverage = np.asarray(curve_or_coverage, dtype=np.float64)
        risk_array = np.asarray(risk, dtype=np.float64)
    if coverage.ndim != 1 or coverage.size == 0 or risk_array.shape != coverage.shape:
        raise ValueError("coverage and risk must be non-empty one-dimensional arrays of equal shape")
    if not np.all(np.isfinite(coverage)) or not np.all(np.isfinite(risk_array)):
        raise ValueError("coverage and risk must be finite")
    if np.any(coverage <= 0.0) or np.any(coverage > 1.0) or np.any(np.diff(coverage) <= 0.0):
        raise ValueError("coverage must be strictly increasing in (0, 1]")
    if np.any(risk_array < 0.0) or np.any(risk_array > 1.0):
        raise ValueError("risk must contain values in [0, 1]")
    widths = np.diff(np.r_[0.0, coverage])
    return float(np.sum(widths * risk_array))


def dual_threshold_frontier(
    labels: np.ndarray,
    success_probability: np.ndarray,
    au: np.ndarray,
    eu: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the lower-risk envelope over every exact AU/EU threshold pair."""
    target, probability = _classification_arrays(labels, success_probability)
    au_score = _finite_scores(au, "au")
    eu_score = _finite_scores(eu, "eu")
    if au_score.shape != target.shape or eu_score.shape != target.shape:
        raise ValueError("au, eu, and labels must have the same shape")
    errors = (probability >= 0.5) != target

    au_values, au_rank = np.unique(au_score, return_inverse=True)
    eu_values, eu_rank = np.unique(eu_score, return_inverse=True)
    shape = (au_values.size, eu_values.size)
    accepted_grid = np.zeros(shape, dtype=np.int32)
    error_grid = np.zeros(shape, dtype=np.int32)
    np.add.at(accepted_grid, (au_rank, eu_rank), 1)
    np.add.at(error_grid, (au_rank, eu_rank), errors.astype(np.int32))
    np.cumsum(accepted_grid, axis=0, out=accepted_grid)
    np.cumsum(accepted_grid, axis=1, out=accepted_grid)
    np.cumsum(error_grid, axis=0, out=error_grid)
    np.cumsum(error_grid, axis=1, out=error_grid)

    # The maximum observed threshold and +inf accept the same records. Keep the
    # less aggressive +inf representation to preserve the original tie break.
    au_thresholds = au_values.astype(np.float64, copy=True)
    eu_thresholds = eu_values.astype(np.float64, copy=True)
    au_thresholds[-1] = np.inf
    eu_thresholds[-1] = np.inf

    flat_count = accepted_grid.reshape(-1)
    flat_error = error_grid.reshape(-1)
    minimum_error = np.full(target.size + 1, target.size + 1, dtype=np.int64)
    np.minimum.at(minimum_error, flat_count, flat_error)
    eligible = (flat_count > 0) & (flat_error == minimum_error[flat_count])
    best_flat_index = np.full(target.size + 1, -1, dtype=np.int64)
    flat_indices = np.arange(flat_count.size, dtype=np.int64)
    np.maximum.at(best_flat_index, flat_count[eligible], flat_indices[eligible])

    counts = np.flatnonzero(best_flat_index > -1).astype(np.int64)
    selected = best_flat_index[counts]
    au_indices, eu_indices = np.divmod(selected, eu_values.size)
    return {
        "accepted_count": _read_only(counts),
        "coverage": _read_only(counts.astype(np.float64) / target.size),
        "risk": _read_only(minimum_error[counts].astype(np.float64) / counts),
        "tau_au": _read_only(au_thresholds[au_indices]),
        "tau_eu": _read_only(eu_thresholds[eu_indices]),
    }


def selective_metrics(
    labels: np.ndarray,
    success_probability: np.ndarray,
    accepted: np.ndarray,
    *,
    suites: np.ndarray | None = None,
    reasons: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute conditional metrics while retaining abstention support beside them."""
    target, probability = _classification_arrays(labels, success_probability)
    accepted_array = _boolean_array(accepted, "accepted")
    if accepted_array.shape != target.shape:
        raise ValueError("accepted and labels must have the same shape")
    predictions = (probability >= 0.5).astype(np.int64)
    errors = predictions != target
    accepted_count = int(np.count_nonzero(accepted_array))
    rejected = ~accepted_array
    rejected_count = int(np.count_nonzero(rejected))
    coverage = accepted_count / target.size

    if accepted_count:
        accepted_metrics = binary_metrics(target[accepted_array], probability[accepted_array])
        selective_error: float | None = float(np.mean(errors[accepted_array]))
        selective_accuracy: float | None = float(1.0 - selective_error)
        selective_balanced_accuracy: float | None = accepted_metrics["balanced_accuracy"]
    else:
        selective_error = None
        selective_accuracy = None
        selective_balanced_accuracy = None

    result: dict[str, Any] = {
        "coverage": float(coverage),
        "abstention_rate": float(1.0 - coverage),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "selective_error": selective_error,
        "selective_accuracy": selective_accuracy,
        "selective_balanced_accuracy": selective_balanced_accuracy,
        "rejected_error_rate": float(np.mean(errors[rejected])) if rejected_count else None,
        "accepted_failure_count": int(np.count_nonzero(accepted_array & (target == 0))),
        "accepted_success_count": int(np.count_nonzero(accepted_array & (target == 1))),
        "rejected_failure_count": int(np.count_nonzero(rejected & (target == 0))),
        "rejected_success_count": int(np.count_nonzero(rejected & (target == 1))),
    }

    if suites is not None:
        suite_array = _string_array(suites, "suites")
        if suite_array.shape != target.shape:
            raise ValueError("suites and labels must have the same shape")
        result["per_suite_coverage"] = {
            str(suite): float(np.mean(accepted_array[suite_array == suite]))
            for suite in np.unique(suite_array)
        }
    if reasons is not None:
        reason_array = _string_array(reasons, "reasons")
        if reason_array.shape != target.shape:
            raise ValueError("reasons and labels must have the same shape")
        rejected_reasons = reason_array[rejected]
        result["rejection_reason_counts"] = {
            str(reason): int(np.count_nonzero(rejected_reasons == reason))
            for reason in np.unique(rejected_reasons)
            if reason != "accepted"
        }
    return result


def metrics_by_absolute_chunk(
    labels: np.ndarray,
    success_probability: np.ndarray,
    absolute_chunk: np.ndarray,
    episode_id: np.ndarray,
    *,
    uncertainty: np.ndarray | None = None,
    accepted: np.ndarray | None = None,
    suites: np.ndarray | None = None,
    reasons: np.ndarray | None = None,
    common_support_horizon: int = 10,
) -> list[dict[str, Any]]:
    """Compute fixed-position rows with explicit active-episode support."""
    target, probability = _classification_arrays(labels, success_probability)
    chunks = _positive_integer_array(absolute_chunk, "absolute_chunk")
    episodes = _string_array(episode_id, "episode_id")
    if chunks.shape != target.shape or episodes.shape != target.shape:
        raise ValueError("absolute_chunk, episode_id, and labels must have the same shape")
    if isinstance(common_support_horizon, (bool, np.bool_)) or not isinstance(
        common_support_horizon, (int, np.integer)
    ) or common_support_horizon <= 0:
        raise ValueError("common_support_horizon must be a positive integer")
    pairs = np.asarray([f"{episode}\0{chunk}" for episode, chunk in zip(episodes, chunks)])
    if np.unique(pairs).size != pairs.size:
        raise ValueError("each episode may contribute at most one record per absolute chunk")
    for episode in np.unique(episodes):
        if np.unique(target[episodes == episode]).size != 1:
            raise ValueError("all chunks of an episode must have the same label")

    optional = {
        "uncertainty": None if uncertainty is None else _finite_scores(uncertainty, "uncertainty"),
        "accepted": None if accepted is None else _boolean_array(accepted, "accepted"),
        "suites": None if suites is None else _string_array(suites, "suites"),
        "reasons": None if reasons is None else _string_array(reasons, "reasons"),
    }
    for name, array in optional.items():
        if array is not None and array.shape != target.shape:
            raise ValueError(f"{name} and labels must have the same shape")

    total_episodes = np.unique(episodes).size
    rows: list[dict[str, Any]] = []
    for chunk in np.unique(chunks):
        selected = chunks == chunk
        active = int(np.count_nonzero(selected))
        row: dict[str, Any] = {
            "absolute_chunk": int(chunk),
            "active_episode_count": active,
            "support": active,
            "active_failure_count": int(np.count_nonzero(target[selected] == 0)),
            "active_success_count": int(np.count_nonzero(target[selected] == 1)),
            "survival_fraction": float(active / total_episodes),
            "incomplete_support": bool(chunk > common_support_horizon or active < total_episodes),
            "classifier_metrics": binary_metrics(target[selected], probability[selected]),
        }
        if optional["uncertainty"] is not None:
            score = optional["uncertainty"][selected]
            row["error_detection_metrics"] = error_detection_metrics(
                target[selected], probability[selected], score
            )
            row["aurc"] = aurc(risk_coverage_curve(target[selected], probability[selected], score))
        if optional["accepted"] is not None:
            row["selective_metrics"] = selective_metrics(
                target[selected],
                probability[selected],
                optional["accepted"][selected],
                suites=None if optional["suites"] is None else optional["suites"][selected],
                reasons=None if optional["reasons"] is None else optional["reasons"][selected],
            )
        rows.append(row)
    return rows


def suite_macro_metrics(
    metrics_by_suite: Mapping[str, Mapping[str, float | int | None]],
) -> dict[str, float | int | None]:
    """Compute an unweighted macro average across suite-level metric rows."""
    if not metrics_by_suite:
        raise ValueError("metrics_by_suite must not be empty")
    rows = list(metrics_by_suite.values())
    keys = list(rows[0])
    if any(set(row) != set(keys) for row in rows[1:]):
        raise ValueError("all suite metric rows must have the same keys")
    if any("selective_accuracy" in row and "coverage" not in row for row in rows):
        raise ValueError("selective_accuracy must be accompanied by coverage")

    result: dict[str, float | int | None] = {"suite_count": len(rows)}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        if not values:
            result[key] = None
            continue
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) for value in values):
            raise ValueError(f"suite metric {key!r} must be numeric or None")
        numeric = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"suite metric {key!r} must be finite or None")
        result[key] = float(np.mean(numeric))
    return result


suite_macro = suite_macro_metrics


def paired_episode_bootstrap(
    episode_records: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float | None],
    *,
    seed: int,
    replicates: int = 10_000,
    confidence_level: float = 0.95,
) -> BootstrapInterval:
    """Bootstrap episode identities while retaining every chunk in each draw."""
    records = list(episode_records)
    if not records:
        raise ValueError("episode_records must not be empty")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if isinstance(replicates, (bool, np.bool_)) or not isinstance(replicates, (int, np.integer)) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if not isinstance(confidence_level, (int, float, np.number)) or not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")

    groups: OrderedDict[str, list[Any]] = OrderedDict()
    for record in records:
        episode = _record_episode_id(record)
        groups.setdefault(episode, []).append(record)
    grouped = list(groups.values())
    rng = np.random.default_rng(int(seed))
    estimate = _optional_statistic(statistic(records))
    values: list[float] = []
    for _ in range(int(replicates)):
        sampled_indices = rng.integers(0, len(grouped), size=len(grouped))
        sample = [record for index in sampled_indices for record in grouped[int(index)]]
        value = _optional_statistic(statistic(sample))
        if value is not None:
            values.append(value)

    if values:
        tail = (1.0 - float(confidence_level)) / 2.0
        lower, upper = np.quantile(np.asarray(values), [tail, 1.0 - tail])
        lower_value: float | None = float(lower)
        upper_value: float | None = float(upper)
    else:
        lower_value = None
        upper_value = None
    return BootstrapInterval(
        estimate=estimate,
        lower=lower_value,
        upper=upper_value,
        confidence_level=float(confidence_level),
        replicates=int(replicates),
        valid_replicates=len(values),
        seed=int(seed),
    )


def _classification_arrays(labels: Any, probability: Any) -> tuple[np.ndarray, np.ndarray]:
    target = _binary_labels(labels)
    score = _unit_scores(probability, "success_probability")
    if target.shape != score.shape:
        raise ValueError("labels and success_probability must have the same shape")
    return target, score


def _binary_labels(value: Any) -> np.ndarray:
    array = _one_dimensional(value, "labels")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise ValueError("labels must be a numeric binary array")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError("labels must be real-valued")
    if not np.all(np.isfinite(array)) or not np.all((array == 0) | (array == 1)):
        raise ValueError("labels must contain only 0 and 1")
    return array.astype(np.int64, copy=False)


def _unit_scores(value: Any, name: str) -> np.ndarray:
    array = _finite_scores(value, name)
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must contain values in [0, 1]")
    return array


def _finite_scores(value: Any, name: str) -> np.ndarray:
    array = _one_dimensional(value, name)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a real-valued numeric array")
    if np.issubdtype(array.dtype, np.complexfloating) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real values")
    return array.astype(np.float64, copy=False)


def _boolean_array(value: Any, name: str) -> np.ndarray:
    array = _one_dimensional(value, name)
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array")
    return array


def _positive_integer_array(value: Any, name: str) -> np.ndarray:
    array = _one_dimensional(value, name)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer) or np.any(array <= 0):
        raise ValueError(f"{name} must contain positive integers")
    return array.astype(np.int64, copy=False)


def _string_array(value: Any, name: str) -> np.ndarray:
    array = _one_dimensional(value, name)
    result = np.asarray([str(item) for item in array], dtype=np.str_)
    if np.any(np.char.str_len(result) == 0):
        raise ValueError(f"{name} values must be non-empty")
    return result


def _one_dimensional(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return array


def _record_episode_id(record: Any) -> str:
    if isinstance(record, Mapping):
        value = record.get("episode_id")
    else:
        value = getattr(record, "episode_id", None)
    if value is None or str(value) == "":
        raise ValueError("every episode record must provide a non-empty episode_id")
    return str(value)


def _optional_statistic(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError("statistic must return a finite real scalar or None")
    result = float(value)
    return result if np.isfinite(result) else None


def _read_only(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result
