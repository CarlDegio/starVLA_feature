"""Evaluate frozen verifier rejection policies on independent LIBERO datasets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
import numpy as np

from .artifacts import PredictionRecord
from .metrics import binary_metrics
from .rejection import PolicyThresholds, apply_rejection, predictive_entropy
from .rejection_artifacts import read_json, write_json_atomic
from .rejection_data import (
    EpisodePrediction,
    collect_dataset_provenance,
    sha256_file,
    validate_independent_test_datasets,
    validate_matched_predictions,
)
from .rejection_inference import load_frozen_verifier, predict_frozen_datasets
from .rejection_metrics import (
    aurc,
    dual_threshold_frontier,
    metrics_by_absolute_chunk,
    risk_coverage_curve,
    selective_metrics,
    suite_macro_metrics,
)


@dataclass(frozen=True)
class _FlatPredictions:
    episode_id: np.ndarray
    suite: np.ndarray
    episode_key: np.ndarray
    label: np.ndarray
    absolute_chunk: np.ndarray
    success_probability: np.ndarray
    au: np.ndarray | None
    eu: np.ndarray | None


def evaluate_prediction_sets(
    calibration: Mapping[str, Any],
    predictions_by_run: Mapping[str, Sequence[EpisodePrediction]],
    *,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    """Apply only frozen calibration thresholds and compute predefined metrics."""
    if calibration.get("schema_version") != "1.0":
        raise ValueError("unsupported rejection calibration schema")
    calibration_runs = calibration.get("runs")
    if not isinstance(calibration_runs, Mapping) or set(calibration_runs) != set(predictions_by_run):
        raise ValueError("calibration and prediction run names differ")
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    reference = None
    for records in predictions_by_run.values():
        if reference is None:
            reference = records
        else:
            validate_matched_predictions(reference, records)
    seed = int(calibration.get("analysis_seed", 20260805))
    result_runs: dict[str, Any] = {}
    flattened: dict[str, _FlatPredictions] = {}
    for run_name in sorted(predictions_by_run):
        run_calibration = calibration_runs[run_name]
        head = run_calibration.get("head")
        flat = _flatten_predictions(predictions_by_run[run_name], expected_head=head)
        flattened[run_name] = flat
        classifier_rows = metrics_by_absolute_chunk(
            flat.label,
            flat.success_probability,
            flat.absolute_chunk,
            flat.episode_id,
            suites=flat.suite,
        )
        classifier_by_suite = _classifier_metrics_by_suite(flat)
        policies_payload = run_calibration.get("policies")
        if not isinstance(policies_payload, Mapping):
            raise ValueError(f"calibration run {run_name!r} is missing policies")
        policies: dict[str, Any] = {}
        for policy_name in sorted(policies_payload):
            policy_payload = policies_payload[policy_name]
            point = policy_payload.get("default")
            if not isinstance(point, Mapping):
                raise ValueError(f"calibration policy {run_name}/{policy_name} is missing default")
            uncertainty = _policy_uncertainty(policy_name, flat)
            primary = flat.absolute_chunk <= 10
            if policy_name == "au_or_eu":
                assert flat.au is not None and flat.eu is not None
                curve = dual_threshold_frontier(
                    flat.label[primary], flat.success_probability[primary], flat.au[primary], flat.eu[primary]
                )
            else:
                assert uncertainty is not None
                curve = risk_coverage_curve(
                    flat.label[primary], flat.success_probability[primary], uncertainty[primary]
                )
            evaluated_default = _evaluate_operating_point(
                flat,
                policy_name,
                point,
                uncertainty,
                primary,
                seed=seed,
                bootstrap_replicates=bootstrap_replicates,
            )
            target_results = {
                str(target): _evaluate_operating_point(
                    flat,
                    policy_name,
                    target_point,
                    uncertainty,
                    primary,
                    seed=seed,
                    bootstrap_replicates=bootstrap_replicates,
                )
                for target, target_point in policy_payload.get("target_operating_points", {}).items()
            }
            policies[policy_name] = {
                **evaluated_default,
                "calibration_default": dict(point),
                "target_operating_points": target_results,
                "primary_aurc": aurc(curve),
                "risk_coverage": _json_curve(curve),
            }
        result_runs[run_name] = {
            "head": head,
            "chunk_encoder": run_calibration.get("chunk_encoder"),
            "token_position_encoding": run_calibration.get("token_position_encoding"),
            "classifier_metrics_by_chunk": classifier_rows,
            "classifier_metrics_by_suite_chunk": classifier_by_suite,
            "policies": policies,
        }
    comparisons = _matched_comparisons(
        calibration_runs,
        result_runs,
        flattened,
        max_selective_error=float(calibration.get("objective", {}).get("max_selective_error", 0.20)),
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    return {
        "schema_version": "1.0",
        "analysis_seed": seed,
        "primary_chunks": list(range(1, 11)),
        "bootstrap_replicates": bootstrap_replicates,
        "runs": result_runs,
        "matched_comparisons": comparisons,
    }


def publish_evaluation(
    output_dir: str | Path,
    evaluation: Mapping[str, Any],
    predictions_by_run: Mapping[str, Sequence[EpisodePrediction]],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically publish structured metrics, predictions, CSV, and plots."""
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"evaluation output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        write_json_atomic(stage / "rejection_metrics.json", evaluation)
        _write_metrics_csv(stage / "rejection_metrics.csv", evaluation)
        _write_prediction_hdf5(stage / "rejection_test_predictions.hdf5", evaluation, predictions_by_run)
        _plot_risk_coverage(stage / "risk_coverage.png", evaluation)
        _plot_metrics_by_chunk(stage / "metrics_by_chunk.png", evaluation)
        _plot_au_eu(stage / "au_eu_quadrants.png", predictions_by_run)
        _plot_edl_vs_softmax(stage / "edl_vs_softmax.png", evaluation)
        if destination.exists():
            shutil.rmtree(destination)
        stage.replace(destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def _flatten_predictions(records: Sequence[EpisodePrediction], *, expected_head: str) -> _FlatPredictions:
    episode_ids: list[str] = []
    suites: list[str] = []
    keys: list[str] = []
    labels: list[int] = []
    chunks: list[int] = []
    probability: list[float] = []
    au: list[float] = []
    eu: list[float] = []
    for record in sorted(records, key=lambda item: item.identity):
        if expected_head == "edl" and (record.verifier_au is None or record.verifier_eu is None):
            raise ValueError("EDL test predictions are missing AU or EU")
        if expected_head == "softmax" and record.verifier_au is not None:
            raise ValueError("softmax test predictions unexpectedly contain EDL fields")
        for index, value in enumerate(record.success_probability):
            episode_ids.append(record.episode_id)
            suites.append(record.suite)
            keys.append(record.episode_key)
            labels.append(record.label)
            chunks.append(index + 1)
            probability.append(float(value))
            if record.verifier_au is not None:
                assert record.verifier_eu is not None
                au.append(float(record.verifier_au[index]))
                eu.append(float(record.verifier_eu[index]))
    return _FlatPredictions(
        episode_id=np.asarray(episode_ids),
        suite=np.asarray(suites),
        episode_key=np.asarray(keys),
        label=np.asarray(labels, dtype=np.int64),
        absolute_chunk=np.asarray(chunks, dtype=np.int64),
        success_probability=np.asarray(probability, dtype=np.float64),
        au=None if not au else np.asarray(au, dtype=np.float64),
        eu=None if not eu else np.asarray(eu, dtype=np.float64),
    )


def _policy_uncertainty(policy_name: str, flat: _FlatPredictions) -> np.ndarray | None:
    if policy_name == "au":
        if flat.au is None:
            raise ValueError("AU policy requires EDL AU")
        return flat.au
    if policy_name == "predictive_entropy":
        return predictive_entropy(flat.success_probability)
    if policy_name == "au_or_eu":
        return None
    raise ValueError(f"unsupported calibrated policy: {policy_name!r}")


def _evaluate_operating_point(
    flat: _FlatPredictions,
    policy_name: str,
    point: Mapping[str, Any],
    uncertainty: np.ndarray | None,
    primary: np.ndarray,
    *,
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if not point.get("available"):
        return {"available": False, "unavailable_reason": point.get("unavailable_reason")}
    thresholds = PolicyThresholds.from_dict(point["thresholds"])
    decision = apply_rejection(
        flat.success_probability,
        thresholds,
        au=flat.au,
        eu=flat.eu,
    )
    chunk_kwargs: dict[str, Any] = {}
    if uncertainty is not None:
        chunk_kwargs["uncertainty"] = uncertainty
    chunk_rows = metrics_by_absolute_chunk(
        flat.label,
        flat.success_probability,
        flat.absolute_chunk,
        flat.episode_id,
        accepted=decision.accepted,
        suites=flat.suite,
        reasons=decision.reason,
        **chunk_kwargs,
    )
    return {
        "available": True,
        "thresholds": thresholds.to_dict(),
        "calibration_point": dict(point),
        "primary_selective_metrics": selective_metrics(
            flat.label[primary],
            flat.success_probability[primary],
            decision.accepted[primary],
            suites=flat.suite[primary],
            reasons=decision.reason[primary],
        ),
        "metrics_by_chunk": chunk_rows,
        "bootstrap": _bootstrap_selective(
            flat,
            decision.accepted,
            primary,
            seed=seed,
            replicates=bootstrap_replicates,
        ),
        "primary_metrics_by_suite": _primary_selective_by_suite(flat, decision.accepted, primary, decision.reason),
        "metrics_by_suite_chunk": _selective_metrics_by_suite_chunk(
            flat,
            decision.accepted,
            decision.reason,
            uncertainty,
        ),
    }


def _json_curve(curve: Mapping[str, np.ndarray]) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for name, values in curve.items():
        serialized: list[Any] = []
        for value in values:
            item = value.item() if isinstance(value, np.generic) else value
            if isinstance(item, float) and not np.isfinite(item):
                item = "+inf" if item > 0 else "-inf"
            serialized.append(item)
        result[name] = serialized
    return result


def _classifier_metrics_by_suite(flat: _FlatPredictions) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for suite in np.unique(flat.suite):
        selected = flat.suite == suite
        result[str(suite)] = metrics_by_absolute_chunk(
            flat.label[selected],
            flat.success_probability[selected],
            flat.absolute_chunk[selected],
            flat.episode_id[selected],
        )
    return result


def _primary_selective_by_suite(
    flat: _FlatPredictions,
    accepted: np.ndarray,
    primary: np.ndarray,
    reasons: np.ndarray,
) -> dict[str, Any]:
    by_suite: dict[str, Any] = {}
    for suite in np.unique(flat.suite):
        selected = primary & (flat.suite == suite)
        by_suite[str(suite)] = selective_metrics(
            flat.label[selected],
            flat.success_probability[selected],
            accepted[selected],
            reasons=reasons[selected],
        )
    macro_fields = {
        suite: {
            key: values[key]
            for key in ("coverage", "selective_error", "selective_accuracy", "selective_balanced_accuracy")
        }
        for suite, values in by_suite.items()
    }
    return {"suites": by_suite, "suite_macro": suite_macro_metrics(macro_fields)}


def _selective_metrics_by_suite_chunk(
    flat: _FlatPredictions,
    accepted: np.ndarray,
    reasons: np.ndarray,
    uncertainty: np.ndarray | None,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for suite in np.unique(flat.suite):
        selected = flat.suite == suite
        kwargs: dict[str, Any] = {}
        if uncertainty is not None:
            kwargs["uncertainty"] = uncertainty[selected]
        result[str(suite)] = metrics_by_absolute_chunk(
            flat.label[selected],
            flat.success_probability[selected],
            flat.absolute_chunk[selected],
            flat.episode_id[selected],
            accepted=accepted[selected],
            reasons=reasons[selected],
            **kwargs,
        )
    return result


def _matched_comparisons(
    calibration_runs: Mapping[str, Any],
    result_runs: Mapping[str, Any],
    flattened: Mapping[str, _FlatPredictions],
    *,
    max_selective_error: float,
    seed: int,
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    by_architecture: dict[tuple[Any, Any], dict[str, str]] = {}
    for run_name, run in calibration_runs.items():
        key = (run.get("chunk_encoder"), run.get("token_position_encoding"))
        if None in key:
            continue
        head = run.get("head")
        if head in {"edl", "softmax"}:
            by_architecture.setdefault(key, {})[head] = run_name
    comparisons: list[dict[str, Any]] = []
    for pair_index, (architecture, names) in enumerate(sorted(by_architecture.items(), key=lambda item: str(item[0]))):
        if set(names) != {"edl", "softmax"}:
            continue
        edl_name = names["edl"]
        softmax_name = names["softmax"]
        edl_policy = result_runs[edl_name]["policies"].get("au")
        softmax_policy = result_runs[softmax_name]["policies"].get("predictive_entropy")
        if not edl_policy or not softmax_policy:
            continue
        comparisons.append(_paired_edl_softmax_comparison(
            edl_name,
            softmax_name,
            architecture,
            flattened[edl_name],
            flattened[softmax_name],
            edl_policy,
            softmax_policy,
            max_selective_error=max_selective_error,
            seed=seed + pair_index,
            replicates=bootstrap_replicates,
        ))
    return comparisons


def _paired_edl_softmax_comparison(
    edl_name: str,
    softmax_name: str,
    architecture: tuple[Any, Any],
    edl: _FlatPredictions,
    softmax: _FlatPredictions,
    edl_policy: Mapping[str, Any],
    softmax_policy: Mapping[str, Any],
    *,
    max_selective_error: float,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if edl.au is None:
        raise ValueError("matched EDL comparison requires AU")
    if not np.array_equal(edl.episode_id, softmax.episode_id) or not np.array_equal(edl.absolute_chunk, softmax.absolute_chunk):
        raise ValueError("matched comparison flattened episode order differs")
    primary = edl.absolute_chunk <= 10
    edl_accepted = _accepted_or_none(edl, edl_policy)
    softmax_accepted = _accepted_or_none(softmax, softmax_policy)

    def statistics(indices: np.ndarray) -> tuple[float | None, float | None, float]:
        coverage_difference = error_difference = None
        if edl_accepted is not None and softmax_accepted is not None:
            edl_metrics = selective_metrics(
                edl.label[indices], edl.success_probability[indices], edl_accepted[indices]
            )
            softmax_metrics = selective_metrics(
                softmax.label[indices], softmax.success_probability[indices], softmax_accepted[indices]
            )
            coverage_difference = float(edl_metrics["coverage"] - softmax_metrics["coverage"])
            if edl_metrics["selective_error"] is not None and softmax_metrics["selective_error"] is not None:
                error_difference = float(edl_metrics["selective_error"] - softmax_metrics["selective_error"])
        aurc_difference = aurc(risk_coverage_curve(
            edl.label[indices], edl.success_probability[indices], edl.au[indices]
        )) - aurc(risk_coverage_curve(
            softmax.label[indices],
            softmax.success_probability[indices],
            predictive_entropy(softmax.success_probability[indices]),
        ))
        return coverage_difference, error_difference, float(aurc_difference)

    primary_indices = np.flatnonzero(primary)
    coverage_difference, error_difference, aurc_difference = statistics(primary_indices)
    episode_ids = np.unique(edl.episode_id[primary])
    episode_codes = np.searchsorted(episode_ids, edl.episode_id[primary_indices])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, episode_ids.size, size=(replicates, episode_ids.size))
    bootstrap_coverage: list[float] = []
    bootstrap_error: list[float] = []
    if edl_accepted is not None and softmax_accepted is not None:
        edl_samples = _bootstrap_selective_samples(
            edl.label[primary_indices],
            edl.success_probability[primary_indices],
            edl_accepted[primary_indices],
            episode_codes,
            draws,
        )
        softmax_samples = _bootstrap_selective_samples(
            softmax.label[primary_indices],
            softmax.success_probability[primary_indices],
            softmax_accepted[primary_indices],
            episode_codes,
            draws,
        )
        bootstrap_coverage = (edl_samples["coverage"] - softmax_samples["coverage"]).tolist()
        valid_error = np.isfinite(edl_samples["selective_error"]) & np.isfinite(
            softmax_samples["selective_error"]
        )
        bootstrap_error = (
            edl_samples["selective_error"][valid_error]
            - softmax_samples["selective_error"][valid_error]
        ).tolist()
    edl_aurc_samples = _bootstrap_aurc_samples(
        edl.label[primary_indices],
        edl.success_probability[primary_indices],
        edl.au[primary_indices],
        episode_codes,
        draws,
    )
    softmax_aurc_samples = _bootstrap_aurc_samples(
        softmax.label[primary_indices],
        softmax.success_probability[primary_indices],
        predictive_entropy(softmax.success_probability[primary_indices]),
        episode_codes,
        draws,
    )
    bootstrap_aurc = (edl_aurc_samples - softmax_aurc_samples).tolist()
    edl_error = None if not edl_policy.get("available") else edl_policy["primary_selective_metrics"]["selective_error"]
    softmax_error = (
        None if not softmax_policy.get("available") else softmax_policy["primary_selective_metrics"]["selective_error"]
    )
    return {
        "chunk_encoder": architecture[0],
        "token_position_encoding": architecture[1],
        "edl_run": edl_name,
        "edl_policy": "au",
        "softmax_run": softmax_name,
        "softmax_policy": "predictive_entropy",
        "max_selective_error": max_selective_error,
        "edl_constraint_satisfied": edl_error is not None and edl_error <= max_selective_error,
        "softmax_constraint_satisfied": softmax_error is not None and softmax_error <= max_selective_error,
        "coverage_difference": _interval(
            coverage_difference,
            bootstrap_coverage,
            replicates,
        ),
        "selective_error_difference": _interval(
            error_difference,
            bootstrap_error,
            replicates,
        ),
        "aurc_difference": _interval(aurc_difference, bootstrap_aurc, replicates),
        "difference_direction": {
            "coverage": "positive favors EDL AU",
            "selective_error": "negative favors EDL AU",
            "aurc": "negative favors EDL AU",
        },
    }


def _accepted_or_none(flat: _FlatPredictions, policy: Mapping[str, Any]) -> np.ndarray | None:
    if not policy.get("available"):
        return None
    thresholds = PolicyThresholds.from_dict(policy["thresholds"])
    return apply_rejection(
        flat.success_probability,
        thresholds,
        au=flat.au,
        eu=flat.eu,
    ).accepted


def _bootstrap_selective_samples(
    labels: np.ndarray,
    probability: np.ndarray,
    accepted: np.ndarray,
    episode_codes: np.ndarray,
    draws: np.ndarray,
) -> dict[str, np.ndarray]:
    episode_count = int(draws.shape[1])
    total = np.bincount(episode_codes, minlength=episode_count)
    accepted_count = np.bincount(
        episode_codes,
        weights=accepted.astype(np.int64),
        minlength=episode_count,
    )
    errors = (probability >= 0.5) != labels
    accepted_errors = np.bincount(
        episode_codes,
        weights=(accepted & errors).astype(np.int64),
        minlength=episode_count,
    )
    sampled_total = total[draws].sum(axis=1)
    sampled_accepted = accepted_count[draws].sum(axis=1)
    sampled_errors = accepted_errors[draws].sum(axis=1)
    selective_error = np.full(draws.shape[0], np.nan, dtype=np.float64)
    valid = sampled_accepted > 0
    selective_error[valid] = sampled_errors[valid] / sampled_accepted[valid]
    return {
        "coverage": sampled_accepted / sampled_total,
        "selective_error": selective_error,
    }


def _bootstrap_aurc_samples(
    labels: np.ndarray,
    probability: np.ndarray,
    uncertainty: np.ndarray,
    episode_codes: np.ndarray,
    draws: np.ndarray,
    *,
    batch_size: int = 128,
) -> np.ndarray:
    """Compute exact episode-bootstrap AURC after sorting uncertainty once."""
    errors = ((probability >= 0.5) != labels).astype(np.int64)
    order = np.argsort(uncertainty, kind="mergesort")
    sorted_uncertainty = uncertainty[order]
    sorted_episode = episode_codes[order]
    sorted_errors = errors[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_uncertainty) != 0) + 1]
    episode_count = int(draws.shape[1])
    result = np.empty(draws.shape[0], dtype=np.float64)
    for start in range(0, draws.shape[0], batch_size):
        stop = min(start + batch_size, draws.shape[0])
        batch_draws = draws[start:stop]
        multiplicity = np.zeros((stop - start, episode_count), dtype=np.int16)
        rows = np.repeat(np.arange(stop - start), episode_count)
        np.add.at(multiplicity, (rows, batch_draws.reshape(-1)), 1)
        weights = multiplicity[:, sorted_episode]
        grouped_count = np.add.reduceat(weights, group_starts, axis=1)
        grouped_errors = np.add.reduceat(weights * sorted_errors, group_starts, axis=1)
        cumulative_count = np.cumsum(grouped_count, axis=1)
        cumulative_errors = np.cumsum(grouped_errors, axis=1)
        risk = np.divide(
            cumulative_errors,
            cumulative_count,
            out=np.zeros_like(cumulative_errors, dtype=np.float64),
            where=cumulative_count > 0,
        )
        total_count = cumulative_count[:, -1]
        result[start:stop] = np.sum((grouped_count / total_count[:, None]) * risk, axis=1)
    return result


def _bootstrap_selective(
    flat: _FlatPredictions,
    accepted: np.ndarray,
    primary: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    episode_ids = np.unique(flat.episode_id[primary])
    primary_indices = np.flatnonzero(primary)
    episode_codes = np.searchsorted(episode_ids, flat.episode_id[primary_indices])
    total_by_episode = np.bincount(episode_codes, minlength=episode_ids.size)
    accepted_by_episode = np.bincount(
        episode_codes,
        weights=accepted[primary_indices].astype(np.int64),
        minlength=episode_ids.size,
    )
    errors = (flat.success_probability[primary_indices] >= 0.5) != flat.label[primary_indices]
    accepted_errors_by_episode = np.bincount(
        episode_codes,
        weights=(accepted[primary_indices] & errors).astype(np.int64),
        minlength=episode_ids.size,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, episode_ids.size, size=(replicates, episode_ids.size))
    sampled_total = total_by_episode[draws].sum(axis=1)
    sampled_accepted = accepted_by_episode[draws].sum(axis=1)
    sampled_errors = accepted_errors_by_episode[draws].sum(axis=1)
    coverage = sampled_accepted / sampled_total
    valid = sampled_accepted > 0
    accuracy = 1.0 - sampled_errors[valid] / sampled_accepted[valid]
    base = selective_metrics(flat.label[primary], flat.success_probability[primary], accepted[primary])
    return {
        "coverage": _interval(base["coverage"], coverage.tolist(), replicates),
        "selective_accuracy": _interval(base["selective_accuracy"], accuracy.tolist(), replicates),
    }


def _interval(estimate: float | None, values: Sequence[float], replicates: int) -> dict[str, Any]:
    if not values:
        return {"estimate": estimate, "lower": None, "upper": None, "replicates": replicates, "valid_replicates": 0}
    lower, upper = np.quantile(np.asarray(values), [0.025, 0.975])
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "replicates": replicates,
        "valid_replicates": len(values),
    }


def _write_metrics_csv(path: Path, evaluation: Mapping[str, Any]) -> None:
    fields = [
        "run_name", "head", "policy", "operating_point", "suite", "absolute_chunk", "incomplete_support", "support",
        "active_failure_count", "active_success_count", "coverage", "selective_error",
        "selective_accuracy", "selective_balanced_accuracy", "rejected_error_rate",
        "classifier_accuracy", "classifier_balanced_accuracy", "classifier_roc_auc", "classifier_pr_auc",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run_name, run in evaluation["runs"].items():
            classifier = {row["absolute_chunk"]: row for row in run["classifier_metrics_by_chunk"]}
            for policy_name, policy in run["policies"].items():
                points = {"default": policy, **{
                    f"coverage_{target}": payload
                    for target, payload in policy.get("target_operating_points", {}).items()
                }}
                for point_name, point in points.items():
                    if not point.get("available"):
                        continue
                    row_sets = {"ALL": point["metrics_by_chunk"], **point["metrics_by_suite_chunk"]}
                    raw_sets = {"ALL": classifier, **{
                        suite: {row["absolute_chunk"]: row for row in rows}
                        for suite, rows in run["classifier_metrics_by_suite_chunk"].items()
                    }}
                    for suite, rows in row_sets.items():
                        for row in rows:
                            selected = row["selective_metrics"]
                            raw = raw_sets[suite][row["absolute_chunk"]]["classifier_metrics"]
                            writer.writerow({
                                "run_name": run_name, "head": run["head"], "policy": policy_name,
                                "operating_point": point_name, "suite": suite,
                                "absolute_chunk": row["absolute_chunk"],
                                "incomplete_support": row["incomplete_support"], "support": row["support"],
                                "active_failure_count": row["active_failure_count"],
                                "active_success_count": row["active_success_count"], "coverage": selected["coverage"],
                                "selective_error": selected["selective_error"],
                                "selective_accuracy": selected["selective_accuracy"],
                                "selective_balanced_accuracy": selected["selective_balanced_accuracy"],
                                "rejected_error_rate": selected["rejected_error_rate"], "classifier_accuracy": raw["accuracy"],
                                "classifier_balanced_accuracy": raw["balanced_accuracy"], "classifier_roc_auc": raw["roc_auc"],
                                "classifier_pr_auc": raw["pr_auc"],
                            })


def _write_prediction_hdf5(
    path: Path,
    evaluation: Mapping[str, Any],
    predictions_by_run: Mapping[str, Sequence[EpisodePrediction]],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "1.0"
        if evaluation.get("test_datasets"):
            handle.attrs["test_datasets"] = json.dumps(evaluation["test_datasets"], sort_keys=True)
        runs_group = handle.create_group("runs")
        for run_name, records in predictions_by_run.items():
            run_group = runs_group.create_group(run_name)
            policies = evaluation["runs"][run_name]["policies"]
            for record in records:
                group = run_group.require_group("episodes").require_group(record.suite).create_group(record.episode_key)
                group.attrs["label"] = record.label
                group.create_dataset("success_probability", data=record.success_probability)
                group.create_dataset("predictive_entropy", data=predictive_entropy(record.success_probability))
                if record.verifier_au is not None:
                    group.create_dataset("verifier_au", data=record.verifier_au)
                    group.create_dataset("verifier_eu", data=record.verifier_eu)
                    if record.class_evidence is not None:
                        group.create_dataset("class_evidence", data=record.class_evidence)
                    if record.verifier_total_evidence is not None:
                        group.create_dataset("verifier_total_evidence", data=record.verifier_total_evidence)
                policy_group = group.create_group("policies")
                for policy_name, payload in policies.items():
                    subgroup = policy_group.create_group(policy_name)
                    subgroup.attrs["default_available"] = bool(payload.get("available"))
                    if payload.get("available"):
                        thresholds = PolicyThresholds.from_dict(payload["thresholds"])
                        decision = apply_rejection(
                            record.success_probability,
                            thresholds,
                            au=record.verifier_au,
                            eu=record.verifier_eu,
                        )
                        subgroup.create_dataset("accepted", data=decision.accepted)
                        subgroup.create_dataset("predicted_label", data=decision.predicted_label)
                        subgroup.create_dataset("decision", data=np.asarray(decision.decision, dtype="S16"))
                        subgroup.create_dataset("reason", data=np.asarray(decision.reason, dtype="S24"))
                    target_group = subgroup.create_group("target_operating_points")
                    for target, target_payload in payload.get("target_operating_points", {}).items():
                        if not target_payload.get("available"):
                            continue
                        target_thresholds = PolicyThresholds.from_dict(target_payload["thresholds"])
                        target_decision = apply_rejection(
                            record.success_probability,
                            target_thresholds,
                            au=record.verifier_au,
                            eu=record.verifier_eu,
                        )
                        target_subgroup = target_group.create_group(str(target))
                        target_subgroup.create_dataset("accepted", data=target_decision.accepted)
                        target_subgroup.create_dataset("predicted_label", data=target_decision.predicted_label)
                        target_subgroup.create_dataset(
                            "decision", data=np.asarray(target_decision.decision, dtype="S16")
                        )
                        target_subgroup.create_dataset(
                            "reason", data=np.asarray(target_decision.reason, dtype="S24")
                        )


def _plot_risk_coverage(path: Path, evaluation: Mapping[str, Any]) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for run_name, run in evaluation["runs"].items():
        for policy_name, policy in run["policies"].items():
            if "risk_coverage" not in policy:
                continue
            curve = policy["risk_coverage"]
            axis.plot(curve["coverage"], curve["risk"], label=f"{run_name}:{policy_name}", alpha=0.8)
    axis.axhline(0.2, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Coverage", ylabel="Selective risk", xlim=(0, 1), ylim=(0, 1))
    axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_metrics_by_chunk(path: Path, evaluation: Mapping[str, Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run_name, run in evaluation["runs"].items():
        for policy_name, policy in run["policies"].items():
            if not policy.get("available"):
                continue
            rows = policy["metrics_by_chunk"]
            chunks = [row["absolute_chunk"] for row in rows]
            coverage = [row["selective_metrics"]["coverage"] for row in rows]
            risk = [np.nan if row["selective_metrics"]["selective_error"] is None else row["selective_metrics"]["selective_error"] for row in rows]
            label = f"{run_name}:{policy_name}"
            axes[0].plot(chunks, coverage, label=label)
            axes[1].plot(chunks, risk, label=label)
    axes[0].set(xlabel="Absolute chunk", ylabel="Coverage", ylim=(0, 1))
    axes[1].set(xlabel="Absolute chunk", ylabel="Selective risk", ylim=(0, 1))
    axes[1].axhline(0.2, color="black", linestyle="--", linewidth=1)
    axes[0].legend(fontsize=5)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_au_eu(path: Path, predictions_by_run: Mapping[str, Sequence[EpisodePrediction]]) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    plotted = False
    for run_name, records in predictions_by_run.items():
        if not records or records[0].verifier_au is None:
            continue
        au = np.concatenate([record.verifier_au for record in records])
        eu = np.concatenate([record.verifier_eu for record in records])
        axis.scatter(au, eu, s=8, alpha=0.25, label=run_name)
        plotted = True
    axis.set(xlabel="Verifier AU", ylabel="Verifier EU / vacuity")
    if plotted:
        axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_edl_vs_softmax(path: Path, evaluation: Mapping[str, Any]) -> None:
    labels: list[str] = []
    values: list[float] = []
    for run_name, run in evaluation["runs"].items():
        for policy_name, policy in run["policies"].items():
            if "primary_aurc" in policy:
                labels.append(f"{run_name}\n{policy_name}")
                values.append(policy["primary_aurc"])
    figure, axis = plt.subplots(figsize=(max(7, len(labels) * 0.9), 4))
    axis.bar(np.arange(len(labels)), values)
    axis.set_ylabel("Primary AURC (lower is better)")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _records_from_predictions(records: Sequence[PredictionRecord]) -> tuple[EpisodePrediction, ...]:
    result = []
    for record in records:
        probabilities = np.asarray(record.class_probabilities)
        result.append(EpisodePrediction(
            suite=record.ref.suite,
            episode_key=record.ref.episode_key,
            label=record.ref.label,
            success_probability=probabilities[:, 1],
            failure_probability=probabilities[:, 0],
            token_lengths=None if record.token_mask is None else np.asarray(record.token_mask).sum(axis=1),
            class_evidence=record.class_evidence,
            verifier_au=record.verifier_au,
            verifier_eu=record.verifier_eu,
            verifier_total_evidence=record.verifier_total_evidence,
        ))
    return tuple(result)


def _parse_datasets(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset must use suite=path")
        suite, raw_path = value.split("=", 1)
        if not suite or not raw_path or suite in result:
            raise ValueError("--dataset suites and paths must be non-empty and unique")
        result[suite] = Path(raw_path).expanduser().resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    calibration = read_json(args.calibration)
    datasets = _parse_datasets(args.dataset)
    test_dataset_provenance = collect_dataset_provenance(
        datasets,
        require_collection_identity=True,
    )
    validate_independent_test_datasets(
        test_dataset_provenance,
        calibration.get("source_datasets", {}),
    )
    predictions: dict[str, tuple[EpisodePrediction, ...]] = {}
    for run_name, run in calibration["runs"].items():
        checkpoint = Path(run["checkpoint_path"])
        if sha256_file(checkpoint) != run["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash changed after calibration: {run_name}")
        frozen = load_frozen_verifier(checkpoint, device=args.device)
        predictions[run_name] = _records_from_predictions(
            predict_frozen_datasets(frozen, datasets, batch_size=args.batch_size)
        )
    evaluation = evaluate_prediction_sets(
        calibration,
        predictions,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    evaluation["test_datasets"] = test_dataset_provenance
    destination = publish_evaluation(args.output_dir, evaluation, predictions, overwrite=args.overwrite)
    print(f"Frozen rejection evaluation written to {destination}")


if __name__ == "__main__":
    main()
