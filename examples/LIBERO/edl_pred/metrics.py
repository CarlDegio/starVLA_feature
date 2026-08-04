"""Dependency-free binary metrics for trajectory verifier validation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)


def binary_metrics(labels: np.ndarray, success_probability: np.ndarray) -> dict[str, float | None]:
    """Compute binary ranking, threshold, and calibration metrics.

    Scores are success probabilities. ROC-AUC and PR-AUC are undefined for a
    one-class label set and are returned as ``None`` in that case.
    """
    target = _binary_labels(labels, "labels")
    probability = _probabilities(success_probability, "success_probability")
    if target.shape != probability.shape:
        raise ValueError("labels and success_probability must have the same shape")

    predicted = probability >= 0.5
    positive = target == 1
    negative = ~positive
    true_positive = int(np.count_nonzero(predicted & positive))
    false_positive = int(np.count_nonzero(predicted & negative))
    false_negative = int(np.count_nonzero(~predicted & positive))
    true_negative = int(np.count_nonzero(~predicted & negative))

    positive_count = true_positive + false_negative
    negative_count = true_negative + false_positive
    positive_recall = true_positive / positive_count if positive_count else None
    negative_recall = true_negative / negative_count if negative_count else None
    recalls = [recall for recall in (positive_recall, negative_recall) if recall is not None]
    balanced_accuracy = float(sum(recalls) / len(recalls))
    f1_denominator = 2 * true_positive + false_positive + false_negative

    if positive_count == 0 or negative_count == 0:
        LOGGER.warning("ROC-AUC and PR-AUC are undefined because labels contain one class")
        roc_auc = None
        pr_auc = None
    else:
        roc_auc = _roc_auc(target, probability, positive_count, negative_count)
        pr_auc = _pr_auc(target, probability, positive_count)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": float((true_positive + true_negative) / target.size),
        "balanced_accuracy": balanced_accuracy,
        "f1": float(2 * true_positive / f1_denominator) if f1_denominator else 0.0,
        "brier": float(np.mean((probability - target) ** 2)),
        "ece": _expected_calibration_error(target, probability),
    }


def sequence_metrics(
    labels: np.ndarray,
    success_probability: np.ndarray,
    chunk_mask: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    """Compute all-valid-chunk and final-valid-chunk metrics per episode."""
    target = _binary_labels(labels, "labels")
    probability = _array(success_probability, "success_probability", ndim=2)
    if probability.shape[0] != target.size:
        raise ValueError("success_probability must have one row per label")
    probability = _probabilities(probability, "success_probability")

    mask = np.asarray(chunk_mask)
    if mask.dtype != np.bool_ or mask.shape != probability.shape:
        raise ValueError("chunk_mask must be boolean with the same shape as success_probability")
    valid_counts = mask.sum(axis=1)
    if np.any(valid_counts == 0):
        raise ValueError("every episode must contain at least one valid chunk")

    all_valid_labels = np.repeat(target, valid_counts)
    all_valid_probability = probability[mask]
    final_indices = probability.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    final_probability = probability[np.arange(target.size), final_indices]
    return {
        "chunk": binary_metrics(all_valid_labels, all_valid_probability),
        "final_chunk": binary_metrics(target, final_probability),
    }


def _roc_auc(labels: np.ndarray, probability: np.ndarray, positive_count: int, negative_count: int) -> float:
    """Compute Mann-Whitney ROC-AUC using average ranks for equal scores."""
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    ranks = np.empty(probability.size, dtype=np.float64)
    start = 0
    while start < probability.size:
        stop = start + 1
        while stop < probability.size and sorted_probability[stop] == sorted_probability[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


def _pr_auc(labels: np.ndarray, probability: np.ndarray, positive_count: int) -> float:
    """Compute average precision over grouped, stable descending thresholds."""
    order = np.argsort(-probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_labels = labels[order]
    true_positive = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < probability.size:
        stop = start + 1
        while stop < probability.size and sorted_probability[stop] == sorted_probability[start]:
            stop += 1
        true_positive += int(sorted_labels[start:stop].sum())
        precision = true_positive / stop
        recall = true_positive / positive_count
        area += precision * (recall - previous_recall)
        previous_recall = recall
        start = stop
    return float(area)


def _expected_calibration_error(labels: np.ndarray, probability: np.ndarray) -> float:
    bin_indices = np.minimum((probability * 10).astype(np.int64), 9)
    error = 0.0
    for index in range(10):
        in_bin = bin_indices == index
        if not np.any(in_bin):
            continue
        error += float(in_bin.mean() * abs(probability[in_bin].mean() - labels[in_bin].mean()))
    return float(error)


def _binary_labels(value: np.ndarray, name: str) -> np.ndarray:
    array = _array(value, name, ndim=1)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must be a one-dimensional numeric binary array")
    if not np.all(np.isfinite(array)) or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only binary values 0 and 1")
    return array.astype(np.int64, copy=False)


def _probabilities(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must be a real-valued numeric array")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    return array.astype(np.float64, copy=False)


def _array(value: Any, name: str, *, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or array.size == 0:
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array")
    return array
