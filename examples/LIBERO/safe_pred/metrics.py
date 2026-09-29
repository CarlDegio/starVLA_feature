"""Trajectory-level metrics for raw QwenFast token uncertainty."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def binary_ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    target = np.asarray(labels)
    score = np.asarray(scores, dtype=np.float64)
    if target.ndim != 1 or score.ndim != 1 or target.size == 0 or target.shape != score.shape:
        raise ValueError("labels and scores must be non-empty one-dimensional arrays of the same shape")
    if not np.all((target == 0) | (target == 1)):
        raise ValueError("labels must contain only 0 and 1")
    if not np.all(np.isfinite(score)):
        raise ValueError("scores must contain only finite values")
    target = target.astype(np.int64, copy=False)
    positive_count = int(target.sum())
    negative_count = int(target.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": _roc_auc(target, score, positive_count, negative_count),
        "pr_auc": _average_precision(target, score, positive_count),
    }


def evaluate_token_uncertainty_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one episode record is required")
    by_task: dict[tuple[str | None, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[_task_key(record)].append(record)
    task_prefixes = {
        task_id: min(_chunk_count(record) for record in task_records)
        for task_id, task_records in by_task.items()
    }
    if any(prefix <= 0 for prefix in task_prefixes.values()):
        raise ValueError("every episode must contain at least one chunk")

    labels: list[int] = []
    score_values: dict[str, list[float]] = {
        "max_nll": [],
        "mean_nll": [],
        "max_entropy": [],
        "mean_entropy": [],
    }
    for record in records:
        labels.append(0 if bool(record["success"]) else 1)
        prefix = task_prefixes[_task_key(record)]
        nll_chunks = _token_chunks(record, "action_token_nll")[:prefix]
        entropy_chunks = _token_chunks(record, "action_token_entropy")[:prefix]
        score_values["max_nll"].append(max(float(np.max(chunk)) for chunk in nll_chunks))
        score_values["mean_nll"].append(max(float(np.mean(chunk)) for chunk in nll_chunks))
        score_values["max_entropy"].append(max(float(np.max(chunk)) for chunk in entropy_chunks))
        score_values["mean_entropy"].append(max(float(np.mean(chunk)) for chunk in entropy_chunks))

    label_array = np.asarray(labels, dtype=np.int64)
    result_scores: dict[str, dict[str, Any]] = {}
    for name, values in score_values.items():
        metrics = binary_ranking_metrics(label_array, np.asarray(values, dtype=np.float64))
        result_scores[name] = {
            **metrics,
            "brier": None,
            "brier_reason": "Raw token uncertainty is a ranking score, not a failure probability.",
        }
    failure_count = int(label_array.sum())
    return {
        "positive_class": "failure",
        "num_episodes": int(label_array.size),
        "num_failures": failure_count,
        "failure_prevalence": float(failure_count / label_array.size),
        "task_prefix_chunks": {
            _format_task_key(key): int(value)
            for key, value in sorted(task_prefixes.items(), key=lambda item: str(item[0]))
        },
        "scores": result_scores,
    }


def _task_key(record: Mapping[str, Any]) -> tuple[str | None, int]:
    suite = record.get("suite")
    return (None if suite is None else str(suite), int(record["task_id"]))


def _format_task_key(key: tuple[str | None, int]) -> str:
    suite, task_id = key
    return str(task_id) if suite is None else f"{suite}:{task_id}"


def _chunk_count(record: Mapping[str, Any]) -> int:
    counts = np.asarray(record["num_action_tokens"])
    if counts.ndim != 1:
        raise ValueError("num_action_tokens must be one-dimensional")
    return int(counts.size)


def _token_chunks(record: Mapping[str, Any], field: str) -> list[np.ndarray]:
    counts = np.asarray(record["num_action_tokens"], dtype=np.int64)
    offsets = np.asarray(record["token_offsets"], dtype=np.int64)
    values = np.asarray(record[field], dtype=np.float64)
    if counts.ndim != 1 or offsets.shape != (counts.size + 1,) or values.ndim != 1:
        raise ValueError(f"invalid token layout for {field}")
    if np.any(counts <= 0) or offsets[0] != 0 or offsets[-1] != values.size:
        raise ValueError(f"invalid token offsets for {field}")
    if not np.array_equal(np.diff(offsets), counts):
        raise ValueError(f"token offsets do not match num_action_tokens for {field}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field} contains non-finite values")
    return [values[offsets[index] : offsets[index + 1]] for index in range(counts.size)]


def _roc_auc(labels: np.ndarray, scores: np.ndarray, positives: int, negatives: int) -> float:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = float(ranks[labels == 1].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray, positives: int) -> float:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    true_positives = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        true_positives += int(sorted_labels[start:stop].sum())
        precision = true_positives / stop
        recall = true_positives / positives
        area += precision * (recall - previous_recall)
        previous_recall = recall
        start = stop
    return float(area)
