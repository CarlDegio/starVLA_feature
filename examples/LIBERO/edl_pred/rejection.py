"""Pure selective-rejection policies and deterministic threshold calibration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np


class PolicyKind(str, Enum):
    """Supported rejection policy families."""

    AU = "au"
    AU_OR_EU = "au_or_eu"
    PREDICTIVE_ENTROPY = "predictive_entropy"


@dataclass(frozen=True)
class PolicyThresholds:
    """Validated immutable thresholds for one rejection policy."""

    kind: PolicyKind | Literal["au", "au_or_eu", "predictive_entropy"]
    tau_au: float | None = None
    tau_eu: float | None = None
    tau_entropy: float | None = None

    def __post_init__(self) -> None:
        try:
            kind = PolicyKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported policy kind: {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)

        required = {
            PolicyKind.AU: ("tau_au",),
            PolicyKind.AU_OR_EU: ("tau_au", "tau_eu"),
            PolicyKind.PREDICTIVE_ENTROPY: ("tau_entropy",),
        }[kind]
        for name in ("tau_au", "tau_eu", "tau_entropy"):
            value = getattr(self, name)
            if name in required:
                object.__setattr__(self, name, _threshold(value, name))
            elif value is not None:
                raise ValueError(f"{name} is not valid for policy kind {kind.value!r}")

    def to_dict(self) -> dict[str, str | float]:
        """Return a stable, JSON-safe representation, including sentinels."""
        result: dict[str, str | float] = {"kind": self.kind.value}
        for name in ("tau_au", "tau_eu", "tau_entropy"):
            value = getattr(self, name)
            if value is not None:
                result[name] = _serialize_threshold(value)
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyThresholds":
        """Restore thresholds from the strict JSON representation."""
        if not isinstance(payload, Mapping):
            raise ValueError("policy thresholds must be a mapping")
        allowed = {"kind", "tau_au", "tau_eu", "tau_entropy"}
        if "kind" not in payload or set(payload) - allowed:
            raise ValueError("policy thresholds contain missing or unknown fields")
        values = {
            name: _deserialize_threshold(payload[name])
            for name in ("tau_au", "tau_eu", "tau_entropy")
            if name in payload
        }
        return cls(kind=payload["kind"], **values)


@dataclass(frozen=True)
class RejectionResult:
    accepted: np.ndarray
    predicted_label: np.ndarray
    decision: np.ndarray
    reason: np.ndarray


@dataclass(frozen=True)
class CalibrationRecords:
    """A balanced calibration population from absolute chunks 1 through 10."""

    labels: np.ndarray
    success_probability: np.ndarray
    episode_id: np.ndarray
    absolute_chunk: np.ndarray
    au: np.ndarray | None = None
    eu: np.ndarray | None = None

    def __post_init__(self) -> None:
        labels = _binary_labels(self.labels, "labels")
        probability = _probabilities(self.success_probability, "success_probability")
        episode_id = _episode_ids(self.episode_id)
        absolute_chunk = _absolute_chunks(self.absolute_chunk)
        arrays = (probability, episode_id, absolute_chunk)
        if any(array.shape != labels.shape for array in arrays):
            raise ValueError("calibration record fields must have the same shape")

        au = _optional_uncertainty(self.au, "au", labels.shape)
        eu = _optional_uncertainty(self.eu, "eu", labels.shape)
        expected_chunks = np.arange(1, 11)
        for episode in np.unique(episode_id):
            in_episode = episode_id == episode
            chunks = np.sort(absolute_chunk[in_episode])
            if not np.array_equal(chunks, expected_chunks):
                raise ValueError(
                    "each calibration episode must contain exactly one record for absolute chunks 1 through 10"
                )
            if np.unique(labels[in_episode]).size != 1:
                raise ValueError("all chunks of a calibration episode must have the same label")

        for name, value in (
            ("labels", labels),
            ("success_probability", probability),
            ("episode_id", episode_id),
            ("absolute_chunk", absolute_chunk),
            ("au", au),
            ("eu", eu),
        ):
            if value is not None:
                object.__setattr__(self, name, _read_only(value))


@dataclass(frozen=True)
class OperatingPoint:
    """One selected calibration operating point."""

    available: bool
    target_coverage: float | None
    thresholds: PolicyThresholds | None
    coverage: float | None
    abstention_rate: float | None
    selective_error: float | None
    selective_accuracy: float | None
    accepted_count: int
    accepted_episode_count: int
    rejected_count: int
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "target_coverage": self.target_coverage,
            "thresholds": None if self.thresholds is None else self.thresholds.to_dict(),
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "selective_error": self.selective_error,
            "selective_accuracy": self.selective_accuracy,
            "accepted_count": self.accepted_count,
            "accepted_episode_count": self.accepted_episode_count,
            "rejected_count": self.rejected_count,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class CalibrationResult:
    kind: PolicyKind
    default: OperatingPoint
    target_operating_points: Mapping[float, OperatingPoint]
    candidate_count: int
    record_count: int
    episode_count: int
    max_selective_error: float
    min_coverage: float
    min_accepted_episodes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_operating_points",
            MappingProxyType(dict(self.target_operating_points)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "default": self.default.to_dict(),
            "target_operating_points": {
                str(target): point.to_dict()
                for target, point in self.target_operating_points.items()
            },
            "candidate_count": self.candidate_count,
            "record_count": self.record_count,
            "episode_count": self.episode_count,
            "max_selective_error": self.max_selective_error,
            "min_coverage": self.min_coverage,
            "min_accepted_episodes": self.min_accepted_episodes,
        }


def predictive_entropy(success_probability: np.ndarray) -> np.ndarray:
    """Return normalized binary entropy, with ``0 log(0)`` defined as zero."""
    probability = _probabilities(success_probability, "success_probability")
    entropy = np.zeros(probability.shape, dtype=np.float64)
    interior = (probability > 0.0) & (probability < 1.0)
    p = probability[interior]
    entropy[interior] = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return entropy


def apply_rejection(
    success_probability: np.ndarray,
    policy: PolicyThresholds,
    *,
    au: np.ndarray | None = None,
    eu: np.ndarray | None = None,
) -> RejectionResult:
    """Apply a global policy without consulting labels."""
    if not isinstance(policy, PolicyThresholds):
        raise TypeError("policy must be a PolicyThresholds instance")
    probability = _probabilities(success_probability, "success_probability")
    au_array = _optional_uncertainty(au, "au", probability.shape)
    eu_array = _optional_uncertainty(eu, "eu", probability.shape)

    if policy.kind is PolicyKind.AU:
        if au_array is None:
            raise ValueError("au is required for policy kind 'au'")
        accepted = au_array <= policy.tau_au
        reason = np.where(accepted, "accepted", "high_au")
    elif policy.kind is PolicyKind.AU_OR_EU:
        if au_array is None:
            raise ValueError("au is required for policy kind 'au_or_eu'")
        if eu_array is None:
            raise ValueError("eu is required for policy kind 'au_or_eu'")
        high_au = au_array > policy.tau_au
        high_eu = eu_array > policy.tau_eu
        accepted = ~(high_au | high_eu)
        reason = np.full(probability.shape, "accepted", dtype="<U19")
        reason[high_au & ~high_eu] = "high_au"
        reason[~high_au & high_eu] = "high_eu"
        reason[high_au & high_eu] = "high_au_and_eu"
    else:
        entropy = predictive_entropy(probability)
        accepted = entropy <= policy.tau_entropy
        reason = np.where(accepted, "accepted", "high_entropy")

    predicted_label = (probability >= 0.5).astype(np.int64)
    decision = np.full(probability.shape, "UNDETERMINED", dtype="<U12")
    decision[accepted & (predicted_label == 1)] = "SUCCESS"
    decision[accepted & (predicted_label == 0)] = "FAILURE"
    return RejectionResult(
        accepted=_read_only(accepted.astype(np.bool_, copy=False)),
        predicted_label=_read_only(predicted_label),
        decision=_read_only(decision),
        reason=_read_only(np.asarray(reason)),
    )


def candidate_thresholds(values: np.ndarray) -> tuple[float, ...]:
    """Return exact observed thresholds with rejecting/accepting sentinels."""
    array = _uncertainty(values, "values")
    return (-np.inf, *(float(value) for value in np.unique(array)), np.inf)


def calibrate_policy(
    records: CalibrationRecords,
    kind: PolicyKind | str,
    *,
    max_selective_error: float = 0.20,
    min_coverage: float = 0.10,
    min_accepted_episodes: int = 10,
    target_coverages: Sequence[float] = (0.25, 0.50, 0.75, 0.90),
) -> CalibrationResult:
    """Exactly calibrate the default and diagnostic target operating points."""
    if not isinstance(records, CalibrationRecords):
        raise TypeError("records must be CalibrationRecords")
    try:
        policy_kind = PolicyKind(kind)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported policy kind: {kind!r}") from exc
    max_error = _unit_interval(max_selective_error, "max_selective_error")
    minimum_coverage = _unit_interval(min_coverage, "min_coverage")
    if isinstance(min_accepted_episodes, (bool, np.bool_)) or not isinstance(
        min_accepted_episodes, (int, np.integer)
    ) or min_accepted_episodes <= 0:
        raise ValueError("min_accepted_episodes must be a positive integer")
    targets = tuple(_unit_interval(value, "target coverage") for value in target_coverages)
    if len(set(targets)) != len(targets):
        raise ValueError("target_coverages must not contain duplicates")

    policies = tuple(_candidate_policies(records, policy_kind))
    evaluated = tuple(_evaluate_candidate(records, policy) for policy in policies)
    supported = [
        point
        for point in evaluated
        if point.coverage >= minimum_coverage
        and point.accepted_episode_count >= min_accepted_episodes
    ]
    feasible = [point for point in supported if point.selective_error <= max_error]
    if feasible:
        default = min(feasible, key=_default_selection_key)
    else:
        default = OperatingPoint(
            available=False,
            target_coverage=None,
            thresholds=None,
            coverage=None,
            abstention_rate=None,
            selective_error=None,
            selective_accuracy=None,
            accepted_count=0,
            accepted_episode_count=0,
            rejected_count=records.labels.size,
            unavailable_reason=(
                "no candidate satisfies max_selective_error, min_coverage, "
                "and min_accepted_episodes"
            ),
        )

    target_points: dict[float, OperatingPoint] = {}
    for target in targets:
        if supported:
            selected = min(supported, key=lambda point: _target_selection_key(point, target))
            target_points[target] = _with_target(selected, target)
        else:
            target_points[target] = OperatingPoint(
                available=False,
                target_coverage=target,
                thresholds=None,
                coverage=None,
                abstention_rate=None,
                selective_error=None,
                selective_accuracy=None,
                accepted_count=0,
                accepted_episode_count=0,
                rejected_count=records.labels.size,
                unavailable_reason="no candidate satisfies min_coverage and min_accepted_episodes",
            )

    return CalibrationResult(
        kind=policy_kind,
        default=default,
        target_operating_points=target_points,
        candidate_count=len(policies),
        record_count=records.labels.size,
        episode_count=np.unique(records.episode_id).size,
        max_selective_error=max_error,
        min_coverage=minimum_coverage,
        min_accepted_episodes=int(min_accepted_episodes),
    )


def calibrate_default_operating_point(
    records: CalibrationRecords,
    kind: PolicyKind | str,
    **kwargs: Any,
) -> OperatingPoint:
    """Convenience wrapper for callers that only need the deployable point."""
    return calibrate_policy(records, kind, **kwargs).default


def _candidate_policies(
    records: CalibrationRecords, kind: PolicyKind
) -> Iterable[PolicyThresholds]:
    if kind is PolicyKind.AU:
        if records.au is None:
            raise ValueError("au is required to calibrate policy kind 'au'")
        for threshold in candidate_thresholds(records.au):
            yield PolicyThresholds(kind, tau_au=threshold)
    elif kind is PolicyKind.AU_OR_EU:
        if records.au is None or records.eu is None:
            raise ValueError("au and eu are required to calibrate policy kind 'au_or_eu'")
        candidates = product(candidate_thresholds(records.au), candidate_thresholds(records.eu))
        for tau_au, tau_eu in candidates:
            yield PolicyThresholds(kind, tau_au=tau_au, tau_eu=tau_eu)
    else:
        for threshold in candidate_thresholds(predictive_entropy(records.success_probability)):
            yield PolicyThresholds(kind, tau_entropy=threshold)


def _evaluate_candidate(records: CalibrationRecords, policy: PolicyThresholds) -> OperatingPoint:
    result = apply_rejection(
        records.success_probability,
        policy,
        au=records.au,
        eu=records.eu,
    )
    accepted_count = int(np.count_nonzero(result.accepted))
    errors = result.predicted_label != records.labels
    selective_error = float(np.mean(errors[result.accepted])) if accepted_count else 0.0
    accepted_episode_count = int(np.unique(records.episode_id[result.accepted]).size)
    coverage = accepted_count / records.labels.size
    return OperatingPoint(
        available=True,
        target_coverage=None,
        thresholds=policy,
        coverage=float(coverage),
        abstention_rate=float(1.0 - coverage),
        selective_error=selective_error,
        selective_accuracy=float(1.0 - selective_error),
        accepted_count=accepted_count,
        accepted_episode_count=accepted_episode_count,
        rejected_count=records.labels.size - accepted_count,
    )


def _default_selection_key(point: OperatingPoint) -> tuple[Any, ...]:
    assert point.coverage is not None and point.selective_error is not None
    assert point.thresholds is not None
    thresholds = _threshold_tuple(point.thresholds)
    return (-point.coverage, point.selective_error, tuple(-value for value in thresholds), thresholds)


def _target_selection_key(point: OperatingPoint, target: float) -> tuple[Any, ...]:
    assert point.coverage is not None and point.selective_error is not None
    assert point.thresholds is not None
    thresholds = _threshold_tuple(point.thresholds)
    return (
        abs(point.coverage - target),
        point.selective_error,
        tuple(-value for value in thresholds),
        thresholds,
    )


def _with_target(point: OperatingPoint, target: float) -> OperatingPoint:
    return OperatingPoint(
        available=point.available,
        target_coverage=target,
        thresholds=point.thresholds,
        coverage=point.coverage,
        abstention_rate=point.abstention_rate,
        selective_error=point.selective_error,
        selective_accuracy=point.selective_accuracy,
        accepted_count=point.accepted_count,
        accepted_episode_count=point.accepted_episode_count,
        rejected_count=point.rejected_count,
        unavailable_reason=point.unavailable_reason,
    )


def _threshold_tuple(policy: PolicyThresholds) -> tuple[float, ...]:
    if policy.kind is PolicyKind.AU:
        return (policy.tau_au,)
    if policy.kind is PolicyKind.AU_OR_EU:
        return (policy.tau_au, policy.tau_eu)
    return (policy.tau_entropy,)


def _threshold(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a real scalar threshold")
    threshold = float(value)
    if np.isnan(threshold):
        raise ValueError(f"{name} must not be NaN")
    if np.isfinite(threshold) and not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be in [0, 1] or a boundary sentinel")
    return threshold


def _serialize_threshold(value: float) -> str | float:
    if value == np.inf:
        return "+inf"
    if value == -np.inf:
        return "-inf"
    return value


def _deserialize_threshold(value: Any) -> float:
    if value == "+inf":
        return np.inf
    if value == "-inf":
        return -np.inf
    if isinstance(value, str):
        raise ValueError(f"unknown threshold sentinel: {value!r}")
    return _threshold(value, "threshold")


def _unit_interval(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a real scalar in [0, 1]")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return result


def _binary_labels(value: Any, name: str) -> np.ndarray:
    array = _one_dimensional(value, name)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a numeric binary array")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must be a real-valued binary array")
    if not np.all(np.isfinite(array)) or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only 0 and 1")
    return array.astype(np.int64, copy=True)


def _probabilities(value: Any, name: str) -> np.ndarray:
    array = _uncertainty(value, name)
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must contain values in [0, 1]")
    return array


def _uncertainty(value: Any, name: str) -> np.ndarray:
    array = _one_dimensional(value, name)
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a real-valued numeric array")
    if np.issubdtype(array.dtype, np.complexfloating) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real values")
    return array.astype(np.float64, copy=True)


def _optional_uncertainty(
    value: Any | None, name: str, expected_shape: tuple[int, ...]
) -> np.ndarray | None:
    if value is None:
        return None
    array = _probabilities(value, name)
    if array.shape != expected_shape:
        raise ValueError(f"{name} and success_probability must have the same shape")
    return array


def _episode_ids(value: Any) -> np.ndarray:
    array = _one_dimensional(value, "episode_id")
    result = np.asarray([str(item) for item in array], dtype=np.str_)
    if np.any(np.char.str_len(result) == 0):
        raise ValueError("episode_id values must be non-empty")
    return result


def _absolute_chunks(value: Any) -> np.ndarray:
    array = _one_dimensional(value, "absolute_chunk")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("absolute_chunk must be an integer array")
    return array.astype(np.int64, copy=True)


def _one_dimensional(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    return array


def _read_only(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result
