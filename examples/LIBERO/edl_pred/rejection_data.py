"""Strict readers and identity helpers for frozen verifier rejection analysis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import h5py
import numpy as np

from .rejection import CalibrationRecords


@dataclass(frozen=True)
class EpisodePrediction:
    suite: str
    episode_key: str
    label: int
    success_probability: np.ndarray
    failure_probability: np.ndarray
    token_lengths: np.ndarray | None
    class_evidence: np.ndarray | None = None
    verifier_au: np.ndarray | None = None
    verifier_eu: np.ndarray | None = None
    verifier_total_evidence: np.ndarray | None = None

    @property
    def episode_id(self) -> str:
        return f"{self.suite}/{self.episode_key}"

    @property
    def num_chunks(self) -> int:
        return int(self.success_probability.size)

    @property
    def identity(self) -> tuple[str, str, int, int]:
        return self.suite, self.episode_key, self.label, self.num_chunks


def read_prediction_file(
    path: str | Path,
    *,
    expected_head: Literal["edl", "softmax"] | None = None,
) -> tuple[EpisodePrediction, ...]:
    """Read one verifier prediction HDF5 into validated immutable records."""
    source = Path(path).expanduser().resolve()
    if expected_head not in {None, "edl", "softmax"}:
        raise ValueError("expected_head must be 'edl', 'softmax', or None")
    records: list[EpisodePrediction] = []
    with h5py.File(source, "r") as handle:
        if "episodes" not in handle or not isinstance(handle["episodes"], h5py.Group):
            raise ValueError(f"prediction file {source} is missing episodes")
        for suite_name in sorted(handle["episodes"]):
            suite_group = handle[f"episodes/{suite_name}"]
            if not isinstance(suite_group, h5py.Group):
                raise ValueError(f"prediction suite {suite_name!r} must be a group")
            for episode_key in sorted(suite_group):
                group = suite_group[episode_key]
                if not isinstance(group, h5py.Group):
                    raise ValueError(f"prediction episode {episode_key!r} must be a group")
                records.append(_read_episode_prediction(group, str(suite_name), str(episode_key), expected_head))
    if not records:
        raise ValueError(f"prediction file {source} contains no episodes")
    identities = [record.identity for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("prediction file contains duplicate episode identities")
    return tuple(records)


def validate_matched_predictions(
    left: Sequence[EpisodePrediction],
    right: Sequence[EpisodePrediction],
) -> None:
    """Require two model outputs to cover exactly the same labeled episodes."""
    left_identity = sorted(record.identity for record in left)
    right_identity = sorted(record.identity for record in right)
    if left_identity != right_identity:
        raise ValueError("matched prediction identities, labels, or chunk counts differ")


def build_calibration_records(
    predictions: Sequence[EpisodePrediction],
    *,
    primary_chunks: Sequence[int] = tuple(range(1, 11)),
) -> CalibrationRecords:
    """Flatten equal absolute chunks per episode into strict calibration records."""
    chunks = tuple(int(value) for value in primary_chunks)
    if chunks != tuple(range(1, 11)):
        raise ValueError("primary_chunks must be exactly absolute chunks 1 through 10")
    if not predictions:
        raise ValueError("predictions must not be empty")
    is_edl = predictions[0].verifier_au is not None
    episode_ids: list[str] = []
    labels: list[int] = []
    probability: list[float] = []
    absolute_chunk: list[int] = []
    au: list[float] = []
    eu: list[float] = []
    for record in sorted(predictions, key=lambda item: item.identity):
        if record.num_chunks < chunks[-1]:
            raise ValueError(f"episode {record.episode_id!r} does not reach absolute chunk 10")
        if (record.verifier_au is not None) != is_edl or (record.verifier_eu is not None) != is_edl:
            raise ValueError("prediction records mix EDL and softmax fields")
        for chunk in chunks:
            index = chunk - 1
            episode_ids.append(record.episode_id)
            labels.append(record.label)
            probability.append(float(record.success_probability[index]))
            absolute_chunk.append(chunk)
            if is_edl:
                assert record.verifier_au is not None and record.verifier_eu is not None
                au.append(float(record.verifier_au[index]))
                eu.append(float(record.verifier_eu[index]))
    return CalibrationRecords(
        labels=np.asarray(labels, dtype=np.int64),
        success_probability=np.asarray(probability, dtype=np.float64),
        episode_id=np.asarray(episode_ids),
        absolute_chunk=np.asarray(absolute_chunk, dtype=np.int64),
        au=None if not is_edl else np.asarray(au, dtype=np.float64),
        eu=None if not is_edl else np.asarray(eu, dtype=np.float64),
    )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 identity for an input artifact."""
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def collect_dataset_provenance(
    datasets: Mapping[str, str | Path],
    *,
    require_collection_identity: bool,
) -> dict[str, dict[str, Any]]:
    """Read immutable identities from a suite-indexed collector dataset bundle."""
    if not datasets:
        raise ValueError("datasets must not be empty")
    result: dict[str, dict[str, Any]] = {}
    for suite, raw_path in sorted(datasets.items()):
        if not isinstance(suite, str) or not suite:
            raise ValueError("dataset suite names must be non-empty strings")
        path = Path(raw_path).expanduser().resolve()
        with h5py.File(path, "r") as handle:
            task_suite = _attribute_string(handle.attrs.get("task_suite"))
            collection_id = _attribute_string(handle.attrs.get("collection_id"))
            seed_namespace = _attribute_string(handle.attrs.get("seed_namespace"))
            raw_seed = handle.attrs.get("seed")
        if task_suite is not None and task_suite != suite:
            raise ValueError(f"dataset suite metadata mismatch: expected {suite!r}, found {task_suite!r}")
        if require_collection_identity and (not collection_id or not seed_namespace):
            raise ValueError(
                f"independent test dataset {path} requires non-empty collection_id and seed_namespace"
            )
        result[suite] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "task_suite": task_suite,
            "collection_id": collection_id,
            "seed_namespace": seed_namespace,
            "seed": None if raw_seed is None else int(raw_seed),
        }
    if require_collection_identity:
        collection_ids = {item["collection_id"] for item in result.values()}
        namespaces = {item["seed_namespace"] for item in result.values()}
        if len(collection_ids) != 1:
            raise ValueError("all independent test suites must share one collection_id")
        if len(namespaces) != 1:
            raise ValueError("all independent test suites must share one seed_namespace")
    return result


def validate_independent_test_datasets(
    test_datasets: Mapping[str, Mapping[str, Any]],
    calibration_datasets: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject reused source files or collection namespaces before test inference."""
    if calibration_datasets and set(test_datasets) != set(calibration_datasets):
        raise ValueError("test and calibration dataset suites differ")
    calibration_paths = {str(item.get("path")) for item in calibration_datasets.values()}
    calibration_hashes = {str(item.get("sha256")) for item in calibration_datasets.values()}
    calibration_ids = {
        str(item["collection_id"])
        for item in calibration_datasets.values()
        if item.get("collection_id")
    }
    calibration_namespaces = {
        str(item["seed_namespace"])
        for item in calibration_datasets.values()
        if item.get("seed_namespace")
    }
    for suite, item in test_datasets.items():
        if not item.get("collection_id") or not item.get("seed_namespace"):
            raise ValueError(f"test dataset {suite!r} lacks collection identity")
        conflicts = (
            str(item.get("path")) in calibration_paths
            or str(item.get("sha256")) in calibration_hashes
            or str(item.get("collection_id")) in calibration_ids
            or str(item.get("seed_namespace")) in calibration_namespaces
        )
        if conflicts:
            raise ValueError(f"test dataset {suite!r} is not independent from calibration data")


def _attribute_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    return text or None


def _read_episode_prediction(
    group: h5py.Group,
    suite_name: str,
    episode_key: str,
    expected_head: str | None,
) -> EpisodePrediction:
    suite = str(group.attrs.get("suite", suite_name))
    key = str(group.attrs.get("episode_key", episode_key))
    if suite != suite_name or key != episode_key:
        raise ValueError("prediction group attributes do not match its HDF5 path")
    raw_label = group.attrs.get("label")
    if isinstance(raw_label, (bool, np.bool_)) or not isinstance(raw_label, (int, np.integer)) or int(raw_label) not in (0, 1):
        raise ValueError(f"prediction episode {suite}/{key} has invalid label")
    success = _vector(group, "success_probability", np.float64)
    failure = _vector(group, "failure_probability", np.float64)
    token_lengths = _vector(group, "token_lengths", np.int64) if "token_lengths" in group else None
    if success.shape != failure.shape:
        raise ValueError("prediction probability shapes must match")
    if token_lengths is not None and (token_lengths.shape != success.shape or np.any(token_lengths <= 0)):
        raise ValueError("prediction token length shape must match probabilities")
    if np.any(success < 0.0) or np.any(success > 1.0) or np.any(failure < 0.0) or np.any(failure > 1.0):
        raise ValueError("prediction probabilities must be in [0, 1]")
    if not np.allclose(success + failure, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("prediction class probabilities must sum to one")
    edl_names = ("class_evidence", "verifier_au", "verifier_eu", "verifier_total_evidence")
    present = tuple(name in group for name in edl_names)
    if any(present) and not all(present):
        raise ValueError("EDL prediction fields must be present together")
    if expected_head == "edl" and not all(present):
        raise ValueError("EDL prediction file is missing EDL fields")
    if expected_head == "softmax" and any(present):
        raise ValueError("softmax prediction file unexpectedly contains EDL fields")
    evidence = au = eu = total = None
    if all(present):
        evidence = _matrix(group, "class_evidence", np.float64)
        au = _vector(group, "verifier_au", np.float64)
        eu = _vector(group, "verifier_eu", np.float64)
        total = _vector(group, "verifier_total_evidence", np.float64)
        if evidence.shape != (success.size, 2) or any(value.shape != success.shape for value in (au, eu, total)):
            raise ValueError("EDL field shapes must match prediction chunks")
        if np.any(evidence < 0.0) or np.any(au < 0.0) or np.any(au > 1.0) or np.any(eu < 0.0) or np.any(eu > 1.0):
            raise ValueError("EDL fields contain invalid values")
    return EpisodePrediction(
        suite=suite,
        episode_key=key,
        label=int(raw_label),
        success_probability=_readonly(success),
        failure_probability=_readonly(failure),
        token_lengths=None if token_lengths is None else _readonly(token_lengths),
        class_evidence=None if evidence is None else _readonly(evidence),
        verifier_au=None if au is None else _readonly(au),
        verifier_eu=None if eu is None else _readonly(eu),
        verifier_total_evidence=None if total is None else _readonly(total),
    )


def _vector(group: h5py.Group, name: str, dtype: np.dtype) -> np.ndarray:
    if name not in group:
        raise ValueError(f"prediction episode is missing {name}")
    value = np.asarray(group[name][...])
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
        raise ValueError(f"prediction field {name} must be a finite numeric vector")
    return value.astype(dtype, copy=False)


def _matrix(group: h5py.Group, name: str, dtype: np.dtype) -> np.ndarray:
    if name not in group:
        raise ValueError(f"prediction episode is missing {name}")
    value = np.asarray(group[name][...])
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
        raise ValueError(f"prediction field {name} must be a finite numeric matrix")
    return value.astype(dtype, copy=False)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result
