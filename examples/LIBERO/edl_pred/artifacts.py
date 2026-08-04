"""Atomic persistence helpers for verifier metrics, checkpoints, and outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
import numpy as np
import torch

from .dataset import EpisodeRef, SplitManifest
from .model import VerifierOutput


@dataclass(frozen=True)
class PredictionRecord:
    """Already-masked per-episode validation output ready for HDF5 export."""

    ref: EpisodeRef
    class_probabilities: np.ndarray  # [valid_chunks, 2], failure then success
    class_evidence: np.ndarray | None = None  # [valid_chunks, 2]
    verifier_au: np.ndarray | None = None  # [valid_chunks]
    verifier_eu: np.ndarray | None = None  # [valid_chunks]
    verifier_total_evidence: np.ndarray | None = None  # [valid_chunks]
    pooling_weights: np.ndarray | None = None  # [valid_chunks, tokens]
    token_mask: np.ndarray | None = None  # [valid_chunks, tokens]


def prediction_records_from_output(
    refs: Sequence[EpisodeRef],
    output: VerifierOutput,
    chunk_mask: Any,
    token_mask: Any,
) -> list[PredictionRecord]:
    """Detach a padded verifier batch into per-episode valid prediction records."""
    if not isinstance(output, VerifierOutput):
        raise ValueError("output must be a VerifierOutput")
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
        raise ValueError("refs must be a sequence of EpisodeRef values")
    probabilities = _tensor_array(output.probabilities, "output.probabilities", ndim=3)
    if probabilities.shape[-1] != 2:
        raise ValueError("output.probabilities must have shape [batch, chunks, 2]")
    chunk_mask_array = np.asarray(_tensor_value(chunk_mask))
    if chunk_mask_array.dtype != np.bool_ or chunk_mask_array.shape != probabilities.shape[:2]:
        raise ValueError("chunk_mask must be boolean with shape [batch, chunks]")
    token_mask_array = np.asarray(_tensor_value(token_mask))
    if token_mask_array.dtype != np.bool_ or token_mask_array.shape[:2] != probabilities.shape[:2]:
        raise ValueError("token_mask must be boolean with shape [batch, chunks, tokens]")
    if len(refs) != probabilities.shape[0]:
        raise ValueError("refs must contain one EpisodeRef per output batch item")
    if np.any(chunk_mask_array.sum(axis=1) == 0):
        raise ValueError("every output batch item must contain a valid chunk")

    optional_edl = (output.evidence, output.verifier_au, output.verifier_eu, output.verifier_total_evidence)
    if any(value is not None for value in optional_edl) and any(value is None for value in optional_edl):
        raise ValueError("VerifierOutput EDL fields must be present together")
    evidence = verifier_au = verifier_eu = verifier_total_evidence = None
    if output.evidence is not None:
        evidence = _tensor_array(output.evidence, "output.evidence", ndim=3)
        verifier_au = _tensor_array(output.verifier_au, "output.verifier_au", ndim=2)
        verifier_eu = _tensor_array(output.verifier_eu, "output.verifier_eu", ndim=2)
        verifier_total_evidence = _tensor_array(
            output.verifier_total_evidence, "output.verifier_total_evidence", ndim=2
        )
        if evidence.shape != probabilities.shape or any(
            value.shape != probabilities.shape[:2] for value in (verifier_au, verifier_eu, verifier_total_evidence)
        ):
            raise ValueError("VerifierOutput EDL field shapes do not match probabilities")

    pooling_weights = None
    if output.pooling_weights is not None:
        pooling_weights = _tensor_array(output.pooling_weights, "output.pooling_weights", ndim=3)
        if pooling_weights.shape != token_mask_array.shape:
            raise ValueError("output.pooling_weights must match token_mask")

    records: list[PredictionRecord] = []
    for row, ref in enumerate(refs):
        if not isinstance(ref, EpisodeRef):
            raise ValueError("refs must contain EpisodeRef values")
        valid = chunk_mask_array[row]
        records.append(
            PredictionRecord(
                ref=ref,
                class_probabilities=probabilities[row, valid],
                class_evidence=None if evidence is None else evidence[row, valid],
                verifier_au=None if verifier_au is None else verifier_au[row, valid],
                verifier_eu=None if verifier_eu is None else verifier_eu[row, valid],
                verifier_total_evidence=None if verifier_total_evidence is None else verifier_total_evidence[row, valid],
                pooling_weights=None if pooling_weights is None else pooling_weights[row, valid],
                token_mask=None if pooling_weights is None else token_mask_array[row, valid],
            )
        )
    return records


def append_metrics(path: str | Path, metrics: Mapping[str, Any]) -> None:
    """Atomically append one strictly increasing epoch record to a JSONL file."""
    destination = _destination(path, allow_existing=True)
    record = _metrics_record(metrics)
    existing = b""
    if destination.exists():
        existing = destination.read_bytes()
        previous_epoch = _last_metrics_epoch(existing, destination)
        if record["epoch"] <= previous_epoch:
            raise ValueError("metrics epochs must strictly increase")
    payload = existing + json.dumps(record, allow_nan=False, sort_keys=True).encode("utf-8") + b"\n"
    _write_atomic_bytes(destination, payload, allow_existing=True)


def save_checkpoint(path: str | Path, checkpoint: Mapping[str, Any]) -> None:
    """Atomically replace a PyTorch checkpoint after fully flushing its payload."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    destination = _destination(path, allow_existing=True)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(checkpoint), handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        _remove_temporary(temporary)
        raise


def write_split_manifest(path: str | Path, manifest: SplitManifest, *, overwrite: bool = False) -> None:
    """Write a portable split manifest without source-machine dataset paths."""
    if not isinstance(manifest, SplitManifest):
        raise ValueError("manifest must be a SplitManifest")
    if isinstance(manifest.max_action_tokens, bool) or manifest.max_action_tokens <= 0:
        raise ValueError("manifest max_action_tokens must be positive")
    train = _manifest_refs(manifest.train, "train")
    validation = _manifest_refs(manifest.validation, "validation")
    train_identity = {(item["suite"], item["episode_key"]) for item in train}
    validation_identity = {(item["suite"], item["episode_key"]) for item in validation}
    if len(train_identity) != len(train) or len(validation_identity) != len(validation):
        raise ValueError("split manifest contains duplicate episode identities")
    if train_identity & validation_identity:
        raise ValueError("split manifest cannot assign an episode to both splits")

    destination = _destination(path, allow_existing=overwrite)
    payload = {
        "max_action_tokens": manifest.max_action_tokens,
        "train": train,
        "validation": validation,
    }
    _write_atomic_bytes(
        destination,
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        allow_existing=overwrite,
    )


def write_validation_predictions(
    path: str | Path,
    records: Sequence[PredictionRecord],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically write valid per-chunk validation predictions to HDF5."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError("records must be a non-empty sequence")
    validated = [_validate_prediction_record(record) for record in records]
    identities = [(record.ref.suite, record.ref.episode_key) for record in validated]
    if len(set(identities)) != len(identities):
        raise ValueError("prediction records contain duplicate suite and episode identities")

    destination = _destination(path, allow_existing=overwrite)
    temporary = _temporary_path(destination)
    try:
        with h5py.File(temporary, "w") as handle:
            episodes = handle.create_group("episodes")
            for record in validated:
                episode = episodes.require_group(record.ref.suite).create_group(record.ref.episode_key)
                episode.attrs["suite"] = record.ref.suite
                episode.attrs["episode_key"] = record.ref.episode_key
                episode.attrs["label"] = record.ref.label
                episode.create_dataset("failure_probability", data=record.class_probabilities[:, 0])
                episode.create_dataset("success_probability", data=record.class_probabilities[:, 1])
                if record.class_evidence is not None:
                    episode.create_dataset("class_evidence", data=record.class_evidence)
                    episode.create_dataset("verifier_au", data=record.verifier_au)
                    episode.create_dataset("verifier_eu", data=record.verifier_eu)
                    episode.create_dataset("verifier_total_evidence", data=record.verifier_total_evidence)
                if record.pooling_weights is not None:
                    compact_weights = _compact_pooling_weights(record.pooling_weights, record.token_mask)
                    episode.create_dataset("pooling_weights", data=compact_weights)
                    episode.create_dataset("token_lengths", data=record.token_mask.sum(axis=1, dtype=np.int64))
            handle.flush()
        _fsync_path(temporary)
        temporary.replace(destination)
    except BaseException:
        _remove_temporary(temporary)
        raise


def plot_training_curves(
    path: str | Path,
    history: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically render total-loss and available validation ranking curves."""
    destination = _destination(path, allow_existing=overwrite)
    epochs, train_loss, validation_loss, roc_auc, pr_auc = _curve_values(history)
    has_ranking = any(value is not None for value in roc_auc) or any(value is not None for value in pr_auc)
    figure, axes = plt.subplots(1, 2 if has_ranking else 1, figsize=(10, 4))
    loss_axis = axes[0] if has_ranking else axes
    loss_axis.plot(epochs, train_loss, label="train total loss")
    loss_axis.plot(epochs, validation_loss, label="validation total loss")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("loss")
    loss_axis.legend()
    if has_ranking:
        metric_axis = axes[1]
        _plot_available(metric_axis, epochs, roc_auc, "validation ROC-AUC")
        _plot_available(metric_axis, epochs, pr_auc, "validation PR-AUC")
        metric_axis.set_xlabel("epoch")
        metric_axis.set_ylabel("score")
        metric_axis.set_ylim(0.0, 1.0)
        metric_axis.legend()
    figure.tight_layout()

    temporary = _temporary_path(destination)
    try:
        figure.savefig(temporary, format="png")
        _fsync_path(temporary)
        temporary.replace(destination)
    except BaseException:
        _remove_temporary(temporary)
        raise
    finally:
        plt.close(figure)


def _validate_prediction_record(record: PredictionRecord) -> PredictionRecord:
    if not isinstance(record, PredictionRecord) or not isinstance(record.ref, EpisodeRef):
        raise ValueError("each prediction record must contain an EpisodeRef")
    _episode_identity(record.ref)
    probabilities = _real_array(record.class_probabilities, "class_probabilities", ndim=2)
    if probabilities.shape[1] != 2 or probabilities.shape[0] == 0:
        raise ValueError("class_probabilities must have shape [valid_chunks, 2]")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("class_probabilities rows must be probabilities that sum to 1")

    optional_edl = (
        record.class_evidence,
        record.verifier_au,
        record.verifier_eu,
        record.verifier_total_evidence,
    )
    if any(value is not None for value in optional_edl) and any(value is None for value in optional_edl):
        raise ValueError("EDL prediction fields must be supplied together")
    class_evidence = verifier_au = verifier_eu = verifier_total_evidence = None
    if record.class_evidence is not None:
        class_evidence = _real_array(record.class_evidence, "class_evidence", ndim=2)
        if class_evidence.shape != probabilities.shape or np.any(class_evidence < 0.0):
            raise ValueError("class_evidence must be non-negative with shape [valid_chunks, 2]")
        verifier_au = _real_array(record.verifier_au, "verifier_au", ndim=1)
        verifier_eu = _real_array(record.verifier_eu, "verifier_eu", ndim=1)
        verifier_total_evidence = _real_array(record.verifier_total_evidence, "verifier_total_evidence", ndim=1)
        if any(value.shape[0] != probabilities.shape[0] for value in (verifier_au, verifier_eu, verifier_total_evidence)):
            raise ValueError("verifier diagnostic arrays must have one value per valid chunk")

    pooling_weights = token_mask = None
    if (record.pooling_weights is None) != (record.token_mask is None):
        raise ValueError("pooling_weights and token_mask must be supplied together")
    if record.pooling_weights is not None:
        pooling_weights = _real_array(record.pooling_weights, "pooling_weights", ndim=2)
        token_mask = np.asarray(record.token_mask)
        if token_mask.dtype != np.bool_ or token_mask.shape != pooling_weights.shape:
            raise ValueError("token_mask must be boolean with the pooling_weights shape")
        if pooling_weights.shape[0] != probabilities.shape[0] or pooling_weights.shape[1] == 0:
            raise ValueError("pooling_weights must have one row per valid chunk")
        token_counts = token_mask.sum(axis=1)
        if np.any(token_counts == 0):
            raise ValueError("every pooling row must contain a valid token")
        if np.any(pooling_weights[token_mask] < 0.0) or not np.allclose(
            (pooling_weights * token_mask).sum(axis=1), 1.0, rtol=1e-5, atol=1e-6
        ):
            raise ValueError("valid pooling weights must be non-negative and sum to 1")

    return PredictionRecord(
        ref=record.ref,
        class_probabilities=probabilities,
        class_evidence=class_evidence,
        verifier_au=verifier_au,
        verifier_eu=verifier_eu,
        verifier_total_evidence=verifier_total_evidence,
        pooling_weights=pooling_weights,
        token_mask=token_mask,
    )


def _curve_values(history: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[float], list[float], list[float | None], list[float | None]]:
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not history:
        raise ValueError("history must be a non-empty sequence")
    epochs: list[int] = []
    train_loss: list[float] = []
    validation_loss: list[float] = []
    roc_auc: list[float | None] = []
    pr_auc: list[float | None] = []
    for record in history:
        if not isinstance(record, Mapping):
            raise ValueError("each history item must be a mapping")
        epoch = record.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("history epoch must be a non-negative integer")
        if epochs and epoch <= epochs[-1]:
            raise ValueError("history epochs must strictly increase")
        epochs.append(epoch)
        train_loss.append(_finite_history_value(record, ("train_total_loss", "train_loss", "train_total"), "train total loss"))
        validation_loss.append(
            _finite_history_value(record, ("validation_total_loss", "validation_loss", "validation_total"), "validation total loss")
        )
        roc_auc.append(_optional_history_value(record, ("validation_chunk_roc_auc", "chunk_roc_auc", "roc_auc")))
        pr_auc.append(_optional_history_value(record, ("validation_chunk_pr_auc", "chunk_pr_auc", "pr_auc")))
    return epochs, train_loss, validation_loss, roc_auc, pr_auc


def _finite_history_value(record: Mapping[str, Any], names: tuple[str, ...], description: str) -> float:
    for name in names:
        if name in record:
            return _finite_scalar(record[name], description)
    raise ValueError(f"history item is missing {description}")


def _optional_history_value(record: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in record:
            if record[name] is None:
                return None
            return _finite_scalar(record[name], name)
    return None


def _plot_available(axis: Any, epochs: list[int], values: list[float | None], label: str) -> None:
    points = [(epoch, value) for epoch, value in zip(epochs, values) if value is not None]
    if points:
        x_values, y_values = zip(*points)
        axis.plot(x_values, y_values, label=label)


def _metrics_record(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping")
    record = dict(metrics)
    epoch = record.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("metrics must contain a non-negative integer epoch")
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("metrics must be JSON serializable without non-finite values") from error
    return record


def _last_metrics_epoch(content: bytes, path: Path) -> int:
    if content and not content.endswith(b"\n"):
        raise ValueError(f"existing metrics JSONL {path} does not end with a newline")
    epochs: list[int] = []
    for line in content.splitlines():
        if not line:
            raise ValueError(f"existing metrics JSONL {path} contains an empty line")
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"existing metrics JSONL {path} is invalid") from error
        parsed = _metrics_record(item)
        epochs.append(parsed["epoch"])
    if not epochs:
        return -1
    if any(current <= previous for previous, current in zip(epochs, epochs[1:])):
        raise ValueError(f"existing metrics JSONL {path} epochs must strictly increase")
    return epochs[-1]


def _manifest_refs(refs: Sequence[EpisodeRef], split: str) -> list[dict[str, Any]]:
    if not isinstance(refs, Sequence):
        raise ValueError(f"manifest {split} refs must be a sequence")
    payload: list[dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, EpisodeRef):
            raise ValueError(f"manifest {split} refs must contain EpisodeRef values")
        _episode_identity(ref)
        payload.append({"suite": ref.suite, "episode_key": ref.episode_key, "label": ref.label})
    return payload


def _episode_identity(ref: EpisodeRef) -> None:
    if (
        not isinstance(ref.suite, str)
        or not ref.suite
        or "/" in ref.suite
        or not isinstance(ref.episode_key, str)
        or not ref.episode_key
        or "/" in ref.episode_key
        or isinstance(ref.label, bool)
        or ref.label not in (0, 1)
    ):
        raise ValueError("EpisodeRef suite, episode_key, and label are invalid for artifact export")


def _real_array(value: Any, name: str, *, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != ndim
        or array.dtype == np.bool_
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real-valued {ndim}-dimensional array")
    return array.astype(np.float32, copy=False)


def _tensor_array(value: Any, name: str, *, ndim: int) -> np.ndarray:
    return _real_array(_tensor_value(value), name, ndim=ndim)


def _tensor_value(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    return value


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _compact_pooling_weights(pooling_weights: np.ndarray, token_mask: np.ndarray) -> np.ndarray:
    """Move each row's selected valid token weights to leading columns."""
    token_lengths = token_mask.sum(axis=1, dtype=np.int64)
    compact = np.zeros((pooling_weights.shape[0], int(token_lengths.max())), dtype=pooling_weights.dtype)
    for row, length in enumerate(token_lengths):
        compact[row, :length] = pooling_weights[row, token_mask[row]]
    return compact


def _destination(path: str | Path, *, allow_existing: bool) -> Path:
    destination = Path(path)
    if not destination.parent.is_dir():
        raise ValueError(f"artifact parent directory does not exist: {destination.parent}")
    if destination.exists() and not allow_existing:
        raise FileExistsError(f"artifact already exists: {destination}")
    if destination.exists() and destination.is_dir():
        raise ValueError(f"artifact destination must be a file: {destination}")
    return destination


def _temporary_path(destination: Path) -> Path:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(temporary)


def _write_atomic_bytes(destination: Path, payload: bytes, *, allow_existing: bool) -> None:
    _destination(destination, allow_existing=allow_existing)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        _remove_temporary(temporary)
        raise


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_temporary(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
