"""Read, split, and collate collector-format LIBERO uncertainty trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
from typing import Any, Sequence

import h5py
import numpy as np

from .config import DataConfig


_COLLECTOR_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, order=True)
class EpisodeRef:
    suite: str
    path: Path
    episode_key: str
    label: int


@dataclass(frozen=True)
class SplitManifest:
    train: tuple[EpisodeRef, ...]
    validation: tuple[EpisodeRef, ...]
    max_action_tokens: int


@dataclass(frozen=True)
class EpisodeSample:
    ref: EpisodeRef
    features: np.ndarray  # float32 [chunks, max_tokens, 2]
    token_mask: np.ndarray  # bool [chunks, max_tokens]
    label: int


def build_episode_splits(data_config: DataConfig, *, seed: int | None = None) -> SplitManifest:
    """Index collector HDF5 files and create a deterministic per-suite split."""
    rng = random.Random(data_config.split_seed if seed is None else seed)
    train: list[EpisodeRef] = []
    validation: list[EpisodeRef] = []
    observed_max = 0

    for suite in sorted(data_config.selected_suites):
        refs, suite_max = _index_suite(suite, data_config.datasets[suite])
        observed_max = max(observed_max, suite_max)
        by_label = {label: [ref for ref in refs if ref.label == label] for label in (0, 1)}
        if not all(by_label.values()):
            raise ValueError(f"suite {suite!r} must contain both labels 0 and 1")
        for label in (0, 1):
            stratum = by_label[label]
            count = len(stratum)
            if count < 2:
                raise ValueError(f"suite {suite!r} label {label} needs at least 2 episodes for splitting")
            shuffled = list(stratum)
            rng.shuffle(shuffled)
            validation_count = min(
                count - 1,
                max(1, round(count * data_config.validation_ratio)),
            )
            validation.extend(shuffled[:validation_count])
            train.extend(shuffled[validation_count:])

    if observed_max == 0:
        raise ValueError("no action tokens were indexed")
    max_action_tokens = data_config.max_action_tokens or observed_max
    if observed_max > max_action_tokens:
        raise ValueError(
            "data.max_action_tokens is smaller than an observed num_action_tokens value: "
            f"{max_action_tokens} < {observed_max}"
        )
    return SplitManifest(tuple(sorted(train)), tuple(sorted(validation)), max_action_tokens)


class TrajectoryDataset:
    """Lazily decode episode references without sharing HDF5 handles between workers."""

    def __init__(self, refs: Sequence[EpisodeRef], max_action_tokens: int) -> None:
        if isinstance(max_action_tokens, bool) or max_action_tokens <= 0:
            raise ValueError("max_action_tokens must be positive")
        self.refs = tuple(refs)
        self.max_action_tokens = max_action_tokens
        self._handles: dict[Path, h5py.File] = {}
        self._handle_owner_pid: int | None = os.getpid()

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> EpisodeSample:
        ref = self.refs[index]
        handle = self._get_handle(ref.path)
        _validate_root_schema(handle, ref.path)
        group = _episode_group(handle, ref.episode_key, ref.path)
        label = _read_label(group, ref.path, ref.episode_key)
        if label != ref.label:
            raise ValueError(f"episode {ref.episode_key!r} label does not match its EpisodeRef")
        lengths = _read_integer_vector(group, "num_action_tokens", ref.path, ref.episode_key)
        offsets = _read_integer_vector(group, "token_offsets", ref.path, ref.episode_key)
        _validate_offsets(lengths, offsets, ref.path, ref.episode_key)
        if int(lengths.max()) > self.max_action_tokens:
            raise ValueError(
                f"episode {ref.episode_key!r} exceeds max_action_tokens={self.max_action_tokens}"
            )
        au = _read_feature_vector(group, "aleatoric_uncertainty", ref.path, ref.episode_key)
        eu = _read_feature_vector(group, "epistemic_uncertainty", ref.path, ref.episode_key)
        total_tokens = int(offsets[-1])
        if au.size != eu.size:
            raise ValueError(
                f"episode {ref.episode_key!r} aleatoric_uncertainty and epistemic_uncertainty lengths must match"
            )
        if au.size != total_tokens:
            raise ValueError(
                f"episode {ref.episode_key!r} uncertainty length does not match token_offsets"
            )

        features = np.zeros((len(lengths), self.max_action_tokens, 2), dtype=np.float32)
        token_mask = np.zeros((len(lengths), self.max_action_tokens), dtype=bool)
        for chunk_index, token_count in enumerate(lengths.tolist()):
            start, stop = int(offsets[chunk_index]), int(offsets[chunk_index + 1])
            features[chunk_index, :token_count, 0] = au[start:stop]
            features[chunk_index, :token_count, 1] = eu[start:stop]
            token_mask[chunk_index, :token_count] = True
        return EpisodeSample(ref, features, token_mask, label)

    def _get_handle(self, path: Path) -> h5py.File:
        self._reset_handles_after_fork()
        handle = self._handles.get(path)
        if handle is None or not handle.id.valid:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def _reset_handles_after_fork(self) -> None:
        current_pid = os.getpid()
        if self._handle_owner_pid != current_pid:
            # Do not close inherited handles in the child; discard and reopen them.
            self._handles = {}
            self._handle_owner_pid = current_pid

    def close(self) -> None:
        self._reset_handles_after_fork()
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_handle_owner_pid"] = None
        return state


def collate_trajectories(samples: Sequence[EpisodeSample]) -> dict[str, Any]:
    """Pad a batch on its chunk axis while retaining fixed token-axis padding."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    max_chunks = max(sample.features.shape[0] for sample in samples)
    max_tokens = samples[0].features.shape[1]
    for sample in samples:
        if sample.features.ndim != 3 or sample.features.shape[2] != 2:
            raise ValueError("sample features must have shape [chunks, max_tokens, 2]")
        if sample.token_mask.shape != sample.features.shape[:2]:
            raise ValueError("sample token_mask shape must match feature chunk and token axes")
        if sample.features.shape[1] != max_tokens:
            raise ValueError("all samples must use the same max_action_tokens")

    batch_size = len(samples)
    features = np.zeros((batch_size, max_chunks, max_tokens, 2), dtype=np.float32)
    token_mask = np.zeros((batch_size, max_chunks, max_tokens), dtype=bool)
    chunk_mask = np.zeros((batch_size, max_chunks), dtype=bool)
    chunk_lengths = np.zeros(batch_size, dtype=np.int64)
    for row, sample in enumerate(samples):
        chunk_count = sample.features.shape[0]
        features[row, :chunk_count] = sample.features
        token_mask[row, :chunk_count] = sample.token_mask
        chunk_mask[row, :chunk_count] = True
        chunk_lengths[row] = chunk_count
    return {
        "features": features,
        "token_mask": token_mask,
        "chunk_mask": chunk_mask,
        "chunk_lengths": chunk_lengths,
        "labels": np.asarray([sample.label for sample in samples], dtype=np.int64),
        "refs": tuple(sample.ref for sample in samples),
    }


def _index_suite(suite: str, path: Path) -> tuple[list[EpisodeRef], int]:
    with h5py.File(path, "r") as handle:
        _validate_root_schema(handle, path)
        if "episodes" not in handle or not isinstance(handle["episodes"], h5py.Group):
            raise ValueError(f"dataset {path} is missing the episodes group")
        refs: list[EpisodeRef] = []
        observed_max = 0
        for episode_key in sorted(handle["episodes"].keys()):
            group = _episode_group(handle, episode_key, path)
            label = _read_label(group, path, episode_key)
            lengths = _read_integer_vector(group, "num_action_tokens", path, episode_key)
            if lengths.size == 0 or np.any(lengths <= 0):
                raise ValueError(f"episode {episode_key!r} num_action_tokens must be positive")
            if "num_chunks" in group.attrs and int(group.attrs["num_chunks"]) != lengths.size:
                raise ValueError(f"episode {episode_key!r} num_chunks does not match num_action_tokens")
            observed_max = max(observed_max, int(lengths.max()))
            refs.append(EpisodeRef(suite, path, episode_key, label))
    return refs, observed_max


def _validate_root_schema(handle: h5py.File, path: Path) -> None:
    if handle.attrs.get("schema_version") != _COLLECTOR_SCHEMA_VERSION:
        raise ValueError(
            f"dataset {path} has unsupported schema_version "
            f"{handle.attrs.get('schema_version')!r}"
        )


def _episode_group(handle: h5py.File, episode_key: str, path: Path) -> h5py.Group:
    location = f"episodes/{episode_key}"
    if location not in handle or not isinstance(handle[location], h5py.Group):
        raise ValueError(f"dataset {path} is missing episode {episode_key!r}")
    return handle[location]


def _read_label(group: h5py.Group, path: Path, episode_key: str) -> int:
    if "success" not in group.attrs:
        raise ValueError(f"episode {episode_key!r} in {path} is missing success")
    raw_label = group.attrs["success"]
    if (
        isinstance(raw_label, np.ndarray)
        or isinstance(raw_label, (bool, np.bool_))
        or not isinstance(raw_label, (int, np.integer))
        or int(raw_label) not in (0, 1)
    ):
        raise ValueError(f"episode {episode_key!r} success must be binary")
    return int(raw_label)


def _read_integer_vector(group: h5py.Group, name: str, path: Path, episode_key: str) -> np.ndarray:
    if name not in group:
        raise ValueError(f"episode {episode_key!r} in {path} is missing {name}")
    values = np.asarray(group[name][...])
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"episode {episode_key!r} {name} must be a one-dimensional integer array")
    return values.astype(np.int64, copy=False)


def _read_feature_vector(group: h5py.Group, name: str, path: Path, episode_key: str) -> np.ndarray:
    if name not in group:
        raise ValueError(f"episode {episode_key!r} in {path} is missing {name}")
    values = np.asarray(group[name][...])
    if (
        values.ndim != 1
        or not np.issubdtype(values.dtype, np.number)
        or np.issubdtype(values.dtype, np.complexfloating)
    ):
        raise ValueError(f"episode {episode_key!r} {name} must be a one-dimensional real-valued numeric array")
    values = values.astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"episode {episode_key!r} {name} must contain only finite values")
    return values


def _validate_offsets(lengths: np.ndarray, offsets: np.ndarray, path: Path, episode_key: str) -> None:
    if lengths.size == 0 or np.any(lengths <= 0):
        raise ValueError(f"episode {episode_key!r} num_action_tokens must be positive")
    if offsets.size != lengths.size + 1 or offsets[0] != 0:
        raise ValueError(f"episode {episode_key!r} token_offsets must start at 0 and have one boundary per chunk")
    differences = np.diff(offsets)
    if np.any(differences <= 0) or not np.array_equal(differences, lengths):
        raise ValueError(f"episode {episode_key!r} token_offsets must match positive num_action_tokens")
