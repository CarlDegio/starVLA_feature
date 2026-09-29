"""Atomic HDF5 storage for QwenFast token and latent diagnostics."""

from __future__ import annotations

import datetime
import json
import pathlib
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


SCHEMA_VERSION = "1.0"


class SafeDiagnosticsDatasetWriter:
    """Append validated diagnostic trajectories to one suite-level HDF5 file."""

    def __init__(
        self,
        path: str | pathlib.Path,
        metadata: Mapping[str, Any],
        *,
        overwrite: bool = False,
        resume: bool = False,
    ) -> None:
        if overwrite and resume:
            raise ValueError("overwrite and resume cannot both be enabled")
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        if exists and not overwrite and not resume:
            raise FileExistsError(f"dataset already exists: {self.path}; use overwrite=True or resume=True")
        mode = "w" if overwrite or not exists else "r+"
        self._file = h5py.File(self.path, mode)
        try:
            if mode == "w":
                self._initialize(metadata)
            else:
                self._validate_metadata(metadata)
            self._file.require_group("episodes")
            in_progress = self._file.require_group("_in_progress")
            for stale_key in list(in_progress.keys()):
                del in_progress[stale_key]
            self._file.flush()
        except Exception:
            self._file.close()
            raise

    def __enter__(self) -> "SafeDiagnosticsDatasetWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._file and self._file.id.valid:
            self._file.flush()
            self._file.close()

    @staticmethod
    def episode_key(task_id: int, episode_idx: int) -> str:
        return f"task_{int(task_id):03d}_episode_{int(episode_idx):04d}"

    def has_episode(self, task_id: int, episode_idx: int) -> bool:
        return f"episodes/{self.episode_key(task_id, episode_idx)}" in self._file

    def append_episode(
        self,
        *,
        task_id: int,
        episode_idx: int,
        task_description: str,
        success: bool,
        executed_steps: int,
        termination_reason: str,
        diagnostic_chunks: Sequence[Mapping[str, Any]],
    ) -> None:
        key = self.episode_key(task_id, episode_idx)
        final_path = f"episodes/{key}"
        if final_path in self._file:
            raise ValueError(f"episode already exists: {key}")
        arrays, hidden_dim = self._prepare_episode_arrays(diagnostic_chunks)
        temporary_path = f"_in_progress/{key}"
        if temporary_path in self._file:
            del self._file[temporary_path]
        group = self._file.create_group(temporary_path)
        try:
            group.attrs["success"] = np.uint8(bool(success))
            group.attrs["task_id"] = int(task_id)
            group.attrs["episode_idx"] = int(episode_idx)
            group.attrs["task_description"] = str(task_description)
            group.attrs["num_chunks"] = len(diagnostic_chunks)
            group.attrs["hidden_dim"] = hidden_dim
            group.attrs["executed_steps"] = int(executed_steps)
            group.attrs["termination_reason"] = str(termination_reason)
            for name, values in arrays.items():
                group.create_dataset(name, data=values, compression="gzip", compression_opts=4, shuffle=True)
            self._file.flush()
            self._file.move(temporary_path, final_path)
            self._file.flush()
        except Exception:
            if temporary_path in self._file:
                del self._file[temporary_path]
                self._file.flush()
            raise

    def _initialize(self, metadata: Mapping[str, Any]) -> None:
        self._file.attrs["schema_version"] = SCHEMA_VERSION
        self._file.attrs["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for key, value in metadata.items():
            self._file.attrs[key] = self._attribute_value(value)

    def _validate_metadata(self, metadata: Mapping[str, Any]) -> None:
        if self._file.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("schema version mismatch")
        for key, expected in metadata.items():
            if key not in self._file.attrs:
                raise ValueError(f"resume metadata missing root attribute: {key}")
            normalized = self._attribute_value(expected)
            if self._file.attrs[key] != normalized:
                raise ValueError(
                    f"resume metadata mismatch for {key}: expected {normalized!r}, found {self._file.attrs[key]!r}"
                )

    @staticmethod
    def _attribute_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, pathlib.Path):
            return str(value)
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True)
        return value

    @staticmethod
    def _prepare_episode_arrays(
        chunks: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, np.ndarray], int]:
        if not chunks:
            raise ValueError("cannot write an episode without diagnostic chunks")

        chunk_idx: list[int] = []
        policy_step: list[int] = []
        env_step: list[int] = []
        token_counts: list[int] = []
        token_offsets = [0]
        token_ids: list[np.ndarray] = []
        token_nll: list[np.ndarray] = []
        token_entropy: list[np.ndarray] = []
        embeddings: dict[str, list[np.ndarray]] = {"first": [], "last": [], "mean": []}
        hidden_dim: int | None = None

        for chunk in chunks:
            ids = np.asarray(chunk.get("action_token_ids", []))
            nll = np.asarray(chunk.get("action_token_nll", []), dtype=np.float32).reshape(-1)
            entropy = np.asarray(chunk.get("action_token_entropy", []), dtype=np.float32).reshape(-1)
            if ids.dtype == np.bool_ or not np.issubdtype(ids.dtype, np.integer):
                raise ValueError("action_token_ids must contain integers")
            ids = ids.astype(np.int64, copy=False).reshape(-1)
            lengths = {ids.size, nll.size, entropy.size}
            if len(lengths) != 1:
                raise ValueError("token field lengths must match for IDs, NLL, and entropy")
            count = int(ids.size)
            if count == 0:
                raise ValueError("each diagnostic chunk must contain at least one action token")
            if int(chunk.get("num_action_tokens", count)) != count:
                raise ValueError("num_action_tokens does not match token field lengths")
            if not np.all(np.isfinite(nll)) or not np.all(np.isfinite(entropy)):
                raise ValueError("token diagnostics must contain finite values")

            for aggregation in embeddings:
                name = f"action_token_embedding_{aggregation}"
                feature = np.asarray(chunk.get(name, []), dtype=np.float32).reshape(-1)
                if feature.size == 0 or not np.all(np.isfinite(feature)):
                    raise ValueError(f"{name} must contain a finite feature vector")
                if hidden_dim is None:
                    hidden_dim = int(feature.size)
                if feature.size != hidden_dim:
                    raise ValueError("all embedding fields must use one hidden dimension")
                embeddings[aggregation].append(feature)

            chunk_idx.append(int(chunk["chunk_idx"]))
            policy_step.append(int(chunk["policy_step"]))
            env_step.append(int(chunk["env_step"]))
            token_counts.append(count)
            token_offsets.append(token_offsets[-1] + count)
            token_ids.append(ids)
            token_nll.append(nll)
            token_entropy.append(entropy)

        chunk_idx_array = np.asarray(chunk_idx, dtype=np.int32)
        if chunk_idx_array.size > 1 and not np.all(np.diff(chunk_idx_array) > 0):
            raise ValueError("chunk_idx must be strictly increasing within an episode")
        assert hidden_dim is not None
        return (
            {
                "chunk_idx": chunk_idx_array,
                "policy_step": np.asarray(policy_step, dtype=np.int32),
                "env_step": np.asarray(env_step, dtype=np.int32),
                "num_action_tokens": np.asarray(token_counts, dtype=np.int32),
                "token_offsets": np.asarray(token_offsets, dtype=np.int64),
                "action_token_ids": np.concatenate(token_ids),
                "action_token_nll": np.concatenate(token_nll),
                "action_token_entropy": np.concatenate(token_entropy),
                "embedding_first": np.stack(embeddings["first"]),
                "embedding_last": np.stack(embeddings["last"]),
                "embedding_mean": np.stack(embeddings["mean"]),
            },
            hidden_dim,
        )
