from __future__ import annotations

import datetime
import json
import pathlib
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


SCHEMA_VERSION = "1.0"


class LiberoUncertaintyDatasetWriter:
    """Append validated LIBERO uncertainty episodes to one suite-level HDF5 file."""

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
            raise FileExistsError(
                f"dataset already exists: {self.path}; use overwrite=True or resume=True"
            )

        mode = "w" if overwrite or not exists else "r+"
        self._file = h5py.File(self.path, mode)
        try:
            if mode == "w":
                self._initialize_file(metadata)
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

    def __enter__(self) -> "LiberoUncertaintyDatasetWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._file:
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
        uncertainty_chunks: Sequence[Mapping[str, Any]],
    ) -> bool:
        key = self.episode_key(task_id, episode_idx)
        final_path = f"episodes/{key}"
        if final_path in self._file:
            raise ValueError(f"episode already exists: {key}")
        if not uncertainty_chunks:
            raise ValueError("cannot write an episode without uncertainty chunks")

        arrays = self._prepare_episode_arrays(uncertainty_chunks)
        temporary_path = f"_in_progress/{key}"
        if temporary_path in self._file:
            del self._file[temporary_path]

        group = self._file.create_group(temporary_path)
        try:
            group.attrs["success"] = np.uint8(bool(success))
            group.attrs["task_id"] = int(task_id)
            group.attrs["episode_idx"] = int(episode_idx)
            group.attrs["task_description"] = str(task_description)
            group.attrs["num_chunks"] = len(uncertainty_chunks)
            group.attrs["executed_steps"] = int(executed_steps)
            group.attrs["termination_reason"] = str(termination_reason)
            for name, values in arrays.items():
                group.create_dataset(
                    name,
                    data=values,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
            self._file.flush()
            self._file.move(temporary_path, final_path)
            self._file.flush()
        except Exception:
            if temporary_path in self._file:
                del self._file[temporary_path]
                self._file.flush()
            raise
        return True

    def _initialize_file(self, metadata: Mapping[str, Any]) -> None:
        self._file.attrs["schema_version"] = SCHEMA_VERSION
        self._file.attrs["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for key, value in metadata.items():
            self._file.attrs[key] = self._attribute_value(value)

    def _validate_metadata(self, metadata: Mapping[str, Any]) -> None:
        actual_version = self._file.attrs.get("schema_version")
        if actual_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema version mismatch: expected {SCHEMA_VERSION}, found {actual_version}"
            )
        for key, expected in metadata.items():
            if key not in self._file.attrs:
                raise ValueError(f"resume metadata missing root attribute: {key}")
            actual = self._file.attrs[key]
            normalized_expected = self._attribute_value(expected)
            if actual != normalized_expected:
                raise ValueError(
                    f"resume metadata mismatch for {key}: expected {normalized_expected!r}, found {actual!r}"
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
        uncertainty_chunks: Sequence[Mapping[str, Any]],
    ) -> dict[str, np.ndarray]:
        chunk_idx = []
        policy_step = []
        env_step = []
        num_generated_tokens = []
        num_action_tokens = []
        token_offsets = [0]
        evidence_rows = []
        au_rows = []
        eu_rows = []
        confidence_rows = []
        rank_rows = []

        for chunk in uncertainty_chunks:
            evidence = np.asarray(chunk.get("action_token_evidence", []), dtype=np.float32).reshape(-1)
            au = np.asarray(
                chunk.get("action_token_aleatoric_uncertainty", []), dtype=np.float32
            ).reshape(-1)
            eu = np.asarray(
                chunk.get("action_token_epistemic_uncertainty", []), dtype=np.float32
            ).reshape(-1)
            confidence = np.asarray(
                chunk.get("action_token_confidence", []), dtype=np.float32
            ).reshape(-1)
            rank = np.asarray(chunk.get("action_token_rank", []), dtype=np.float32).reshape(-1)
            lengths = {evidence.size, au.size, eu.size, confidence.size, rank.size}
            if len(lengths) != 1:
                raise ValueError(
                    "action-token field lengths must match for evidence, AU, EU, confidence, and rank"
                )
            token_count = int(evidence.size)
            if token_count == 0:
                raise ValueError("each uncertainty chunk must contain at least one action token")
            if not all(np.all(np.isfinite(values)) for values in (evidence, au, eu, confidence, rank)):
                raise ValueError("action-token fields must contain only finite values")
            rounded_rank = np.rint(rank)
            if not np.allclose(rank, rounded_rank):
                raise ValueError("action-token rank values must be integers")
            recorded_count = chunk.get("num_action_tokens")
            if recorded_count is not None and int(recorded_count) != token_count:
                raise ValueError(
                    f"num_action_tokens mismatch: expected {token_count}, found {recorded_count}"
                )

            chunk_idx.append(int(chunk["chunk_idx"]))
            policy_step.append(int(chunk["policy_step"]))
            env_step.append(int(chunk["env_step"]))
            num_generated_tokens.append(int(chunk.get("num_tokens", token_count)))
            num_action_tokens.append(token_count)
            token_offsets.append(token_offsets[-1] + token_count)
            evidence_rows.append(evidence)
            au_rows.append(au)
            eu_rows.append(eu)
            confidence_rows.append(confidence)
            rank_rows.append(rounded_rank.astype(np.int32))

        chunk_idx_array = np.asarray(chunk_idx, dtype=np.int32)
        if chunk_idx_array.size > 1 and not np.all(np.diff(chunk_idx_array) > 0):
            raise ValueError("chunk_idx must be strictly increasing within an episode")

        return {
            "chunk_idx": chunk_idx_array,
            "policy_step": np.asarray(policy_step, dtype=np.int32),
            "env_step": np.asarray(env_step, dtype=np.int32),
            "num_generated_tokens": np.asarray(num_generated_tokens, dtype=np.int32),
            "num_action_tokens": np.asarray(num_action_tokens, dtype=np.int32),
            "token_offsets": np.asarray(token_offsets, dtype=np.int64),
            "evidence": np.concatenate(evidence_rows).astype(np.float32, copy=False),
            "aleatoric_uncertainty": np.concatenate(au_rows).astype(np.float32, copy=False),
            "epistemic_uncertainty": np.concatenate(eu_rows).astype(np.float32, copy=False),
            "confidence": np.concatenate(confidence_rows).astype(np.float32, copy=False),
            "rank": np.concatenate(rank_rows).astype(np.int32, copy=False),
        }
