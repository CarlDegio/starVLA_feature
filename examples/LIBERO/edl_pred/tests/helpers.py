"""Synthetic collector-schema fixtures for EDL trajectory dataset tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


def decode_preloaded_dataset_after_fork(dataset: object, result_queue: object) -> None:
    """Decode one sample in a forked process and report cache ownership."""
    try:
        sample = dataset[0]  # type: ignore[index]
        result_queue.put(("ok", os.getpid(), dataset._handle_owner_pid, sample.label))
    except BaseException as error:
        result_queue.put(("error", type(error).__name__, str(error)))


def write_uncertainty_hdf5(
    path: Path,
    episodes: Sequence[tuple[str, int, Sequence[int]]],
) -> Path:
    """Write episode key, binary label, and per-chunk token lengths.

    AU and EU values are deterministic nonzero ramps. The helper writes the
    same flattened fields and offsets used by the collector schema.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "1.0"
        episodes_group = handle.create_group("episodes")
        for episode_index, (key, label, chunk_lengths) in enumerate(episodes):
            lengths = np.asarray(chunk_lengths, dtype=np.int32)
            total_tokens = int(lengths.sum())
            offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
            values = np.arange(total_tokens, dtype=np.float32) + 1.0 + episode_index * 100.0
            group = episodes_group.create_group(key)
            group.attrs["success"] = np.uint8(label)
            group.attrs["task_id"] = episode_index
            group.attrs["episode_idx"] = episode_index
            group.attrs["task_description"] = f"synthetic {key}"
            group.attrs["num_chunks"] = len(lengths)
            group.attrs["executed_steps"] = len(lengths)
            group.attrs["termination_reason"] = "success"
            group.create_dataset("chunk_idx", data=np.arange(len(lengths), dtype=np.int32))
            group.create_dataset("policy_step", data=np.arange(len(lengths), dtype=np.int32))
            group.create_dataset("env_step", data=np.arange(len(lengths), dtype=np.int32))
            group.create_dataset("num_generated_tokens", data=lengths)
            group.create_dataset("num_action_tokens", data=lengths)
            group.create_dataset("token_offsets", data=offsets)
            group.create_dataset("evidence", data=values + 2000.0)
            group.create_dataset("aleatoric_uncertainty", data=values)
            group.create_dataset("epistemic_uncertainty", data=values + 1000.0)
            group.create_dataset("confidence", data=values + 3000.0)
            group.create_dataset("rank", data=np.arange(total_tokens, dtype=np.int32))
    return path
