"""Load and batch SAFE-format latent trajectories."""

from __future__ import annotations

import dataclasses
import pathlib
from collections import defaultdict
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclasses.dataclass(frozen=True)
class SafeEpisode:
    episode_id: str
    source_path: str
    suite: str
    task_id: int
    success: bool
    features: np.ndarray

    @property
    def failure_label(self) -> float:
        return 0.0 if self.success else 1.0


@dataclasses.dataclass(frozen=True)
class SafeBatch:
    features: torch.Tensor
    chunk_mask: torch.Tensor
    failure_labels: torch.Tensor
    episodes: tuple[SafeEpisode, ...]


class SafeEpisodeDataset(Dataset):
    def __init__(self, episodes: Sequence[SafeEpisode]) -> None:
        if not episodes:
            raise ValueError("SafeEpisodeDataset requires at least one episode")
        self.episodes = tuple(episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> SafeEpisode:
        return self.episodes[index]


def load_episodes(
    paths: Sequence[str | pathlib.Path],
    *,
    feature_aggregation: str,
) -> list[SafeEpisode]:
    if feature_aggregation not in {"first", "last", "mean"}:
        raise ValueError("feature_aggregation must be first, last, or mean")
    if not paths:
        raise ValueError("at least one dataset path is required")
    dataset_name = f"embedding_{feature_aggregation}"
    episodes: list[SafeEpisode] = []
    hidden_dim: int | None = None
    for path_value in paths:
        path = pathlib.Path(path_value).expanduser().resolve()
        with h5py.File(path, "r") as dataset:
            suite = str(dataset.attrs.get("task_suite", path.stem))
            if "episodes" not in dataset:
                raise ValueError(f"dataset has no episodes group: {path}")
            for key in sorted(dataset["episodes"].keys()):
                group = dataset["episodes"][key]
                if dataset_name not in group:
                    raise ValueError(f"episode {key} is missing {dataset_name}")
                features = np.asarray(group[dataset_name][:], dtype=np.float32)
                if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
                    raise ValueError(f"episode {key} has invalid feature shape {features.shape}")
                if not np.all(np.isfinite(features)):
                    raise ValueError(f"episode {key} contains non-finite features")
                if hidden_dim is None:
                    hidden_dim = int(features.shape[1])
                if features.shape[1] != hidden_dim:
                    raise ValueError(
                        f"hidden dimension mismatch: expected {hidden_dim}, found {features.shape[1]} in {path}"
                    )
                episodes.append(
                    SafeEpisode(
                        episode_id=f"{suite}:{path.name}:{key}",
                        source_path=str(path),
                        suite=suite,
                        task_id=int(group.attrs["task_id"]),
                        success=bool(group.attrs["success"]),
                        features=features,
                    )
                )
    if not episodes:
        raise ValueError("datasets contain no episodes")
    return episodes


def split_episodes_by_suite(
    episodes: Sequence[SafeEpisode],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[SafeEpisode], list[SafeEpisode]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    by_suite: dict[str, list[SafeEpisode]] = defaultdict(list)
    for episode in episodes:
        by_suite[episode.suite].append(episode)
    rng = np.random.default_rng(seed)
    train: list[SafeEpisode] = []
    validation: list[SafeEpisode] = []
    for suite in sorted(by_suite):
        suite_episodes = sorted(by_suite[suite], key=lambda episode: episode.episode_id)
        if len(suite_episodes) < 2:
            raise ValueError(f"suite {suite} requires at least two episodes for a train/validation split")
        order = rng.permutation(len(suite_episodes))
        val_count = max(1, int(round(len(suite_episodes) * val_fraction)))
        val_count = min(val_count, len(suite_episodes) - 1)
        val_indices = set(int(index) for index in order[:val_count])
        for index, episode in enumerate(suite_episodes):
            (validation if index in val_indices else train).append(episode)
    return train, validation


def collate_episodes(episodes: Sequence[SafeEpisode]) -> SafeBatch:
    if not episodes:
        raise ValueError("cannot collate an empty episode batch")
    hidden_dims = {episode.features.shape[1] for episode in episodes}
    if len(hidden_dims) != 1:
        raise ValueError("all episodes in a batch must use one hidden dimension")
    max_chunks = max(episode.features.shape[0] for episode in episodes)
    hidden_dim = hidden_dims.pop()
    features = torch.zeros((len(episodes), max_chunks, hidden_dim), dtype=torch.float32)
    mask = torch.zeros((len(episodes), max_chunks), dtype=torch.bool)
    labels = torch.empty((len(episodes),), dtype=torch.float32)
    for index, episode in enumerate(episodes):
        chunk_count = episode.features.shape[0]
        features[index, :chunk_count] = torch.from_numpy(episode.features)
        mask[index, :chunk_count] = True
        labels[index] = episode.failure_label
    return SafeBatch(features, mask, labels, tuple(episodes))
