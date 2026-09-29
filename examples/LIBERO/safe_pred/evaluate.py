"""Offline evaluation for trained SAFE-style failure detectors."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch

from examples.LIBERO.safe_pred.dataset import SafeEpisode, collate_episodes, load_episodes
from examples.LIBERO.safe_pred.metrics import binary_ranking_metrics
from examples.LIBERO.safe_pred.model import SafeFailureDetector


@dataclasses.dataclass
class Args:
    checkpoint_path: str
    dataset_paths: list[str]
    output_path: str = ""
    device: str = "auto"


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def evaluate_model(
    model: SafeFailureDetector,
    episodes: Sequence[SafeEpisode],
    *,
    device: str | torch.device,
) -> dict:
    if not episodes:
        raise ValueError("evaluation requires at least one episode")
    torch_device = resolve_device(device) if isinstance(device, str) else device
    task_lengths: dict[tuple[str, int], list[int]] = defaultdict(list)
    for episode in episodes:
        task_lengths[(episode.suite, episode.task_id)].append(episode.features.shape[0])
    prefixes = {key: min(lengths) for key, lengths in task_lengths.items()}

    model = model.to(torch_device).eval()
    labels = []
    scores = []
    with torch.inference_mode():
        for episode in episodes:
            batch = collate_episodes([episode])
            output = model(batch.features.to(torch_device), batch.chunk_mask.to(torch_device))
            prefix = prefixes[(episode.suite, episode.task_id)]
            scores.append(float(output.failure_score[0, :prefix].max().cpu()))
            labels.append(int(episode.failure_label))
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    metrics = binary_ranking_metrics(label_array, score_array)
    metrics["brier"] = (
        float(np.mean((score_array - label_array) ** 2)) if model.backbone == "lstm" else None
    )
    return {
        "episode_ids": [episode.episode_id for episode in episodes],
        "failure_labels": label_array,
        "failure_scores": score_array,
        "task_prefix_chunks": {f"{suite}:{task_id}": value for (suite, task_id), value in sorted(prefixes.items())},
        "metrics": metrics,
    }


def load_detector_checkpoint(
    checkpoint_path: str | pathlib.Path,
    *,
    device: str | torch.device,
) -> tuple[SafeFailureDetector, dict]:
    torch_device = resolve_device(device) if isinstance(device, str) else device
    payload = torch.load(pathlib.Path(checkpoint_path), map_location=torch_device, weights_only=False)
    model_args = dict(payload["model_args"])
    model = SafeFailureDetector(**model_args)
    model.load_state_dict(payload["model_state_dict"])
    return model.to(torch_device), payload


def evaluate_checkpoint(
    checkpoint_path: str | pathlib.Path,
    dataset_paths: Sequence[str | pathlib.Path],
    *,
    device: str = "auto",
) -> dict:
    model, payload = load_detector_checkpoint(checkpoint_path, device=device)
    episodes = load_episodes(
        dataset_paths,
        feature_aggregation=str(payload["feature_aggregation"]),
    )
    return evaluate_model(model, episodes, device=device)


def _json_ready(result: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }


def main(args: Args) -> None:
    result = evaluate_checkpoint(args.checkpoint_path, args.dataset_paths, device=args.device)
    serializable = _json_ready(result)
    if args.output_path:
        path = pathlib.Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(serializable, indent=2, sort_keys=True))


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
