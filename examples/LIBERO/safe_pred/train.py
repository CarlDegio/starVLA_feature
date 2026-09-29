"""Train SAFE-style MLP or LSTM detectors on frozen QwenFast features."""

from __future__ import annotations

import json
import pathlib
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from examples.LIBERO.safe_pred.config import TrainConfig, load_config
from examples.LIBERO.safe_pred.dataset import (
    SafeBatch,
    SafeEpisode,
    SafeEpisodeDataset,
    collate_episodes,
    load_episodes,
    split_episodes_by_suite,
)
from examples.LIBERO.safe_pred.evaluate import evaluate_model, resolve_device
from examples.LIBERO.safe_pred.model import DetectorOutput, SafeFailureDetector


def train(config: TrainConfig) -> dict:
    _seed_everything(config.seed)
    episodes = load_episodes(config.dataset_paths, feature_aggregation=config.feature_aggregation)
    train_episodes, val_episodes = split_episodes_by_suite(
        episodes,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    run_dir = pathlib.Path(config.output_dir).expanduser() / config.run_name
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(f"run directory already exists: {run_dir}") from exc

    device = resolve_device(config.device)
    input_dim = int(train_episodes[0].features.shape[1])
    model_args = {
        "backbone": config.backbone,
        "input_dim": input_dim,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
    }
    model = SafeFailureDetector(**model_args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = DataLoader(
        SafeEpisodeDataset(train_episodes),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_episodes,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(
        SafeEpisodeDataset(val_episodes),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_episodes,
    )
    class_weights = _class_weights(train_episodes, device)

    _write_json(run_dir / "config.json", config.as_dict())
    _write_json(
        run_dir / "split_manifest.json",
        {
            "train": [episode.episode_id for episode in train_episodes],
            "validation": [episode.episode_id for episode in val_episodes],
        },
    )
    history_path = run_dir / "epochs.jsonl"
    best_selection = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(model, train_loader, device, class_weights, optimizer=optimizer)
        val_loss = _run_epoch(model, val_loader, device, class_weights, optimizer=None)
        val_result = evaluate_model(model, val_episodes, device=device)
        val_auc = val_result["metrics"]["roc_auc"]
        selection = float(val_auc) if val_auc is not None else -float(val_loss)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_roc_auc": val_auc,
            "val_pr_auc": val_result["metrics"]["pr_auc"],
            "val_brier": val_result["metrics"]["brier"],
        }
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(row, sort_keys=True) + "\n")
        if selection > best_selection:
            best_selection = selection
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(run_dir / "best.pt", model, model_args, config, epoch)
        else:
            epochs_without_improvement += 1
        if config.early_stopping_patience and epochs_without_improvement >= config.early_stopping_patience:
            break

    with history_path.open(encoding="utf-8") as history_file:
        completed_epochs = sum(1 for _ in history_file)
    _save_checkpoint(run_dir / "final.pt", model, model_args, config, completed_epochs)
    summary = {
        "run_dir": str(run_dir.resolve()),
        "backbone": config.backbone,
        "feature_aggregation": config.feature_aggregation,
        "num_train_episodes": len(train_episodes),
        "num_validation_episodes": len(val_episodes),
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def _run_epoch(
    model: SafeFailureDetector,
    loader: DataLoader,
    device: torch.device,
    class_weights: tuple[float, float],
    *,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    model.train(optimizer is not None)
    losses = []
    for batch in loader:
        assert isinstance(batch, SafeBatch)
        features = batch.features.to(device)
        mask = batch.chunk_mask.to(device)
        labels = batch.failure_labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        output = model(features, mask)
        loss = _detector_loss(model.backbone, output, labels, class_weights)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _detector_loss(
    backbone: str,
    output: DetectorOutput,
    labels: torch.Tensor,
    class_weights: tuple[float, float],
) -> torch.Tensor:
    mask = output.chunk_mask
    expanded_labels = labels[:, None].expand_as(output.logits)
    negative_weight, positive_weight = class_weights
    if backbone == "lstm":
        raw = F.binary_cross_entropy_with_logits(output.logits, expanded_labels, reduction="none")
    else:
        upper = torch.arange(1, output.failure_score.shape[1] + 1, device=labels.device)[None, :]
        raw = torch.where(
            expanded_labels.bool(),
            upper - output.failure_score,
            output.failure_score,
        )
    episode_weights = torch.where(
        labels.bool(),
        torch.as_tensor(positive_weight, device=labels.device),
        torch.as_tensor(negative_weight, device=labels.device),
    )
    weighted = raw * episode_weights[:, None]
    return weighted[mask].mean()


def _class_weights(episodes: Sequence[SafeEpisode], device: torch.device) -> tuple[float, float]:
    del device
    labels = np.asarray([episode.failure_label for episode in episodes], dtype=np.int64)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return 1.0, 1.0
    return labels.size / (2.0 * negatives), labels.size / (2.0 * positives)


def _save_checkpoint(
    path: pathlib.Path,
    model: SafeFailureDetector,
    model_args: dict,
    config: TrainConfig,
    epoch: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_args": model_args,
            "feature_aggregation": config.feature_aggregation,
            "epoch": int(epoch),
        },
        path,
    )


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config_path: str) -> None:
    summary = train(load_config(config_path))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
