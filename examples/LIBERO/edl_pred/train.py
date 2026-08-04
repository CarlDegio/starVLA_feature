"""Deterministic training entrypoint for the LIBERO trajectory verifier."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import logging
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

# Make documented direct-script execution resolve the top-level ``examples`` package.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.LIBERO.edl_pred.artifacts import (
    PredictionRecord,
    append_metrics,
    plot_training_curves,
    prediction_records_from_output,
    save_checkpoint,
    write_split_manifest,
    write_validation_predictions,
)
from examples.LIBERO.edl_pred.config import ExperimentConfig, config_to_dict, load_config
from examples.LIBERO.edl_pred.dataset import SplitManifest, TrajectoryDataset, build_episode_splits, collate_trajectories
from examples.LIBERO.edl_pred.losses import LossOutput, compute_verifier_loss
from examples.LIBERO.edl_pred.metrics import binary_metrics
from examples.LIBERO.edl_pred.model import TrajectoryVerifier, build_verifier


LOGGER = logging.getLogger(__name__)


def train(config: ExperimentConfig) -> Path:
    """Run deterministic training and write all verifier artifacts for one run."""
    _seed_everything(config.run.seed)
    device = _resolve_device(config.training.device)
    output_dir = _preflight_run_directory(config)
    manifest = build_episode_splits(config.data)
    class_weights = _class_weights(manifest, config, device)
    resolved_config = _resolved_config(config, manifest, device, class_weights)
    _prepare_run_directory(output_dir, config.run.overwrite)
    _write_yaml_atomic(output_dir / "resolved_config.yaml", resolved_config)
    write_split_manifest(output_dir / "split_manifest.json", manifest)

    train_dataset = TrajectoryDataset(manifest.train, manifest.max_action_tokens)
    validation_dataset = TrajectoryDataset(manifest.validation, manifest.max_action_tokens)
    try:
        train_loader = _build_loader(
            train_dataset,
            config,
            shuffle=True,
            seed=config.run.seed,
        )
        validation_loader = _build_loader(
            validation_dataset,
            config,
            shuffle=False,
            seed=config.run.seed + 1,
        )
        model = build_verifier(config.model, config.edl, manifest.max_action_tokens).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir()
        best_path = checkpoint_dir / "best.pt"
        last_path = checkpoint_dir / "last.pt"
        history: list[dict[str, Any]] = []
        best_loss = float("inf")
        non_improving_epochs = 0

        for epoch in range(1, config.training.epochs + 1):
            train_summary = run_epoch(model, train_loader, optimizer, config, epoch, class_weights)
            validation_summary, _ = validate(
                model,
                validation_loader,
                config,
                epoch,
                class_weights,
                collect_predictions=False,
            )
            record = _epoch_record(epoch, train_summary, validation_summary)
            append_metrics(output_dir / "metrics.jsonl", record)
            history.append(record)
            checkpoint = _checkpoint_payload(
                model,
                optimizer,
                epoch,
                resolved_config,
                manifest,
                record["validation_total_loss"],
            )
            save_checkpoint(last_path, checkpoint)

            if record["validation_total_loss"] < best_loss:
                best_loss = record["validation_total_loss"]
                non_improving_epochs = 0
                save_checkpoint(best_path, checkpoint)
            else:
                non_improving_epochs += 1
                if non_improving_epochs >= config.training.early_stopping_patience:
                    LOGGER.info("Early stopping after epoch %d", epoch)
                    break

        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        final_summary, records = validate(
            model,
            validation_loader,
            config,
            int(best_checkpoint["epoch"]),
            class_weights,
            collect_predictions=True,
        )
        if not records:
            raise RuntimeError("best-checkpoint validation produced no prediction records")
        write_validation_predictions(output_dir / "validation_predictions.hdf5", records)
        plot_training_curves(output_dir / "training_curves.png", history)
        LOGGER.info(
            "Restored epoch %d with validation total loss %.6f",
            best_checkpoint["epoch"],
            final_summary["total_loss"],
        )
        return output_dir
    finally:
        train_dataset.close()
        validation_dataset.close()


def run_epoch(
    model: TrajectoryVerifier,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
    class_weights: torch.Tensor | None,
) -> dict[str, float]:
    """Train one epoch and return episode-weighted loss summaries."""
    model.train()
    totals = _loss_totals()
    episode_count = 0
    device = _model_device(model)
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["features"], batch["token_mask"], batch["chunk_lengths"])
        loss = compute_verifier_loss(
            output,
            batch["labels"],
            batch["chunk_mask"],
            config.model.head,
            config.edl,
            epoch,
            class_weights,
        )
        _require_finite(loss.total, "training loss")
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
        _require_finite_gradients(model)
        optimizer.step()
        batch_episodes = int(batch["labels"].shape[0])
        _accumulate_loss(totals, loss, batch_episodes)
        episode_count += batch_episodes
    return _mean_loss_totals(totals, episode_count)


@torch.inference_mode()
def validate(
    model: TrajectoryVerifier,
    loader: DataLoader[Any],
    config: ExperimentConfig,
    epoch: int,
    class_weights: torch.Tensor | None,
    collect_predictions: bool,
) -> tuple[dict[str, Any], list[PredictionRecord]]:
    """Evaluate one epoch and optionally retain valid per-episode outputs."""
    model.eval()
    totals = _loss_totals()
    episode_count = 0
    records: list[PredictionRecord] = []
    chunk_labels: list[np.ndarray] = []
    chunk_probabilities: list[np.ndarray] = []
    final_labels: list[np.ndarray] = []
    final_probabilities: list[np.ndarray] = []
    edl_values = {name: {0: [], 1: []} for name in ("verifier_au", "verifier_eu", "verifier_total_evidence")}
    device = _model_device(model)

    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        output = model(batch["features"], batch["token_mask"], batch["chunk_lengths"])
        loss = compute_verifier_loss(
            output,
            batch["labels"],
            batch["chunk_mask"],
            config.model.head,
            config.edl,
            epoch,
            class_weights,
        )
        _require_finite(loss.total, "validation loss")
        batch_episodes = int(batch["labels"].shape[0])
        _accumulate_loss(totals, loss, batch_episodes)
        episode_count += batch_episodes
        labels = batch["labels"].detach().cpu().numpy()
        mask = batch["chunk_mask"].detach().cpu().numpy()
        probabilities = output.probabilities[..., 1].detach().cpu().numpy()
        for row, label in enumerate(labels):
            valid = mask[row]
            chunk_labels.append(np.full(int(valid.sum()), label, dtype=np.int64))
            chunk_probabilities.append(probabilities[row, valid])
            final_labels.append(np.asarray([label], dtype=np.int64))
            final_probabilities.append(np.asarray([probabilities[row, np.flatnonzero(valid)[-1]]], dtype=np.float64))
            if output.verifier_au is not None:
                for name in edl_values:
                    value = getattr(output, name)
                    assert value is not None
                    edl_values[name][int(label)].append(value[row, valid].detach().cpu().numpy())
        if collect_predictions:
            records.extend(prediction_records_from_output(raw_batch["refs"], output, batch["chunk_mask"], batch["token_mask"]))

    summary: dict[str, Any] = _mean_loss_totals(totals, episode_count)
    summary.update(_prefixed("chunk", binary_metrics(np.concatenate(chunk_labels), np.concatenate(chunk_probabilities))))
    summary.update(_prefixed("final_chunk", binary_metrics(np.concatenate(final_labels), np.concatenate(final_probabilities))))
    if config.model.head == "edl":
        for name, values_by_label in edl_values.items():
            for label, values in values_by_label.items():
                summary[f"{name}_label_{label}"] = float(np.concatenate(values).mean()) if values else None
    return summary, records


def _preflight_run_directory(config: ExperimentConfig) -> Path:
    """Validate every destructive-path invariant without changing the filesystem."""
    output_root = config.run.output_root.resolve()
    _validate_run_name(config.run.name)
    output_dir = output_root / config.run.name
    if output_dir.parent != output_root:  # Defensive: preserve the lexical-child invariant.
        raise ValueError("run.name must be a direct child of run.output_root")
    if output_dir.is_symlink():
        raise ValueError(f"run destination must not be a symlink: {output_dir}")
    for suite, source in config.data.datasets.items():
        resolved_source = source.resolve()
        if _is_equal_or_below(resolved_source, output_dir):
            raise ValueError(
                f"data.datasets.{suite} resolves inside run destination: {resolved_source}"
            )
    return output_dir


def _prepare_run_directory(output_dir: Path, overwrite: bool) -> None:
    """Create the preflighted run directory, deleting only that lexical child when allowed."""
    if output_dir.is_symlink():
        raise ValueError(f"run destination must not be a symlink: {output_dir}")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"run destination is not a directory: {output_dir}")
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"non-empty run directory exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=True)


def _validate_run_name(name: str) -> None:
    separators = {"/", "\\"}
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "\x00" in name
        or any(separator in name for separator in separators)
        or Path(name).parts != (name,)
    ):
        raise ValueError("run.name must be exactly one normal lexical path component")


def _is_equal_or_below(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device requests CUDA but CUDA is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_loader(dataset: TrajectoryDataset, config: ExperimentConfig, *, shuffle: bool, seed: int) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        collate_fn=collate_trajectories,
        generator=generator,
        worker_init_fn=_seed_worker if config.training.num_workers else None,
    )


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _class_weights(manifest: SplitManifest, config: ExperimentConfig, device: torch.device) -> torch.Tensor | None:
    if config.loss.class_balance == "none":
        return None
    counts = np.bincount([ref.label for ref in manifest.train], minlength=2)
    if np.any(counts == 0):
        raise ValueError("inverse_frequency class balancing requires both training labels")
    inverse = 1.0 / counts.astype(np.float64)
    normalized = inverse / inverse.mean()
    weights = torch.tensor(normalized, device=device, dtype=torch.float32)
    LOGGER.warning(
        "inverse_frequency class balancing changes the effective class prior; probability calibration may be biased"
    )
    return weights


def _resolved_config(
    config: ExperimentConfig,
    manifest: SplitManifest,
    device: torch.device,
    class_weights: torch.Tensor | None,
) -> dict[str, Any]:
    payload = config_to_dict(config)
    payload["metadata"] = {
        "device": str(device),
        "max_action_tokens": manifest.max_action_tokens,
        "class_weights": None if class_weights is None else [float(value) for value in class_weights.cpu().tolist()],
        "class_balance_warning": (
            None
            if class_weights is None
            else "inverse_frequency changes the effective class prior and can bias probability calibration"
        ),
    }
    return payload


def _write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            yaml.safe_dump(dict(payload), handle, sort_keys=True)
            handle.flush()
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _to_device(raw_batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "features": torch.as_tensor(raw_batch["features"], dtype=torch.float32, device=device),
        "token_mask": torch.as_tensor(raw_batch["token_mask"], dtype=torch.bool, device=device),
        "chunk_mask": torch.as_tensor(raw_batch["chunk_mask"], dtype=torch.bool, device=device),
        "chunk_lengths": torch.as_tensor(raw_batch["chunk_lengths"], dtype=torch.long, device=device),
        "labels": torch.as_tensor(raw_batch["labels"], dtype=torch.long, device=device),
    }


def _loss_totals() -> dict[str, float]:
    return {"total_loss": 0.0, "data_loss": 0.0, "kl_loss": 0.0, "weighted_kl_loss": 0.0, "annealing_coefficient": 0.0}


def _accumulate_loss(totals: dict[str, float], loss: LossOutput, episodes: int) -> None:
    totals["total_loss"] += float(loss.total.detach().cpu()) * episodes
    totals["data_loss"] += float(loss.data.detach().cpu()) * episodes
    totals["kl_loss"] += float(loss.kl.detach().cpu()) * episodes
    totals["weighted_kl_loss"] += float(loss.weighted_kl.detach().cpu()) * episodes
    totals["annealing_coefficient"] += loss.annealing_coefficient * episodes


def _mean_loss_totals(totals: dict[str, float], episodes: int) -> dict[str, float]:
    if episodes <= 0:
        raise ValueError("loader produced no episodes")
    return {name: value / episodes for name, value in totals.items()}


def _prefixed(prefix: str, values: Mapping[str, float | None]) -> dict[str, float | None]:
    return {f"{prefix}_{name}": value for name, value in values.items()}


def _epoch_record(epoch: int, train_summary: Mapping[str, float], validation_summary: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {"epoch": epoch}
    record.update({f"train_{name}": value for name, value in train_summary.items()})
    record.update({f"validation_{name}": value for name, value in validation_summary.items()})
    return record


def _checkpoint_payload(
    model: TrajectoryVerifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    resolved_config: Mapping[str, Any],
    manifest: SplitManifest,
    validation_total_loss: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "resolved_config": dict(resolved_config),
        "max_action_tokens": manifest.max_action_tokens,
        "split_manifest_identity": _manifest_identity(manifest),
        "validation_total_loss": validation_total_loss,
    }


def _manifest_identity(manifest: SplitManifest) -> str:
    payload = {
        "max_action_tokens": manifest.max_action_tokens,
        "train": [{"suite": ref.suite, "episode_key": ref.episode_key, "label": ref.label} for ref in manifest.train],
        "validation": [{"suite": ref.suite, "episode_key": ref.episode_key, "label": ref.label} for ref in manifest.validation],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _require_finite(value: torch.Tensor, description: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"non-finite {description}")


def _require_finite_gradients(model: TrajectoryVerifier) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"non-finite gradient for {name}")


def _model_device(model: TrajectoryVerifier) -> torch.device:
    return next(model.parameters()).device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output_dir = train(load_config(args.config))
    print(f"Verifier run written to {output_dir}")


if __name__ == "__main__":
    main()
