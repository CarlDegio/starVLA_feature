"""Frozen trajectory-verifier reconstruction and independent dataset inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .artifacts import PredictionRecord, prediction_records_from_output
from .config import EDLConfig, ModelConfig
from .dataset import TrajectoryDataset, collate_trajectories, index_suite_episodes
from .model import TrajectoryVerifier, build_verifier
from .rejection_data import sha256_file


@dataclass(frozen=True)
class FrozenVerifier:
    checkpoint_path: Path
    checkpoint_sha256: str
    model: TrajectoryVerifier
    model_config: ModelConfig
    edl_config: EDLConfig
    max_action_tokens: int
    split_manifest_identity: str
    device: torch.device


def load_frozen_verifier(checkpoint_path: str | Path, *, device: str = "auto") -> FrozenVerifier:
    """Restore a strict best-checkpoint payload for inference only."""
    path = Path(checkpoint_path).expanduser().resolve()
    resolved_device = _resolve_device(device)
    payload = torch.load(path, map_location=resolved_device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("verifier checkpoint must contain a mapping")
    required = {"model_state_dict", "resolved_config", "max_action_tokens", "split_manifest_identity"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"verifier checkpoint missing fields: {sorted(missing)}")
    resolved = payload["resolved_config"]
    if not isinstance(resolved, Mapping) or not isinstance(resolved.get("model"), Mapping) or not isinstance(resolved.get("edl"), Mapping):
        raise ValueError("verifier checkpoint resolved_config must contain model and edl mappings")
    try:
        model_config = ModelConfig(**dict(resolved["model"]))
        edl_config = EDLConfig(**dict(resolved["edl"]))
    except TypeError as error:
        raise ValueError("verifier checkpoint contains an invalid model or EDL config") from error
    max_action_tokens = payload["max_action_tokens"]
    if isinstance(max_action_tokens, bool) or not isinstance(max_action_tokens, int) or max_action_tokens <= 0:
        raise ValueError("verifier checkpoint max_action_tokens must be positive")
    identity = payload["split_manifest_identity"]
    if not isinstance(identity, str) or not identity:
        raise ValueError("verifier checkpoint split_manifest_identity must be non-empty")
    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise ValueError("verifier checkpoint model_state_dict must be a mapping")
    model = build_verifier(model_config, edl_config, max_action_tokens).to(resolved_device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("verifier checkpoint state dict does not match resolved config") from error
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return FrozenVerifier(
        checkpoint_path=path,
        checkpoint_sha256=sha256_file(path),
        model=model,
        model_config=model_config,
        edl_config=edl_config,
        max_action_tokens=max_action_tokens,
        split_manifest_identity=identity,
        device=resolved_device,
    )


@torch.inference_mode()
def predict_frozen_datasets(
    frozen: FrozenVerifier,
    datasets: Mapping[str, str | Path],
    *,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[PredictionRecord, ...]:
    """Run one frozen verifier over every episode in independent suite files."""
    if not isinstance(frozen, FrozenVerifier):
        raise TypeError("frozen must be a FrozenVerifier")
    if not datasets:
        raise ValueError("datasets must not be empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    records: list[PredictionRecord] = []
    for suite in sorted(datasets):
        refs, observed_max = index_suite_episodes(suite, datasets[suite])
        if observed_max > frozen.max_action_tokens:
            raise ValueError(
                f"test suite {suite!r} exceeds checkpoint max_action_tokens: "
                f"{observed_max} > {frozen.max_action_tokens}"
            )
        dataset = TrajectoryDataset(refs, frozen.max_action_tokens)
        try:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_trajectories,
            )
            for raw_batch in loader:
                features = torch.as_tensor(raw_batch["features"], dtype=torch.float32, device=frozen.device)
                token_mask = torch.as_tensor(raw_batch["token_mask"], dtype=torch.bool, device=frozen.device)
                chunk_lengths = torch.as_tensor(raw_batch["chunk_lengths"], dtype=torch.long, device=frozen.device)
                output = frozen.model(features, token_mask, chunk_lengths)
                records.extend(
                    prediction_records_from_output(
                        raw_batch["refs"],
                        output,
                        raw_batch["chunk_mask"],
                        raw_batch["token_mask"],
                    )
                )
        finally:
            dataset.close()
    return tuple(records)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid inference device: {value!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("inference device requests CUDA but CUDA is unavailable")
    return device
