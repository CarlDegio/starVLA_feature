"""Configuration for SAFE-style detector training."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import yaml


@dataclasses.dataclass
class TrainConfig:
    dataset_paths: list[str]
    output_dir: str = "examples/LIBERO/safe_pred/runs"
    run_name: str = "safe_lstm"
    backbone: str = "lstm"
    feature_aggregation: str = "last"
    hidden_dim: int = 256
    num_layers: int = 1
    epochs: int = 1000
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    val_fraction: float = 0.1
    seed: int = 7
    device: str = "auto"
    early_stopping_patience: int = 0

    def __post_init__(self) -> None:
        if not self.dataset_paths:
            raise ValueError("dataset_paths must not be empty")
        if self.backbone not in {"lstm", "mlp"}:
            raise ValueError("backbone must be lstm or mlp")
        if self.feature_aggregation not in {"first", "last", "mean"}:
            raise ValueError("feature_aggregation must be first, last, or mean")
        if min(self.hidden_dim, self.num_layers, self.epochs, self.batch_size) <= 0:
            raise ValueError("model dimensions, epochs, and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_config(path: str | pathlib.Path) -> TrainConfig:
    payload = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must be a YAML mapping")
    return TrainConfig(**payload)
