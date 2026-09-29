"""SAFE-style MLP and LSTM failure detectors."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn


@dataclasses.dataclass(frozen=True)
class DetectorOutput:
    logits: torch.Tensor
    failure_score: torch.Tensor
    failure_probability: torch.Tensor | None
    chunk_mask: torch.Tensor


class SafeFailureDetector(nn.Module):
    def __init__(
        self,
        backbone: str,
        *,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if backbone not in {"lstm", "mlp"}:
            raise ValueError("backbone must be lstm or mlp")
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("model dimensions and num_layers must be positive")
        self.backbone = backbone
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        if backbone == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_dim, 1)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor, chunk_mask: torch.Tensor) -> DetectorOutput:
        self._validate_inputs(features, chunk_mask)
        encoded = self.encoder(features)[0] if self.backbone == "lstm" else self.encoder(features)
        logits = self.head(encoded).squeeze(-1)
        if self.backbone == "lstm":
            probability = torch.sigmoid(logits) * chunk_mask.to(logits.dtype)
            score = probability
        else:
            local_score = torch.sigmoid(logits) * chunk_mask.to(logits.dtype)
            score = torch.cumsum(local_score, dim=1) * chunk_mask.to(logits.dtype)
            probability = None
        return DetectorOutput(logits=logits, failure_score=score, failure_probability=probability, chunk_mask=chunk_mask)

    def _validate_inputs(self, features: torch.Tensor, chunk_mask: torch.Tensor) -> None:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [batch, chunks, {self.input_dim}]")
        if chunk_mask.dtype != torch.bool or chunk_mask.shape != features.shape[:2]:
            raise ValueError("chunk_mask must be boolean with shape [batch, chunks]")
        counts = chunk_mask.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError("every row must contain at least one valid chunk")
        expected = torch.arange(chunk_mask.shape[1], device=chunk_mask.device)[None, :] < counts[:, None]
        if not torch.equal(expected, chunk_mask):
            raise ValueError("chunk_mask must mark a contiguous valid prefix")
