"""Causal recurrent trajectory verifier with softmax and EDL heads."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import EDLConfig, ModelConfig
from .encoders import build_chunk_encoder


@dataclass
class VerifierOutput:
    """Per-chunk classifier outputs and optional evidential diagnostics."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    evidence: torch.Tensor | None
    alpha: torch.Tensor | None
    verifier_au: torch.Tensor | None
    verifier_eu: torch.Tensor | None
    verifier_total_evidence: torch.Tensor | None
    pooling_weights: torch.Tensor | None


class TrajectoryVerifier(nn.Module):
    """Encode AU/EU chunks causally and classify every valid chunk."""

    def __init__(self, model_config: ModelConfig, edl_config: EDLConfig, max_action_tokens: int) -> None:
        super().__init__()
        self.model_config = model_config
        self.edl_config = edl_config
        self.chunk_encoder = build_chunk_encoder(model_config, max_action_tokens)
        self.lstm = nn.LSTM(
            input_size=model_config.chunk_embed_dim,
            hidden_size=model_config.lstm_hidden_dim,
            num_layers=model_config.lstm_layers,
            batch_first=True,
            dropout=model_config.lstm_dropout if model_config.lstm_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Linear(model_config.lstm_hidden_dim, 2)

    def forward(
        self,
        features: torch.Tensor,
        token_mask: torch.Tensor,
        chunk_lengths: torch.Tensor,
    ) -> VerifierOutput:
        """Return zero-padded per-chunk predictions for a batch of trajectories."""
        chunk_mask, packed_lengths = self._validate_chunk_lengths(features, token_mask, chunk_lengths)
        encoder_output = self.chunk_encoder(features, token_mask)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            encoder_output.embedding,
            packed_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_hidden, _ = self.lstm(packed_embeddings)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden,
            batch_first=True,
            total_length=features.size(1),
        )
        logits = self.classifier(hidden)

        if self.model_config.head == "softmax":
            probabilities = torch.softmax(logits, dim=-1)
            evidence = None
            alpha = None
            verifier_au = None
            verifier_eu = None
            verifier_total_evidence = None
        elif self.model_config.head == "edl":
            evidence = self._evidence(logits)
            alpha = evidence + 1.0
            strength = alpha.sum(dim=-1, keepdim=True)
            probabilities = alpha / strength
            verifier_au = (
                probabilities
                * (torch.digamma(strength + 1.0) - torch.digamma(alpha + 1.0))
            ).sum(dim=-1) / math.log(2.0)
            verifier_eu = 2.0 / strength.squeeze(-1)
            verifier_total_evidence = evidence.sum(dim=-1)
        else:
            raise ValueError(f"unknown verifier head: {self.model_config.head!r}")

        logits = self._zero_padded_chunks(logits, chunk_mask)
        probabilities = self._zero_padded_chunks(probabilities, chunk_mask)
        evidence = self._zero_padded_chunks(evidence, chunk_mask)
        alpha = self._zero_padded_chunks(alpha, chunk_mask)
        verifier_au = self._zero_padded_chunks(verifier_au, chunk_mask)
        verifier_eu = self._zero_padded_chunks(verifier_eu, chunk_mask)
        verifier_total_evidence = self._zero_padded_chunks(verifier_total_evidence, chunk_mask)
        pooling_weights = encoder_output.pooling_weights
        if pooling_weights is not None:
            pooling_weights = pooling_weights.masked_fill(~token_mask, 0.0)

        return VerifierOutput(
            logits=logits,
            probabilities=probabilities,
            evidence=evidence,
            alpha=alpha,
            verifier_au=verifier_au,
            verifier_eu=verifier_eu,
            verifier_total_evidence=verifier_total_evidence,
            pooling_weights=pooling_weights,
        )

    def _evidence(self, logits: torch.Tensor) -> torch.Tensor:
        if self.edl_config.evidence_activation == "softplus":
            return F.softplus(logits)
        if self.edl_config.evidence_activation == "relu":
            return F.relu(logits)
        if self.edl_config.evidence_activation == "exp":
            return torch.exp(torch.clamp(logits, min=-10.0, max=10.0))
        raise ValueError(f"unknown EDL evidence activation: {self.edl_config.evidence_activation!r}")

    @staticmethod
    def _zero_padded_chunks(value: torch.Tensor | None, chunk_mask: torch.Tensor) -> torch.Tensor | None:
        if value is None:
            return None
        padding_mask = ~chunk_mask
        while padding_mask.ndim < value.ndim:
            padding_mask = padding_mask.unsqueeze(-1)
        return value.masked_fill(padding_mask, 0.0)

    @staticmethod
    def _validate_chunk_lengths(
        features: torch.Tensor,
        token_mask: torch.Tensor,
        chunk_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 4 or features.shape[-1] != 2:
            raise ValueError("features must have shape [batch, chunks, max_tokens, 2]")
        if token_mask.shape != features.shape[:-1] or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be boolean with shape [batch, chunks, max_tokens]")
        if not isinstance(chunk_lengths, torch.Tensor) or chunk_lengths.ndim != 1:
            raise ValueError("chunk_lengths must be a one-dimensional integer tensor")
        if chunk_lengths.shape[0] != features.shape[0]:
            raise ValueError("chunk_lengths must have one value per batch item")
        if chunk_lengths.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("chunk_lengths must be an integer tensor")

        batch_size, chunk_count = features.shape[:2]
        packed_lengths = chunk_lengths.to(dtype=torch.long)
        if torch.any(packed_lengths < 1) or torch.any(packed_lengths > chunk_count):
            raise ValueError(f"chunk_lengths must be in [1, {chunk_count}]")

        chunk_mask = token_mask.any(dim=-1)
        expected_mask = torch.arange(chunk_count, device=token_mask.device).unsqueeze(0) < packed_lengths.to(
            token_mask.device
        ).unsqueeze(1)
        if expected_mask.shape != (batch_size, chunk_count) or not torch.equal(chunk_mask, expected_mask):
            raise ValueError("chunk_lengths must exactly match the prefix-valid chunks in token_mask")
        return chunk_mask, packed_lengths


def build_verifier(
    model_config: ModelConfig,
    edl_config: EDLConfig,
    max_action_tokens: int,
) -> TrajectoryVerifier:
    """Build the configured causal trajectory verifier."""
    return TrajectoryVerifier(model_config, edl_config, max_action_tokens)
