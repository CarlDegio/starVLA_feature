"""Masked encoders for fixed-width LIBERO action-token chunks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .config import ModelConfig


_FEATURE_DIM = 2


@dataclass
class ChunkEncoderOutput:
    embedding: torch.Tensor
    pooling_weights: torch.Tensor | None


class _IdentityTokenPosition(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens


class SinusoidalTokenPosition(nn.Module):
    def __init__(self, max_tokens: int, dim: int) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        position = torch.arange(max_tokens, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        values = torch.zeros(max_tokens, dim, dtype=torch.float32)
        values[:, 0::2] = torch.sin(position * frequencies)
        values[:, 1::2] = torch.cos(position * frequencies[: values[:, 1::2].shape[1]])
        self.register_buffer("values", values, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        token_count = tokens.shape[-2]
        if token_count > self.max_tokens:
            raise ValueError(
                f"token sequence length {token_count} exceeds max_tokens={self.max_tokens}"
            )
        return tokens + self.values[:token_count].to(dtype=tokens.dtype)


class LearnedTokenPosition(nn.Module):
    def __init__(self, max_tokens: int, dim: int) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(max_tokens, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        token_count = tokens.shape[-2]
        if token_count > self.max_tokens:
            raise ValueError(
                f"token sequence length {token_count} exceeds max_tokens={self.max_tokens}"
            )
        positions = torch.arange(token_count, device=tokens.device)
        return tokens + self.embedding(positions).to(dtype=tokens.dtype)


def build_token_position(kind: str, max_tokens: int, dim: int) -> nn.Module:
    """Build a token position module for a fixed action-token axis."""
    if kind == "none":
        return _IdentityTokenPosition()
    if kind == "sinusoidal":
        return SinusoidalTokenPosition(max_tokens, dim)
    if kind == "learned":
        return LearnedTokenPosition(max_tokens, dim)
    raise ValueError(f"unknown token position encoding: {kind!r}")


class _AttentionPool(nn.Module):
    """Pool valid token rows with a learned query and masked attention."""

    def __init__(self, token_embed_dim: int, attention_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, token_embed_dim))
        nn.init.normal_(self.query, std=token_embed_dim**-0.5)
        self.attention = nn.MultiheadAttention(
            token_embed_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, weights = self.attention(
            query,
            tokens,
            tokens,
            key_padding_mask=~token_mask,
            need_weights=True,
        )
        weights = weights.squeeze(1).masked_fill(~token_mask, 0.0)
        weight_sum = weights.sum(dim=-1, keepdim=True)
        uniform_weights = token_mask.to(dtype=weights.dtype)
        uniform_weights = uniform_weights / uniform_weights.sum(dim=-1, keepdim=True)
        weights = torch.where(
            weight_sum > 0,
            weights / weight_sum.clamp_min(torch.finfo(weights.dtype).tiny),
            uniform_weights,
        )
        return pooled.squeeze(1), weights


class _MaskedChunkEncoder(nn.Module):
    def __init__(self, config: ModelConfig, max_action_tokens: int) -> None:
        super().__init__()
        self.chunk_embed_dim = config.chunk_embed_dim
        self.max_action_tokens = max_action_tokens

    def _prepare(
        self,
        features: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        if features.ndim != 4 or features.shape[-1] != _FEATURE_DIM:
            raise ValueError("features must have shape [batch, chunks, max_tokens, 2]")
        if token_mask.shape != features.shape[:-1] or token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be boolean with shape [batch, chunks, max_tokens]")
        if features.shape[-2] != self.max_action_tokens:
            raise ValueError(f"expected max_action_tokens={self.max_action_tokens}")

        batch_size, chunk_count, token_count, _ = features.shape
        masked_features = features.masked_fill(~token_mask[..., None], 0.0)
        flat_features = masked_features.reshape(-1, token_count, _FEATURE_DIM)
        flat_mask = token_mask.reshape(-1, token_count)
        valid_rows = flat_mask.any(dim=-1)
        return flat_features, flat_mask, valid_rows, batch_size, chunk_count, token_count

    def _scatter_embeddings(
        self,
        encoded: torch.Tensor,
        valid_rows: torch.Tensor,
        batch_size: int,
        chunk_count: int,
    ) -> torch.Tensor:
        flattened = encoded.new_zeros((valid_rows.numel(), self.chunk_embed_dim))
        flattened[valid_rows] = encoded
        return flattened.reshape(batch_size, chunk_count, self.chunk_embed_dim)

    def _zero_output(
        self,
        features: torch.Tensor,
        batch_size: int,
        chunk_count: int,
        token_count: int,
        *,
        has_pooling_weights: bool,
    ) -> ChunkEncoderOutput:
        return ChunkEncoderOutput(
            embedding=features.new_zeros((batch_size, chunk_count, self.chunk_embed_dim)),
            pooling_weights=(
                features.new_zeros((batch_size, chunk_count, token_count))
                if has_pooling_weights
                else None
            ),
        )

    @staticmethod
    def _scatter_weights(
        weights: torch.Tensor,
        valid_rows: torch.Tensor,
        batch_size: int,
        chunk_count: int,
        token_count: int,
    ) -> torch.Tensor:
        flattened = weights.new_zeros((valid_rows.numel(), token_count))
        flattened[valid_rows] = weights
        return flattened.reshape(batch_size, chunk_count, token_count)


class MLPFlatEncoder(_MaskedChunkEncoder):
    """Encode each masked token chunk by flattening its fixed token axis."""

    def __init__(self, config: ModelConfig, max_action_tokens: int) -> None:
        super().__init__(config, max_action_tokens)
        self.network = nn.Sequential(
            nn.Linear(max_action_tokens * _FEATURE_DIM, config.token_embed_dim),
            nn.GELU(),
            nn.Dropout(config.encoder_dropout),
            nn.Linear(config.token_embed_dim, config.chunk_embed_dim),
        )

    def forward(self, features: torch.Tensor, token_mask: torch.Tensor) -> ChunkEncoderOutput:
        flat_features, _, valid_rows, batch_size, chunk_count, token_count = self._prepare(features, token_mask)
        if not valid_rows.any():
            return self._zero_output(
                flat_features,
                batch_size,
                chunk_count,
                token_count,
                has_pooling_weights=False,
            )
        embedding = self._scatter_embeddings(
            self.network(flat_features[valid_rows].flatten(start_dim=1)),
            valid_rows,
            batch_size,
            chunk_count,
        )
        return ChunkEncoderOutput(embedding=embedding, pooling_weights=None)


class TokenAttentionPoolEncoder(_MaskedChunkEncoder):
    """Project token features, add positions, and pool each valid chunk."""

    def __init__(self, config: ModelConfig, max_action_tokens: int) -> None:
        super().__init__(config, max_action_tokens)
        self.token_projection = nn.Linear(_FEATURE_DIM, config.token_embed_dim)
        self.token_position = build_token_position(
            config.token_position_encoding,
            max_action_tokens,
            config.token_embed_dim,
        )
        self.pool = _AttentionPool(config.token_embed_dim, config.attention_heads, config.encoder_dropout)
        self.output_projection = nn.Linear(config.token_embed_dim, config.chunk_embed_dim)

    def _encode_valid_rows(
        self,
        features: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.token_position(self.token_projection(features))
        return self.pool(tokens, token_mask)

    def forward(self, features: torch.Tensor, token_mask: torch.Tensor) -> ChunkEncoderOutput:
        flat_features, flat_mask, valid_rows, batch_size, chunk_count, token_count = self._prepare(features, token_mask)
        if not valid_rows.any():
            return self._zero_output(
                flat_features,
                batch_size,
                chunk_count,
                token_count,
                has_pooling_weights=True,
            )
        pooled, weights = self._encode_valid_rows(flat_features[valid_rows], flat_mask[valid_rows])
        embedding = self._scatter_embeddings(
            self.output_projection(pooled),
            valid_rows,
            batch_size,
            chunk_count,
        )
        return ChunkEncoderOutput(
            embedding=embedding,
            pooling_weights=self._scatter_weights(weights, valid_rows, batch_size, chunk_count, token_count),
        )


class TokenSelfAttentionEncoder(TokenAttentionPoolEncoder):
    """Contextualize valid token rows before learned-query attention pooling."""

    def __init__(self, config: ModelConfig, max_action_tokens: int) -> None:
        super().__init__(config, max_action_tokens)
        layer = nn.TransformerEncoderLayer(
            d_model=config.token_embed_dim,
            nhead=config.attention_heads,
            dim_feedforward=4 * config.token_embed_dim,
            dropout=config.encoder_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.self_attention = nn.TransformerEncoder(layer, num_layers=config.self_attention_layers)

    def _encode_valid_rows(
        self,
        features: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.token_position(self.token_projection(features))
        tokens = self.self_attention(tokens, src_key_padding_mask=~token_mask)
        return self.pool(tokens, token_mask)


def build_chunk_encoder(config: ModelConfig, max_action_tokens: int) -> nn.Module:
    """Build the configured action-chunk encoder."""
    if max_action_tokens <= 0:
        raise ValueError("max_action_tokens must be positive")
    if config.chunk_encoder == "mlp_flat":
        return MLPFlatEncoder(config, max_action_tokens)
    if config.chunk_encoder == "token_attention_pool":
        return TokenAttentionPoolEncoder(config, max_action_tokens)
    if config.chunk_encoder == "token_self_attention":
        return TokenSelfAttentionEncoder(config, max_action_tokens)
    raise ValueError(f"unknown chunk encoder: {config.chunk_encoder!r}")
