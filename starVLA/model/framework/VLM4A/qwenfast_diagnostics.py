"""Inference-only diagnostics for autoregressive QwenFast generation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def compute_generation_diagnostics(
    generated: Any,
    *,
    action_token_min: int,
    action_token_max: int,
    include_token_uncertainty: bool,
    include_latent_features: bool,
) -> dict[str, np.ndarray]:
    """Extract action-token softmax statistics and final-layer features.

    Token arrays are padded with finite zeros and accompanied by an explicit
    boolean mask and per-row count. Token IDs use ``-1`` for padding.
    """
    if not include_token_uncertainty and not include_latent_features:
        raise ValueError("at least one diagnostic must be requested")
    if action_token_min > action_token_max:
        raise ValueError("action_token_min must not exceed action_token_max")
    if not hasattr(generated, "sequences") or generated.sequences is None:
        raise ValueError("generation output is missing sequences")

    sequences = generated.sequences
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
        raise ValueError("generation sequences must be a rank-2 tensor")

    scores = getattr(generated, "scores", None)
    hidden_states = getattr(generated, "hidden_states", None)
    if include_token_uncertainty and (scores is None or len(scores) == 0):
        raise ValueError("generation output is missing scores")
    if include_latent_features and (hidden_states is None or len(hidden_states) == 0):
        raise ValueError("generation output is missing hidden_states")

    step_counts = []
    if include_token_uncertainty:
        step_counts.append(len(scores))
    if include_latent_features:
        step_counts.append(len(hidden_states))
    if len(set(step_counts)) != 1:
        raise ValueError("generation scores and hidden_states must have the same number of steps")
    num_steps = step_counts[0]
    if sequences.shape[1] < num_steps:
        raise ValueError("generation sequences are shorter than the diagnostic step count")

    generated_ids = sequences[:, -num_steps:].long()
    batch_size = generated_ids.shape[0]
    action_mask = (generated_ids >= int(action_token_min)) & (generated_ids <= int(action_token_max))
    action_counts = action_mask.sum(dim=1).to(dtype=torch.int64)
    max_action_tokens = int(action_counts.max().item()) if batch_size else 0

    token_nll = None
    token_entropy = None
    if include_token_uncertainty:
        step_nll = []
        step_entropy = []
        for step_idx, step_scores in enumerate(scores):
            if not isinstance(step_scores, torch.Tensor) or step_scores.ndim != 2:
                raise ValueError("each generation score must be a rank-2 tensor")
            if step_scores.shape[0] != batch_size:
                raise ValueError("generation score batch size does not match sequences")
            log_probs = torch.log_softmax(step_scores.float(), dim=-1)
            selected_ids = generated_ids[:, step_idx].to(device=log_probs.device)
            if torch.any(selected_ids < 0) or torch.any(selected_ids >= log_probs.shape[-1]):
                raise ValueError("generated token ID falls outside the score vocabulary")
            step_nll.append(-log_probs.gather(dim=-1, index=selected_ids[:, None]).squeeze(-1))
            step_entropy.append(-(log_probs.exp() * log_probs).sum(dim=-1))
        token_nll = torch.stack(step_nll, dim=1)
        token_entropy = torch.stack(step_entropy, dim=1)

    step_features = None
    hidden_dim = 0
    if include_latent_features:
        final_features = []
        for step_hidden_states in hidden_states:
            if not isinstance(step_hidden_states, (tuple, list)) or len(step_hidden_states) == 0:
                raise ValueError("each generation hidden_states entry must contain transformer layers")
            final_hidden = step_hidden_states[-1]
            if not isinstance(final_hidden, torch.Tensor) or final_hidden.ndim != 3:
                raise ValueError("final generation hidden state must be a rank-3 tensor")
            if final_hidden.shape[0] != batch_size:
                raise ValueError("generation hidden-state batch size does not match sequences")
            final_features.append(final_hidden[:, -1, :].float())
        step_features = torch.stack(final_features, dim=1)
        hidden_dim = int(step_features.shape[-1])

    result: dict[str, np.ndarray] = {
        "action_token_ids": np.full((batch_size, max_action_tokens), -1, dtype=np.int64),
        "action_token_mask": np.zeros((batch_size, max_action_tokens), dtype=np.bool_),
        "num_action_tokens": action_counts.cpu().numpy(),
    }
    if include_token_uncertainty:
        result["action_token_nll"] = np.zeros((batch_size, max_action_tokens), dtype=np.float32)
        result["action_token_entropy"] = np.zeros((batch_size, max_action_tokens), dtype=np.float32)
    if include_latent_features:
        for aggregation in ("first", "last", "mean"):
            result[f"action_token_embedding_{aggregation}"] = np.zeros(
                (batch_size, hidden_dim), dtype=np.float32
            )

    for batch_idx in range(batch_size):
        row_mask = action_mask[batch_idx]
        count = int(action_counts[batch_idx].item())
        if count == 0:
            continue
        result["action_token_mask"][batch_idx, :count] = True
        result["action_token_ids"][batch_idx, :count] = generated_ids[batch_idx, row_mask].cpu().numpy()
        if include_token_uncertainty:
            result["action_token_nll"][batch_idx, :count] = (
                token_nll[batch_idx, row_mask].detach().cpu().numpy().astype(np.float32, copy=False)
            )
            result["action_token_entropy"][batch_idx, :count] = (
                token_entropy[batch_idx, row_mask].detach().cpu().numpy().astype(np.float32, copy=False)
            )
        if include_latent_features:
            row_features = step_features[batch_idx, row_mask]
            result["action_token_embedding_first"][batch_idx] = row_features[0].detach().cpu().numpy()
            result["action_token_embedding_last"][batch_idx] = row_features[-1].detach().cpu().numpy()
            result["action_token_embedding_mean"][batch_idx] = row_features.mean(dim=0).detach().cpu().numpy()

    return result
