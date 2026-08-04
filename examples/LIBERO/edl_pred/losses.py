"""Masked, episode-mean objectives for the LIBERO trajectory verifier."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .config import EDLConfig
from .model import VerifierOutput


@dataclass
class LossOutput:
    """Scalar verifier loss components after episode-level aggregation."""

    total: torch.Tensor
    data: torch.Tensor
    kl: torch.Tensor
    weighted_kl: torch.Tensor
    annealing_coefficient: float


def compute_verifier_loss(
    output: VerifierOutput,
    labels: torch.Tensor,
    chunk_mask: torch.Tensor,
    head: str,
    edl_config: EDLConfig,
    epoch: int,
    class_weights: torch.Tensor | None = None,
) -> LossOutput:
    """Return masked episode-mean data, KL, weighted KL, and total losses.

    Each component is first averaged across the valid chunks of an episode,
    then optionally reweighted by that episode's target class, and finally
    averaged across the batch of episodes.
    """
    _validate_inputs(output, labels, chunk_mask, head, edl_config, epoch, class_weights)

    logits = output.logits
    labels = labels.to(dtype=torch.long)
    batch_size = logits.size(0)
    episode_indices = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand_as(chunk_mask)[chunk_mask]
    valid_labels = labels[episode_indices]

    if head == "softmax":
        data_chunks = F.cross_entropy(logits[chunk_mask], valid_labels, reduction="none")
        data = _mean_episodes(data_chunks, episode_indices, labels, chunk_mask, class_weights)
        zero = data.new_zeros(())
        return LossOutput(total=data, data=data, kl=zero, weighted_kl=zero, annealing_coefficient=0.0)

    alpha = output.alpha
    assert alpha is not None  # Guaranteed by _validate_inputs.
    alpha = alpha[chunk_mask]
    one_hot = F.one_hot(valid_labels, num_classes=2).to(dtype=alpha.dtype)
    strength = alpha.sum(dim=-1, keepdim=True)
    data_chunks = (one_hot * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=-1)

    adjusted_alpha = (alpha - 1.0) * (1.0 - one_hot) + 1.0
    kl_chunks = _dirichlet_kl_to_uniform(adjusted_alpha)
    data = _mean_episodes(data_chunks, episode_indices, labels, chunk_mask, class_weights)
    kl = _mean_episodes(kl_chunks, episode_indices, labels, chunk_mask, class_weights)
    annealing_coefficient = _annealing_coefficient(epoch, edl_config.kl_anneal_epochs)
    weighted_kl = kl * (annealing_coefficient * edl_config.kl_weight)
    return LossOutput(
        total=data + weighted_kl,
        data=data,
        kl=kl,
        weighted_kl=weighted_kl,
        annealing_coefficient=annealing_coefficient,
    )


def _mean_episodes(
    chunk_values: torch.Tensor,
    episode_indices: torch.Tensor,
    labels: torch.Tensor,
    chunk_mask: torch.Tensor,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Reduce valid chunk values according to the public loss contract."""
    episode_sums = chunk_values.new_zeros(labels.size(0)).scatter_add(0, episode_indices, chunk_values)
    valid_chunk_counts = chunk_mask.sum(dim=1).to(dtype=chunk_values.dtype)
    episode_values = episode_sums / valid_chunk_counts
    if class_weights is not None:
        episode_values = episode_values * class_weights[labels].to(dtype=chunk_values.dtype)
    return episode_values.mean()


def _dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """Compute KL(Dirichlet(alpha) || Dirichlet(ones)) for binary alpha."""
    strength = alpha.sum(dim=-1, keepdim=True)
    uniform_log_beta = torch.lgamma(torch.tensor(2.0, dtype=alpha.dtype, device=alpha.device))
    return (
        torch.lgamma(strength).squeeze(-1)
        - torch.lgamma(alpha).sum(dim=-1)
        - uniform_log_beta
        + ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(strength))).sum(dim=-1)
    )


def _annealing_coefficient(epoch: int, anneal_epochs: int) -> float:
    if anneal_epochs <= 0:
        return 1.0
    return min(float(epoch) / float(anneal_epochs), 1.0)


def _validate_inputs(
    output: VerifierOutput,
    labels: torch.Tensor,
    chunk_mask: torch.Tensor,
    head: str,
    edl_config: EDLConfig,
    epoch: int,
    class_weights: torch.Tensor | None,
) -> None:
    if not isinstance(output, VerifierOutput):
        raise ValueError("output must be a VerifierOutput")
    if head not in {"softmax", "edl"}:
        raise ValueError("head must be either 'softmax' or 'edl'")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if not isinstance(edl_config, EDLConfig):
        raise ValueError("edl_config must be an EDLConfig")
    if not math.isfinite(edl_config.kl_weight) or edl_config.kl_weight < 0.0:
        raise ValueError("edl_config.kl_weight must be finite and non-negative")
    if not isinstance(edl_config.kl_anneal_epochs, int) or isinstance(edl_config.kl_anneal_epochs, bool):
        raise ValueError("edl_config.kl_anneal_epochs must be an integer")

    _validate_output_tensor(output.logits, "logits")
    logits = output.logits
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("logits must have shape [batch, chunks, 2]")
    if logits.shape[0] == 0:
        raise ValueError("batch must contain at least one episode")
    _validate_output_tensor(output.probabilities, "probabilities")
    if output.probabilities.shape != logits.shape:
        raise ValueError("probabilities must have the same shape as logits")
    if output.probabilities.device != logits.device:
        raise ValueError("probabilities must have the same device as logits")
    if output.probabilities.dtype != logits.dtype:
        raise ValueError("probabilities must have the same dtype as logits")

    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional integer tensor")
    if labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have one value per batch item")
    if labels.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
        raise ValueError("labels must be an integer tensor")
    if labels.device != logits.device:
        raise ValueError("labels and logits must be on the same device")
    if not bool(torch.all((labels == 0) | (labels == 1))):
        raise ValueError("labels must contain only binary class values 0 and 1")

    if not isinstance(chunk_mask, torch.Tensor) or chunk_mask.dtype != torch.bool:
        raise ValueError("chunk_mask must be a boolean tensor")
    if chunk_mask.shape != logits.shape[:2]:
        raise ValueError("chunk_mask must have shape [batch, chunks]")
    if chunk_mask.device != logits.device:
        raise ValueError("chunk_mask and logits must be on the same device")
    if not bool(chunk_mask.any(dim=1).all()):
        raise ValueError("each episode must contain at least one valid chunk")

    if head == "edl":
        _validate_output_tensor(output.alpha, "alpha")
        if output.alpha.shape != logits.shape:
            raise ValueError("alpha must have the same shape as logits")
        if output.alpha.device != logits.device:
            raise ValueError("alpha must have the same device as logits")
        if output.alpha.dtype != logits.dtype:
            raise ValueError("alpha must have the same dtype as logits")
        if not bool(torch.all(output.alpha[chunk_mask] > 0.0)):
            raise ValueError("alpha must be positive at valid chunks")

    if class_weights is not None:
        if not isinstance(class_weights, torch.Tensor) or class_weights.ndim != 1 or class_weights.shape[0] != 2:
            raise ValueError("class_weights must be a one-dimensional tensor with two values")
        if class_weights.device != logits.device:
            raise ValueError("class_weights and logits must be on the same device")
        if not torch.is_floating_point(class_weights):
            raise ValueError("class_weights must be floating point")
        if not bool(torch.isfinite(class_weights).all()) or bool((class_weights < 0.0).any()):
            raise ValueError("class_weights must be finite and non-negative")


def _validate_output_tensor(value: torch.Tensor | None, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} is required")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
