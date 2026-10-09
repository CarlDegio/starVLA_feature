# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-EDL Framework

A lightweight implementation for autoregressive discrete action prediction conditioned on multi-view images + instruction.
fast tokenizer is copyright from physical-intelligence/fast

Key Points:
  - Qwen2.5 vision-language backbone
  - Evidential top-k full-vocabulary next-token learning
  - Autoregressive action tokens derived from discretized / symbolized continuous actions
  - Dirichlet uncertainty estimates for generated next-token decisions

Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model
from starVLA.model.modules.vlm import get_vlm_model


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenEDL
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenEDLDefaultConfig:
    """QwenEDL framework default parameters.

    Autoregressive evidential discrete action prediction via FAST tokenizer.
    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.
    """

    # --- Registry identifier ---
    name: str = "QwenEDL"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL with action special tokens) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (must include FAST action tokens)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
        }
    )

    # === Action head (FAST tokenizer - evidential discrete next-token prediction) ===
    action_model: dict = field(
        default_factory=lambda: {
            # Action head architecture type
            "action_model_type": "FAST",
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # How many future steps to predict
            "future_action_window_size": 15,
            # How many past steps included in action chunk (usually 0)
            "past_action_window_size": 0,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenEDL")
class Qwenvl_EDL(baseframework):
    """
    Multimodal vision-language-action model (FAST + EDL variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - FAST tokenizer for discretized / symbolized continuous action encoding
      - Evidential next-token prediction over top-k full-vocabulary candidates

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenEDLDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.action_model = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        # self.hidden_dim = config.framework.action_model.action_hidden_dim

        self.action_model.fast_tokenizer.time_horizon = self.action_horizon
        self.action_model.fast_tokenizer.action_dim = self.config.framework.action_model.action_dim

        # Internal EDL controls. Keep local for now so existing configs do not need
        # to plumb new fields through the training scripts.
        self.edl_loss_type = "digamma"  # "mse" | "log" | "digamma"
        self.edl_topk = 25
        self.edl_kl_weight = 0.0000
        self.edl_annealing_steps = 15000
        self.edl_evidence_fn = "softplus"  # "softplus" | "relu" | "exp"
        self.register_buffer("_edl_step", torch.zeros((), dtype=torch.long), persistent=False)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: predict response tokens with top-k evidential classification.

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Build FAST action-token labels from continuous actions
          3. Run QwenVL and collect vocabulary logits
          4. Build GT+top-k full-vocabulary candidates and compute EDL loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar evidential action-token loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B, [PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, 7]

        # step 0: map_raw_action_to_vlm_action
        batch_fast_tokens = self.action_model.encoder_action2fastoken(actions)  # List[str]

        # batch_fast_tokens = [self.fast_tokenizer(raw_action)[0] for raw_action in raw_actions]
        vlm_action_tokens = [self.map_fast_token_to_vlm_action(fast_tokens) for fast_tokens in batch_fast_tokens]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions, solutions=vlm_action_tokens
        )
        labels = qwen_inputs.pop("labels")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

        action_loss, loss_stats = self._compute_edl_action_loss(qwenvl_outputs.logits, labels)
        if action_loss is None or torch.isnan(action_loss):
            action_loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)

        annealed_kl_loss = loss_stats["annealing_coef"] * self.edl_kl_weight * loss_stats["kl_loss"]
        return {
            "action_loss": action_loss,
            "edl/data_loss": self._metric_item(loss_stats["data_loss"]),
            "edl/kl_loss": self._metric_item(loss_stats["kl_loss"]),
            "edl/annealed_kl_loss": self._metric_item(annealed_kl_loss),
            "edl/annealing_coef": self._metric_item(loss_stats["annealing_coef"]),
            "edl/mean_uncertainty": self._metric_item(loss_stats["mean_uncertainty"]),
            "edl/topk": self._metric_item(loss_stats["topk"]),
            "edl/valid_token_count": self._metric_item(loss_stats["valid_token_count"]),
            "edl/gt_in_topk_rate": self._metric_item(loss_stats["gt_in_topk_rate"]),
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: decode FAST action tokens using the original Qwen generate path.

        Steps:
          1. Resize images to training resolution (if specified)
          2. Generate full-vocabulary tokens with QwenVL
          3. Estimate full-vocabulary top-k Dirichlet uncertainty for generated steps
          4. Decode FAST tokens into a normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim].
                uncertainty (np.ndarray): Shape [B], mean Dirichlet uncertainty over generated tokens.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        # train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        # if train_obs_image_size:
        #     batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        instructions = [instruction for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        eval_max_new_tokens = int(self.config.trainer.get("eval_max_new_tokens", 64))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated = self.qwen_vl_interface.model.generate(
                **qwen_inputs,
                max_new_tokens=eval_max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = generated.sequences
        batch_vlm_action_token_ids = self._extract_action_token_ids(generated_ids)
        batch_fast_action_token_idx = self._decode_action_tokens(batch_vlm_action_token_ids)
        normalized_actions = self.action_model.fast_tokenizer.decode(batch_fast_action_token_idx)

        (
            token_uncertainty,
            uncertainty,
            action_token_confidence,
            action_token_confidence_mean,
            action_token_rank,
            action_token_evidence,
            action_token_aleatoric_uncertainty,
            action_token_epistemic_uncertainty,
            action_token_confidence_threshold,
            action_token_confidence_above_threshold_ratio,
        ) = self._compute_generation_uncertainty(generated)

        result = {
            "normalized_actions": normalized_actions,
            "uncertainty": uncertainty,
            "token_uncertainty": token_uncertainty,
            "action_token_confidence": action_token_confidence,
            "action_token_confidence_mean": action_token_confidence_mean,
            "action_token_rank": action_token_rank,
            "action_token_evidence": action_token_evidence,
            "action_token_aleatoric_uncertainty": action_token_aleatoric_uncertainty,
            "action_token_epistemic_uncertainty": action_token_epistemic_uncertainty,
            "action_token_confidence_threshold": action_token_confidence_threshold,
            "action_token_confidence_above_threshold_ratio": action_token_confidence_above_threshold_ratio,
        }
        if kwargs.get("return_edl_details", False):
            result.update(self._generation_edl_details(generated))
        return result

    def _generation_edl_details(self, generated) -> dict:
        """Optional replayable action-vocabulary top-k Dirichlet parameters.

        This adds no parameters and does not change decoding. Selected-token
        evidence alone is insufficient to reconstruct AU/EU; retain all top-k
        alpha values and their vocabulary IDs for each generated action token.
        """
        act_min, act_max = self._action_token_range()
        batch = generated.sequences.shape[0]
        steps = len(generated.scores)
        k = min(int(self.edl_topk), act_max - act_min + 1)
        ids = generated.sequences[:, -steps:] if steps else generated.sequences[:, :0]
        mask = (ids >= act_min) & (ids <= act_max)
        lengths = mask.sum(dim=1).cpu().numpy().astype(np.int32)
        width = int(lengths.max()) if len(lengths) else 0
        alpha_out = np.full((batch, width, k), np.nan, dtype=np.float32)
        topk_ids = np.full((batch, width, k), -1, dtype=np.int64)
        action_ids = np.full((batch, width), -1, dtype=np.int64)
        if steps:
            alphas, candidates = [], []
            for scores in generated.scores:
                values, indices = scores.float()[:, act_min:act_max + 1].topk(k, dim=-1)
                alphas.append(self._logits_to_alpha(values))
                candidates.append(indices + act_min)
            alphas = torch.stack(alphas, dim=1)
            candidates = torch.stack(candidates, dim=1)
            for row, length in enumerate(lengths):
                alpha_out[row, :length] = alphas[row][mask[row]].detach().cpu().numpy()
                topk_ids[row, :length] = candidates[row][mask[row]].cpu().numpy()
                action_ids[row, :length] = ids[row][mask[row]].cpu().numpy()
        return {
            "action_token_topk_alpha": alpha_out,
            "action_token_topk_ids": topk_ids,
            "action_token_ids": action_ids,
            "action_token_mask": np.arange(width)[None, :] < lengths[:, None],
            "num_action_tokens": lengths,
        }

    def _action_token_range(self) -> Tuple[int, int]:
        if not hasattr(self.qwen_vl_interface, "_ACTION_TOKEN_MIN") or not hasattr(
            self.qwen_vl_interface, "_ACTION_TOKEN_MAX"
        ):
            raise RuntimeError(
                "QwenEDL requires a Qwen action-token checkpoint with _ACTION_TOKEN_MIN/_ACTION_TOKEN_MAX. "
                "Use a `*-Action` VLM checkpoint with FAST action tokens added."
            )
        return int(self.qwen_vl_interface._ACTION_TOKEN_MIN), int(self.qwen_vl_interface._ACTION_TOKEN_MAX)

    def _logits_to_alpha(self, logits: torch.Tensor) -> torch.Tensor:
        if self.edl_evidence_fn == "relu":
            evidence = F.relu(logits)
        elif self.edl_evidence_fn == "exp":
            evidence = torch.exp(torch.clamp(logits, min=-10.0, max=10.0))
        elif self.edl_evidence_fn == "softplus":
            evidence = F.softplus(logits)
        else:
            raise ValueError(f"Unknown EDL evidence function: {self.edl_evidence_fn}")
        return evidence + 1.0

    def _compute_edl_action_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Causal LM alignment: logits at t predict label at t+1.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != IGNORE_INDEX

        if not torch.any(valid_mask):
            zero = logits.sum() * 0.0
            return zero, {
                "data_loss": zero,
                "kl_loss": zero,
                "annealing_coef": zero,
                "mean_uncertainty": zero,
                "topk": zero,
                "valid_token_count": zero,
                "gt_in_topk_rate": zero,
            }

        valid_logits = shift_logits[valid_mask].float()
        target = shift_labels[valid_mask].long()
        candidate_logits, target_pos, gt_in_topk = self._build_topk_edl_candidates(valid_logits, target)
        num_candidates = candidate_logits.size(-1)
        target_onehot = F.one_hot(target_pos, num_classes=num_candidates).to(candidate_logits.dtype)

        alpha = self._logits_to_alpha(candidate_logits)
        data_loss = self._edl_data_loss(alpha, target_onehot)

        annealing_coef = self._edl_annealing_coef(device=candidate_logits.device, dtype=candidate_logits.dtype)
        kl_alpha = (alpha - 1.0) * (1.0 - target_onehot) + 1.0
        kl_loss = self._dirichlet_kl_to_uniform(kl_alpha).mean()
        total_loss = data_loss + annealing_coef * self.edl_kl_weight * kl_loss

        uncertainty = num_candidates / alpha.sum(dim=-1)
        return total_loss, {
            "data_loss": data_loss,
            "kl_loss": kl_loss,
            "annealing_coef": annealing_coef,
            "mean_uncertainty": uncertainty.mean(),
            "topk": torch.tensor(float(num_candidates), dtype=candidate_logits.dtype, device=candidate_logits.device),
            "valid_token_count": torch.tensor(
                float(target.numel()), dtype=candidate_logits.dtype, device=candidate_logits.device
            ),
            "gt_in_topk_rate": gt_in_topk.float().mean(),
        }

    def _build_topk_edl_candidates(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vocab_size = logits.size(-1)
        topk = min(int(self.edl_topk), vocab_size)
        candidate_logits, candidate_ids = logits.topk(topk, dim=-1)

        target_in_topk_mask = candidate_ids.eq(target[:, None])
        gt_in_topk = target_in_topk_mask.any(dim=-1)
        target_pos = target_in_topk_mask.float().argmax(dim=-1).long()

        missing_gt = ~gt_in_topk
        if torch.any(missing_gt):
            target_logits = logits.gather(dim=-1, index=target[:, None]).squeeze(-1)
            candidate_logits = candidate_logits.clone()
            candidate_ids = candidate_ids.clone()
            candidate_logits[missing_gt, -1] = target_logits[missing_gt]
            candidate_ids[missing_gt, -1] = target[missing_gt]
            target_pos[missing_gt] = topk - 1

        return candidate_logits, target_pos, gt_in_topk

    def _compute_generation_uncertainty(
        self, generated: Any
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if not hasattr(generated, "scores") or generated.scores is None or len(generated.scores) == 0:
            batch_size = int(generated.sequences.size(0)) if hasattr(generated, "sequences") else 0
            empty = np.zeros((batch_size, 0), dtype=np.float32)
            empty_scalar = np.zeros((batch_size,), dtype=np.float32)
            return (
                empty,
                np.ones((batch_size,), dtype=np.float32),
                empty,
                empty_scalar,
                empty,
                empty,
                empty,
                empty,
                empty_scalar,
                empty_scalar,
            )

        step_uncertainties = []
        step_selected_confidences = []
        step_selected_evidences = []
        step_action_aleatoric_uncertainties = []
        step_action_epistemic_uncertainties = []
        step_action_ranks = []
        topk = int(self.edl_topk)
        num_generated_tokens = len(generated.scores)
        generated_token_ids = generated.sequences[:, -num_generated_tokens:]
        act_min, act_max = self._action_token_range()
        num_action_tokens = act_max - act_min + 1
        for step_scores in generated.scores:
            step_idx = len(step_uncertainties)
            scores = step_scores.float()
            step_topk = min(topk, scores.size(-1))
            topk_scores, _ = scores.topk(step_topk, dim=-1)
            alpha = self._logits_to_alpha(topk_scores)
            step_uncertainties.append((step_topk / alpha.sum(dim=-1)).detach())
            selected_ids = generated_token_ids[:, step_idx].to(device=scores.device)
            action_logits = scores[:, act_min : act_max + 1]
            action_topk = min(topk, num_action_tokens)
            action_topk_scores, _ = action_logits.topk(action_topk, dim=-1)
            action_alpha = self._logits_to_alpha(action_topk_scores)
            action_strength = action_alpha.sum(dim=-1, keepdim=True)
            action_probs = action_alpha / action_strength
            action_au = (
                action_probs
                * (torch.digamma(action_strength + 1.0) - torch.digamma(action_alpha + 1.0))
            ).sum(dim=-1)
            if action_topk > 1:
                action_au = action_au / float(np.log(action_topk))
            action_eu = float(action_topk) / action_strength.squeeze(-1)
            step_action_aleatoric_uncertainties.append(action_au.detach())
            step_action_epistemic_uncertainties.append(action_eu.detach())
            selected_action_idx = (selected_ids - act_min).clamp(min=0, max=num_action_tokens - 1)
            selected_action_scores = action_logits.gather(dim=-1, index=selected_action_idx[:, None]).squeeze(-1)
            selected_action_alpha = self._logits_to_alpha(selected_action_scores)
            step_selected_confidences.append((selected_action_alpha / action_alpha.sum(dim=-1)).detach())
            step_selected_evidences.append((selected_action_alpha - 1.0).detach())
            step_action_ranks.append((action_logits.gt(selected_action_scores[:, None]).sum(dim=-1) + 1).detach())

        token_uncertainty = torch.stack(step_uncertainties, dim=1).float().cpu().numpy()
        selected_token_confidence = torch.stack(step_selected_confidences, dim=1)
        action_token_mask = (generated_token_ids >= act_min) & (generated_token_ids <= act_max)
        action_token_confidence = self._select_ragged_with_padding(
            selected_token_confidence.float(), action_token_mask
        )
        action_token_rank = self._select_ragged_with_padding(
            torch.stack(step_action_ranks, dim=1).float(), action_token_mask
        )
        action_token_evidence = self._select_ragged_with_padding(
            torch.stack(step_selected_evidences, dim=1).float(), action_token_mask
        )
        action_token_aleatoric_uncertainty = self._select_ragged_with_padding(
            torch.stack(step_action_aleatoric_uncertainties, dim=1).float(), action_token_mask
        )
        action_token_epistemic_uncertainty = self._select_ragged_with_padding(
            torch.stack(step_action_epistemic_uncertainties, dim=1).float(), action_token_mask
        )
        uncertainty = token_uncertainty.mean(axis=1)
        action_token_confidence_mean = self._nanmean_with_empty_zero(action_token_confidence, axis=1)
        action_token_confidence_threshold = np.full(
            (action_token_confidence.shape[0],), 1.0 / float(action_topk) + 0.01, dtype=np.float32
        )
        action_token_confidence_above_threshold_ratio = self._above_threshold_ratio_with_empty_zero(
            action_token_confidence, action_token_confidence_threshold
        )
        return (
            token_uncertainty,
            uncertainty,
            action_token_confidence,
            action_token_confidence_mean,
            action_token_rank,
            action_token_evidence,
            action_token_aleatoric_uncertainty,
            action_token_epistemic_uncertainty,
            action_token_confidence_threshold,
            action_token_confidence_above_threshold_ratio,
        )

    def _select_ragged_with_padding(self, values: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        rows = []
        max_len = 0
        for batch_idx in range(values.size(0)):
            row = values[batch_idx][mask[batch_idx]].detach().cpu().numpy().astype(np.float32)
            rows.append(row)
            max_len = max(max_len, row.shape[0])

        if max_len == 0:
            return np.zeros((values.size(0), 0), dtype=np.float32)

        padded = np.full((values.size(0), max_len), np.nan, dtype=np.float32)
        for batch_idx, row in enumerate(rows):
            padded[batch_idx, : row.shape[0]] = row
        return padded

    def _nanmean_with_empty_zero(self, values: np.ndarray, axis: int) -> np.ndarray:
        if values.size == 0:
            return np.zeros((values.shape[0],), dtype=np.float32)
        counts = np.sum(np.isfinite(values), axis=axis)
        sums = np.nansum(values, axis=axis)
        return np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float32), where=counts > 0)

    def _above_threshold_ratio_with_empty_zero(self, values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.zeros((values.shape[0],), dtype=np.float32)
        finite = np.isfinite(values)
        counts = np.sum(finite, axis=1)
        above = np.sum((values > thresholds[:, None]) & finite, axis=1)
        return np.divide(above, counts, out=np.zeros_like(thresholds, dtype=np.float32), where=counts > 0)

    def _edl_data_loss(self, alpha: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        S = alpha.sum(dim=-1, keepdim=True)

        if self.edl_loss_type == "mse":
            probs = alpha / S
            err = (target_onehot - probs).pow(2).sum(dim=-1)
            var = (alpha * (S - alpha) / (S * S * (S + 1.0))).sum(dim=-1)
            return (err + var).mean()

        if self.edl_loss_type == "log":
            return (target_onehot * (torch.log(S) - torch.log(alpha))).sum(dim=-1).mean()

        if self.edl_loss_type == "digamma":
            return (target_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1).mean()

        raise ValueError(f"Unknown EDL loss type: {self.edl_loss_type}")

    def _dirichlet_kl_to_uniform(self, alpha: torch.Tensor) -> torch.Tensor:
        num_classes = alpha.size(-1)
        beta = torch.ones((1, num_classes), dtype=alpha.dtype, device=alpha.device)

        sum_alpha = alpha.sum(dim=-1, keepdim=True)
        sum_beta = beta.sum(dim=-1, keepdim=True)

        lnB_alpha = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
        lnB_beta = torch.lgamma(beta).sum(dim=-1, keepdim=True) - torch.lgamma(sum_beta)
        digamma_term = ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(
            dim=-1, keepdim=True
        )
        return (digamma_term + lnB_alpha + lnB_beta).squeeze(-1)

    def _edl_annealing_coef(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.training:
            self._edl_step.add_(1)
        step = self._edl_step.to(device=device, dtype=dtype)
        if self.edl_annealing_steps <= 0:
            return torch.ones((), dtype=dtype, device=device)
        return torch.clamp(step / float(self.edl_annealing_steps), max=1.0)

    def _metric_item(self, value: torch.Tensor) -> float:
        return float(value.detach().float().item())

    def _extract_action_token_ids(
        self,
        generated_ids: torch.LongTensor,
    ) -> List[List[int]]:
        """
        Extract action tokens (with offset) from the generated token sequence and return a 2D list:
        ret[b] = [vlm_action_token_id_0, vlm_action_token_id_1, ...]
        Rule: keep all tokens falling within [_ACTION_TOKEN_MIN, _ACTION_TOKEN_MAX] in order of appearance.
        You may change it to "take only the first occurrence followed by continuous segment" as needed.
        """
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        act_max = self.qwen_vl_interface._ACTION_TOKEN_MAX
        mask = (generated_ids >= act_min) & (generated_ids <= act_max)  # [B, L]
        results = []
        for b in range(generated_ids.size(0)):
            idx = mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                results.append([])
                continue
            # all action tokens
            tokens = generated_ids[b, idx].tolist()
            results.append(tokens)
        return results

    def _decode_action_tokens(self, batch_vlm_tokens: List[List[int]]) -> List[Any]:
        """
        Decode the offset VLM action token list back to fast tokenizer semantics.
        fast_tokenizer.decode expects the original fast token id sequence (without offset).
        """
        act_min = self.qwen_vl_interface._ACTION_TOKEN_MIN
        batch_fast_token_ids = []
        for seq in batch_vlm_tokens:
            if not seq:
                batch_fast_token_ids.append(None)
                continue
            fast_ids = [t - act_min for t in seq]

            batch_fast_token_ids.append(fast_ids)

        return batch_fast_token_ids

    def map_fast_token_to_vlm_action(self, tokens) -> str:
        """Maps fast action tokens to the VLM action format.
        Action token 0 is mapped to the string <robot_action_0>  ... and so on
        """
        return "".join(
            [f"<robot_action_{token}>" for token in tokens]
        )  # you should add <robot_action_{token}> to VLM as special tokens,


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwenvl_EDL(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    # Untrained models haven't learned the action tokens, so predictions may be empty.
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
