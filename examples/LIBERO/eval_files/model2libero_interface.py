# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""LIBERO env-side adapter (thin client).

After the server-side refactor (see `deployment/model_server/policy_wrapper.py`),
the websocket *server* now returns already-unnormalized actions and ships
model-invariant fields (`action_chunk_size`, `available_unnorm_keys`) at
handshake. This client therefore no longer needs to:
  - load `dataset_statistics.json`
  - know `future_action_window_size`
  - perform un-normalization

It only handles env-specific adaptation: image history bookkeeping, action
ensembling, gripper sticky logic, and chunk-cache scheduling.
"""

from collections import deque
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler


PROVISIONAL_LOW_EVIDENCE_THRESHOLD = 4.0
WORST_TOKEN_COUNT = 3


class ModelClient:
    def __init__(
        self,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble: bool = True,
        action_ensemble_horizon: Optional[int] = 3,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha: float = 0.1,
        host: str = "0.0.0.0",
        port: int = 10095,
        image_size: Sequence[int] = (224, 224),
    ) -> None:
        # Connect & receive handshake metadata (action_chunk_size, etc.)
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        self._server_metadata = meta

        self.image_size: tuple = tuple(image_size)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_chunk_size: {self.action_chunk_size}, "
            f"server_meta: {meta} ***"
        )

        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon

        # Gripper sticky state (kept for parity with the previous client; not
        # currently consumed by LIBERO but other policy_setup paths use it).
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(
                self.action_ensemble_horizon, self.adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        # Cached unnormalized chunk; refreshed every `action_chunk_size` steps.
        self.raw_actions: Optional[np.ndarray] = None
        self.chunk_uncertainty: Optional[dict] = None

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    @staticmethod
    def _first_batch_scalar(value) -> Optional[float]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return None
        return float(arr.reshape(-1)[0])

    @staticmethod
    def _first_batch_sequence(value) -> Optional[list[float]]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            return []
        if arr.ndim <= 1:
            seq = arr.reshape(-1)
        else:
            seq = arr[0].reshape(-1)
        seq = seq[np.isfinite(seq)]
        return [float(x) for x in seq]

    @staticmethod
    def _max_consecutive_true(mask: np.ndarray) -> int:
        max_run = 0
        current_run = 0
        for value in mask:
            current_run = current_run + 1 if bool(value) else 0
            max_run = max(max_run, current_run)
        return max_run

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.raw_actions = None
        self.chunk_uncertainty = None

    def step(self, example: dict, step: int = 0, **kwargs) -> dict:
        """One env step.

        Args:
            example: dict with keys ``image`` (list of np.uint8 HWC arrays) and ``lang`` (str).
            step: env step counter; used for chunk caching.

        Returns:
            ``{"raw_action": {"world_vector": ..., "rotation_delta": ..., "open_gripper": ...}}``
        """
        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)

        # Resize images to self.image_size if needed.
        if self.image_size and example.get("image"):
            resized = []
            target_hw = self.image_size  # (H, W)
            for img in example["image"]:
                arr = np.asarray(img)
                if arr.shape[:2] != target_hw:
                    arr = np.asarray(
                        Image.fromarray(arr).resize(
                            (target_hw[1], target_hw[0]), Image.BILINEAR
                        )
                    )
                resized.append(arr)
            example = {**example, "image": resized}

        # Refresh chunk if needed.
        refresh_chunk = step % self.action_chunk_size == 0 or self.raw_actions is None
        if refresh_chunk:
            vla_input = {
                "examples": [example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            response = self.client.predict_action(vla_input)
            try:
                actions_batch = response["data"]["actions"]  # (B, T, D), unnormalized server-side
            except KeyError:
                raise KeyError(
                    f"Key 'actions' not found in response data: keys={list(response.get('data', {}).keys())}, "
                    f"full response={response}"
                )
            self.raw_actions = np.asarray(actions_batch)[0]  # (T, D)
            data = response.get("data", {})
            chunk_idx = int(step // self.action_chunk_size)
            chunk_mean = self._first_batch_scalar(data.get("uncertainty"))
            token_uncertainty = self._first_batch_sequence(data.get("token_uncertainty"))
            action_token_confidence_mean = self._first_batch_scalar(
                data.get(
                    "action_token_confidence_mean",
                    data.get("selected_token_confidence_mean", data.get("selected_evidence_mean")),
                )
            )
            action_token_confidence = self._first_batch_sequence(
                data.get("action_token_confidence", data.get("selected_token_confidence", data.get("selected_evidence")))
            )
            action_token_rank = self._first_batch_sequence(data.get("action_token_rank"))
            action_token_evidence = self._first_batch_sequence(data.get("action_token_evidence"))
            action_token_aleatoric_uncertainty = self._first_batch_sequence(
                data.get("action_token_aleatoric_uncertainty")
            )
            action_token_epistemic_uncertainty = self._first_batch_sequence(
                data.get("action_token_epistemic_uncertainty")
            )
            action_token_confidence_threshold = self._first_batch_scalar(
                data.get("action_token_confidence_threshold")
            )
            action_token_confidence_above_threshold_ratio = self._first_batch_scalar(
                data.get("action_token_confidence_above_threshold_ratio")
            )
            if chunk_mean is None and token_uncertainty:
                chunk_mean = float(np.mean(token_uncertainty))
            if action_token_confidence_mean is None and action_token_confidence:
                action_token_confidence_mean = float(np.mean(action_token_confidence))
            if action_token_confidence_threshold is None and action_token_confidence:
                action_token_confidence_threshold = 1.0 / 25.0 + 0.01
            if (
                action_token_confidence_above_threshold_ratio is None
                and action_token_confidence
                and action_token_confidence_threshold is not None
            ):
                action_token_confidence_above_threshold_ratio = float(
                    np.mean(np.asarray(action_token_confidence, dtype=np.float32) > action_token_confidence_threshold)
                )

            low_evidence_count = None
            low_evidence_ratio = None
            low_evidence_max_consecutive = None
            if action_token_evidence is not None:
                evidence = np.asarray(action_token_evidence, dtype=np.float32)
                low_evidence_mask = evidence < PROVISIONAL_LOW_EVIDENCE_THRESHOLD
                low_evidence_count = int(np.sum(low_evidence_mask))
                low_evidence_ratio = float(np.mean(low_evidence_mask)) if evidence.size > 0 else 0.0
                low_evidence_max_consecutive = self._max_consecutive_true(low_evidence_mask)

            worst_token_eu_mean = None
            if action_token_epistemic_uncertainty is not None:
                epistemic = np.asarray(action_token_epistemic_uncertainty, dtype=np.float32)
                if epistemic.size > 0:
                    tail_size = min(WORST_TOKEN_COUNT, epistemic.size)
                    worst_token_eu_mean = float(np.mean(np.sort(epistemic)[-tail_size:]))
                else:
                    worst_token_eu_mean = 0.0

            self.chunk_uncertainty = None
            if (
                chunk_mean is not None
                or token_uncertainty is not None
                or action_token_confidence_mean is not None
                or action_token_confidence is not None
                or action_token_rank is not None
                or action_token_evidence is not None
                or action_token_aleatoric_uncertainty is not None
                or action_token_epistemic_uncertainty is not None
                or action_token_confidence_above_threshold_ratio is not None
            ):
                action_token_lengths = [
                    len(values)
                    for values in (
                        action_token_confidence,
                        action_token_evidence,
                        action_token_aleatoric_uncertainty,
                        action_token_epistemic_uncertainty,
                    )
                    if values is not None
                ]
                self.chunk_uncertainty = {
                    "chunk_idx": chunk_idx,
                    "chunk_mean": chunk_mean,
                    "num_tokens": len(token_uncertainty or []),
                    "num_action_tokens": max(action_token_lengths, default=0),
                    "token_uncertainty": token_uncertainty or [],
                    "action_token_confidence_mean": action_token_confidence_mean,
                    "action_token_confidence": action_token_confidence or [],
                    "action_token_rank": action_token_rank or [],
                    "action_token_evidence": action_token_evidence or [],
                    "action_token_aleatoric_uncertainty": action_token_aleatoric_uncertainty or [],
                    "action_token_epistemic_uncertainty": action_token_epistemic_uncertainty or [],
                    "low_evidence_threshold": PROVISIONAL_LOW_EVIDENCE_THRESHOLD,
                    "low_evidence_count": low_evidence_count,
                    "low_evidence_ratio": low_evidence_ratio,
                    "low_evidence_max_consecutive": low_evidence_max_consecutive,
                    "worst_token_count": WORST_TOKEN_COUNT,
                    "worst_token_eu_mean": worst_token_eu_mean,
                    "action_token_confidence_threshold": action_token_confidence_threshold,
                    "action_token_confidence_above_threshold_ratio": action_token_confidence_above_threshold_ratio,
                }

        raw_actions = self.raw_actions[step % self.action_chunk_size][None]
        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # 1 = open; 0 = close
        }
        return {
            "raw_action": raw_action,
            "new_chunk": refresh_chunk,
            "uncertainty": self.chunk_uncertainty if refresh_chunk else None,
        }

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
