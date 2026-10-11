"""Adapt StarVLA real joint-target checkpoints to the YAM/OpenPI contract."""
from collections.abc import Mapping
import time

import numpy as np
from PIL import Image

CAMERAS = ("top", "left", "right")
STAR_ACTION_KEYS = ["action.left_joints", "action.right_joints",
                    "action.left_gripper", "action.right_gripper"]
STAR_TO_YAM = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
YAM_TO_STAR = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]
STAR_STATE_KEYS = [key.replace("action.", "state.") for key in STAR_ACTION_KEYS]
# Real-data converter records consecutive targets at 30 Hz; no resampling.
CONTROL_HZ = 30.0


class YAMStarVLAPolicy:
    def __init__(self, wrapper, *, task, default_prompt, state_processor=None, execution_horizon=None,
                 fast_decode_mode=None):
        self.wrapper = wrapper
        self.state_processor = state_processor
        if fast_decode_mode not in (None, "truncate", "strict"):
            raise ValueError("fast_decode_mode must be truncate or strict")
        self.fast_decode_mode = fast_decode_mode
        source = wrapper.metadata
        if list(source.get("action_keys", [])) != STAR_ACTION_KEYS:
            raise ValueError(f"Unsupported checkpoint action layout: {source.get('action_keys')}")
        self.model_horizon = int(source["action_chunk_size"])
        if self.model_horizon < 1:
            raise ValueError("Checkpoint horizon must be positive")
        self.horizon = self.model_horizon if execution_horizon is None else execution_horizon
        if (not isinstance(self.horizon, int) or isinstance(self.horizon, bool)
                or not 1 <= self.horizon <= self.model_horizon):
            raise ValueError("execution_horizon must be an integer between 1 and the checkpoint horizon")
        if state_processor is not None and list(state_processor.state_keys) != STAR_STATE_KEYS:
            raise ValueError("State normalization keys must match the training left/right joints, then grippers order")
        self.metadata = {
            **source,
            "model_horizon": self.model_horizon,
            "action_chunk_size": self.horizon,
            "execution_horizon": self.horizon,
            "starvla_action_keys": list(source["action_keys"]),
            "starvla_state_keys": list(source.get("state_keys", [])),
            "action_keys": ["action.left_joints", "action.left_gripper",
                            "action.right_joints", "action.right_gripper"],
            "state_keys": ["state.left_joints", "state.left_gripper",
                           "state.right_joints", "state.right_gripper"],
            "backend": "starvla", "protocol": "openpi", "task": task,
            "horizon": self.horizon, "state_dim": 14, "action_dim": 14,
            "control_hz": CONTROL_HZ, "action_dt": 1.0 / CONTROL_HZ,
            "action_stride": 1, "open_loop_horizon": self.horizon,
            "execution_speed": 1.0, "supports_rtc": False,
            "uses_state": state_processor is not None, "action_space": "absolute_joint_radians",
            "gripper_range": [0.0, 1.0],
            "action_order": ["left_joints[0:6]", "left_gripper",
                             "right_joints[0:6]", "right_gripper"],
            "camera_order": list(CAMERAS), "image_size": [224, 224],
            "default_prompt": default_prompt,
            "diagnostics_layout": "StarVLA batch/token axes, unchanged",
            "diagnostics_horizon": self.model_horizon,
            "fast_decode_mode": fast_decode_mode or "original",
        }

    def infer(self, obs):
        if not isinstance(obs, Mapping):
            raise ValueError("Observation must be a mapping")
        if "action_prefix" in obs or obs.get("prefix_length", 0) != 0:
            raise ValueError("StarVLA real inference does not support RTC; disable RTC in YAM")
        if obs.get("execution_speed", 1.0) != 1.0:
            raise ValueError("StarVLA real inference requires execution_speed=1.0")
        state = np.asarray(obs.get("state"), dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("state must contain 14 finite values in YAM order")
        prompt = obs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        raw_images = obs.get("images")
        if not isinstance(raw_images, Mapping) or set(raw_images) != set(CAMERAS):
            raise ValueError("images must contain exactly top, left, right")
        images = []
        for role in CAMERAS:
            frame = np.asarray(raw_images[role])
            if (frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3
                    or min(frame.shape[:2]) < 1):
                raise ValueError(f"images[{role}] must be a nonempty HWC uint8 RGB array")
            # Match gr00t_lerobot/datasets.py::_pack_sample (PIL default resize).
            images.append(Image.fromarray(frame).resize((224, 224)))
        started = time.perf_counter()
        example = {"image": images, "lang": prompt}
        if self.state_processor is not None:
            raw_state = state[YAM_TO_STAR][None, :]
            normalized_state = self.state_processor.apply_state(raw_state)
            # Dataset _pack_sample casts normalized state to float16 before
            # QwenEDL converts it back to float32 for text quantization.
            normalized_state = np.asarray(normalized_state, dtype=np.float16)
            if (normalized_state.shape != (1, 14) or not np.isfinite(normalized_state).all()
                    or np.any(np.abs(normalized_state) > 1)):
                raise ValueError("Training state transform must return finite (1, 14) state in [-1, 1]")
            example["state"] = normalized_state
        decode_diagnostics = {}
        decode_kwargs = {}
        if self.fast_decode_mode is not None:
            from deployment.real.fast_decode import decode_fast_chunk

            def decode(tokenizer, tokens):
                normalized, audit = decode_fast_chunk(
                    tokenizer, tokens, model_horizon=self.model_horizon,
                    execution_horizon=self.horizon, mode=self.fast_decode_mode)
                decode_diagnostics.update(audit)
                return normalized

            decode_kwargs["fast_action_decoder"] = decode
        result = self.wrapper.predict_action(
            examples=[example],
            return_edl_details=bool(obs.get("return_edl_details", False)),
            **decode_kwargs,
        )
        if self.fast_decode_mode is not None and not decode_diagnostics:
            raise RuntimeError("Framework did not invoke the requested FAST decoder")
        actions = np.asarray(result["actions"], dtype=np.float32)
        decoded_horizon = int(decode_diagnostics["fast_decode_horizon"][0]) if decode_diagnostics else self.model_horizon
        if actions.shape != (1, decoded_horizon, 14) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite (1, {decoded_horizon}, 14) actions, got {actions.shape}")
        # Wrapper already applied this checkpoint's inverse normalization.
        # Do not clip, interpolate, convert to deltas, or normalize a second time.
        response = {
            "actions": np.ascontiguousarray(actions[0, :self.horizon][:, STAR_TO_YAM]),
            "policy_timing": {"infer_ms": (time.perf_counter() - started) * 1000},
            "diagnostics": {**{key: value for key, value in result.items() if key != "actions"},
                            **decode_diagnostics},
        }
        if obs.get("return_edl_details", False):
            # Keep the full decoded prediction for audits; token evidence still
            # describes the model horizon, not just the executed prefix.
            response["diagnostics"]["full_model_actions"] = np.ascontiguousarray(actions[0][:, STAR_TO_YAM])
            # Exact PIL images supplied to Qwen; its processor performs the
            # subsequent vision-token preprocessing internally.
            response["model_images"] = {role: np.asarray(image) for role, image in zip(CAMERAS, images)}
        return response
