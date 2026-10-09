"""Adapt StarVLA real joint-target checkpoints to the YAM/OpenPI contract."""
from collections.abc import Mapping
import time

import numpy as np
from PIL import Image

CAMERAS = ("top", "left", "right")
STAR_ACTION_KEYS = ["action.left_joints", "action.right_joints",
                    "action.left_gripper", "action.right_gripper"]
STAR_TO_YAM = [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]
# Real-data converter records consecutive targets at 30 Hz; no resampling.
CONTROL_HZ = 30.0


class YAMStarVLAPolicy:
    def __init__(self, wrapper, *, task, default_prompt):
        self.wrapper = wrapper
        source = wrapper.metadata
        if list(source.get("action_keys", [])) != STAR_ACTION_KEYS:
            raise ValueError(f"Unsupported checkpoint action layout: {source.get('action_keys')}")
        self.horizon = int(source["action_chunk_size"])
        if self.horizon < 1:
            raise ValueError("Checkpoint horizon must be positive")
        self.metadata = {
            **source,
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
            "uses_state": False, "action_space": "absolute_joint_radians",
            "gripper_range": [0.0, 1.0],
            "action_order": ["left_joints[0:6]", "left_gripper",
                             "right_joints[0:6]", "right_gripper"],
            "camera_order": list(CAMERAS), "image_size": [224, 224],
            "default_prompt": default_prompt,
            "diagnostics_layout": "StarVLA batch/token axes, unchanged",
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
        # Current real QwenEDL/QwenFast checkpoints do not condition on state.
        result = self.wrapper.predict_action(
            examples=[{"image": images, "lang": prompt}],
            return_edl_details=bool(obs.get("return_edl_details", False)),
        )
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (1, self.horizon, 14) or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite (1, {self.horizon}, 14) actions, got {actions.shape}")
        # Wrapper already applied this checkpoint's inverse normalization.
        # Do not clip, interpolate, convert to deltas, or normalize a second time.
        response = {
            "actions": np.ascontiguousarray(actions[0][:, STAR_TO_YAM]),
            "policy_timing": {"infer_ms": (time.perf_counter() - started) * 1000},
            "diagnostics": {key: value for key, value in result.items() if key != "actions"},
        }
        if obs.get("return_edl_details", False):
            # Exact PIL images supplied to Qwen; its processor performs the
            # subsequent vision-token preprocessing internally.
            response["model_images"] = {role: np.asarray(image) for role, image in zip(CAMERAS, images)}
        return response
