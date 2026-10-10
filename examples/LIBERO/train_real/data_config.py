"""Three independent real-robot task mixtures using the existing LeRobot loader."""

import torch

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


ACTION_HORIZON = 30


class RealStateTransform(StateActionTransform):
    """Normalize measured state with its own quantiles for text quantization."""

    def apply(self, data):
        data = super().apply(data)
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            stats = self.normalization_statistics[key]
            q01 = torch.as_tensor(stats["q01"], device=value.device, dtype=value.dtype)
            q99 = torch.as_tensor(stats["q99"], device=value.device, dtype=value.dtype)
            # A constant state dimension carries no scale information. Do not
            # leave its raw joint angle in an otherwise normalized vector.
            data[key] = torch.where(q99 != q01, value, torch.zeros_like(value)).clamp(-1, 1)
        return data


class EDLRealDualArmDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.top", "video.left", "video.right"]
    state_keys = ["state.left_joints", "state.right_joints", "state.left_gripper", "state.right_gripper"]
    action_keys = [key.replace("state.", "action.") for key in state_keys]
    # Match convert_edl_real.SLICES when splitting saved 14-D normalization stats.
    state_key_dims = {
        "state.left_joints": 6,
        "state.right_joints": 6,
        "state.left_gripper": 1,
        "state.right_gripper": 1,
    }
    action_key_dims = {key.replace("state.", "action."): dim for key, dim in state_key_dims.items()}
    language_keys = ["annotation.human.action.task_description"]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=[0], modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=[0], modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=list(range(ACTION_HORIZON)), modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=[0], modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys + self.state_keys),
            # Match QwenEDL's existing q01/q99 action unnormalization at inference.
            StateActionTransform(apply_to=self.action_keys,
                                 normalization_modes={key: "q99" for key in self.action_keys}),
            RealStateTransform(apply_to=self.state_keys,
                               normalization_modes={key: "q99" for key in self.state_keys}),
        ])


ROBOT_TYPE_CONFIG_MAP = {"edl_real_dual_arm": EDLRealDualArmDataConfig()}
DATASET_NAMED_MIXTURES = {
    f"edl_real_{task}": [(task, 1.0, "edl_real_dual_arm")]
    for task in ("classification_the_blocks", "insert_the_two_tubes_into_the_rack_one_by_one",
                 "place_the_slippers_on_the_shoe_rack")
}
