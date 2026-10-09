"""Three independent real-robot task mixtures using the existing LeRobot loader."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


ACTION_HORIZON = 15


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
            StateActionToTensor(apply_to=self.action_keys),
            # Match QwenEDL's existing q01/q99 action unnormalization at inference.
            StateActionTransform(apply_to=self.action_keys,
                                 normalization_modes={key: "q99" for key in self.action_keys}),
        ])


ROBOT_TYPE_CONFIG_MAP = {"edl_real_dual_arm": EDLRealDualArmDataConfig()}
DATASET_NAMED_MIXTURES = {
    f"edl_real_{task}": [(task, 1.0, "edl_real_dual_arm")]
    for task in ("classification_the_blocks", "insert_the_two_tubes_into_the_rack_one_by_one",
                 "place_the_slippers_on_the_shoe_rack")
}
