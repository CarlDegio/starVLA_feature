"""Shared YAM execution/recording helpers used by the manual GUI profile."""
from dataclasses import replace
import math
import sys

import numpy as np



class RecordingRobot:
    """Log each arm's accepted command separately, including partial failures."""
    def __init__(self, robot, name):
        self.robot, self.name = robot, name
        self.context = None

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def command_joint_pos(self, command):
        recorder, chunk, index, cancel = self.context
        if cancel.is_set():
            raise RuntimeError("Execution cancelled")
        recorder.event(chunk, "command_attempt", row=index, arm=self.name,
                       target=np.asarray(command).tolist())
        self.robot.command_joint_pos(command)
        readback = getattr(self.robot, "last_effective_command", None)
        effective = readback() if readback else None
        recorder.event(chunk, "command_returned", row=index, arm=self.name,
                       effective_target=None if effective is None else np.asarray(effective).tolist())


class YAMBackend:
    def __init__(self, cfg, metadata, max_joint_speed, mock):
        self.cfg, self.metadata, self.max_joint_speed, self.mock = cfg, metadata, max_joint_speed, mock
        self.units, self.loop = [], None
        if not math.isclose(cfg.control_hz, metadata["control_hz"], abs_tol=1e-6):
            raise ValueError("Station control_hz must match server control_hz")
        if [r.type for r in cfg.robot.robots] != ["yam_left", "yam_right"] or cfg.robot.num_arm_joints != 6:
            raise ValueError("Manual collector requires left then right YAM arms with six joints each")

    def initialize(self):
        if self.loop is not None:
            return
        from yam_abc_reproduce.runtime import build_arm_units, build_cameras_from_config
        from yam_abc_reproduce.deploy.loop import DeployLoop
        # Only reached after the first explicit n. Building real drivers may
        # enable motors/calibrate grippers; the model service never does this.
        self.units = build_arm_units(self.cfg, mock=self.mock, followers_only=True)
        self.units = [replace(unit, robot=RecordingRobot(unit.robot, unit.name)) for unit in self.units]
        cameras = build_cameras_from_config(self.cfg, mock=self.mock)
        self.loop = DeployLoop(self.units, cameras, None, prompt="", control_hz=self.cfg.control_hz,
                               max_joint_speed=self.max_joint_speed)
        self.loop._start_cameras()

    def observe(self, prompt):
        from yam_abc_reproduce.deploy import contract
        self.initialize()
        frames, observations = self.loop._read()
        obs = contract.build_observation(observations, frames, self.loop.cameras, prompt)
        if set(obs["images"]) != {"top", "left", "right"}:
            raise ValueError("All top/left/right camera frames must be present")
        return obs

    def execute(self, row, index, recorder, chunk, cancel):
        from yam_abc_reproduce.deploy import contract
        if cancel.is_set():
            return False
        self.loop._cancel = cancel
        _, measured = self.loop._read()
        recorder.event(chunk, "measured_before", row=index,
                       state=contract.build_state(measured).tolist())
        cmds = contract.split_action(row, self.loop._arm_dims)
        # Keep existing joint slew limits and normalized-gripper saturation.
        cmds = self.loop._clamp_cmds(cmds, measured)
        for cmd in cmds:
            cmd[-1] = np.clip(cmd[-1], 0.0, 1.0)
        for unit in self.units:
            unit.robot.context = recorder, chunk, index, cancel
        if not self.loop._command(cmds):
            return False
        _, measured = self.loop._read()
        recorder.event(chunk, "measured_after", row=index,
                       state=contract.build_state(measured).tolist())
        return True

    def close(self):
        if self.loop is not None:
            self.loop.stop()
        for unit in reversed(self.units):
            # Follow the driver's normal shutdown (may release motor torque).
            close = getattr(unit.robot, "close", None) or unit.robot.stop
            close()
        self.units = []


def discard_queued_terminal_input():
    # Characters typed during inference/motion must not launch another chunk.
    if sys.stdin.isatty():
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)


