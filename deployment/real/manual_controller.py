"""One human trigger -> one inference -> at most one horizon; no background loop."""
import threading
import time

import numpy as np

from examples.LIBERO.edl_pred_real.recording import validate_response


class ManualChunkController:
    def __init__(self, policy, backend, recorder, prompt):
        self.policy, self.backend, self.recorder, self.prompt = policy, backend, recorder, prompt
        self.horizon = int(policy.metadata["horizon"])
        self.control_hz = float(policy.metadata["control_hz"])
        self.cancel = threading.Event()
        self._busy = threading.Lock()
        self.failed = False
        self._progress_lock = threading.Lock()
        self._progress = None

    def progress(self):
        with self._progress_lock:
            return None if self._progress is None else dict(self._progress)

    def _update_progress(self, **values):
        with self._progress_lock:
            self._progress = {**(self._progress or {}), **values}

    def execute_once(self):
        # Concurrent button presses are rejected, never queued for later motion.
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("A chunk is already running; trigger again after it finishes")
        chunk = None
        completed = 0
        try:
            if self.cancel.is_set() or self.failed:
                raise RuntimeError("Session stopped or failed; restart after checking the cause")
            self._update_progress(chunk_number=self.recorder.count + 1, horizon=self.horizon,
                                  status="inferring", completed_rows=0, last_action=None,
                                  all_actions=None,
                                  max_model_speed=None, max_model_speed_joint=None,
                                  max_model_speed_joint_index=None, max_model_speed_step=None,
                                  max_model_arm_speed=None, max_model_arm_speed_joint=None,
                                  max_model_arm_speed_step=None, max_chunk_target_speed=None,
                                  max_chunk_target_speed_joint=None, max_chunk_target_speed_from=None,
                                  max_chunk_target_speed_to=None, max_chunk_arm_speed=None,
                                  max_chunk_arm_speed_joint=None, max_chunk_arm_speed_from=None,
                                  max_chunk_arm_speed_to=None)
            observation = self.backend.observe(self.prompt)
            observation["return_edl_details"] = True
            chunk = self.recorder.begin_chunk(observation)
            response = self.policy.infer(observation)  # Exactly ONE request.
            validate_response(response, self.horizon)
            # The adapter has already inverse-normalized these absolute joint
            # targets. Measure the model trajectory before any YAM slew limit
            # or gripper clipping is applied.
            actions = np.asarray(response["actions"], dtype=float)
            state = np.asarray(observation["state"], dtype=float)
            trajectory = np.vstack((state, actions))
            chunk_speeds = np.abs(np.diff(trajectory, axis=0)) * self.control_hz
            if len(actions) > 1:
                speeds = np.abs(np.diff(actions, axis=0)) * self.control_hz
                flat_index = int(np.argmax(speeds))
                speed_row, speed_joint = np.unravel_index(flat_index, speeds.shape)
                max_speed = float(speeds[speed_row, speed_joint])
                arm_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
                arm_flat = np.argmax(speeds[:, arm_indices])
                arm_row, arm_col = np.unravel_index(int(arm_flat), (speeds.shape[0], len(arm_indices)))
                max_arm_speed = float(speeds[arm_row, arm_indices[arm_col]])
            else:
                speed_row, speed_joint, max_speed = 0, 0, 0.0
                arm_indices, max_arm_speed, arm_row, arm_col = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], 0.0, 0, 0
            chunk_flat = int(np.argmax(chunk_speeds))
            chunk_row, chunk_joint = np.unravel_index(chunk_flat, chunk_speeds.shape)
            chunk_max_speed = float(chunk_speeds[chunk_row, chunk_joint])
            chunk_arm_flat = int(np.argmax(chunk_speeds[:, arm_indices]))
            chunk_arm_row, chunk_arm_col = np.unravel_index(
                chunk_arm_flat, (chunk_speeds.shape[0], len(arm_indices)))
            chunk_max_arm_speed = float(chunk_speeds[chunk_arm_row, arm_indices[chunk_arm_col]])
            action_names = ([f"left_joint_{i}" for i in range(6)] + ["left_gripper"] +
                            [f"right_joint_{i}" for i in range(6)] + ["right_gripper"])
            self.recorder.save_prediction(chunk, response)  # Fail closed if recording fails.
            self._update_progress(
                status="executing", last_action=[float(v) for v in actions[-1]],
                all_actions=[[float(v) for v in row] for row in actions],
                max_model_speed=max_speed, max_model_speed_joint=action_names[speed_joint],
                max_model_speed_joint_index=int(speed_joint),
                max_model_speed_step=int(speed_row + 1),
                max_model_arm_speed=max_arm_speed,
                max_model_arm_speed_joint=action_names[arm_indices[arm_col]],
                max_model_arm_speed_step=int(arm_row + 1),
                max_chunk_target_speed=chunk_max_speed,
                max_chunk_target_speed_joint=action_names[chunk_joint],
                max_chunk_target_speed_from="measured_state" if chunk_row == 0 else f"action_{chunk_row}",
                max_chunk_target_speed_to=f"action_{chunk_row + 1}",
                max_chunk_arm_speed=chunk_max_arm_speed,
                max_chunk_arm_speed_joint=action_names[arm_indices[chunk_arm_col]],
                max_chunk_arm_speed_from="measured_state" if chunk_arm_row == 0 else f"action_{chunk_arm_row}",
                max_chunk_arm_speed_to=f"action_{chunk_arm_row + 1}",
            )
            for index, row in enumerate(response["actions"]):
                if self.cancel.is_set():
                    break
                started = time.monotonic()
                if not self.backend.execute(row, index, self.recorder, chunk, self.cancel):
                    break
                completed += 1
                self._update_progress(completed_rows=completed)
                self.cancel.wait(max(0, 1.0 / self.control_hz - (time.monotonic() - started)))
            status = "completed" if completed == self.horizon else "cancelled"
            self.recorder.finish_chunk(chunk, status=status, completed_rows=completed)
            self._update_progress(status=status)
            return {"status": status, "completed_rows": completed, "path": str(chunk)}
        except BaseException as exc:
            self.failed = True
            self.cancel.set()
            self._update_progress(status="error", completed_rows=completed)
            if chunk is not None:
                self.recorder.finish_chunk(chunk, status="error", completed_rows=completed,
                                           error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._busy.release()

    def stop(self):
        self.cancel.set()
