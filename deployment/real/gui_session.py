"""Manual collection attached to an existing YAM GUI session and its E-stop."""
from dataclasses import replace
import threading
import time

import numpy as np

from deployment.real.infer import RecordedClient
from deployment.real.manual_client import RecordingRobot, YAMBackend
from deployment.real.manual_controller import ManualChunkController
from examples.LIBERO.edl_pred_real.recording import EpisodeRecorder, write_json



class SharedBackend(YAMBackend):
    def __init__(self, session, metadata, speed, generation):
        super().__init__(session.cfg, metadata, speed, session.mock)
        self.session, self.generation = session, generation
        self.cancel = None
        self.on_first_observation = None

    def observe(self, prompt):
        observation = super().observe(prompt)
        if self.on_first_observation is not None:
            self.on_first_observation(observation)
            self.on_first_observation = None
        return observation

    def initialize(self):
        if self.loop is not None:
            return
        from yam_abc_reproduce.deploy.loop import DeployLoop
        session = self.session
        if session.live and not session.followers_only:
            raise RuntimeError("请先结束遥操并 Reset Session，再开始手动推理")
        if not session.live:
            session._teardown_units()
            session.go_live(session.cfg, followers_only=True)
        if session._estop_generation != self.generation or self.cancel.is_set():
            session.estop()
            raise RuntimeError("启动已被停止/急停取消")
        self.units = [replace(unit, robot=RecordingRobot(unit.robot, unit.name)) for unit in session.units]
        self.loop = DeployLoop(self.units, session.workers, None, prompt="", control_hz=session.cfg.control_hz,
                               max_joint_speed=self.max_joint_speed)
        self.loop._cancel = self.cancel
        session.deploy_loop = self.loop  # Existing E-STOP reaches the shared cancel event.

    def close(self):
        # GUI owns hardware and cameras; never close them from the collector.
        if self.loop is not None:
            self.loop.stop()


class ManualEpisodeSession:
    def __init__(self, session, output, cfg_loader, client_factory=RecordedClient):
        self.session, self.output, self.cfg_loader = session, output, cfg_loader
        self.client_factory = client_factory
        self.phase = "idle"
        self.error = None
        self.recorder = self.controller = self.policy = self.backend = None
        self.generation = None
        self.home = None
        self._operation = threading.Lock()
        self._state = threading.Lock()
        self._end_thread = None
        self._stop_requested = threading.Event()
        self.last_result = None

    def status(self):
        with self._state:
            return {"phase": self.phase, "error": self.error,
                    "episode_id": self.recorder.metadata["episode_id"] if self.recorder else None,
                    "path": str(self.recorder.path) if self.recorder else None,
                    "chunks": self.recorder.count if self.recorder else 0,
                    "latest_chunk": self.controller.progress() if self.controller else None,
                    "last_result": self.last_result}

    def step(self, *, host, port, prompt, collection_id, seed_namespace, max_joint_speed):
        if not self._operation.acquire(blocking=False):
            raise RuntimeError("正在执行或结束 episode，请勿重复触发")
        locked = False
        try:
            locked = self.session._deploy_lock.acquire(blocking=False)
            if not locked:
                raise RuntimeError("YAM 正在进行其他操作")
            with self._state:
                if self.phase not in ("idle", "paused"):
                    raise RuntimeError(f"当前状态 {self.phase} 不允许执行；请先结束并标记本轮")
                self.phase = "busy"
                self.error = None
                self._stop_requested.clear()
            if self.recorder is None:
                if self.session.deploy_loop is not None or self.session._estopped:
                    raise RuntimeError("先结束其他执行；急停后需现场 Reset Session")
                if self.session.hardware_error():
                    raise RuntimeError(self.session.hardware_error())
                cfg = self.cfg_loader()
                self.home = None  # Freeze measured pose immediately before first inference.
                self.generation = self.session._estop_generation
                self.policy = self.client_factory(host, port, 300)
                if self.session._estop_generation != self.generation or self.session._estopped:
                    raise RuntimeError("模型连接期间发生急停，启动已取消")
                meta = self.policy.metadata
                if (meta.get("horizon") != 15 or meta.get("control_hz") != 30 or meta.get("action_stride") != 1
                        or not meta.get("edl_details_available")):
                    raise ValueError("需要更新后的 QwenEDL real server（15 步、30 Hz、支持 top-k alpha 记录）")
                self.session.cfg = cfg
                self.recorder = EpisodeRecorder(self.output, meta, collection_id=collection_id,
                                               seed_namespace=seed_namespace, mock=self.session.mock)
                self.recorder.metadata["max_joint_speed"] = max_joint_speed
                write_json(self.recorder.path / "episode.json", self.recorder.metadata)
                self.backend = SharedBackend(self.session, meta, max_joint_speed, self.generation)
                def save_start(observation):
                    pose = np.asarray(observation["state"], dtype=float)
                    if pose.shape != (14,) or not np.isfinite(pose).all():
                        raise ValueError("无法记录有效的 episode 起始姿态")
                    self.home = pose.tolist()
                    self.recorder.metadata["return_pose"] = self.home
                    self.recorder.metadata["return_pose_source"] = "measured_before_first_inference"
                    write_json(self.recorder.path / "episode.json", self.recorder.metadata)
                self.backend.on_first_observation = save_start
                self.controller = ManualChunkController(self.policy, self.backend, self.recorder,
                                                        prompt.strip() or meta["default_prompt"])
                self.backend.cancel = self.controller.cancel
            elif self.session._estop_generation != self.generation:
                raise RuntimeError("本轮已经急停；请结束并标记，不可继续执行")
            # Bind this operation to the episode's original prompt/model/settings.
            if self._stop_requested.is_set():
                self.controller.stop()
            result = self.controller.execute_once()
            with self._state:
                if self.phase == "busy":
                    self.phase = "paused"
                self.last_result = result
            return result
        except Exception as exc:
            with self._state:
                if self.phase != "stopping":
                    self.phase = "error" if self.recorder else "idle"
                self.error = str(exc)
            if self.recorder is None and self.policy is not None:
                self.policy.close()
                self.policy = None
            raise
        finally:
            if locked:
                self.session._deploy_lock.release()
            self._operation.release()

    def end(self):
        # Cancel immediately, even while inference is blocked in another thread.
        with self._state:
            if self.phase in ("stopping", "await_label"):
                return {"phase": self.phase}
            if self.recorder is None and self.phase != "busy":
                raise RuntimeError("没有正在记录的 episode")
            self.phase = "stopping"
            self._stop_requested.set()
            if self.controller is not None:
                self.controller.stop()
            self._end_thread = threading.Thread(target=self._end_worker, daemon=True)
            self._end_thread.start()
        return {"phase": "stopping"}

    def _end_worker(self):
        with self._operation, self.session._deploy_lock:
            try:
                if self.controller is None or self.controller.failed:
                    raise RuntimeError("本轮发生错误，不自动回位；数据保留，可标记或 drop")
                if self.session._estop_generation != self.generation or self.session._estopped:
                    raise RuntimeError("已急停，不自动重新使能/回位；数据保留，可标记或 drop")
                if self.backend.loop is not None:
                    self.backend.loop.stop()
                self._return_home()
                self.error = None
            except Exception as exc:
                self.error = f"回位未完成：{exc}"
            finally:
                try:
                    if self.recorder is not None:
                        write_json(self.recorder.path / "end_episode.json", {"return_error": self.error,
                                   "return_pose": self.home, "label_pending": True})
                except Exception as exc:
                    self.error = f"{self.error or ''} 结束记录写入失败：{exc}"
                try:
                    if self.policy is not None:
                        self.policy.close()
                except Exception as exc:
                    self.error = f"{self.error or ''} 连接关闭失败：{exc}"
                finally:
                    self.policy = None
                    with self._state:
                        self.phase = "await_label" if self.recorder else "idle"

    def _return_home(self):
        from yam_abc_reproduce.deploy.loop import DeployLoop
        from yam_abc_reproduce.deploy.contract import build_state
        session = self.session
        if not session.live:
            raise RuntimeError("硬件会话已关闭")
        if self.home is None:
            raise RuntimeError("尚未取得本轮起始姿态，没有可用的回位目标")
        # Freeze the pose before the FIRST inference, never the most recent chunk.
        speed = min(self.recorder.metadata["max_joint_speed"], 0.3)
        loop = DeployLoop(session.units, session.workers, None, prompt="", control_hz=session.cfg.control_hz,
                          max_joint_speed=speed)
        session.deploy_loop = loop
        path = self.recorder.path / "return_home.json"
        write_json(path, {"status": "started", "target": self.home})
        try:
            session._check_deploy_generation(self.generation)
            loop.move_to_home(self.home)
            deadline = time.monotonic() + 3
            joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            while True:
                session._check_deploy_generation(self.generation)
                _, feedback = loop._read()
                measured = build_state(feedback)
                difference = np.abs(measured - self.home)
                if np.all(difference[joints] <= 0.05) and np.all(difference[[6, 13]] <= 0.1):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("已发送 home 目标，但反馈尚未到达容差范围")
                time.sleep(0.05)
            write_json(path, {"status": "returned", "target": self.home, "measured": measured.tolist(),
                              "joint_tolerance_rad": 0.05, "gripper_tolerance": 0.1})
        except Exception as exc:
            write_json(path, {"status": "error", "target": self.home, "error": str(exc)})
            raise
        finally:
            loop.stop()
            # Keep the loop reservation until the label/drop decision, preventing
            # another policy rollout from interleaving with an unlabeled episode.

    def label(self, episode_id, decision):
        if decision not in ("y", "n", "drop"):
            raise ValueError("结果只能为 y、n 或 drop")
        if not self._operation.acquire(blocking=False):
            raise RuntimeError("episode 仍在结束中")
        try:
            with self._state:
                if self.phase != "await_label" or self.recorder.metadata["episode_id"] != episode_id:
                    raise RuntimeError("episode 标识或状态不匹配")
                path = str(self.recorder.path)
                if decision == "drop":
                    self.recorder.discard()
                else:
                    write_json(self.recorder.path / "end_episode.json", {"return_error": self.error,
                               "return_pose": self.home, "label_pending": False, "decision": decision})
                    self.recorder.finish(1 if decision == "y" else 0, "operator_gui_" + decision)
                self.session.deploy_loop = None
                self.recorder = self.controller = self.backend = None
                self.home = None
                self.phase = "idle"
                self.last_result = {"decision": decision, "path": path}
                return self.last_result
        finally:
            self._operation.release()
