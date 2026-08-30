"""Render LIBERO rollouts with live EDL trajectory-verifier predictions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from .rejection_inference import FrozenVerifier, load_frozen_verifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY_CHECKPOINT = (
    PROJECT_ROOT
    / "playground/Checkpoints/qwen3fast_libero_all_edl_1e-2/checkpoints/steps_30000_pytorch_model.pt"
)
DEFAULT_VERIFIER_CHECKPOINT = (
    PROJECT_ROOT
    / "examples/LIBERO/edl_pred/outputs/sweeps/all_suites_seed7_v1/runs"
    / "all_mlp_flat_edl_seed7/checkpoints/best.pt"
)
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass(frozen=True)
class VerifierDisplay:
    """Latest EDL verifier prediction for the observed chunk prefix."""

    au: float
    failure_probability: float
    success_probability: float


class OnlineEDLVerifier:
    """Accumulate source-policy AU/EU chunks and score every observed prefix."""

    def __init__(self, frozen: FrozenVerifier) -> None:
        if frozen.model_config.head != "edl":
            raise ValueError("video overlay requires an EDL verifier checkpoint")
        self.frozen = frozen
        self._chunks: list[np.ndarray] = []
        self.latest: VerifierDisplay | None = None

    def reset(self) -> None:
        self._chunks.clear()
        self.latest = None

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @torch.inference_mode()
    def append_chunk(self, uncertainty: Mapping[str, Any]) -> VerifierDisplay:
        au = self._token_values(uncertainty, "action_token_aleatoric_uncertainty")
        eu = self._token_values(uncertainty, "action_token_epistemic_uncertainty")
        if au.size == 0 or au.size != eu.size:
            raise ValueError("action-token AU and EU must have the same non-zero length")
        if au.size > self.frozen.max_action_tokens:
            raise ValueError(
                "action-token count exceeds verifier max_action_tokens: "
                f"{au.size} > {self.frozen.max_action_tokens}"
            )

        self._chunks.append(np.stack((au, eu), axis=-1))
        chunk_count = len(self._chunks)
        max_tokens = self.frozen.max_action_tokens
        features = torch.zeros(
            (1, chunk_count, max_tokens, 2),
            dtype=torch.float32,
            device=self.frozen.device,
        )
        token_mask = torch.zeros(
            (1, chunk_count, max_tokens),
            dtype=torch.bool,
            device=self.frozen.device,
        )
        for chunk_idx, chunk in enumerate(self._chunks):
            token_count = chunk.shape[0]
            features[0, chunk_idx, :token_count] = torch.as_tensor(
                chunk,
                dtype=torch.float32,
                device=self.frozen.device,
            )
            token_mask[0, chunk_idx, :token_count] = True
        chunk_lengths = torch.tensor([chunk_count], dtype=torch.long, device=self.frozen.device)

        output = self.frozen.model(features, token_mask, chunk_lengths)
        if output.verifier_au is None:
            raise ValueError("EDL verifier did not return verifier_au")
        probabilities = output.probabilities[0, chunk_count - 1]
        self.latest = VerifierDisplay(
            au=float(output.verifier_au[0, chunk_count - 1].item()),
            failure_probability=float(probabilities[0].item()),
            success_probability=float(probabilities[1].item()),
        )
        return self.latest

    @staticmethod
    def _token_values(uncertainty: Mapping[str, Any], key: str) -> np.ndarray:
        if key not in uncertainty:
            raise ValueError(f"policy uncertainty record is missing {key!r}")
        values = np.asarray(uncertainty[key], dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"policy uncertainty field {key!r} contains non-finite values")
        if np.any(values < 0.0):
            raise ValueError(f"policy uncertainty field {key!r} contains negative values")
        return values


def format_overlay_lines(display: VerifierDisplay | None) -> tuple[str, str, str]:
    if display is None:
        return ("AU: --", "Failure: --", "Success: --")
    return (
        f"AU: {display.au:.3f}",
        f"Failure: {display.failure_probability:.3f}",
        f"Success: {display.success_probability:.3f}",
    )


def overlay_verifier_text(frame: np.ndarray, display: VerifierDisplay | None) -> np.ndarray:
    """Return a copy of an RGB frame with a compact top-right text overlay."""
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("frame must be an HWC uint8 RGB array")
    image = Image.fromarray(array.copy(), mode="RGB")
    draw = ImageDraw.Draw(image)
    font_size = max(14, int(round(image.height * 0.055)))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    lines = format_overlay_lines(display)
    padding = max(5, font_size // 3)
    line_gap = max(2, font_size // 6)
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    widths = [box[2] - box[0] for box in boxes]
    heights = [box[3] - box[1] for box in boxes]
    panel_width = max(widths) + 2 * padding
    panel_height = sum(heights) + (len(lines) - 1) * line_gap + 2 * padding
    panel_x = max(0, image.width - panel_width - padding)
    panel_y = padding
    draw.rectangle(
        (panel_x, panel_y, image.width - padding, panel_y + panel_height),
        fill=(255, 255, 255),
    )
    y = panel_y + padding
    for line, height in zip(lines, heights):
        draw.text((panel_x + padding, y), line, fill=(0, 0, 0), font=font)
        y += height + line_gap
    return np.asarray(image, dtype=np.uint8)


def validate_server_checkpoint(
    server_metadata: Mapping[str, Any],
    expected_checkpoint: str | Path,
) -> Path:
    server_checkpoint = server_metadata.get("ckpt_path")
    if not server_checkpoint:
        raise ValueError("policy server metadata does not include ckpt_path")
    expected = Path(expected_checkpoint).expanduser().resolve()
    actual = Path(str(server_checkpoint)).expanduser().resolve()
    if actual != expected:
        raise ValueError(
            f"checkpoint mismatch: renderer requested {expected}, policy server loaded {actual}"
        )
    return actual


@dataclass(frozen=True)
class RenderArgs:
    host: str
    port: int
    task_suite_name: str
    task_id: int
    num_videos: int
    episode_start_index: int
    num_steps_wait: int
    seed: int
    fps: int
    policy_checkpoint: Path
    verifier_checkpoint: Path
    verifier_device: str
    output_dir: Path
    unnorm_key: str | None
    overwrite: bool


def _parse_args(argv: Sequence[str] | None = None) -> RenderArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6694)
    parser.add_argument("--task-suite-name", choices=tuple(MAX_STEPS_BY_SUITE), default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--episode-start-index", type=int, default=0)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--policy-checkpoint", type=Path, default=DEFAULT_POLICY_CHECKPOINT)
    parser.add_argument("--verifier-checkpoint", type=Path, default=DEFAULT_VERIFIER_CHECKPOINT)
    parser.add_argument("--verifier-device", default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--unnorm-key")
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args(argv)
    policy_checkpoint = parsed.policy_checkpoint.expanduser().resolve()
    output_dir = parsed.output_dir
    if output_dir is None:
        model_root = policy_checkpoint.parent.parent
        output_dir = (
            model_root
            / "results/edl_pred_videos"
            / parsed.task_suite_name
            / f"task_{parsed.task_id:02d}"
        )
    return RenderArgs(
        host=parsed.host,
        port=parsed.port,
        task_suite_name=parsed.task_suite_name,
        task_id=parsed.task_id,
        num_videos=parsed.num_videos,
        episode_start_index=parsed.episode_start_index,
        num_steps_wait=parsed.num_steps_wait,
        seed=parsed.seed,
        fps=parsed.fps,
        policy_checkpoint=policy_checkpoint,
        verifier_checkpoint=parsed.verifier_checkpoint.expanduser().resolve(),
        verifier_device=parsed.verifier_device,
        output_dir=output_dir.expanduser().resolve(),
        unnorm_key=parsed.unnorm_key,
        overwrite=parsed.overwrite,
    )


def _validate_args(args: RenderArgs) -> None:
    if args.task_id < 0:
        raise ValueError("task_id must be non-negative")
    if args.num_videos <= 0:
        raise ValueError("num_videos must be positive")
    if args.episode_start_index < 0:
        raise ValueError("episode_start_index must be non-negative")
    if args.num_steps_wait < 0:
        raise ValueError("num_steps_wait must be non-negative")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if not args.policy_checkpoint.is_file():
        raise FileNotFoundError(f"policy checkpoint does not exist: {args.policy_checkpoint}")
    if not args.verifier_checkpoint.is_file():
        raise FileNotFoundError(f"verifier checkpoint does not exist: {args.verifier_checkpoint}")


def _reserve_outputs(args: RenderArgs) -> tuple[list[Path], Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_paths = [
        args.output_dir / f"task_{args.task_id:02d}_episode_{episode_idx:03d}.mp4"
        for episode_idx in range(
            args.episode_start_index,
            args.episode_start_index + args.num_videos,
        )
    ]
    manifest_path = args.output_dir / "manifest.json"
    for path in [*video_paths, manifest_path]:
        if path.exists() and (not args.overwrite or not path.is_file() or path.is_symlink()):
            raise FileExistsError(f"output already exists: {path}; use --overwrite for regular files")
    return video_paths, manifest_path


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def render_videos(args: RenderArgs) -> Path:
    """Run one LIBERO task and return the completed manifest path."""
    _validate_args(args)
    video_paths, manifest_path = _reserve_outputs(args)

    import imageio
    from libero.libero import benchmark

    from examples.LIBERO.eval_files.eval_libero import (
        LIBERO_DUMMY_ACTION,
        LIBERO_ENV_RESOLUTION,
        _binarize_gripper_open,
        _get_libero_env,
    )
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient

    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if args.task_id >= task_suite.n_tasks:
        raise ValueError(
            f"task_id {args.task_id} is outside {args.task_suite_name} range [0, {task_suite.n_tasks})"
        )
    initial_states = task_suite.get_task_init_states(args.task_id)
    episode_stop = args.episode_start_index + args.num_videos
    if episode_stop > len(initial_states):
        raise ValueError(
            f"task has {len(initial_states)} initial states, but requested indices "
            f"[{args.episode_start_index}, {episode_stop})"
        )

    client = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
    )
    server_checkpoint = validate_server_checkpoint(client.server_metadata, args.policy_checkpoint)
    frozen = load_frozen_verifier(args.verifier_checkpoint, device=args.verifier_device)
    online = OnlineEDLVerifier(frozen)
    task = task_suite.get_task(args.task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    max_steps = MAX_STEPS_BY_SUITE[args.task_suite_name]
    episodes: list[dict[str, Any]] = []

    try:
        for video_index, episode_idx in enumerate(
            range(args.episode_start_index, episode_stop)
        ):
            target_path = video_paths[video_index]
            temporary_path = target_path.with_name(f".{target_path.stem}.tmp.mp4")
            if temporary_path.exists():
                raise FileExistsError(f"temporary output already exists: {temporary_path}")
            client.reset(task_description=task_description)
            online.reset()
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            done = False
            frame_count = 0
            policy_step = 0
            env_step = 0
            writer = imageio.get_writer(temporary_path, fps=args.fps)
            try:
                while env_step < max_steps + args.num_steps_wait:
                    if env_step < args.num_steps_wait:
                        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                        env_step += 1
                        continue

                    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_image = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )
                    response = client.step(
                        example={"image": [image, wrist_image], "lang": task_description},
                        step=policy_step,
                    )
                    if response.get("new_chunk"):
                        uncertainty = response.get("uncertainty")
                        if uncertainty is None:
                            raise RuntimeError(
                                "policy returned a new action chunk without AU/EU uncertainty"
                            )
                        online.append_chunk(uncertainty)

                    writer.append_data(overlay_verifier_text(image, online.latest))
                    frame_count += 1
                    raw_action = response["raw_action"]
                    world_vector = np.asarray(
                        raw_action["world_vector"], dtype=np.float32
                    ).reshape(-1)
                    rotation_delta = np.asarray(
                        raw_action["rotation_delta"], dtype=np.float32
                    ).reshape(-1)
                    open_gripper = np.asarray(
                        raw_action["open_gripper"], dtype=np.float32
                    ).reshape(-1)
                    if world_vector.size != 3 or rotation_delta.size != 3 or open_gripper.size != 1:
                        raise ValueError(
                            "policy action must contain 3D translation, 3D rotation, and 1D gripper"
                        )
                    action = np.concatenate(
                        (world_vector, rotation_delta, _binarize_gripper_open(open_gripper))
                    )
                    obs, _, done, _ = env.step(action.tolist())
                    policy_step += 1
                    env_step += 1
                    if done:
                        break
            except BaseException:
                writer.close()
                temporary_path.unlink(missing_ok=True)
                raise
            else:
                writer.close()
                temporary_path.replace(target_path)

            latest = online.latest
            episodes.append(
                {
                    "episode_index": episode_idx,
                    "video": target_path.name,
                    "success": bool(done),
                    "frame_count": frame_count,
                    "action_chunk_count": online.chunk_count,
                    "final_verifier": None if latest is None else asdict(latest),
                }
            )
            logging.info(
                "Rendered %s (%s, %d frames)",
                target_path,
                "success" if done else "failure",
                frame_count,
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    manifest = {
        "task_suite": args.task_suite_name,
        "task_id": args.task_id,
        "task_description": task_description,
        "seed": args.seed,
        "policy_checkpoint": str(server_checkpoint),
        "verifier_checkpoint": str(frozen.checkpoint_path),
        "verifier_checkpoint_sha256": frozen.checkpoint_sha256,
        "verifier_split_manifest_identity": frozen.split_manifest_identity,
        "class_order": ["failure", "success"],
        "action_chunk_size": client.action_chunk_size,
        "episodes": episodes,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    manifest = render_videos(args)
    logging.info("Saved five-video verifier render manifest: %s", manifest)


if __name__ == "__main__":
    main()
