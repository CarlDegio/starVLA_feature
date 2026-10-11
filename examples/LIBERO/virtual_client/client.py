"""Replay LeRobot real episodes through the YAM policy protocol, without hardware."""
import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import time

import av
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from deployment.real.infer import RecordedClient, ROOT, TASKS
from deployment.real.yam_policy import STAR_TO_YAM
from examples.LIBERO.edl_pred_real.recording import write_json

ROLES = ("top", "left", "right")
STAR_NAMES = ([f"left_joint_{i}" for i in range(6)] + [f"right_joint_{i}" for i in range(6)]
              + ["left_gripper", "right_gripper"])
NAMES = [STAR_NAMES[i] for i in STAR_TO_YAM]
JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


class Episode:
    def __init__(self, root, index):
        self.root, self.index = Path(root), index
        self.info = json.loads((self.root / "meta/info.json").read_text())
        info = self.info
        if info["codebase_version"] != "v2.1" or info["robot_type"] != "edl_real_dual_arm":
            raise ValueError("Expected converted edl_real LeRobot v2.1 dataset")
        for field in ("action", "observation.state"):
            if info["features"][field]["names"] != STAR_NAMES:
                raise ValueError(f"Unexpected {field} order; refusing to compare mismatched joints")
        self.fps = float(info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Dataset fps must be positive")
        entries = [json.loads(line) for line in (self.root / "meta/episodes.jsonl").read_text().splitlines() if line]
        self.entry = next((item for item in entries if item["episode_index"] == index), None)
        if self.entry is None:
            raise ValueError(f"Episode {index} does not exist")
        self.format_args = {"episode_chunk": index // info["chunks_size"], "episode_index": index}
        table = pq.read_table(self.root / info["data_path"].format(**self.format_args))
        self.length = len(table)
        if self.length != self.entry["length"] or not self.length:
            raise ValueError("Episode length does not match metadata")
        for name, expected in (("frame_index", np.arange(self.length)),
                               ("episode_index", np.full(self.length, index))):
            if not np.array_equal(table[name].to_numpy(), expected):
                raise ValueError(f"Unexpected {name}; cannot align images with GT")
        self.timestamps = table["timestamp"].to_numpy()
        np.testing.assert_allclose(self.timestamps, np.arange(self.length) / self.fps, atol=1e-5, rtol=1e-6)
        self.states, self.actions = [np.asarray(table[key].to_pylist(), dtype=np.float32)[:, STAR_TO_YAM]
                                    for key in ("observation.state", "action")]
        if any(a.shape != (self.length, 14) or not np.isfinite(a).all() for a in (self.states, self.actions)):
            raise ValueError("Expected finite (N,14) raw states and absolute GT actions")
        tasks = {t["task_index"]: t["task"] for t in map(json.loads, (self.root / "meta/tasks.jsonl").read_text().splitlines())}
        self.prompts = [tasks[int(i)] for i in table["task_index"].to_numpy()]
        self.source_frames = (table["source_frame_index"].to_numpy() if "source_frame_index" in table.column_names
                              else np.arange(self.length))
        self._videos, self._frames = {}, {}
        self._position = -1

    def observation(self, frame):
        if not self._position < frame < self.length:
            raise ValueError("Video frames must advance within the episode")
        if not self._videos:
            for role in ROLES:
                path = self.root / self.info["video_path"].format(
                    **self.format_args, video_key=f"observation.images.{role}")
                self._videos[role] = av.open(str(path))
                self._frames[role] = iter(self._videos[role].decode(video=0))
        images = {}
        for role in ROLES:
            try:
                for _ in range(self._position + 1, frame + 1):
                    decoded = next(self._frames[role])
            except StopIteration as exc:
                raise ValueError(f"{role} video ends before frame {frame}") from exc
            images[role] = decoded.to_ndarray(format="rgb24")
        self._position = frame
        return {"images": images, "state": self.states[frame].copy(), "prompt": self.prompts[frame]}

    def close(self):
        for video in self._videos.values():
            video.close()


def errors(prediction, gt):
    delta = np.asarray(prediction, dtype=np.float64) - gt
    return {"joint_mae_rad": float(np.abs(delta[:, JOINTS]).mean()),
            "joint_rmse_rad": float(np.sqrt(np.square(delta[:, JOINTS]).mean())),
            "joint_max_error_rad": float(np.abs(delta[:, JOINTS]).max()),
            "gripper_mae": float(np.abs(delta[:, [6, 13]]).mean()),
            "mae_per_dim": np.abs(delta).mean(axis=0).tolist()}


def save_charts(output, frames, prediction, gt, fps, starts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(7, 2, figsize=(16, 19), sharex=True)
    for col, ax in enumerate(axes.flat):
        ax.plot(frames / fps, gt[:, col], label="GT action", linewidth=1.3)
        ax.plot(frames / fps, prediction[:, col], label="Prediction", linewidth=1.1, alpha=.85)
        if len(starts) <= 100:
            for frame in starts[1:]:
                ax.axvline(frame / fps, color="gray", alpha=.15)
        ax.set_title(NAMES[col])
        ax.set_ylabel("normalized gripper" if col in (6, 13) else "rad")
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("Dataset time (s)")
    fig.suptitle("Recorded episode: predicted absolute actions vs GT (no robot execution)")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=130)
    plt.close(fig)


def save_page(output, summary, chunks):
    parts = ['<!doctype html><meta charset="utf-8"><title>离线轨迹比较</title>',
             '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:15px}img{max-width:100%}pre{overflow:auto}details{margin:15px 0}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #bbb}</style>',
             '<h1>预测 action / GT action 离线比较</h1>',
             '<p>按录制轨迹读取图像和 GT state；预测动作不会改变下一次观测。不代表闭环成功率。</p>',
             '<pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>',
             '<a href="comparison.png"><img src="comparison.png" alt="14维动作曲线"></a>',
             '<p><a href="comparison.csv">全部有效帧的 GT 与预测 CSV</a> · <a href="comparison.npz">NPZ</a></p>']
    for chunk in chunks:
        i = chunk["chunk_index"]
        parts.append(f'<details><summary>Chunk {i + 1} · 帧 {chunk["frame_start"]} · '
                     f'joint MAE {chunk["joint_mae_rad"]:.5f} rad · '
                     f'chunk 内完全恒定：{chunk["all_actions_identical"]} · '
                     f'与上次输出完全相同：{chunk["same_as_previous_prediction"]}</summary>')
        parts.append(f'<p>top / left / right</p><img loading="lazy" src="chunk_{i:06d}_images.png">')
        parts.append('<pre>' + html.escape(json.dumps(chunk, ensure_ascii=False, indent=2)) + '</pre>')
        with np.load(output / f"chunk_{i:06d}.npz") as data:
            parts.append('<table><tr><th>帧</th><th>预测（YAM顺序）</th><th>GT（YAM顺序）</th></tr>')
            for step, row in enumerate(data["predicted_actions"]):
                ground = data["gt_actions"][step].tolist() if step < len(data["gt_actions"]) else "超出 episode，不评分"
                parts.append(f'<tr><td>{chunk["frame_start"] + step}</td><td>{row.tolist()}</td><td>{ground}</td></tr>')
            parts.append('</table></details>')
    (output / "report.html").write_text("\n".join(parts), encoding="utf-8")


def replay(client, episode, output, *, task, start_frame=0, max_chunks=None, step=False, plots=True):
    meta = client.metadata
    horizon = meta.get("horizon")
    expected_keys = ["action.left_joints", "action.left_gripper", "action.right_joints", "action.right_gripper"]
    if (meta.get("backend") != "starvla" or meta.get("task") != task
            or not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1
            or meta.get("action_dim") != 14 or meta.get("state_dim") != 14
            or meta.get("action_keys") != expected_keys or meta.get("action_space") != "absolute_joint_radians"
            or not np.isclose(meta.get("control_hz", 0), episode.fps)
            or meta.get("action_stride") != 1 or meta.get("execution_speed") != 1):
        raise ValueError("Policy task/layout/frequency must match the edl_real dataset")
    if not 0 <= start_frame < episode.length or (max_chunks is not None and max_chunks < 1):
        raise ValueError("Invalid start-frame or max-chunks")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "dataset": str(episode.root.resolve()), "episode_index": episode.index,
               "episode_metadata": episode.entry, "server_metadata": meta, "fps": episode.fps,
               "frame_stride": horizon, "model_horizon": meta.get("model_horizon", horizon),
               "action_order": NAMES, "start_frame": start_frame, "robot_commands_sent": False,
               "observation_source": "recorded_images_and_GT_state", "split": episode.info.get("splits")}
    write_json(output / "summary.json", summary)
    print(f"Model horizon={summary['model_horizon']}; returned actions / frame stride={horizon}", flush=True)
    chunks, predictions, truths, indices, states = [], [], [], [], []
    previous, previous_state, previous_hashes = None, None, None
    reason = "episode_end"
    try:
        for index, frame in enumerate(range(start_frame, episode.length, horizon)):
            obs = episode.observation(frame)
            obs["return_edl_details"] = bool(meta.get("edl_details_available", False))
            hashes = {role: hashlib.sha256(obs["images"][role].tobytes()).hexdigest() for role in ROLES}
            started = time.perf_counter()
            result = client.infer(obs)  # One request at a time. Never use predictions as observations.
            elapsed = time.perf_counter() - started
            pred = np.asarray(result["actions"], dtype=np.float32)
            if pred.shape != (horizon, 14) or not np.isfinite(pred).all():
                raise ValueError(f"Invalid prediction shape/values at frame {frame}: {pred.shape}")
            valid = min(horizon, episode.length - frame)
            gt = episode.actions[frame:frame + valid]
            chunk = {"chunk_index": index, "frame_start": frame, "valid_steps": valid,
                     "source_frame": int(episode.source_frames[frame]), "prompt": obs["prompt"],
                     "latency_s": elapsed, **errors(pred[:valid], gt),
                     "all_actions_identical": bool(np.all(pred == pred[:1])),
                     "max_model_joint_speed_rad_s": float(np.abs(np.diff(pred[:, JOINTS], axis=0)).max() * episode.fps) if horizon > 1 else 0.,
                     "same_as_previous_prediction": None if previous is None else bool(np.array_equal(pred, previous)),
                     "max_change_from_previous_prediction": None if previous is None else float(np.abs(pred - previous).max()),
                     "raw_state_changed": None if previous_state is None else not np.array_equal(obs["state"], previous_state),
                     "image_sha256": hashes,
                     "image_changed": None if previous_hashes is None else {r: hashes[r] != previous_hashes[r] for r in ROLES}}
            extra = {"diagnostic__" + key: np.asarray(value) for key, value in result.get("diagnostics", {}).items()
                     if np.asarray(value).dtype.kind in "biuf"}
            np.savez_compressed(output / f"chunk_{index:06d}.npz", predicted_actions=pred, gt_actions=gt,
                                state=obs["state"], frame_indices=np.arange(frame, frame + valid),
                                **{f"image_{r}": obs["images"][r] for r in ROLES}, **extra)
            images = [Image.fromarray(obs["images"][r]) for r in ROLES]
            strip = Image.new("RGB", (sum(im.width for im in images), max(im.height for im in images)))
            x = 0
            for im in images:
                strip.paste(im, (x, 0)); x += im.width
            strip.save(output / f"chunk_{index:06d}_images.png")
            with (output / "chunks.jsonl").open("a") as stream:
                stream.write(json.dumps(chunk, ensure_ascii=False, allow_nan=False) + "\n")
            chunks.append(chunk)
            predictions.append(pred[:valid].copy()); truths.append(gt.copy())
            indices.append(np.arange(frame, frame + valid)); states.append(episode.states[frame:frame + valid])
            previous, previous_state, previous_hashes = pred.copy(), obs["state"].copy(), hashes
            print(f"chunk {index + 1}: frames {frame}:{frame + valid}, joint MAE={chunk['joint_mae_rad']:.6f} rad, "
                  f"gripper MAE={chunk['gripper_mae']:.6f}, constant={chunk['all_actions_identical']}, "
                  f"same_previous={chunk['same_as_previous_prediction']}", flush=True)
            if frame + valid == episode.length:
                break
            if max_chunks is not None and len(chunks) >= max_chunks:
                reason = "max_chunks"; break
            if step and input("Enter: next chunk; q: finish report > ").strip().lower() == "q":
                reason = "operator_stop"; break
        prediction, gt, frames = np.concatenate(predictions), np.concatenate(truths), np.concatenate(indices)
        np.savez_compressed(output / "comparison.npz", predicted_actions=prediction, gt_actions=gt,
                            frame_indices=frames, gt_states=np.concatenate(states), action_names=np.array(NAMES))
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "time_s"] + [f"pred_{n}" for n in NAMES] + [f"gt_{n}" for n in NAMES])
            for frame, pred_row, gt_row in zip(frames, prediction, gt):
                writer.writerow([int(frame), frame / episode.fps, *pred_row.tolist(), *gt_row.tolist()])
        summary.update(status="completed", stop_reason=reason, episode_complete=reason == "episode_end",
                       chunks=len(chunks), compared_frames=len(frames), **errors(prediction, gt),
                       constant_chunks=sum(c["all_actions_identical"] for c in chunks),
                       repeated_chunks=sum(c["same_as_previous_prediction"] is True for c in chunks))
        if plots:
            save_charts(output, frames, prediction, gt, episode.fps, [c["frame_start"] for c in chunks])
            save_page(output, summary, chunks)
        write_json(output / "summary.json", summary)
        return summary
    except BaseException as exc:
        summary.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, EOFError)) else "error",
                       error=str(exc), chunks=len(chunks))
        write_json(output / "summary.json", summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="slippers")
    parser.add_argument("--data-root", type=Path, default=ROOT / "playground/Datasets/edl_real")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--step", action="store_true", help="Wait for Enter after each chunk")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--invalid-token-retries", type=int, default=3,
                        help="Additional requests after invalid FAST Unicode; same recorded observation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.invalid_token_retries < 0:
        parser.error("invalid-token-retries must be nonnegative")
    root = args.data_root if (args.data_root / "meta/info.json").is_file() else args.data_root / TASKS[args.task][0]
    if root.name != TASKS[args.task][0]:
        parser.error("data-root subset name must match --task")
    output = args.output or Path(__file__).parent / "outputs" / f"{args.task}_ep{args.episode:06d}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    episode = Episode(root, args.episode)
    client = None
    try:
        client = RecordedClient(args.host, args.port, args.timeout,
                                invalid_token_retries=args.invalid_token_retries)
        summary = replay(client, episode, output, task=args.task, start_frame=args.start_frame,
                         max_chunks=args.max_chunks, step=args.step)
        print(json.dumps({"output": str(output.resolve()), "report": str((output / 'report.html').resolve()),
                          "joint_mae_rad": summary["joint_mae_rad"], "chunks": summary["chunks"]}), flush=True)
    finally:
        episode.close()
        if client is not None:
            client.close()


if __name__ == "__main__":
    main()
