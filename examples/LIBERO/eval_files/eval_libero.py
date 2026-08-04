import dataclasses
import atexit
import json
import logging
import math
import os
import pathlib
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio
import matplotlib
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    save_artifacts: bool = True
    dataset_output_path: str | None = None
    dataset_overwrite: bool = False
    dataset_resume: bool = False
    collection_id: str | None = None
    seed_namespace: str | None = None

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True

    job_name: str = "test"


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    if not args.save_artifacts and args.dataset_output_path is None:
        raise ValueError("dataset_output_path is required when save_artifacts is disabled")
    if (args.dataset_overwrite or args.dataset_resume) and args.dataset_output_path is None:
        raise ValueError("dataset overwrite/resume requires dataset_output_path")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    if args.save_artifacts:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
    )
    dataset_writer = _create_dataset_writer(args, client_model, max_steps)
    if dataset_writer is not None:
        atexit.register(dataset_writer.close)

    # Optional smoke-test cap (still useful for quick verification with -1 = full run).
    n_eval_tasks = num_tasks_in_suite if args.max_tasks <= 0 else min(args.max_tasks, num_tasks_in_suite)
    logging.info(f"Evaluating {n_eval_tasks} of {num_tasks_in_suite} tasks (max_tasks={args.max_tasks})")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(n_eval_tasks)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            if dataset_writer is not None and dataset_writer.has_episode(task_id, episode_idx):
                logging.info(
                    "Skipping collected episode task_id=%d episode_idx=%d",
                    task_id,
                    episode_idx,
                )
                continue
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []
            uncertainty_chunks = []
            executed_steps = 0
            done = False

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Save preprocessed image for replay video
                if args.save_artifacts:
                    replay_images.append(img)

                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                )

                observation = {  #
                    "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                    "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                    "observation.state": np.expand_dims(state, axis=0),
                    "instruction": [str(task_description)],
                }

                # align key with model API --> two images provided here --> check training
                example_dict = {
                    "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                    "lang": observation["instruction"][0],
                }

                start_time = time.time()

                response = client_model.step(example=example_dict, step=step)
                if response.get("new_chunk") and response.get("uncertainty") is not None:
                    uncertainty_record = dict(response["uncertainty"])
                    uncertainty_record["env_step"] = int(t)
                    uncertainty_record["policy_step"] = int(step)
                    uncertainty_chunks.append(uncertainty_record)

                end_time = time.time()
                # print(f"time: {end_time - start_time}")

                # #
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(
                        f"Unexpected action sizes: "
                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                        f"Falling back to LIBERO_DUMMY_ACTION."
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                executed_steps += 1
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            if args.save_artifacts:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                rollout_base = (
                    pathlib.Path(args.video_out_path)
                    / f"rollout_{task_segment}_episode{episode_idx}_{suffix}"
                )
                imageio.mimwrite(
                    rollout_base.parent / f"{rollout_base.name}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                _save_uncertainty_artifacts(
                    uncertainty_chunks,
                    rollout_base,
                    max_steps=max_steps,
                    action_chunk_size=client_model.action_chunk_size,
                )

            if dataset_writer is not None:
                dataset_writer.append_episode(
                    task_id=task_id,
                    episode_idx=episode_idx,
                    task_description=task_description,
                    success=bool(done),
                    executed_steps=executed_steps,
                    termination_reason="success" if done else "max_steps",
                    uncertainty_chunks=uncertainty_chunks,
                )

            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        if task_episodes > 0:
            logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        else:
            logging.info("No new episodes evaluated for this task")
        if total_episodes > 0:
            logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    if total_episodes > 0:
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    else:
        logging.info("No new episodes evaluated")
    logging.info(f"Total episodes: {total_episodes}")
    if dataset_writer is not None:
        dataset_writer.close()


def _create_dataset_writer(args: Args, client_model: ModelClient, max_steps: int):
    if args.dataset_output_path is None:
        return None
    if not args.pretrained_path:
        raise ValueError("pretrained_path is required for dataset provenance")
    collection_id = "" if args.collection_id is None else args.collection_id.strip()
    seed_namespace = "" if args.seed_namespace is None else args.seed_namespace.strip()
    if not collection_id or not seed_namespace:
        raise ValueError("collection_id and seed_namespace are required for dataset collection")

    server_metadata = client_model.server_metadata
    server_checkpoint = server_metadata.get("ckpt_path")
    if not server_checkpoint:
        raise ValueError("policy server metadata does not include ckpt_path")
    requested_checkpoint = pathlib.Path(args.pretrained_path).expanduser().resolve()
    actual_checkpoint = pathlib.Path(server_checkpoint).expanduser().resolve()
    if requested_checkpoint != actual_checkpoint:
        raise ValueError(
            f"checkpoint mismatch: collector requested {requested_checkpoint}, "
            f"policy server loaded {actual_checkpoint}"
        )

    from examples.LIBERO.eval_files.libero_uncertainty_dataset import (
        LiberoUncertaintyDatasetWriter,
    )

    metadata = {
        "checkpoint_path": str(requested_checkpoint),
        "server_checkpoint_path": str(actual_checkpoint),
        "task_suite": args.task_suite_name,
        "collection_id": collection_id,
        "seed_namespace": seed_namespace,
        "seed": args.seed,
        "max_steps": max_steps,
        "action_chunk_size": client_model.action_chunk_size,
        "server_metadata": server_metadata,
    }
    return LiberoUncertaintyDatasetWriter(
        args.dataset_output_path,
        metadata,
        overwrite=args.dataset_overwrite,
        resume=args.dataset_resume,
    )


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _save_uncertainty_artifacts(
    uncertainty_chunks: list[dict],
    rollout_base: pathlib.Path,
    max_steps: int,
    action_chunk_size: int,
) -> None:
    """Save chunk/token uncertainty as JSONL and a compact diagnostic plot."""
    if action_chunk_size <= 0:
        raise ValueError("action_chunk_size must be positive")

    jsonl_path = rollout_base.parent / f"{rollout_base.name}.jsonl"
    png_path = rollout_base.parent / f"{rollout_base.name}.png"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in uncertainty_chunks:
            f.write(json.dumps(record) + "\n")

    fig, axes = plt.subplots(2, 4, figsize=(24, 9), sharex=False)
    axes[0, 3].axis("off")

    au_token_x, au_token_y = [], []
    eu_token_x, eu_token_y = [], []
    evidence_token_x, evidence_token_y = [], []
    chunk_x = []
    action_token_count_y, low_evidence_count_y, low_evidence_ratio_y = [], [], []
    top_eu_20pct_y = []
    quadrant_au, quadrant_eu, quadrant_is_low_evidence, quadrant_chunk_idx = [], [], [], []
    for record in uncertainty_chunks:
        chunk_idx = int(record.get("chunk_idx", len(chunk_x)))
        action_token_au = np.asarray(
            record.get("action_token_aleatoric_uncertainty", []), dtype=np.float32
        ).reshape(-1)
        action_token_eu = np.asarray(
            record.get("action_token_epistemic_uncertainty", []), dtype=np.float32
        ).reshape(-1)
        action_token_evidence = np.asarray(record.get("action_token_evidence", []), dtype=np.float32).reshape(-1)
        action_token_au = action_token_au[np.isfinite(action_token_au)]
        action_token_eu = action_token_eu[np.isfinite(action_token_eu)]
        action_token_evidence = action_token_evidence[np.isfinite(action_token_evidence)]

        for values, xs_out, ys_out in (
            (action_token_au, au_token_x, au_token_y),
            (action_token_eu, eu_token_x, eu_token_y),
        ):
            if values.size > 0:
                xs = chunk_idx + (np.arange(values.size, dtype=np.float32) + 0.5) / float(values.size)
                xs_out.extend(xs.tolist())
                ys_out.extend(values.tolist())

        low_evidence_threshold = float(record.get("low_evidence_threshold", 4.0))
        low_evidence_mask = action_token_evidence < low_evidence_threshold
        if action_token_evidence.size > 0:
            clipped_evidence = np.clip(action_token_evidence, 0.0, 50.0)
            xs = chunk_idx + (np.arange(clipped_evidence.size, dtype=np.float32) + 0.5) / float(
                clipped_evidence.size
            )
            evidence_token_x.extend(xs.tolist())
            evidence_token_y.extend(clipped_evidence.tolist())

        action_token_count = int(action_token_evidence.size)
        if action_token_count > 0:
            low_evidence_count = int(np.sum(low_evidence_mask))
            low_evidence_ratio = float(low_evidence_count / action_token_count)
        else:
            action_token_count = int(record.get("num_action_tokens", 0))
            low_evidence_count = int(record.get("low_evidence_count") or 0)
            low_evidence_ratio = float(
                record.get(
                    "low_evidence_ratio",
                    low_evidence_count / action_token_count if action_token_count > 0 else 0.0,
                )
            )

        top_eu_20pct_mean = None
        if action_token_eu.size > 0:
            top_eu_count = max(1, int(math.ceil(0.2 * action_token_eu.size)))
            top_eu_20pct_mean = float(np.mean(np.sort(action_token_eu)[-top_eu_count:]))

        chunk_x.append(chunk_idx)
        action_token_count_y.append(float(action_token_count))
        low_evidence_count_y.append(float(low_evidence_count))
        low_evidence_ratio_y.append(low_evidence_ratio)
        top_eu_20pct_y.append(float(top_eu_20pct_mean) if top_eu_20pct_mean is not None else np.nan)

        num_quadrant_tokens = min(action_token_au.size, action_token_eu.size)
        if num_quadrant_tokens > 0:
            quadrant_au.extend(action_token_au[:num_quadrant_tokens].tolist())
            quadrant_eu.extend(action_token_eu[:num_quadrant_tokens].tolist())
            quadrant_chunk_idx.extend([chunk_idx] * num_quadrant_tokens)
            if action_token_evidence.size >= num_quadrant_tokens:
                quadrant_is_low_evidence.extend(low_evidence_mask[:num_quadrant_tokens].tolist())
            else:
                quadrant_is_low_evidence.extend([False] * num_quadrant_tokens)

    for axis, xs, ys, title, ylabel, color in (
        (
            axes[0, 0],
            au_token_x,
            au_token_y,
            "Action-token aleatoric uncertainty",
            "Normalized AU",
            "tab:orange",
        ),
        (
            axes[0, 1],
            eu_token_x,
            eu_token_y,
            "Action-token epistemic uncertainty",
            "EU",
            "tab:red",
        ),
    ):
        if xs:
            axis.scatter(xs, ys, s=10, alpha=0.75, color=color)
        else:
            axis.text(0.5, 0.5, "No action-token uncertainty", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xlabel("Chunk index")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, alpha=0.25)

    if evidence_token_x:
        axes[0, 2].scatter(evidence_token_x, evidence_token_y, s=10, alpha=0.75, color="tab:green")
        evidence_thresholds = [
            float(record.get("low_evidence_threshold", 4.0)) for record in uncertainty_chunks
        ]
        axes[0, 2].axhline(
            float(np.median(evidence_thresholds)),
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label="Provisional low-evidence threshold",
        )
        axes[0, 2].legend(loc="upper right", fontsize=8)
    else:
        axes[0, 2].text(
            0.5, 0.5, "No action-token evidence", ha="center", va="center", transform=axes[0, 2].transAxes
        )
    axes[0, 2].set_title("Selected action-token evidence (display clipped)")
    axes[0, 2].set_xlabel("Chunk index")
    axes[0, 2].set_ylabel("Evidence")
    axes[0, 2].set_ylim(0.0, 50.0)
    axes[0, 2].grid(True, alpha=0.25)

    if chunk_x:
        axes[1, 0].bar(
            chunk_x,
            action_token_count_y,
            alpha=0.35,
            color="gray",
            label="Total action tokens",
            zorder=1,
        )
        axes[1, 0].bar(
            chunk_x,
            low_evidence_count_y,
            alpha=0.55,
            color="tab:red",
            label="Low-evidence tokens",
            zorder=2,
        )
        ratio_axis = axes[1, 0].twinx()
        ratio_axis.plot(
            chunk_x,
            low_evidence_ratio_y,
            marker="o",
            markersize=3,
            linewidth=1.2,
            color="black",
            label="Low-evidence ratio",
            zorder=3,
        )
        ratio_axis.set_ylabel("Low-evidence ratio")
        ratio_axis.set_ylim(0.0, 1.0)
        ratio_axis.grid(False)
        count_handles, count_labels = axes[1, 0].get_legend_handles_labels()
        ratio_handles, ratio_labels = ratio_axis.get_legend_handles_labels()
        axes[1, 0].legend(
            count_handles + ratio_handles,
            count_labels + ratio_labels,
            loc="upper right",
            fontsize=8,
        )
    else:
        axes[1, 0].text(
            0.5, 0.5, "No low-evidence diagnostics", ha="center", va="center", transform=axes[1, 0].transAxes
        )
    axes[1, 0].set_title("Low-evidence action tokens per chunk")
    axes[1, 0].set_xlabel("Chunk index")
    axes[1, 0].set_ylabel("Token count")
    axes[1, 0].grid(True, alpha=0.25)

    finite_top_eu = np.isfinite(np.asarray(top_eu_20pct_y, dtype=np.float32))
    if chunk_x and np.any(finite_top_eu):
        chunk_array = np.asarray(chunk_x, dtype=np.float32)
        top_eu_array = np.asarray(top_eu_20pct_y, dtype=np.float32)
        axes[1, 1].plot(
            chunk_array[finite_top_eu],
            top_eu_array[finite_top_eu],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color="tab:red",
        )
    else:
        axes[1, 1].text(
            0.5, 0.5, "No chunk epistemic risk", ha="center", va="center", transform=axes[1, 1].transAxes
        )
    axes[1, 1].set_title("Mean EU of top 20% action tokens per chunk")
    axes[1, 1].set_xlabel("Chunk index")
    axes[1, 1].set_ylabel("Top-20% mean EU")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].grid(True, alpha=0.25)

    if quadrant_au:
        quadrant_au_array = np.asarray(quadrant_au, dtype=np.float32)
        quadrant_eu_array = np.asarray(quadrant_eu, dtype=np.float32)
        low_evidence_array = np.asarray(quadrant_is_low_evidence, dtype=bool)
        axes[1, 2].scatter(
            quadrant_au_array[~low_evidence_array],
            quadrant_eu_array[~low_evidence_array],
            s=12,
            alpha=0.55,
            color="tab:blue",
            label="Normal evidence",
        )
        axes[1, 2].scatter(
            quadrant_au_array[low_evidence_array],
            quadrant_eu_array[low_evidence_array],
            s=18,
            alpha=0.85,
            color="tab:red",
            label="Low evidence",
        )
        au_split = float(np.median(quadrant_au_array))
        eu_split = float(np.median(quadrant_eu_array))
        axes[1, 2].axvline(au_split, color="gray", linestyle="--", linewidth=1.0)
        axes[1, 2].axhline(eu_split, color="gray", linestyle="--", linewidth=1.0)
        axes[1, 2].legend(loc="center right", fontsize=8)
        axes[1, 2].text(0.02, 0.97, "Low AU / High EU", transform=axes[1, 2].transAxes, va="top", fontsize=8)
        axes[1, 2].text(
            0.98, 0.97, "High AU / High EU", transform=axes[1, 2].transAxes, ha="right", va="top", fontsize=8
        )
        axes[1, 2].text(0.02, 0.03, "Low AU / Low EU", transform=axes[1, 2].transAxes, va="bottom", fontsize=8)
        axes[1, 2].text(
            0.98,
            0.03,
            "High AU / Low EU",
            transform=axes[1, 2].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
    else:
        axes[1, 2].text(
            0.5, 0.5, "No AU-EU quadrant data", ha="center", va="center", transform=axes[1, 2].transAxes
        )
    axes[1, 2].set_title("Action-token AU-EU quadrants (median splits)")
    axes[1, 2].set_xlabel("Normalized AU")
    axes[1, 2].set_ylabel("EU")
    axes[1, 2].set_xlim(0.0, 1.0)
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].grid(True, alpha=0.25)

    if quadrant_au:
        quadrant_chunk_idx_array = np.asarray(quadrant_chunk_idx, dtype=np.float32)
        max_chunk_idx = max(math.ceil(max_steps / action_chunk_size) - 1, 1)
        chunk_norm = matplotlib.colors.Normalize(vmin=0, vmax=max_chunk_idx)
        chunk_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "chunk_blues",
            plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)),
        )
        temporal_scatter = axes[1, 3].scatter(
            quadrant_au_array,
            quadrant_eu_array,
            c=quadrant_chunk_idx_array,
            cmap=chunk_cmap,
            norm=chunk_norm,
            s=14,
            alpha=0.75,
        )
        colorbar = fig.colorbar(temporal_scatter, ax=axes[1, 3])
        colorbar.set_label("Chunk index")
    else:
        axes[1, 3].text(
            0.5, 0.5, "No AU-EU temporal data", ha="center", va="center", transform=axes[1, 3].transAxes
        )
    axes[1, 3].set_title("Action-token AU-EU by chunk time")
    axes[1, 3].set_xlabel("Normalized AU")
    axes[1, 3].set_ylabel("EU")
    axes[1, 3].set_xlim(0.0, 1.0)
    axes[1, 3].set_ylim(0.0, 1.0)
    axes[1, 3].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
