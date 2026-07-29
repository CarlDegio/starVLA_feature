import dataclasses
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

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True

    job_name: str = "test"


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

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
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            rollout_base = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}"
            imageio.mimwrite(
                rollout_base.parent / f"{rollout_base.name}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            _save_uncertainty_artifacts(uncertainty_chunks, rollout_base)

            full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


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


def _save_uncertainty_artifacts(uncertainty_chunks: list[dict], rollout_base: pathlib.Path) -> None:
    """Save chunk/token uncertainty as JSONL and a compact diagnostic plot."""
    jsonl_path = rollout_base.parent / f"{rollout_base.name}.jsonl"
    png_path = rollout_base.parent / f"{rollout_base.name}.png"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in uncertainty_chunks:
            f.write(json.dumps(record) + "\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=False)

    au_token_x, au_token_y = [], []
    eu_token_x, eu_token_y = [], []
    evidence_token_x, evidence_token_y = [], []
    chunk_x = []
    low_evidence_count_y, low_evidence_run_y = [], []
    worst_token_eu_y = []
    quadrant_au, quadrant_eu, quadrant_is_low_evidence = [], [], []
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

        low_evidence_count = record.get("low_evidence_count")
        if low_evidence_count is None:
            low_evidence_count = int(np.sum(low_evidence_mask))
        low_evidence_max_consecutive = record.get("low_evidence_max_consecutive")
        if low_evidence_max_consecutive is None:
            current_run = 0
            low_evidence_max_consecutive = 0
            for is_low in low_evidence_mask:
                current_run = current_run + 1 if bool(is_low) else 0
                low_evidence_max_consecutive = max(low_evidence_max_consecutive, current_run)

        worst_token_eu_mean = record.get("worst_token_eu_mean")
        if worst_token_eu_mean is None and action_token_eu.size > 0:
            worst_token_count = min(int(record.get("worst_token_count", 3)), action_token_eu.size)
            worst_token_eu_mean = float(np.mean(np.sort(action_token_eu)[-worst_token_count:]))

        chunk_x.append(chunk_idx)
        low_evidence_count_y.append(float(low_evidence_count))
        low_evidence_run_y.append(float(low_evidence_max_consecutive))
        worst_token_eu_y.append(float(worst_token_eu_mean) if worst_token_eu_mean is not None else np.nan)

        num_quadrant_tokens = min(action_token_au.size, action_token_eu.size)
        if num_quadrant_tokens > 0:
            quadrant_au.extend(action_token_au[:num_quadrant_tokens].tolist())
            quadrant_eu.extend(action_token_eu[:num_quadrant_tokens].tolist())
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
        axes[1, 0].bar(chunk_x, low_evidence_count_y, alpha=0.45, color="tab:red", label="Low count")
        axes[1, 0].plot(
            chunk_x,
            low_evidence_run_y,
            marker="o",
            markersize=3,
            linewidth=1.2,
            color="black",
            label="Longest consecutive run",
        )
        axes[1, 0].legend(loc="upper right", fontsize=8)
    else:
        axes[1, 0].text(
            0.5, 0.5, "No low-evidence diagnostics", ha="center", va="center", transform=axes[1, 0].transAxes
        )
    axes[1, 0].set_title("Low-evidence action tokens per chunk")
    axes[1, 0].set_xlabel("Chunk index")
    axes[1, 0].set_ylabel("Token count")
    axes[1, 0].grid(True, alpha=0.25)

    finite_worst_eu = np.isfinite(np.asarray(worst_token_eu_y, dtype=np.float32))
    if chunk_x and np.any(finite_worst_eu):
        chunk_array = np.asarray(chunk_x, dtype=np.float32)
        worst_eu_array = np.asarray(worst_token_eu_y, dtype=np.float32)
        axes[1, 1].plot(
            chunk_array[finite_worst_eu],
            worst_eu_array[finite_worst_eu],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color="tab:red",
        )
    else:
        axes[1, 1].text(
            0.5, 0.5, "No chunk epistemic risk", ha="center", va="center", transform=axes[1, 1].transAxes
        )
    axes[1, 1].set_title("Mean EU of worst action tokens per chunk")
    axes[1, 1].set_xlabel("Chunk index")
    axes[1, 1].set_ylabel("Worst-token mean EU")
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
