"""Collect QwenFast token and latent diagnostics from LIBERO rollouts."""

from __future__ import annotations

import atexit
import dataclasses
import json
import logging
import pathlib

import numpy as np

from examples.LIBERO.safe_pred.storage import SafeDiagnosticsDatasetWriter


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 6694
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    episode_start_index: int = 0
    max_tasks: int = -1
    seed: int = 7
    pretrained_path: str = ""
    unnorm_key: str | None = None
    dataset_output_path: str = ""
    collection_id: str = "safe_qwenfast"
    seed_namespace: str = "safe_qwenfast_seed7"
    dataset_overwrite: bool = False
    dataset_resume: bool = False


def validate_args(args: Args) -> None:
    if not args.pretrained_path:
        raise ValueError("pretrained_path is required for dataset provenance")
    if not args.dataset_output_path:
        raise ValueError("dataset_output_path is required")
    if args.num_trials_per_task <= 0:
        raise ValueError("num_trials_per_task must be positive")
    if args.episode_start_index < 0:
        raise ValueError("episode_start_index must be non-negative")
    if args.dataset_overwrite and args.dataset_resume:
        raise ValueError("dataset_overwrite and dataset_resume cannot both be enabled")
    if not args.collection_id.strip() or not args.seed_namespace.strip():
        raise ValueError("collection_id and seed_namespace must be non-empty")
    max_steps_for_suite(args.task_suite_name)


def max_steps_for_suite(task_suite_name: str) -> int:
    horizons = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if task_suite_name not in horizons:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return horizons[task_suite_name]


def _create_writer(args: Args, client_model, max_steps: int) -> SafeDiagnosticsDatasetWriter:
    requested_checkpoint = pathlib.Path(args.pretrained_path).expanduser().resolve()
    server_metadata = client_model.server_metadata
    server_checkpoint_value = server_metadata.get("ckpt_path")
    if not server_checkpoint_value:
        raise ValueError("policy server metadata does not include ckpt_path")
    server_checkpoint = pathlib.Path(server_checkpoint_value).expanduser().resolve()
    if requested_checkpoint != server_checkpoint:
        raise ValueError(
            f"checkpoint mismatch: collector requested {requested_checkpoint}, policy server loaded {server_checkpoint}"
        )
    metadata = {
        "checkpoint_path": str(requested_checkpoint),
        "server_checkpoint_path": str(server_checkpoint),
        "task_suite": args.task_suite_name,
        "collection_id": args.collection_id.strip(),
        "seed_namespace": args.seed_namespace.strip(),
        "seed": args.seed,
        "episode_start_index": args.episode_start_index,
        "episode_stop_index_exclusive": args.episode_start_index + args.num_trials_per_task,
        "max_steps": max_steps,
        "action_chunk_size": client_model.action_chunk_size,
        "server_metadata": server_metadata,
    }
    return SafeDiagnosticsDatasetWriter(
        args.dataset_output_path,
        metadata,
        overwrite=args.dataset_overwrite,
        resume=args.dataset_resume,
    )


def collect_libero_safe(args: Args) -> None:
    validate_args(args)
    logging.info("Arguments: %s", json.dumps(dataclasses.asdict(args), indent=2))

    from libero.libero import benchmark

    from examples.LIBERO.eval_files.eval_libero import (
        LIBERO_DUMMY_ACTION,
        LIBERO_ENV_RESOLUTION,
        _binarize_gripper_open,
        _get_libero_env,
        _quat2axisangle,
    )
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = max_steps_for_suite(args.task_suite_name)
    client = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
        return_token_uncertainty=True,
        return_latent_features=True,
    )
    writer = _create_writer(args, client, max_steps)
    atexit.register(writer.close)
    output_path = pathlib.Path(args.dataset_output_path).expanduser().resolve()
    logging.info("Writing SAFE diagnostics to %s", output_path)

    num_tasks = suite.n_tasks if args.max_tasks <= 0 else min(args.max_tasks, suite.n_tasks)
    total_episodes = 0
    total_successes = 0
    try:
        for task_id in range(num_tasks):
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            stop_index = args.episode_start_index + args.num_trials_per_task
            if stop_index > len(initial_states):
                raise ValueError(
                    f"task {task_id} has {len(initial_states)} initial states, requested "
                    f"[{args.episode_start_index}, {stop_index})"
                )
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            try:
                for episode_idx in range(args.episode_start_index, stop_index):
                    if writer.has_episode(task_id, episode_idx):
                        logging.info("Skipping existing task=%d episode=%d", task_id, episode_idx)
                        continue
                    client.reset(task_description)
                    env.reset()
                    obs = env.set_init_state(initial_states[episode_idx])
                    diagnostics: list[dict] = []
                    executed_steps = 0
                    done = False
                    t = 0
                    policy_step = 0
                    while t < max_steps + args.num_steps_wait:
                        if t < args.num_steps_wait:
                            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        )
                        example = {
                            "image": [image, wrist_image],
                            "lang": str(task_description),
                            "state": state,
                        }
                        response = client.step(example=example, step=policy_step)
                        if response.get("new_chunk"):
                            record = response.get("diagnostics")
                            if record is None:
                                raise RuntimeError(
                                    "policy returned a new action chunk without QwenFast diagnostics"
                                )
                            record = dict(record)
                            record["env_step"] = int(t)
                            record["policy_step"] = int(policy_step)
                            diagnostics.append(record)

                        raw_action = response["raw_action"]
                        world = np.asarray(raw_action["world_vector"], dtype=np.float32).reshape(-1)
                        rotation = np.asarray(raw_action["rotation_delta"], dtype=np.float32).reshape(-1)
                        gripper = _binarize_gripper_open(raw_action["open_gripper"])
                        if world.size != 3 or rotation.size != 3 or gripper.size != 1:
                            raise ValueError("policy returned an invalid 7-DoF LIBERO action")
                        action = np.concatenate((world, rotation, gripper))
                        obs, _, done, _ = env.step(action.tolist())
                        executed_steps += 1
                        if done:
                            break
                        t += 1
                        policy_step += 1

                    writer.append_episode(
                        task_id=task_id,
                        episode_idx=episode_idx,
                        task_description=task_description,
                        success=bool(done),
                        executed_steps=executed_steps,
                        termination_reason="success" if done else "max_steps",
                        diagnostic_chunks=diagnostics,
                    )
                    total_episodes += 1
                    total_successes += int(bool(done))
                    logging.info(
                        "Saved %s task=%d episode=%d success=%s (%d/%d)",
                        output_path,
                        task_id,
                        episode_idx,
                        done,
                        total_successes,
                        total_episodes,
                    )
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
    finally:
        writer.close()
    print(f"Dataset written to {output_path}")


if __name__ == "__main__":
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    tyro.cli(collect_libero_safe)
