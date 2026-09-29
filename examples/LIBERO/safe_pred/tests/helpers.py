from __future__ import annotations

import pathlib

from examples.LIBERO.safe_pred.storage import SafeDiagnosticsDatasetWriter


def write_safe_dataset(path: pathlib.Path, suite: str, labels: list[bool], hidden_dim: int = 4) -> pathlib.Path:
    metadata = {
        "checkpoint_path": "/checkpoints/qwenfast.pt",
        "server_checkpoint_path": "/checkpoints/qwenfast.pt",
        "task_suite": suite,
        "collection_id": "test",
        "seed": 7,
        "max_steps": 100,
        "action_chunk_size": 8,
    }
    with SafeDiagnosticsDatasetWriter(path, metadata) as writer:
        for episode_idx, success in enumerate(labels):
            chunks = []
            for chunk_idx in range(episode_idx % 3 + 1):
                base = float(episode_idx * 10 + chunk_idx)
                feature = [base + offset for offset in range(hidden_dim)]
                chunks.append(
                    {
                        "chunk_idx": chunk_idx,
                        "policy_step": chunk_idx * 8,
                        "env_step": 10 + chunk_idx * 8,
                        "num_action_tokens": 1,
                        "action_token_ids": [100 + chunk_idx],
                        "action_token_nll": [0.1 + base],
                        "action_token_entropy": [0.2 + base],
                        "action_token_embedding_first": feature,
                        "action_token_embedding_last": [value + 0.5 for value in feature],
                        "action_token_embedding_mean": [value + 0.25 for value in feature],
                    }
                )
            writer.append_episode(
                task_id=episode_idx % 2,
                episode_idx=episode_idx,
                task_description=f"task {episode_idx % 2}",
                success=success,
                executed_steps=len(chunks) * 8,
                termination_reason="success" if success else "max_steps",
                diagnostic_chunks=chunks,
            )
    return path
