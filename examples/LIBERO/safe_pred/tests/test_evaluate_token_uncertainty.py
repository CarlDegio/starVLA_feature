from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from examples.LIBERO.safe_pred.evaluate_token_uncertainty import evaluate_files
from examples.LIBERO.safe_pred.storage import SafeDiagnosticsDatasetWriter


class EvaluateTokenUncertaintyTest(unittest.TestCase):
    def test_evaluates_hdf5_and_writes_deterministic_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            dataset_path = root / "safe.hdf5"
            output_path = root / "metrics.json"
            metadata = {
                "checkpoint_path": "/checkpoints/qwenfast.pt",
                "server_checkpoint_path": "/checkpoints/qwenfast.pt",
                "task_suite": "libero_goal",
                "collection_id": "safe",
                "seed": 7,
                "max_steps": 300,
                "action_chunk_size": 8,
            }
            feature = [1.0, 2.0, 3.0]
            chunks = [
                {
                    "chunk_idx": 0,
                    "policy_step": 0,
                    "env_step": 10,
                    "num_action_tokens": 2,
                    "action_token_ids": [100, 101],
                    "action_token_nll": [0.2, 0.4],
                    "action_token_entropy": [1.2, 1.4],
                    "action_token_embedding_first": feature,
                    "action_token_embedding_last": feature,
                    "action_token_embedding_mean": feature,
                }
            ]
            with SafeDiagnosticsDatasetWriter(dataset_path, metadata) as writer:
                writer.append_episode(
                    task_id=0,
                    episode_idx=0,
                    task_description="task",
                    success=True,
                    executed_steps=10,
                    termination_reason="success",
                    diagnostic_chunks=chunks,
                )
                failed = [dict(chunk) for chunk in chunks]
                failed[0] = dict(failed[0], action_token_nll=[2.0, 2.1], action_token_entropy=[2.2, 2.3])
                writer.append_episode(
                    task_id=0,
                    episode_idx=1,
                    task_description="task",
                    success=False,
                    executed_steps=300,
                    termination_reason="max_steps",
                    diagnostic_chunks=failed,
                )

            result = evaluate_files([dataset_path], output_path=output_path)

            self.assertEqual(result["scores"]["max_nll"]["roc_auc"], 1.0)
            self.assertEqual(json.loads(output_path.read_text()), result)
            self.assertIn("not a failure probability", result["scores"]["max_nll"]["brier_reason"])


if __name__ == "__main__":
    unittest.main()
