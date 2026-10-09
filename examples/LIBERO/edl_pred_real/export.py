"""Export explicitly labeled real episodes to the existing EDL verifier schema."""
import argparse
import json
import os
from pathlib import Path
import uuid

import h5py
import numpy as np
from scipy.special import digamma

from examples.LIBERO.eval_files.libero_uncertainty_dataset import LiberoUncertaintyDatasetWriter


def recompute_uncertainty(alpha):
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 2 or not np.isfinite(alpha).all() or np.any(alpha < 1):
        raise ValueError("Expected finite action-token top-k alpha [N,K] >= 1")
    strength = alpha.sum(axis=-1, keepdims=True)
    k = alpha.shape[-1]
    au = ((alpha / strength) * (digamma(strength + 1) - digamma(alpha + 1))).sum(axis=-1)
    if k > 1:
        au /= np.log(k)
    return au.astype(np.float32), (k / strength[:, 0]).astype(np.float32)


def export_dataset(root, output, *, task, collection_id, seed_namespace, allow_mock=False):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    episodes, skipped = [], []
    checkpoint = None
    for path in sorted(root.glob("*/episode.json")):
        info = json.loads(path.read_text())
        meta = info["server_metadata"]
        if meta["task"] != task or info["collection_id"] != collection_id or info["seed_namespace"] != seed_namespace:
            continue
        if info.get("mock") and not allow_mock:
            skipped.append({"episode": info["episode_id"], "reason": "mock"})
            continue
        if info["status"] != "closed" or type(info.get("success")) is not int or info["success"] not in (0, 1):
            skipped.append({"episode": info["episode_id"], "reason": "unlabeled or open"})
            continue
        if checkpoint is not None and checkpoint != meta["ckpt_path"]:
            raise ValueError("Do not mix policy checkpoints in one dataset")
        checkpoint = meta["ckpt_path"]
        chunks, details = [], []
        executed_steps = 0
        for directory in sorted(path.parent.glob("chunk_*")):
            result_path = directory / "result.json"
            if not result_path.exists():
                raise ValueError(f"Labeled episode has an incomplete chunk: {directory}")
            result = json.loads(result_path.read_text())
            if result["status"] == "error":
                raise ValueError(f"Error chunk requires review before export: {directory}")
            if result["completed_rows"] == 0:
                continue  # Predicted but never executed; keep raw record only.
            with np.load(directory / "prediction.npz", allow_pickle=False) as file:
                prediction = {key: file[key] for key in file.files}
            diag = {key.removeprefix("diagnostic__"): value for key, value in prediction.items()
                    if key.startswith("diagnostic__")}
            au, eu = recompute_uncertainty(diag["action_token_topk_alpha"][0])
            np.testing.assert_allclose(au, diag["action_token_aleatoric_uncertainty"][0], rtol=1e-4, atol=2e-6)
            np.testing.assert_allclose(eu, diag["action_token_epistemic_uncertainty"][0], rtol=1e-4, atol=2e-6)
            request = json.loads((directory / "request.json").read_text())
            chunks.append({"chunk_idx": len(chunks), "policy_step": executed_steps,
                           "env_step": executed_steps, "num_tokens": diag["token_uncertainty"].shape[1],
                           **{name: diag[name][0] for name in (
                               "action_token_evidence", "action_token_aleatoric_uncertainty",
                               "action_token_epistemic_uncertainty", "action_token_confidence", "action_token_rank")}})
            details.append((directory, request, result, prediction))
            executed_steps += result["completed_rows"]
        if not chunks:
            skipped.append({"episode": info["episode_id"], "reason": "no executed chunks"})
            continue
        episodes.append((info, chunks, details, executed_steps))
    if not episodes:
        raise ValueError("No closed, explicitly labeled, executed episodes match this collection")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        metadata = {"task_suite": f"real_{task}", "collection_id": collection_id,
                    "seed_namespace": seed_namespace, "checkpoint": checkpoint,
                    "action_chunk_size": 15, "control_hz": 30.0, "real_schema_version": "1.0",
                    "contains_mock": any(e[0]["mock"] for e in episodes), "skipped": skipped,
                    "source_root": str(root.resolve())}
        with LiberoUncertaintyDatasetWriter(temporary, metadata) as writer:
            for index, (info, chunks, details, steps) in enumerate(episodes):
                writer.append_episode(task_id=0, episode_idx=index,
                                      task_description=details[0][1]["prompt"], success=bool(info["success"]),
                                      executed_steps=steps, termination_reason=info["termination_reason"],
                                      uncertainty_chunks=chunks)
        with h5py.File(temporary, "r+") as handle:
            for index, (info, chunks, details, steps) in enumerate(episodes):
                group = handle[f"episodes/{LiberoUncertaintyDatasetWriter.episode_key(0, index)}"]
                group.attrs["source_episode_id"] = info["episode_id"]
                group.attrs["server_metadata"] = json.dumps(info["server_metadata"])
                extra = group.create_group("real_chunks")
                for chunk_index, (directory, request, result, prediction) in enumerate(details):
                    child = extra.create_group(f"{chunk_index:06d}")
                    child.attrs["source_path"] = str(directory.resolve())
                    child.attrs["request"] = json.dumps(request)
                    child.attrs["execution_result"] = json.dumps(result)
                    child.create_dataset("execution_events_json", data=(directory / "events.jsonl").read_text())
                    with np.load(directory / "observation.npz", allow_pickle=False) as observation:
                        for name in observation.files:
                            child.create_dataset(name, data=observation[name], compression="gzip")
                    for name, value in prediction.items():
                        child.create_dataset(name, data=value, compression="gzip" if value.ndim else None)
            handle.flush()
        # Publish without overwriting a concurrently created destination.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"output": str(output), "episodes": len(episodes), "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("blocks", "tubes", "slippers"), required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--seed-namespace", required=True)
    parser.add_argument("--allow-mock", action="store_true", help="Tests only; never mix with real training data")
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.records, args.output, task=args.task,
                     collection_id=args.collection_id, seed_namespace=args.seed_namespace, allow_mock=args.allow_mock)))


if __name__ == "__main__":
    main()
