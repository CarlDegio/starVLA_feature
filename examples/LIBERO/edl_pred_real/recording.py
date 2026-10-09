"""Durable per-request records. NumPy + stdlib only, usable in the YAM env."""
from datetime import datetime, timezone
import json
import os
import shutil
from pathlib import Path
import time
import uuid

import numpy as np

TOKEN_FIELDS = ("action_token_evidence", "action_token_aleatoric_uncertainty",
                "action_token_epistemic_uncertainty", "action_token_confidence", "action_token_rank")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_npz(path, **arrays):
    path = Path(path)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def validate_response(response, horizon):
    actions = np.asarray(response["actions"])
    if actions.shape != (horizon, 14) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite ({horizon}, 14) actions")
    diagnostics = response["diagnostics"]
    lengths = []
    for name in TOKEN_FIELDS:
        value = np.asarray(diagnostics[name])
        if value.ndim != 2 or value.shape[0] != 1 or not np.isfinite(value).all():
            raise ValueError(f"Invalid diagnostics {name}")
        lengths.append(value.shape[1])
    if len(set(lengths)) != 1 or lengths[0] == 0:
        raise ValueError("EDL token arrays must have equal nonzero lengths")
    count = lengths[0]
    alpha = np.asarray(diagnostics["action_token_topk_alpha"])
    if alpha.ndim != 3 or alpha.shape[:2] != (1, count) or alpha.shape[2] < 1:
        raise ValueError("Missing or malformed top-k alpha; restart the updated real server")
    if not np.isfinite(alpha).all() or np.any(alpha < 1):
        raise ValueError("Dirichlet alpha must be finite and >= 1")
    for name, shape in (("action_token_topk_ids", alpha.shape),
                        ("action_token_ids", (1, count)), ("action_token_mask", (1, count))):
        if np.asarray(diagnostics[name]).shape != shape:
            raise ValueError(f"Malformed {name}")
    if not np.asarray(diagnostics["action_token_mask"]).all():
        raise ValueError("Single-observation recording must have no padded action tokens")
    if np.asarray(diagnostics["num_action_tokens"]).tolist() != [count]:
        raise ValueError("num_action_tokens mismatch")
    for role in ("top", "left", "right"):
        frame = np.asarray(response["model_images"][role])
        if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
            raise ValueError(f"Invalid processed image {role}")


class EpisodeRecorder:
    def __init__(self, root, metadata, *, collection_id, seed_namespace, mock=False):
        if not collection_id.strip() or not seed_namespace.strip():
            raise ValueError("Collection identity and seed namespace must be nonempty")
        self.path = Path(root) / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:12])
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {"real_schema_version": "1.0", "episode_id": self.path.name,
                         "created_at": now(), "server_metadata": metadata,
                         "collection_id": collection_id, "seed_namespace": seed_namespace,
                         "mock": bool(mock), "success": None, "status": "open"}
        write_json(self.path / "episode.json", self.metadata)
        self.count = 0
        self.closed = False

    def begin_chunk(self, observation):
        if self.closed:
            raise RuntimeError("Episode is closed; start a new episode")
        chunk = self.path / f"chunk_{self.count:06d}"
        chunk.mkdir()
        self.count += 1
        # Persist the exact request BEFORE inference and before any action.
        write_npz(chunk / "observation.npz", state=observation["state"],
                  **{f"image_{role}": observation["images"][role] for role in ("top", "left", "right")})
        write_json(chunk / "request.json", {"chunk_idx": self.count - 1,
                   "prompt": observation["prompt"], "captured_at": now(),
                   "captured_monotonic_ns": time.monotonic_ns()})
        self.event(chunk, "request_saved")
        return chunk

    def save_prediction(self, chunk, response):
        arrays = {"actions": response["actions"],
                  **{f"diagnostic__{key}": value for key, value in response["diagnostics"].items()},
                  **{f"model_image_{role}": value for role, value in response["model_images"].items()}}
        write_npz(chunk / "prediction.npz", **arrays)
        self.event(chunk, "prediction_saved", policy_timing=response.get("policy_timing", {}),
                   server_timing=response.get("server_timing", {}))

    def event(self, chunk, kind, **payload):
        with (chunk / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": kind, "utc": now(),
                                    "monotonic_ns": time.monotonic_ns(), **payload}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def finish_chunk(self, chunk, *, status, completed_rows, error=None):
        write_json(chunk / "result.json", {"status": status, "completed_rows": completed_rows,
                   "finished_at": now(), "error": error})

    def finish(self, success=None, reason="operator_unlabeled"):
        if self.closed:
            raise RuntimeError("Episode is already closed")
        if success is not None and (type(success) is not int or success not in (0, 1)):
            raise ValueError("Episode success must be 0, 1, or None")
        self.metadata.update(success=success, status="closed", termination_reason=reason,
                             num_chunks=self.count, finished_at=now())
        write_json(self.path / "episode.json", self.metadata)
        self.closed = True

    def discard(self):
        """Explicit operator drop: remove only this recorder's own episode."""
        if self.closed:
            raise RuntimeError("Episode is already closed")
        if self.path.is_symlink() or self.path.name != self.metadata["episode_id"]:
            raise RuntimeError("Refusing to discard an unexpected episode path")
        shutil.rmtree(self.path)
        self.closed = True
