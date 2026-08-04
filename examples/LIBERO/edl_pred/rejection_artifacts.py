"""Atomic artifact helpers for selective rejection calibration and evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


def write_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Write canonical strict JSON by atomic replacement."""
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {source}")
    return payload
