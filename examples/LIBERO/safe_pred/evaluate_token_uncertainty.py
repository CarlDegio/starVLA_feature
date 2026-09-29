"""Evaluate SAFE-format HDF5 files with training-free token uncertainty."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Sequence

import h5py

from examples.LIBERO.safe_pred.metrics import evaluate_token_uncertainty_records


@dataclasses.dataclass
class Args:
    dataset_paths: list[str]
    output_path: str


def evaluate_files(
    dataset_paths: Sequence[str | pathlib.Path],
    *,
    output_path: str | pathlib.Path | None = None,
) -> dict:
    if not dataset_paths:
        raise ValueError("at least one dataset path is required")
    records = []
    sources = []
    for path_value in dataset_paths:
        path = pathlib.Path(path_value).expanduser().resolve()
        sources.append(str(path))
        with h5py.File(path, "r") as dataset:
            if "episodes" not in dataset:
                raise ValueError(f"dataset has no episodes group: {path}")
            suite = str(dataset.attrs.get("task_suite", path.stem))
            for episode_key in sorted(dataset["episodes"].keys()):
                episode = dataset["episodes"][episode_key]
                records.append(
                    {
                        "suite": suite,
                        "task_id": int(episode.attrs["task_id"]),
                        "success": bool(episode.attrs["success"]),
                        "num_action_tokens": episode["num_action_tokens"][:],
                        "token_offsets": episode["token_offsets"][:],
                        "action_token_nll": episode["action_token_nll"][:],
                        "action_token_entropy": episode["action_token_entropy"][:],
                    }
                )
    result = {"source_files": sources, **evaluate_token_uncertainty_records(records)}
    if output_path is not None:
        destination = pathlib.Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(args: Args) -> None:
    result = evaluate_files(args.dataset_paths, output_path=args.output_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
