"""Calibrate frozen selective-rejection policies from completed sweep validation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .rejection import calibrate_policy
from .rejection_artifacts import write_json_atomic
from .rejection_data import (
    build_calibration_records,
    collect_dataset_provenance,
    read_prediction_file,
    sha256_file,
    validate_matched_predictions,
)


SCHEMA_VERSION = "1.0"


def calibrate_sweep(
    sweep_root: str | Path,
    output: str | Path,
    *,
    max_selective_error: float = 0.20,
    min_coverage: float = 0.10,
    min_accepted_episodes: int = 10,
    target_coverages: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90),
    analysis_seed: int = 20260805,
    overwrite: bool = False,
) -> Path:
    """Calibrate every canonical run while enforcing one shared episode population."""
    root = Path(sweep_root).expanduser().resolve()
    manifest_path = root / "sweep_manifest.json"
    manifest = _load_manifest(manifest_path)
    run_payloads: dict[str, Any] = {}
    reference_predictions = None
    source_datasets: dict[str, dict[str, Any]] | None = None
    for run in sorted(manifest["runs"], key=lambda item: item["run_name"]):
        run_name = run["run_name"]
        run_dir = Path(run["output_path"]).expanduser().resolve()
        if run_dir.parent != (root / "runs").resolve():
            raise ValueError(f"run {run_name!r} output path is outside sweep runs directory")
        resolved_path = run_dir / "resolved_config.yaml"
        prediction_path = run_dir / "validation_predictions.hdf5"
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
        resolved = _load_resolved_config(resolved_path)
        resolved_datasets = resolved.get("data", {}).get("datasets", {})
        if resolved_datasets:
            current_sources = collect_dataset_provenance(
                resolved_datasets,
                require_collection_identity=False,
            )
            if source_datasets is None:
                source_datasets = current_sources
            elif source_datasets != current_sources:
                raise ValueError("sweep runs do not share identical source datasets")
        model = resolved["model"]
        head = model["head"]
        predictions = read_prediction_file(prediction_path, expected_head=head)
        if reference_predictions is None:
            reference_predictions = predictions
        else:
            validate_matched_predictions(reference_predictions, predictions)
        records = build_calibration_records(predictions)
        kinds = ("au", "au_or_eu", "predictive_entropy") if head == "edl" else ("predictive_entropy",)
        policies = {
            kind: calibrate_policy(
                records,
                kind,
                max_selective_error=max_selective_error,
                min_coverage=min_coverage,
                min_accepted_episodes=min_accepted_episodes,
                target_coverages=target_coverages,
            ).to_dict()
            for kind in kinds
        }
        run_payloads[run_name] = {
            "head": head,
            "chunk_encoder": model["chunk_encoder"],
            "token_position_encoding": model["token_position_encoding"],
            "run_dir": str(run_dir),
            "resolved_config_path": str(resolved_path),
            "resolved_config_sha256": sha256_file(resolved_path),
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "policies": policies,
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_seed": int(analysis_seed),
        "sweep_id": manifest["sweep_id"],
        "sweep_root": str(root),
        "sweep_manifest_path": str(manifest_path),
        "sweep_manifest_sha256": sha256_file(manifest_path),
        "primary_chunks": list(range(1, 11)),
        "objective": {
            "max_selective_error": float(max_selective_error),
            "min_coverage": float(min_coverage),
            "min_accepted_episodes": int(min_accepted_episodes),
            "target_coverages": [float(value) for value in target_coverages],
        },
        "source_datasets": {} if source_datasets is None else source_datasets,
        "runs": run_payloads,
    }
    return write_json_atomic(output, payload, overwrite=overwrite)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid sweep manifest: {path}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("sweep_id"), str):
        raise ValueError("sweep manifest must contain a string sweep_id")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("sweep manifest must contain non-empty runs")
    required = {"run_name", "output_path"}
    for run in runs:
        if not isinstance(run, dict) or not required <= set(run):
            raise ValueError("every sweep run must contain run_name and output_path")
        if not isinstance(run["run_name"], str) or not run["run_name"]:
            raise ValueError("sweep run_name must be non-empty")
    if len({run["run_name"] for run in runs}) != len(runs):
        raise ValueError("sweep run names must be unique")
    return payload


def _load_resolved_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"resolved config {path} is missing model")
    model = payload["model"]
    for field in ("head", "chunk_encoder", "token_position_encoding"):
        if not isinstance(model.get(field), str):
            raise ValueError(f"resolved config model is missing {field}")
    if model["head"] not in {"edl", "softmax"}:
        raise ValueError("resolved config model.head must be edl or softmax")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-selective-error", type=float, default=0.20)
    parser.add_argument("--min-coverage", type=float, default=0.10)
    parser.add_argument("--min-accepted-episodes", type=int, default=10)
    parser.add_argument("--analysis-seed", type=int, default=20260805)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = calibrate_sweep(
        args.sweep_root,
        args.output,
        max_selective_error=args.max_selective_error,
        min_coverage=args.min_coverage,
        min_accepted_episodes=args.min_accepted_episodes,
        analysis_seed=args.analysis_seed,
        overwrite=args.overwrite,
    )
    print(f"Rejection calibration written to {destination}")


if __name__ == "__main__":
    main()
