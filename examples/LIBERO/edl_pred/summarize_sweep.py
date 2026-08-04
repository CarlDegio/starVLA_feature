"""Strict, deterministic publication for the fixed LIBERO EDL sweep."""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any, Mapping, Sequence

import h5py
import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


PACKAGE_ROOT = Path(__file__).resolve().parent
SWEEP_ID = "all_suites_seed7_v1"
CANONICAL_RUNS = {
    0: ("all_mlp_flat_softmax.yaml", "all_mlp_flat_softmax_seed7", "mlp_flat", "softmax", "none"),
    1: ("all_mlp_flat_edl.yaml", "all_mlp_flat_edl_seed7", "mlp_flat", "edl", "none"),
    2: ("all_attention_pool_softmax_sinusoidal.yaml", "all_attention_pool_softmax_sinusoidal_seed7", "token_attention_pool", "softmax", "sinusoidal"),
    3: ("all_attention_pool_edl_sinusoidal.yaml", "all_attention_pool_edl_sinusoidal_seed7", "token_attention_pool", "edl", "sinusoidal"),
    4: ("all_self_attention_softmax_sinusoidal.yaml", "all_self_attention_softmax_sinusoidal_seed7", "token_self_attention", "softmax", "sinusoidal"),
    5: ("all_self_attention_edl_sinusoidal.yaml", "all_self_attention_edl_sinusoidal_seed7", "token_self_attention", "edl", "sinusoidal"),
    6: ("all_attention_pool_edl_none.yaml", "all_attention_pool_edl_none_seed7", "token_attention_pool", "edl", "none"),
    7: ("all_attention_pool_edl_learned.yaml", "all_attention_pool_edl_learned_seed7", "token_attention_pool", "edl", "learned"),
}
REQUIRED_ARTIFACTS = (
    "resolved_config.yaml",
    "split_manifest.json",
    "metrics.jsonl",
    "training_curves.png",
    "checkpoints/best.pt",
    "checkpoints/last.pt",
    "validation_predictions.hdf5",
)
METRIC_FIELDS = (
    "chunk_roc_auc",
    "chunk_pr_auc",
    "chunk_brier",
    "chunk_ece",
    "final_chunk_roc_auc",
    "final_chunk_pr_auc",
    "final_chunk_accuracy",
)
EDL_FIELDS = (
    "verifier_au_label_0",
    "verifier_au_label_1",
    "verifier_eu_label_0",
    "verifier_eu_label_1",
    "verifier_total_evidence_label_0",
    "verifier_total_evidence_label_1",
)
CSV_FIELDS = (
    "sweep_id", "gpu_index", "run_name", "config_path", "output_path", "log_path", "encoder", "head",
    "position_encoding", "best_epoch", "best_validation_loss", *METRIC_FIELDS, *EDL_FIELDS,
)
PLOT_FIELDS = (
    ("best_validation_loss", "Best validation loss"),
    ("chunk_roc_auc", "Chunk ROC-AUC"),
    ("final_chunk_roc_auc", "Final chunk ROC-AUC"),
    ("chunk_pr_auc", "Chunk PR-AUC"),
    ("final_chunk_pr_auc", "Final chunk PR-AUC"),
    ("chunk_brier", "Chunk Brier"),
    ("chunk_ece", "Chunk ECE"),
)
SUMMARY_FILENAMES = ("sweep_summary.json", "sweep_summary.csv", "sweep_comparison.png")


def summarize_sweep(sweep_root: Path, *, overwrite: bool = False) -> list[dict[str, Any]]:
    """Validate the canonical completed sweep and publish deterministic summaries.

    Existing summaries are refused unless ``overwrite=True``. Every output is
    rendered to a regular same-directory temporary file before publication.
    """
    root = Path(sweep_root).expanduser().absolute()
    _require_directory(root, "sweep root")
    with _summary_reservation(root):
        destinations = _preflight_summary_destinations(root, overwrite)
        manifest = _load_manifest(root / "sweep_manifest.json", root)
        rows = [_summarize_run(root, manifest["sweep_id"], run) for run in manifest["runs"]]
        rows.sort(key=lambda row: (row["best_validation_loss"], row["gpu_index"]))
        staged: dict[str, Path] = {}
        try:
            staged = _stage_summaries(root, rows)
            _publish_staged(staged, destinations, overwrite)
        finally:
            _cleanup_paths(staged.values())
        return rows


def _load_manifest(path: Path, root: Path) -> dict[str, Any]:
    payload = _load_json(path, "sweep manifest")
    if set(payload) != {"sweep_id", "started_at", "runs"} or payload["sweep_id"] != SWEEP_ID:
        raise ValueError(f"sweep manifest must use canonical sweep ID {SWEEP_ID!r}")
    if not isinstance(payload["started_at"], str) or not payload["started_at"]:
        raise ValueError("sweep manifest started_at must be a non-empty string")
    runs = payload["runs"]
    if not isinstance(runs, list) or len(runs) != len(CANONICAL_RUNS):
        raise ValueError("sweep manifest must contain exactly eight canonical runs")
    expected_fields = {"gpu_index", "config_path", "run_name", "output_path", "log_path"}
    canonical: list[dict[str, Any]] = []
    seen_gpus: set[int] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping) or set(run) != expected_fields:
            raise ValueError(f"sweep manifest run {index} has unexpected fields")
        gpu = run["gpu_index"]
        if not isinstance(gpu, int) or isinstance(gpu, bool) or gpu not in CANONICAL_RUNS or gpu in seen_gpus:
            raise ValueError(f"sweep manifest run {index} has non-canonical gpu_index")
        seen_gpus.add(gpu)
        filename, run_name, encoder, head, position = CANONICAL_RUNS[gpu]
        expected = {
            "gpu_index": gpu,
            "config_path": str((PACKAGE_ROOT / "configs/sweep" / filename).absolute()),
            "run_name": run_name,
            "output_path": str((root / "runs" / run_name).absolute()),
            "log_path": str((root / "logs" / f"{run_name}.log").absolute()),
        }
        if dict(run) != expected:
            raise ValueError(f"sweep manifest run {gpu} does not match the canonical matrix")
        canonical.append({**expected, "encoder": encoder, "head": head, "position_encoding": position})
    if seen_gpus != set(CANONICAL_RUNS):
        raise ValueError("sweep manifest GPU indices must be exactly 0 through 7")
    return {"sweep_id": SWEEP_ID, "runs": canonical}


def _summarize_run(root: Path, sweep_id: str, run: Mapping[str, Any]) -> dict[str, Any]:
    config_path = Path(run["config_path"])
    output_path = Path(run["output_path"])
    log_path = Path(run["log_path"])
    _require_regular_file(config_path, "manifest config_path")
    _require_directory(output_path, "manifest output_path")
    _require_regular_file(log_path, "manifest log_path")
    for artifact_name in REQUIRED_ARTIFACTS:
        _require_regular_file(output_path / artifact_name, f"run {run['run_name']} required artifact")

    resolved = _load_yaml(output_path / "resolved_config.yaml", "resolved config")
    try:
        model = resolved["model"]
        resolved_values = (resolved["run"]["name"], model["chunk_encoder"], model["head"], model["token_position_encoding"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"run {run['run_name']} resolved config is missing model/run metadata") from error
    expected_values = (run["run_name"], run["encoder"], run["head"], run["position_encoding"])
    if resolved_values != expected_values:
        raise ValueError(f"run {run['run_name']} resolved model metadata does not match canonical matrix")

    validation_identities = _load_split_manifest(output_path / "split_manifest.json", run["run_name"])
    _validate_png(output_path / "training_curves.png", run["run_name"])
    _validate_predictions(output_path / "validation_predictions.hdf5", validation_identities, run["run_name"])
    metrics = _load_metrics(output_path / "metrics.jsonl", run["run_name"])
    best = min(metrics, key=lambda record: (record["validation_total_loss"], record["epoch"]))
    _validate_checkpoint(output_path / "checkpoints/best.pt", best, "best", run["run_name"])
    _validate_checkpoint(output_path / "checkpoints/last.pt", metrics[-1], "last", run["run_name"])

    row: dict[str, Any] = {
        "sweep_id": sweep_id, "gpu_index": run["gpu_index"], "run_name": run["run_name"],
        "config_path": str(config_path), "output_path": str(output_path), "log_path": str(log_path),
        "encoder": run["encoder"], "head": run["head"], "position_encoding": run["position_encoding"],
        "best_epoch": best["epoch"], "best_validation_loss": best["validation_total_loss"],
    }
    for field in METRIC_FIELDS:
        row[field] = best[f"validation_{field}"]
    for field in EDL_FIELDS:
        row[field] = best.get(f"validation_{field}")
    return row


def _load_split_manifest(path: Path, run_name: str) -> set[tuple[str, str]]:
    payload = _load_json(path, f"run {run_name} split manifest")
    validation = payload.get("validation")
    if not isinstance(validation, list) or not validation:
        raise ValueError(f"run {run_name} split manifest must contain validation identities")
    identities: set[tuple[str, str]] = set()
    for item in validation:
        if not isinstance(item, Mapping) or not isinstance(item.get("suite"), str) or not isinstance(item.get("episode_key"), str):
            raise ValueError(f"run {run_name} split manifest has malformed validation identity")
        identity = (item["suite"], item["episode_key"])
        if identity in identities:
            raise ValueError(f"run {run_name} split manifest duplicates validation identity")
        identities.add(identity)
    return identities


def _validate_png(path: Path, run_name: str) -> None:
    if path.stat().st_size < 8 or path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"run {run_name} training_curves.png is not a non-empty PNG")


def _validate_predictions(path: Path, expected: set[tuple[str, str]], run_name: str) -> None:
    try:
        with h5py.File(path, "r") as handle:
            episodes = handle.get("episodes")
            if not isinstance(episodes, h5py.Group):
                raise ValueError("missing episodes group")
            actual: set[tuple[str, str]] = set()
            for suite, suite_group in episodes.items():
                if not isinstance(suite_group, h5py.Group):
                    raise ValueError("suite entry is not a group")
                for episode_key, episode in suite_group.items():
                    if not isinstance(episode, h5py.Group) or "success_probability" not in episode:
                        raise ValueError("episode lacks success_probability")
                    probabilities = episode["success_probability"]
                    if probabilities.ndim != 1 or probabilities.shape[0] == 0:
                        raise ValueError("episode success_probability is empty or malformed")
                    actual.add((suite, episode_key))
    except (OSError, ValueError) as error:
        raise ValueError(f"run {run_name} validation predictions are not a valid HDF5 artifact") from error
    if actual != expected:
        raise ValueError(f"run {run_name} validation prediction identities do not match split manifest")


def _validate_checkpoint(path: Path, expected: Mapping[str, Any], kind: str, run_name: str) -> None:
    if path.stat().st_size == 0:
        raise ValueError(f"run {run_name} {kind} checkpoint is empty")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise ValueError(f"run {run_name} {kind} checkpoint cannot be parsed") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"run {run_name} {kind} checkpoint must be a mapping")
    if payload.get("epoch") != expected["epoch"]:
        raise ValueError(f"run {run_name} {kind} checkpoint epoch does not match metrics")
    loss = payload.get("validation_total_loss")
    if not isinstance(loss, (int, float)) or isinstance(loss, bool) or not math.isclose(float(loss), float(expected["validation_total_loss"]), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"run {run_name} {kind} checkpoint loss does not match metrics")


def _load_metrics(path: Path, run_name: str) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    if not content.endswith("\n"):
        raise ValueError(f"run {run_name} metrics.jsonl must end with a newline")
    records: list[dict[str, Any]] = []
    previous_epoch = 0
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line:
            raise ValueError(f"run {run_name} metrics.jsonl has an empty line at {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"run {run_name} metrics.jsonl has invalid JSON at line {line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"run {run_name} metrics.jsonl line {line_number} must be an object")
        epoch = record.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= previous_epoch:
            raise ValueError(f"run {run_name} metrics.jsonl epochs must be positive and strictly increasing")
        previous_epoch = epoch
        _finite_number(record.get("validation_total_loss"), f"run {run_name} validation_total_loss")
        for field in METRIC_FIELDS:
            _optional_finite_number(record.get(f"validation_{field}"), f"run {run_name} validation_{field}")
        for field in EDL_FIELDS:
            if f"validation_{field}" in record:
                _optional_finite_number(record[f"validation_{field}"], f"run {run_name} validation_{field}")
        records.append(record)
    if not records:
        raise ValueError(f"run {run_name} metrics.jsonl must contain at least one record")
    return records


def _preflight_summary_destinations(root: Path, overwrite: bool) -> dict[str, Path]:
    destinations = {name: root / name for name in SUMMARY_FILENAMES}
    for path in destinations.values():
        _assert_no_symlink_ancestors(path)
        if path.is_symlink() or path.is_dir():
            raise ValueError(f"summary destination must not be a directory or symlink: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"summary destination exists; pass overwrite=True: {path}")
    return destinations


def _stage_summaries(root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    staged: dict[str, Path] = {}
    try:
        staged["sweep_summary.json"] = _temporary_path(root, ".stage-sweep_summary.json-")
        staged["sweep_summary.json"].write_text(json.dumps(list(rows), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        staged["sweep_summary.csv"] = _temporary_path(root, ".stage-sweep_summary.csv-")
        with staged["sweep_summary.csv"].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="raise", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        staged["sweep_comparison.png"] = _temporary_path(root, ".stage-sweep_comparison.png-")
        _plot_comparison(staged["sweep_comparison.png"], rows)
        for temporary in staged.values():
            _require_regular_file(temporary, "staged summary")
        return staged
    except BaseException:
        _cleanup_paths(staged.values())
        raise


def _publish_staged(staged: Mapping[str, Path], destinations: Mapping[str, Path], overwrite: bool) -> None:
    backups: dict[str, Path] = {}
    published: list[str] = []
    try:
        for name in SUMMARY_FILENAMES:
            destination = destinations[name]
            _assert_no_symlink_ancestors(destination)
            if destination.is_symlink() or destination.is_dir() or (destination.exists() and not overwrite):
                raise ValueError(f"summary destination changed during publication: {destination}")
            if overwrite and destination.exists():
                backup = _temporary_path(destination.parent, f".backup-{name}-")
                backup.unlink()
                os.link(destination, backup)
                backups[name] = backup
        for name in SUMMARY_FILENAMES:
            _publish_regular(staged[name], destinations[name], overwrite=overwrite)
            published.append(name)
    except BaseException:
        _rollback_publication(published, destinations, backups)
        raise
    finally:
        _cleanup_paths(staged.values())
        _cleanup_paths(backups.values())


def _publish_regular(stage: Path, destination: Path, *, overwrite: bool) -> None:
    """Publish one staged regular file without following a destination symlink."""
    if overwrite:
        os.replace(stage, destination)
    else:
        os.link(stage, destination)


def _rollback_publication(
    published: Sequence[str], destinations: Mapping[str, Path], backups: Mapping[str, Path]
) -> None:
    rollback_error: OSError | None = None
    for name in reversed(published):
        destination = destinations[name]
        try:
            backup = backups.get(name)
            if backup is None:
                destination.unlink(missing_ok=True)
            else:
                os.replace(backup, destination)
        except OSError as error:
            rollback_error = error
    if rollback_error is not None:
        raise RuntimeError("summary publication rollback failed") from rollback_error


def _temporary_path(root: Path, prefix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, dir=root)
    os.close(descriptor)
    return Path(name)


def _cleanup_paths(paths: Sequence[Path] | Any) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _plot_comparison(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(18, 12), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, PLOT_FIELDS):
        available = [(row["run_name"], row[field]) for row in rows if row[field] is not None]
        axis.set_title(title)
        axis.set_ylabel(title)
        if available:
            labels, values = zip(*available)
            axis.bar(range(len(values)), values)
            axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
            axis.grid(axis="y", alpha=0.25)
        else:
            axis.text(0.5, 0.5, "Unavailable", ha="center", va="center", transform=axis.transAxes)
            axis.set_xticks([])
    for axis in axes.flat[len(PLOT_FIELDS):]:
        axis.set_visible(False)
    figure.savefig(path, dpi=160, format="png")
    plt.close(figure)


def _load_json(path: Path, description: str) -> dict[str, Any]:
    _require_regular_file(path, description)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} has invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return payload


def _load_yaml(path: Path, description: str) -> Mapping[str, Any]:
    _require_regular_file(path, description)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"{description} has invalid YAML: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} must be a mapping: {path}")
    return payload


@contextmanager
def _summary_reservation(root: Path) -> Any:
    reservation = root / ".summary-reservation"
    _assert_no_symlink_ancestors(reservation)
    try:
        reservation.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"summary reservation already exists: {reservation}") from error
    try:
        yield
    finally:
        reservation.rmdir()


def require_lexical_regular_file(path: Path, description: str) -> Path:
    """Reject symlinks before any caller resolves or parses a canonical input."""
    lexical = Path(path).expanduser().absolute()
    _assert_no_symlink_ancestors(lexical)
    try:
        mode = os.lstat(lexical).st_mode
    except FileNotFoundError as error:
        raise ValueError(f"{description} does not exist: {lexical}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"{description} is not a regular file: {lexical}")
    return lexical


def _require_regular_file(path: Path, description: str) -> None:
    require_lexical_regular_file(path, description)


def _require_directory(path: Path, description: str) -> None:
    _assert_no_symlink_ancestors(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{description} is not a regular directory: {path}")


def _assert_no_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    for ancestor in (absolute, *absolute.parents):
        if ancestor.is_symlink():
            raise ValueError(f"symlink ancestor is not allowed: {ancestor}")


def _finite_number(value: Any, description: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{description} must be a finite number")
    return float(value)


def _optional_finite_number(value: Any, description: str) -> float | None:
    return None if value is None else _finite_number(value, description)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly summarize the completed canonical LIBERO EDL sweep.")
    parser.add_argument("--sweep-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true", help="replace existing regular summary files")
    args = parser.parse_args()
    print(json.dumps(summarize_sweep(args.sweep_root, overwrite=args.overwrite), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
