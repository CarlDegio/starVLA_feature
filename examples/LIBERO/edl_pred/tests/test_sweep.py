"""Contract tests for the non-launching eight-GPU sweep tooling."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import tempfile
from threading import Barrier
import unittest
from unittest import mock

import h5py
import torch
import yaml

from examples.LIBERO.edl_pred.config import load_config
import examples.LIBERO.edl_pred.summarize_sweep as summary_module
from examples.LIBERO.edl_pred.summarize_sweep import (
    METRIC_FIELDS,
    PLOT_FIELDS,
    _plot_comparison,
    require_lexical_regular_file,
    summarize_sweep,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SWEEP_CONFIG_DIR = PACKAGE_ROOT / "configs" / "sweep"
LAUNCHER_PATH = PACKAGE_ROOT / "run_gpu_sweep.sh"
SWEEP_ID = "all_suites_seed7_v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

EXPECTED_MATRIX = {
    0: ("all_mlp_flat_softmax.yaml", "all_mlp_flat_softmax_seed7", "mlp_flat", "softmax", "none"),
    1: ("all_mlp_flat_edl.yaml", "all_mlp_flat_edl_seed7", "mlp_flat", "edl", "none"),
    2: (
        "all_attention_pool_softmax_sinusoidal.yaml",
        "all_attention_pool_softmax_sinusoidal_seed7",
        "token_attention_pool",
        "softmax",
        "sinusoidal",
    ),
    3: (
        "all_attention_pool_edl_sinusoidal.yaml",
        "all_attention_pool_edl_sinusoidal_seed7",
        "token_attention_pool",
        "edl",
        "sinusoidal",
    ),
    4: (
        "all_self_attention_softmax_sinusoidal.yaml",
        "all_self_attention_softmax_sinusoidal_seed7",
        "token_self_attention",
        "softmax",
        "sinusoidal",
    ),
    5: (
        "all_self_attention_edl_sinusoidal.yaml",
        "all_self_attention_edl_sinusoidal_seed7",
        "token_self_attention",
        "edl",
        "sinusoidal",
    ),
    6: (
        "all_attention_pool_edl_none.yaml",
        "all_attention_pool_edl_none_seed7",
        "token_attention_pool",
        "edl",
        "none",
    ),
    7: (
        "all_attention_pool_edl_learned.yaml",
        "all_attention_pool_edl_learned_seed7",
        "token_attention_pool",
        "edl",
        "learned",
    ),
}


class SweepConfigTest(unittest.TestCase):
    def test_sweep_configs_have_exact_unique_non_overwriting_matrix(self) -> None:
        configs = {path.name: load_config(path) for path in sorted(SWEEP_CONFIG_DIR.glob("*.yaml"))}
        default = load_config(PACKAGE_ROOT / "configs" / "default.yaml")

        self.assertEqual(set(configs), {row[0] for row in EXPECTED_MATRIX.values()})
        self.assertEqual(len({config.run.name for config in configs.values()}), 8)
        self.assertEqual(len({config.run.output_root / config.run.name for config in configs.values()}), 8)
        for gpu, (filename, run_name, encoder, head, position) in EXPECTED_MATRIX.items():
            self.assertEqual(gpu, list(EXPECTED_MATRIX).index(gpu))
            config = configs[filename]
            self.assertEqual(config.run.name, run_name)
            self.assertFalse(config.run.overwrite)
            self.assertEqual(config.run.seed, 7)
            self.assertEqual(config.data.split_seed, 7)
            self.assertEqual(config.data.selected_suites, SUITES)
            self.assertEqual(tuple(config.data.datasets), SUITES)
            self.assertEqual(config.training, default.training)
            self.assertEqual(config.edl, default.edl)
            self.assertEqual(config.loss, default.loss)
            self.assertEqual(config.model.chunk_encoder, encoder)
            self.assertEqual(config.model.head, head)
            self.assertEqual(config.model.token_position_encoding, position)
            self.assertEqual(config.run.output_root, (PACKAGE_ROOT / "outputs/sweeps/all_suites_seed7_v1/runs").resolve())
            for path in config.data.datasets.values():
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())

    def test_launcher_has_exact_gpu_file_pairs_and_hardening_gates(self) -> None:
        source = LAUNCHER_PATH.read_text(encoding="utf-8")
        pairs = [(int(gpu), filename) for gpu, filename in re.findall(r'^\s*"(\d+)\|([^"|]+\.yaml)\|', source, re.MULTILINE)]

        self.assertEqual(pairs, [(gpu, row[0]) for gpu, row in EXPECTED_MATRIX.items()])
        self.assertIn("CANONICAL_RECORDS", source)
        self.assertIn("DEFAULT_CONFIG", source)
        self.assertIn("assert_no_symlink_ancestors", source)
        self.assertIn("LAUNCH_RESERVATION", source)
        self.assertIn("set -C", source)
        self.assertIn("noclobber", source)
        self.assertIn("sweep_summary.json", source)
        self.assertIn("sweep_summary.csv", source)
        self.assertIn("sweep_comparison.png", source)
        self.assertIn('[[ -e "${SWEEP_ROOT}" || -L "${SWEEP_ROOT}" ]]', source)
        self.assertIn("malformed GPU query row", source)
        self.assertIn("duplicate GPU query index", source)
        self.assertIn("raw YAML dataset paths do not match canonical relative paths", source)
        self.assertIn("require_lexical_regular_file", source)
        self.assertIn("CUDA_VISIBLE_DEVICES=\"${gpu}\" conda run -n starvla python", source)
        self.assertIn("if wait \"${pid}\"; then", source)


class SweepSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sweep_root = Path(self.temp_dir.name) / SWEEP_ID
        self._write_completed_sweep()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_summary_enforces_canonical_runs_and_preserves_null_metrics(self) -> None:
        rows = summarize_sweep(self.sweep_root)

        self.assertEqual(len(rows), 8)
        self.assertEqual([row["gpu_index"] for row in rows], list(range(8)))
        tied = next(row for row in rows if row["gpu_index"] == 0)
        self.assertEqual(tied["best_epoch"], 2)
        self.assertEqual(tied["best_validation_loss"], 0.2)
        self.assertIsNone(tied["chunk_roc_auc"])
        self.assertIsNone(tied["final_chunk_pr_auc"])
        self.assertIsNone(next(row for row in rows if row["gpu_index"] == 1)["verifier_eu_label_0"])

        summary_json = json.loads((self.sweep_root / "sweep_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary_json, rows)
        with (self.sweep_root / "sweep_summary.csv").open(newline="", encoding="utf-8") as handle:
            self.assertEqual([row["gpu_index"] for row in csv.DictReader(handle)], [str(index) for index in range(8)])
        self.assertEqual((self.sweep_root / "sweep_comparison.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_summary_rejects_noncanonical_manifest_probes(self) -> None:
        manifest_path = self.sweep_root / "sweep_manifest.json"
        base = json.loads(manifest_path.read_text(encoding="utf-8"))
        outside_log = Path(self.temp_dir.name) / "outside.log"
        outside_log.write_text("outside\n", encoding="utf-8")
        probes = {
            "wrong_sweep_id": {**base, "sweep_id": "other"},
            "seven_runs": {**base, "runs": base["runs"][:-1]},
            "wrong_gpu": {**base, "runs": [{**row, "gpu_index": 8} if row["gpu_index"] == 7 else row for row in base["runs"]]},
            "outside_log": {
                **base,
                "runs": [{**row, "log_path": str(outside_log)} if row["gpu_index"] == 0 else row for row in base["runs"]],
            },
        }
        for name, payload in probes.items():
            with self.subTest(name=name):
                manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    summarize_sweep(self.sweep_root)
        manifest_path.write_text(json.dumps(base), encoding="utf-8")

    def test_summary_rejects_symlinked_ancestor_and_publication_destination(self) -> None:
        runs = self.sweep_root / "runs"
        external = Path(self.temp_dir.name) / "external-runs"
        runs.rename(external)
        runs.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink ancestor"):
            summarize_sweep(self.sweep_root)

        runs.unlink()
        external.rename(runs)
        sentinel = Path(self.temp_dir.name) / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        (self.sweep_root / "sweep_comparison.png").symlink_to(sentinel)
        with self.assertRaisesRegex(ValueError, "symlink"):
            summarize_sweep(self.sweep_root, overwrite=True)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_summary_rejects_invalid_completed_run_artifacts(self) -> None:
        run_name = EXPECTED_MATRIX[0][1]
        run_dir = self.sweep_root / "runs" / run_name
        probes = {
            "bad_split": (run_dir / "split_manifest.json", b"not-json"),
            "empty_png": (run_dir / "training_curves.png", b""),
            "bad_hdf5": (run_dir / "validation_predictions.hdf5", b"not-hdf5"),
            "empty_checkpoint": (run_dir / "checkpoints/best.pt", b""),
        }
        for name, (path, content) in probes.items():
            with self.subTest(name=name):
                original = path.read_bytes()
                path.write_bytes(content)
                with self.assertRaises(ValueError):
                    summarize_sweep(self.sweep_root)
                path.write_bytes(original)

    def test_summary_rejects_prediction_and_checkpoint_metadata_mismatch(self) -> None:
        run_name = EXPECTED_MATRIX[0][1]
        run_dir = self.sweep_root / "runs" / run_name
        prediction_path = run_dir / "validation_predictions.hdf5"
        with h5py.File(prediction_path, "w") as handle:
            handle.create_group("episodes")
        with self.assertRaisesRegex(ValueError, "prediction identities"):
            summarize_sweep(self.sweep_root)
        self._write_predictions(prediction_path, run_name)
        torch.save({"epoch": 99, "validation_total_loss": 0.2}, run_dir / "checkpoints/best.pt")
        with self.assertRaisesRegex(ValueError, "best checkpoint epoch"):
            summarize_sweep(self.sweep_root)

    def test_summary_rejects_resolved_model_drift_and_refuses_implicit_overwrite(self) -> None:
        run_name = EXPECTED_MATRIX[0][1]
        resolved_path = self.sweep_root / "runs" / run_name / "resolved_config.yaml"
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        resolved["model"]["head"] = "edl"
        resolved_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "resolved model metadata"):
            summarize_sweep(self.sweep_root)

        self._write_run_artifacts(self.sweep_root / "runs" / run_name, run_name, 0, "mlp_flat", "softmax", "none")
        summarize_sweep(self.sweep_root)
        with self.assertRaises(FileExistsError):
            summarize_sweep(self.sweep_root)
        self.assertEqual(len(summarize_sweep(self.sweep_root, overwrite=True)), 8)

    def test_all_null_plot_panels_are_annotated_without_zero_bars(self) -> None:
        rows = [{"run_name": "null-run", **{field: None for field in METRIC_FIELDS}, "best_validation_loss": None}]
        output = Path(self.temp_dir.name) / "all-null.png"
        with mock.patch("matplotlib.axes._axes.Axes.text", autospec=True) as text:
            _plot_comparison(output, rows)
        self.assertEqual(text.call_count, len(PLOT_FIELDS))
        self.assertTrue(output.is_file())

    def test_lexical_regular_file_probe_rejects_final_component_symlinks(self) -> None:
        directory = Path(self.temp_dir.name) / "lexical"
        directory.mkdir()
        target = directory / "target"
        target.write_text("content", encoding="utf-8")
        config_link = directory / "canonical-config.yaml"
        dataset_link = directory / "canonical-dataset.hdf5"
        config_link.symlink_to(target)
        dataset_link.symlink_to(target)

        for path in (config_link, dataset_link):
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, "symlink"):
                require_lexical_regular_file(path, "canonical input")
        require_lexical_regular_file(target, "canonical input")

    def test_concurrent_default_summary_has_exactly_one_success(self) -> None:
        barrier = Barrier(2)

        def publish() -> str:
            barrier.wait(timeout=5)
            try:
                summarize_sweep(self.sweep_root)
            except FileExistsError:
                return "refused"
            return "success"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [future.result(timeout=20) for future in (executor.submit(publish), executor.submit(publish))]
        self.assertEqual(sorted(results), ["refused", "success"])
        self.assertFalse((self.sweep_root / ".summary-reservation").exists())

    def test_publication_failure_rolls_back_every_position_in_both_modes(self) -> None:
        for overwrite in (False, True):
            for position in range(1, 4):
                with self.subTest(overwrite=overwrite, position=position):
                    self._remove_summary_outputs()
                    old = {}
                    if overwrite:
                        for name in summary_module.SUMMARY_FILENAMES:
                            path = self.sweep_root / name
                            path.write_bytes(f"old-{name}".encode("utf-8"))
                            old[name] = path.read_bytes()
                    original = summary_module._publish_regular
                    calls = 0

                    def fail_at_position(stage: Path, destination: Path, *, overwrite: bool) -> None:
                        nonlocal calls
                        calls += 1
                        if calls == position:
                            raise OSError("injected publication failure")
                        original(stage, destination, overwrite=overwrite)

                    with mock.patch.object(summary_module, "_publish_regular", side_effect=fail_at_position):
                        with self.assertRaisesRegex(OSError, "injected publication failure"):
                            summarize_sweep(self.sweep_root, overwrite=overwrite)
                    for name in summary_module.SUMMARY_FILENAMES:
                        path = self.sweep_root / name
                        if overwrite:
                            self.assertEqual(path.read_bytes(), old[name])
                        else:
                            self.assertFalse(path.exists())
                    self.assertEqual(self._publication_leftovers(), [])
                    self.assertFalse((self.sweep_root / ".summary-reservation").exists())

    def _write_completed_sweep(self) -> None:
        self.sweep_root.mkdir(parents=True)
        runs: list[dict[str, object]] = []
        for gpu, (filename, run_name, encoder, head, position) in EXPECTED_MATRIX.items():
            output_path = self.sweep_root / "runs" / run_name
            log_path = self.sweep_root / "logs" / f"{run_name}.log"
            output_path.mkdir(parents=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("fake log\n", encoding="utf-8")
            self._write_run_artifacts(output_path, run_name, gpu, encoder, head, position)
            runs.append(
                {
                    "gpu_index": gpu,
                    "config_path": str((SWEEP_CONFIG_DIR / filename).resolve()),
                    "run_name": run_name,
                    "output_path": str(output_path.resolve()),
                    "log_path": str(log_path.resolve()),
                }
            )
        (self.sweep_root / "sweep_manifest.json").write_text(
            json.dumps({"sweep_id": SWEEP_ID, "started_at": "2026-08-02T00:00:00Z", "runs": runs}), encoding="utf-8"
        )

    def _write_run_artifacts(
        self, output_path: Path, run_name: str, gpu: int, encoder: str, head: str, position: str
    ) -> None:
        resolved_config = {
            "run": {"name": run_name},
            "model": {"chunk_encoder": encoder, "head": head, "token_position_encoding": position},
        }
        (output_path / "resolved_config.yaml").write_text(yaml.safe_dump(resolved_config), encoding="utf-8")
        split = {"max_action_tokens": 1, "train": [], "validation": [{"suite": "libero_spatial", "episode_key": run_name, "label": 1}]}
        (output_path / "split_manifest.json").write_text(json.dumps(split), encoding="utf-8")
        records = [self._record(1, 0.4 + gpu / 100.0, gpu, head), self._record(2, 0.2 + gpu / 100.0, gpu, head), self._record(3, 0.2 + gpu / 100.0, gpu, head)]
        (output_path / "metrics.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        (output_path / "training_curves.png").write_bytes(b"\x89PNG\r\n\x1a\nplaceholder")
        checkpoint_dir = output_path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save({"epoch": 2, "validation_total_loss": 0.2 + gpu / 100.0}, checkpoint_dir / "best.pt")
        torch.save({"epoch": 3, "validation_total_loss": 0.2 + gpu / 100.0}, checkpoint_dir / "last.pt")
        self._write_predictions(output_path / "validation_predictions.hdf5", run_name)

    @staticmethod
    def _write_predictions(path: Path, run_name: str) -> None:
        with h5py.File(path, "w") as handle:
            episodes = handle.create_group("episodes")
            episode = episodes.create_group("libero_spatial").create_group(run_name)
            episode.create_dataset("success_probability", data=[0.5])

    @staticmethod
    def _record(epoch: int, loss: float, gpu: int, head: str) -> dict[str, object]:
        record: dict[str, object] = {
            "epoch": epoch,
            "validation_total_loss": loss,
            "validation_chunk_roc_auc": None if gpu == 0 else 0.6,
            "validation_chunk_pr_auc": 0.5,
            "validation_chunk_brier": 0.2,
            "validation_chunk_ece": 0.1,
            "validation_final_chunk_roc_auc": 0.7,
            "validation_final_chunk_pr_auc": None if gpu == 0 else 0.65,
            "validation_final_chunk_accuracy": 0.8,
        }
        if head == "edl":
            record.update({"validation_verifier_au_label_0": 0.3, "validation_verifier_au_label_1": 0.4, "validation_verifier_eu_label_0": None if gpu == 1 else 0.2, "validation_verifier_eu_label_1": 0.1, "validation_verifier_total_evidence_label_0": 2.0, "validation_verifier_total_evidence_label_1": 3.0})
        return record

    def _remove_summary_outputs(self) -> None:
        for name in summary_module.SUMMARY_FILENAMES:
            (self.sweep_root / name).unlink(missing_ok=True)

    def _publication_leftovers(self) -> list[str]:
        return sorted(path.name for path in self.sweep_root.iterdir() if path.name.startswith(".stage-") or path.name.startswith(".backup-"))


if __name__ == "__main__":
    unittest.main()
