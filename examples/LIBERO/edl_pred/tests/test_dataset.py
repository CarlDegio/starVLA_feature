from __future__ import annotations

import multiprocessing
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from examples.LIBERO.edl_pred.config import DataConfig
from examples.LIBERO.edl_pred.dataset import (
    EpisodeRef,
    TrajectoryDataset,
    build_episode_splits,
    collate_trajectories,
)
from examples.LIBERO.edl_pred.tests.helpers import (
    decode_preloaded_dataset_after_fork,
    write_uncertainty_hdf5,
)


class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.spatial_path = write_uncertainty_hdf5(
            root / "spatial.hdf5",
            [
                ("task_000_episode_0000", 0, [3, 5]),
                ("task_000_episode_0001", 0, [2, 4]),
                ("task_001_episode_0000", 1, [2, 3]),
                ("task_001_episode_0001", 1, [1, 4, 5]),
            ],
        )
        self.object_path = write_uncertainty_hdf5(
            root / "object.hdf5",
            [
                ("task_000_episode_0000", 0, [1, 5]),
                ("task_000_episode_0001", 0, [3, 4]),
                ("task_001_episode_0000", 1, [2, 4]),
                ("task_001_episode_0001", 1, [3, 5]),
            ],
        )
        self.data_config = DataConfig(
            datasets={"libero_object": self.object_path, "libero_spatial": self.spatial_path},
            selected_suites=("libero_object", "libero_spatial"),
            validation_ratio=0.5,
            split_seed=7,
            max_action_tokens=None,
        )
        self.ref = EpisodeRef("libero_spatial", self.spatial_path, "task_000_episode_0000", 0)
        self.short_sample = TrajectoryDataset([self.ref], max_action_tokens=5)[0]
        long_ref = EpisodeRef("libero_spatial", self.spatial_path, "task_001_episode_0001", 1)
        self.long_sample = TrajectoryDataset([long_ref], max_action_tokens=5)[0]

    def test_split_is_stratified_deterministic_and_episode_disjoint(self) -> None:
        # A changed seed application or cross-label split would make this fail.
        first = build_episode_splits(self.data_config, seed=11)
        second = build_episode_splits(self.data_config, seed=11)

        self.assertEqual(first, second)
        self.assertFalse(set(first.train) & set(first.validation))
        self.assertEqual(first.max_action_tokens, 5)
        for suite in self.data_config.selected_suites:
            labels = {ref.label for ref in first.validation if ref.suite == suite}
            self.assertEqual(labels, {0, 1})

    def test_decodes_offsets_into_ordered_au_eu_chunks(self) -> None:
        # A decoder that ignores offsets or swaps AU/EU would make this fail.
        sample = TrajectoryDataset([self.ref], max_action_tokens=5)[0]

        self.assertEqual(sample.features.shape, (2, 5, 2))
        self.assertEqual(sample.token_mask.sum(axis=1).tolist(), [3, 5])
        np.testing.assert_array_equal(sample.features[0, :3, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(sample.features[0, :3, 1], [1001.0, 1002.0, 1003.0])
        self.assertTrue(np.all(sample.features[~sample.token_mask] == 0))

    def test_collate_builds_independent_token_and_chunk_masks(self) -> None:
        # A collator that derives chunk presence from token padding would make this fail.
        batch = collate_trajectories([self.short_sample, self.long_sample])

        self.assertEqual(tuple(batch["features"].shape), (2, 3, 5, 2))
        self.assertEqual(batch["chunk_lengths"].tolist(), [2, 3])
        self.assertFalse(batch["chunk_mask"][0, 2])
        self.assertTrue(np.all(batch["token_mask"][0, 2] == 0))

    def test_rejects_malformed_offsets(self) -> None:
        # Missing offset boundaries must not be silently reinterpreted.
        self._replace_dataset(self.spatial_path, "task_000_episode_0000", "token_offsets", [0, 3])

        with self.assertRaisesRegex(ValueError, "token_offsets"):
            TrajectoryDataset([self.ref], max_action_tokens=5)[0]

    def test_rejects_au_eu_length_mismatch(self) -> None:
        # A decoder that truncates one uncertainty stream would make this fail.
        self._replace_dataset(
            self.spatial_path,
            "task_000_episode_0000",
            "epistemic_uncertainty",
            np.arange(7, dtype=np.float32),
        )

        with self.assertRaisesRegex(ValueError, "aleatoric_uncertainty.*epistemic_uncertainty"):
            TrajectoryDataset([self.ref], max_action_tokens=5)[0]

    def test_rejects_nonfinite_uncertainty_values(self) -> None:
        # A missing finite-value check would permit an invalid model feature.
        self._replace_dataset(
            self.spatial_path,
            "task_000_episode_0000",
            "aleatoric_uncertainty",
            [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            TrajectoryDataset([self.ref], max_action_tokens=5)[0]

    def test_rejects_complex_aleatoric_uncertainty_values(self) -> None:
        # Casting complex AU values to float would silently lose their imaginary component.
        self._replace_dataset(
            self.spatial_path,
            "task_000_episode_0000",
            "aleatoric_uncertainty",
            np.arange(8, dtype=np.complex64) + 1j,
        )

        with self.assertRaisesRegex(ValueError, "real-valued"):
            TrajectoryDataset([self.ref], max_action_tokens=5)[0]

    def test_rejects_complex_epistemic_uncertainty_values(self) -> None:
        # Casting complex EU values to float would silently lose their imaginary component.
        self._replace_dataset(
            self.spatial_path,
            "task_000_episode_0000",
            "epistemic_uncertainty",
            np.arange(8, dtype=np.complex64) + 1j,
        )

        with self.assertRaisesRegex(ValueError, "real-valued"):
            TrajectoryDataset([self.ref], max_action_tokens=5)[0]

    def test_rejects_missing_label_while_indexing(self) -> None:
        # An indexer that defaults missing success labels would make this fail.
        with h5py.File(self.spatial_path, "r+") as handle:
            del handle["episodes/task_000_episode_0000"].attrs["success"]

        with self.assertRaisesRegex(ValueError, "success"):
            build_episode_splits(self.data_config)

    def test_rejects_noninteger_label_while_indexing(self) -> None:
        # Coercing a float label would accept data outside the collector schema.
        with h5py.File(self.spatial_path, "r+") as handle:
            handle["episodes/task_000_episode_0000"].attrs["success"] = 0.0

        with self.assertRaisesRegex(ValueError, "success"):
            build_episode_splits(self.data_config)

    def test_rejects_unknown_collector_schema_version(self) -> None:
        # Parsing a future collector layout as version 1.0 would be unsafe.
        with h5py.File(self.spatial_path, "r+") as handle:
            handle.attrs["schema_version"] = "2.0"

        with self.assertRaisesRegex(ValueError, "schema_version"):
            build_episode_splits(self.data_config)

    def test_direct_decode_rejects_unknown_collector_schema_version(self) -> None:
        # A direct EpisodeRef must not bypass the root schema compatibility check.
        dataset = TrajectoryDataset([self.ref], max_action_tokens=5)
        self.addCleanup(dataset.close)
        dataset[0]
        dataset.close()
        with h5py.File(self.spatial_path, "r+") as handle:
            handle.attrs["schema_version"] = "2.0"

        with self.assertRaisesRegex(ValueError, "schema_version"):
            dataset[0]

    @unittest.skipUnless("fork" in multiprocessing.get_all_start_methods(), "requires fork")
    def test_forked_decoder_reopens_parent_preloaded_handle(self) -> None:
        # Reusing a parent's h5py handle after fork can hang or corrupt HDF5 state.
        dataset = TrajectoryDataset([self.ref], max_action_tokens=5)
        self.addCleanup(dataset.close)
        dataset[0]
        parent_pid = dataset._handle_owner_pid
        context = multiprocessing.get_context("fork")
        result_queue = context.Queue()
        worker = context.Process(
            target=decode_preloaded_dataset_after_fork,
            args=(dataset, result_queue),
        )
        worker.start()
        worker.join(timeout=10)
        try:
            self.assertFalse(worker.is_alive(), "forked decoder did not complete")
            self.assertEqual(worker.exitcode, 0)
            result = result_queue.get(timeout=2)
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join()
            result_queue.close()
            result_queue.join_thread()

        self.assertEqual(result[0], "ok", result)
        self.assertNotEqual(result[1], parent_pid)
        self.assertEqual(result[2], result[1])
        self.assertEqual(result[3], 0)

    def test_rejects_zero_token_length_while_indexing(self) -> None:
        # An indexer that admits empty chunks would make this fail.
        self._replace_dataset(self.spatial_path, "task_000_episode_0000", "num_action_tokens", [0, 5])

        with self.assertRaisesRegex(ValueError, "num_action_tokens"):
            build_episode_splits(self.data_config)

    def test_rejects_explicit_maximum_overflow(self) -> None:
        # An explicit maximum smaller than observed data must not silently truncate samples.
        config = DataConfig(
            datasets=self.data_config.datasets,
            selected_suites=self.data_config.selected_suites,
            validation_ratio=self.data_config.validation_ratio,
            split_seed=self.data_config.split_seed,
            max_action_tokens=4,
        )

        with self.assertRaisesRegex(ValueError, "max_action_tokens"):
            build_episode_splits(config)

    def test_rejects_suite_with_only_one_label(self) -> None:
        # A stratified split without both classes would produce unusable validation metrics.
        single_label_path = write_uncertainty_hdf5(
            Path(self.temp_dir.name) / "single_label.hdf5",
            [
                ("task_000_episode_0000", 0, [2]),
                ("task_000_episode_0001", 0, [3]),
            ],
        )
        config = DataConfig(
            datasets={"libero_spatial": single_label_path},
            selected_suites=("libero_spatial",),
            validation_ratio=0.5,
            split_seed=7,
            max_action_tokens=None,
        )

        with self.assertRaisesRegex(ValueError, "both labels"):
            build_episode_splits(config)

    @staticmethod
    def _replace_dataset(path: Path, episode_key: str, name: str, values: object) -> None:
        with h5py.File(path, "r+") as handle:
            group = handle[f"episodes/{episode_key}"]
            del group[name]
            group.create_dataset(name, data=values)


if __name__ == "__main__":
    unittest.main()
