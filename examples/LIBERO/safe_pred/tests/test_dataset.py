from __future__ import annotations

import pathlib
import tempfile
import unittest

import numpy as np

from examples.LIBERO.safe_pred.dataset import collate_episodes, load_episodes, split_episodes_by_suite
from examples.LIBERO.safe_pred.tests.helpers import write_safe_dataset


class SafeDatasetTest(unittest.TestCase):
    def test_loads_selected_aggregation_and_collates_variable_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_safe_dataset(
                pathlib.Path(temp_dir) / "goal.hdf5",
                "libero_goal",
                [True, False, True, False],
            )
            episodes = load_episodes([path], feature_aggregation="last")

        self.assertEqual(len(episodes), 4)
        self.assertEqual(episodes[0].features.shape, (1, 4))
        np.testing.assert_allclose(episodes[0].features[0], [0.5, 1.5, 2.5, 3.5])
        longest = max(episodes, key=lambda episode: episode.features.shape[0])
        batch = collate_episodes([episodes[0], longest])
        self.assertEqual(tuple(batch.features.shape), (2, 3, 4))
        np.testing.assert_array_equal(batch.chunk_mask.numpy(), [[True, False, False], [True, True, True]])
        np.testing.assert_array_equal(batch.failure_labels.numpy(), [0.0, 0.0])

    def test_splits_each_suite_at_episode_level_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            paths = [
                write_safe_dataset(root / "goal.hdf5", "libero_goal", [True, False, True, False]),
                write_safe_dataset(root / "object.hdf5", "libero_object", [False, True, False, True]),
            ]
            episodes = load_episodes(paths, feature_aggregation="mean")
            train_a, val_a = split_episodes_by_suite(episodes, val_fraction=0.25, seed=17)
            train_b, val_b = split_episodes_by_suite(episodes, val_fraction=0.25, seed=17)

        self.assertEqual([episode.episode_id for episode in val_a], [episode.episode_id for episode in val_b])
        self.assertEqual(len(train_a), 6)
        self.assertEqual(len(val_a), 2)
        self.assertEqual({episode.suite for episode in val_a}, {"libero_goal", "libero_object"})

    def test_rejects_hidden_dimension_mismatch_across_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            paths = [
                write_safe_dataset(root / "a.hdf5", "libero_goal", [True, False], hidden_dim=4),
                write_safe_dataset(root / "b.hdf5", "libero_object", [True, False], hidden_dim=3),
            ]
            with self.assertRaisesRegex(ValueError, "hidden dimension"):
                load_episodes(paths, feature_aggregation="last")


if __name__ == "__main__":
    unittest.main()
