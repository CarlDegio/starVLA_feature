import importlib
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def _import_eval_libero():
    libero_module = types.ModuleType("libero")
    libero_libero_module = types.ModuleType("libero.libero")
    libero_libero_module.benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {})
    libero_libero_module.get_libero_path = lambda _: ""
    libero_envs_module = types.ModuleType("libero.libero.envs")
    libero_envs_module.OffScreenRenderEnv = object

    model_interface_module = types.ModuleType("examples.LIBERO.eval_files.model2libero_interface")
    model_interface_module.ModelClient = object

    module_stubs = {
        "libero": libero_module,
        "libero.libero": libero_libero_module,
        "libero.libero.envs": libero_envs_module,
        "examples.LIBERO.eval_files.model2libero_interface": model_interface_module,
    }
    with mock.patch.dict(sys.modules, module_stubs):
        sys.modules.pop("examples.LIBERO.eval_files.eval_libero", None)
        return importlib.import_module("examples.LIBERO.eval_files.eval_libero")


class UncertaintyPlotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval_libero = _import_eval_libero()

    def test_chunk_time_panel_uses_fixed_chunk_index_scale(self):
        captured = {}

        def capture_savefig(figure, path, **kwargs):
            captured["figure"] = figure
            captured["path"] = pathlib.Path(path)

        chunks = [
            {
                "chunk_idx": 0,
                "action_token_aleatoric_uncertainty": [0.1, 0.2, 0.3, 0.4, 0.5],
                "action_token_epistemic_uncertainty": [0.1, 0.2, 0.3, 0.4, 0.9],
                "action_token_evidence": [3.0, 5.0, 3.0, 5.0, 5.0],
                "worst_token_count": 3,
                "worst_token_eu_mean": 0.123,
            },
            {
                "chunk_idx": 2,
                "action_token_aleatoric_uncertainty": [0.5],
                "action_token_epistemic_uncertainty": [0.6],
                "action_token_evidence": [7.0],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            rollout_base = pathlib.Path(tmp_dir) / "rollout"
            with mock.patch.object(Figure, "savefig", new=capture_savefig):
                self.eval_libero._save_uncertainty_artifacts(
                    chunks,
                    rollout_base,
                    max_steps=300,
                    action_chunk_size=8,
                )

        figure = captured["figure"]
        self.addCleanup(plt.close, figure)
        self.assertEqual(len(figure.axes), 10)
        self.assertFalse(figure.axes[3].axison)

        low_evidence_axis = figure.axes[4]
        self.assertEqual(len(low_evidence_axis.containers), 2)
        np.testing.assert_array_equal(
            [bar.get_height() for bar in low_evidence_axis.containers[0]],
            [5.0, 1.0],
        )
        np.testing.assert_array_equal(
            [bar.get_height() for bar in low_evidence_axis.containers[1]],
            [2.0, 0.0],
        )
        ratio_axis = next(axis for axis in figure.axes if axis.get_ylabel() == "Low-evidence ratio")
        np.testing.assert_allclose(ratio_axis.lines[0].get_ydata(), [0.4, 0.0])
        self.assertEqual(ratio_axis.get_ylim(), (0.0, 1.0))

        top_eu_axis = figure.axes[5]
        np.testing.assert_allclose(top_eu_axis.lines[0].get_ydata(), [0.9, 0.6])
        self.assertEqual(top_eu_axis.get_title(), "Mean EU of top 20% action tokens per chunk")

        temporal_axis = figure.axes[7]
        temporal_scatter = temporal_axis.collections[0]
        np.testing.assert_array_equal(
            temporal_scatter.get_array(),
            np.asarray([0.0] * 5 + [2.0], dtype=np.float32),
        )
        self.assertEqual(temporal_scatter.norm.vmin, 0)
        self.assertEqual(temporal_scatter.norm.vmax, 37)
        earliest_color = temporal_scatter.cmap(0.0)
        self.assertGreater(earliest_color[2] - earliest_color[0], 0.1)
        self.assertEqual(temporal_axis.get_xlabel(), "Normalized AU")
        self.assertEqual(temporal_axis.get_ylabel(), "EU")

    def test_chunk_time_panel_rejects_non_positive_action_chunk_size(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "action_chunk_size must be positive"):
                self.eval_libero._save_uncertainty_artifacts(
                    [],
                    pathlib.Path(tmp_dir) / "rollout",
                    max_steps=300,
                    action_chunk_size=0,
                )


if __name__ == "__main__":
    unittest.main()
