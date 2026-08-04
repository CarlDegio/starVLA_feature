import pathlib
import tempfile
import unittest

import yaml

from examples.LIBERO.edl_pred.config import config_to_dict, load_config


def write_minimal_config(
    temp_dir: tempfile.TemporaryDirectory[str],
    *,
    chunk_encoder: object = "token_attention_pool",
    position: str = "sinusoidal",
    token_embed_dim: int = 32,
    attention_heads: int = 1,
    learning_rate: object = 0.001,
    kl_weight: object = 0.001,
) -> pathlib.Path:
    """Write a complete valid config, with only tested fields overrideable."""
    path = pathlib.Path(temp_dir.name) / "config.yaml"
    config = {
        "run": {
            "name": "test",
            "output_root": "outputs",
            "seed": 7,
            "overwrite": False,
        },
        "data": {
            "datasets": {"libero_spatial": "data/spatial.hdf5"},
            "selected_suites": ["libero_spatial"],
            "validation_ratio": 0.1,
            "split_seed": 7,
            "max_action_tokens": "auto",
        },
        "model": {
            "chunk_encoder": chunk_encoder,
            "token_position_encoding": position,
            "token_embed_dim": token_embed_dim,
            "chunk_embed_dim": 64,
            "attention_heads": attention_heads,
            "self_attention_layers": 1,
            "encoder_dropout": 0.1,
            "lstm_hidden_dim": 64,
            "lstm_layers": 1,
            "lstm_dropout": 0.0,
            "head": "edl",
        },
        "edl": {
            "evidence_activation": "softplus",
            "kl_weight": kl_weight,
            "kl_anneal_epochs": 20,
        },
        "loss": {"class_balance": "none"},
        "training": {
            "device": "auto",
            "epochs": 200,
            "batch_size": 64,
            "learning_rate": learning_rate,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 30,
            "num_workers": 0,
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def test_paths_resolve_relative_to_yaml(self) -> None:
        config_path = write_minimal_config(self.temp_dir)
        config = load_config(config_path)
        self.assertEqual(
            config.data.datasets["libero_spatial"],
            (config_path.parent / "data/spatial.hdf5").resolve(),
        )
        self.assertEqual(config.data.max_action_tokens, None)

    def test_rejects_unknown_encoder_and_position_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_encoder"):
            load_config(write_minimal_config(self.temp_dir, chunk_encoder="transformer"))
        with self.assertRaisesRegex(ValueError, "token_position_encoding"):
            load_config(write_minimal_config(self.temp_dir, position="absolute_time"))

    def test_attention_dimension_must_divide_head_count(self) -> None:
        path = write_minimal_config(self.temp_dir, token_embed_dim=30, attention_heads=4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            load_config(path)

    def test_rejects_nonfinite_numeric_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "training.learning_rate"):
            load_config(write_minimal_config(self.temp_dir, learning_rate=float("nan")))
        with self.assertRaisesRegex(ValueError, "edl.kl_weight"):
            load_config(write_minimal_config(self.temp_dir, kl_weight=float("inf")))

    def test_rejects_non_string_enum_values_with_field_qualified_error(self) -> None:
        path = write_minimal_config(self.temp_dir, chunk_encoder=["token_attention_pool"])
        with self.assertRaisesRegex(ValueError, "model.chunk_encoder"):
            load_config(path)

    def test_rejects_missing_and_unknown_top_level_sections(self) -> None:
        missing_path = write_minimal_config(self.temp_dir)
        missing = yaml.safe_load(missing_path.read_text(encoding="utf-8"))
        del missing["training"]
        missing_path.write_text(yaml.safe_dump(missing), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            load_config(missing_path)

        unknown_path = write_minimal_config(self.temp_dir)
        unknown = yaml.safe_load(unknown_path.read_text(encoding="utf-8"))
        unknown["unexpected"] = True
        unknown_path.write_text(yaml.safe_dump(unknown), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            load_config(unknown_path)

    def test_config_to_dict_serializes_paths_and_tuples(self) -> None:
        config_path = write_minimal_config(self.temp_dir)
        config_dict = config_to_dict(load_config(config_path))
        self.assertEqual(config_dict["source_path"], str(config_path.resolve()))
        self.assertEqual(
            config_dict["data"]["datasets"]["libero_spatial"],
            str((config_path.parent / "data/spatial.hdf5").resolve()),
        )
        self.assertEqual(config_dict["data"]["selected_suites"], ["libero_spatial"])

    def test_shipped_configs_load_with_expected_batch_sizes(self) -> None:
        config_dir = pathlib.Path(__file__).resolve().parents[1] / "configs"
        default = load_config(config_dir / "default.yaml")
        smoke = load_config(config_dir / "smoke.yaml")
        self.assertEqual(default.training.batch_size, 64)
        self.assertEqual(smoke.training.batch_size, 8)


if __name__ == "__main__":
    unittest.main()
