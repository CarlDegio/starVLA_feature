"""Typed configuration loading for the standalone LIBERO EDL verifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


@dataclass(frozen=True)
class RunConfig:
    name: str
    output_root: Path
    seed: int
    overwrite: bool


@dataclass(frozen=True)
class DataConfig:
    datasets: dict[str, Path]
    selected_suites: tuple[str, ...]
    validation_ratio: float
    split_seed: int
    max_action_tokens: int | None


@dataclass(frozen=True)
class ModelConfig:
    chunk_encoder: Literal["mlp_flat", "token_attention_pool", "token_self_attention"]
    token_position_encoding: Literal["none", "sinusoidal", "learned"]
    token_embed_dim: int
    chunk_embed_dim: int
    attention_heads: int
    self_attention_layers: int
    encoder_dropout: float
    lstm_hidden_dim: int
    lstm_layers: int
    lstm_dropout: float
    head: Literal["softmax", "edl"]


@dataclass(frozen=True)
class EDLConfig:
    evidence_activation: Literal["softplus", "relu", "exp"]
    kl_weight: float
    kl_anneal_epochs: int


@dataclass(frozen=True)
class LossConfig:
    class_balance: Literal["none", "inverse_frequency"]


@dataclass(frozen=True)
class TrainingConfig:
    device: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    early_stopping_patience: int
    num_workers: int


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    edl: EDLConfig
    loss: LossConfig
    training: TrainingConfig
    source_path: Path


_TOP_LEVEL_SECTIONS = {"run", "data", "model", "edl", "loss", "training"}
_SECTION_FIELDS = {
    "run": {"name", "output_root", "seed", "overwrite"},
    "data": {"datasets", "selected_suites", "validation_ratio", "split_seed", "max_action_tokens"},
    "model": {
        "chunk_encoder",
        "token_position_encoding",
        "token_embed_dim",
        "chunk_embed_dim",
        "attention_heads",
        "self_attention_layers",
        "encoder_dropout",
        "lstm_hidden_dim",
        "lstm_layers",
        "lstm_dropout",
        "head",
    },
    "edl": {"evidence_activation", "kl_weight", "kl_anneal_epochs"},
    "loss": {"class_balance"},
    "training": {
        "device",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "early_stopping_patience",
        "num_workers",
    },
}


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a verifier YAML config, resolving paths from its directory."""
    source_path = Path(path).expanduser().resolve()
    with source_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    config = _mapping(raw, "top-level config")
    _require_exact_keys(config, _TOP_LEVEL_SECTIONS, "top-level config")
    sections = {name: _mapping(config[name], name) for name in _TOP_LEVEL_SECTIONS}
    for name, fields in _SECTION_FIELDS.items():
        _require_exact_keys(sections[name], fields, name)

    base_dir = source_path.parent
    run = _parse_run(sections["run"], base_dir)
    data = _parse_data(sections["data"], base_dir)
    model = _parse_model(sections["model"])
    edl = _parse_edl(sections["edl"])
    loss = _parse_loss(sections["loss"])
    training = _parse_training(sections["training"])
    return ExperimentConfig(run, data, model, edl, loss, training, source_path)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert a configuration to plain, serialization-friendly Python values."""
    return _to_plain(asdict(config))


def _parse_run(raw: Mapping[str, Any], base_dir: Path) -> RunConfig:
    name = _nonempty_string(raw["name"], "run.name")
    output_root = _resolve_path(raw["output_root"], base_dir, "run.output_root")
    return RunConfig(name, output_root, _integer(raw["seed"], "run.seed"), _boolean(raw["overwrite"], "run.overwrite"))


def _parse_data(raw: Mapping[str, Any], base_dir: Path) -> DataConfig:
    raw_datasets = _mapping(raw["datasets"], "data.datasets")
    if not raw_datasets:
        raise ValueError("data.datasets must not be empty")
    datasets = {
        _nonempty_string(suite, "data.datasets key"): _resolve_path(value, base_dir, f"data.datasets.{suite}")
        for suite, value in raw_datasets.items()
    }
    suites_raw = raw["selected_suites"]
    if not isinstance(suites_raw, list) or not suites_raw:
        raise ValueError("data.selected_suites must be a non-empty list")
    selected_suites = tuple(_nonempty_string(suite, "data.selected_suites item") for suite in suites_raw)
    if len(set(selected_suites)) != len(selected_suites):
        raise ValueError("data.selected_suites must not contain duplicates")
    missing_suites = set(selected_suites) - set(datasets)
    if missing_suites:
        raise ValueError(f"data.selected_suites missing dataset paths: {sorted(missing_suites)}")
    validation_ratio = _number(raw["validation_ratio"], "data.validation_ratio")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("data.validation_ratio must be between 0 and 1")
    max_action_tokens = raw["max_action_tokens"]
    if max_action_tokens == "auto":
        parsed_max_action_tokens = None
    else:
        parsed_max_action_tokens = _positive_integer(max_action_tokens, "data.max_action_tokens")
    return DataConfig(
        datasets,
        selected_suites,
        validation_ratio,
        _integer(raw["split_seed"], "data.split_seed"),
        parsed_max_action_tokens,
    )


def _parse_model(raw: Mapping[str, Any]) -> ModelConfig:
    chunk_encoder = _choice(raw["chunk_encoder"], "model.chunk_encoder", {"mlp_flat", "token_attention_pool", "token_self_attention"})
    token_position_encoding = _choice(raw["token_position_encoding"], "model.token_position_encoding", {"none", "sinusoidal", "learned"})
    token_embed_dim = _positive_integer(raw["token_embed_dim"], "model.token_embed_dim")
    attention_heads = _positive_integer(raw["attention_heads"], "model.attention_heads")
    if token_embed_dim % attention_heads != 0:
        raise ValueError("model.token_embed_dim must be divisible by model.attention_heads")
    return ModelConfig(
        chunk_encoder,
        token_position_encoding,
        token_embed_dim,
        _positive_integer(raw["chunk_embed_dim"], "model.chunk_embed_dim"),
        attention_heads,
        _positive_integer(raw["self_attention_layers"], "model.self_attention_layers"),
        _dropout(raw["encoder_dropout"], "model.encoder_dropout"),
        _positive_integer(raw["lstm_hidden_dim"], "model.lstm_hidden_dim"),
        _positive_integer(raw["lstm_layers"], "model.lstm_layers"),
        _dropout(raw["lstm_dropout"], "model.lstm_dropout"),
        _choice(raw["head"], "model.head", {"softmax", "edl"}),
    )


def _parse_edl(raw: Mapping[str, Any]) -> EDLConfig:
    return EDLConfig(
        _choice(raw["evidence_activation"], "edl.evidence_activation", {"softplus", "relu", "exp"}),
        _nonnegative_number(raw["kl_weight"], "edl.kl_weight"),
        _integer(raw["kl_anneal_epochs"], "edl.kl_anneal_epochs"),
    )


def _parse_loss(raw: Mapping[str, Any]) -> LossConfig:
    return LossConfig(_choice(raw["class_balance"], "loss.class_balance", {"none", "inverse_frequency"}))


def _parse_training(raw: Mapping[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        _nonempty_string(raw["device"], "training.device"),
        _positive_integer(raw["epochs"], "training.epochs"),
        _positive_integer(raw["batch_size"], "training.batch_size"),
        _positive_number(raw["learning_rate"], "training.learning_rate"),
        _nonnegative_number(raw["weight_decay"], "training.weight_decay"),
        _positive_number(raw["gradient_clip_norm"], "training.gradient_clip_norm"),
        _positive_integer(raw["early_stopping_patience"], "training.early_stopping_patience"),
        _nonnegative_integer(raw["num_workers"], "training.num_workers"),
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _require_exact_keys(raw: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = expected - set(raw)
    unknown = set(raw) - expected
    if missing:
        raise ValueError(f"{field} missing required fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{field} has unknown fields: {sorted(unknown)}")


def _resolve_path(value: Any, base_dir: Path, field: str) -> Path:
    return (base_dir / _nonempty_string(value, field)).resolve()


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _positive_integer(value: Any, field: str) -> int:
    value = _integer(value, field)
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _nonnegative_integer(value: Any, field: str) -> int:
    value = _integer(value, field)
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _positive_number(value: Any, field: str) -> float:
    value = _number(value, field)
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def _nonnegative_number(value: Any, field: str) -> float:
    value = _number(value, field)
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _dropout(value: Any, field: str) -> float:
    value = _number(value, field)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{field} must be in [0, 1)")
    return value


def _choice(value: Any, field: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}")
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, Mapping):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value
