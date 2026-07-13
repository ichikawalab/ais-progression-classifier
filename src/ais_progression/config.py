"""Configuration loading: YAML defaults + CLI overrides -> typed dataclasses.

Override priority: CLI arguments > YAML file > dataclass defaults below.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    csv_path: str | None = None
    num_workers: int = 4
    batch_size: int = 32


@dataclass
class CrossValidationConfig:
    num_folds: int = 10


@dataclass
class ModelConfig:
    arch: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 2
    hidden_dim: int = 512
    dropout: float = 0.5
    freeze_backbone: bool = False


@dataclass
class TrainConfig:
    max_epochs: int = 100
    min_epochs: int = 10
    lr: float = 1.0e-5
    weight_decay: float = 1.0e-3
    warmup_epochs: int = 5
    early_stopping_patience: int = 5
    use_class_weights: bool = True
    precision: str = "bf16-mixed"
    deterministic: bool = True
    seed: int = 42


@dataclass
class AugmentConfig:
    horizontal_flip: bool = True
    random_resized_crop: bool = True
    rrc_scale: list[float] = field(default_factory=lambda: [0.5, 1.0])
    rrc_ratio: list[float] = field(default_factory=lambda: [1.0, 1.0])


@dataclass
class OutputConfig:
    dir: str = "outputs"
    run_name: str | None = None


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


_SECTION_TYPES: dict[str, type] = {
    "data": DataConfig,
    "cross_validation": CrossValidationConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "augment": AugmentConfig,
    "output": OutputConfig,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any], path: str = "") -> dict[str, Any]:
    """Merge `override` into `base` in place, raising ValueError on unknown keys."""
    for key, value in override.items():
        full_key = f"{path}.{key}" if path else key
        if key not in base:
            raise ValueError(f"Unknown config key: '{full_key}'")
        if isinstance(value, dict) and isinstance(base[key], dict):
            _deep_merge(base[key], value, full_key)
        else:
            base[key] = value
    return base


def _set_dotted(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key path (e.g. 'train.lr')."""
    parts = dotted_key.split(".")
    cursor = d
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            raise ValueError(f"Unknown config key: '{dotted_key}'")
        cursor = cursor[part]
    last = parts[-1]
    if last not in cursor:
        raise ValueError(f"Unknown config key: '{dotted_key}'")
    cursor[last] = value


def _dataclass_to_default_dict() -> dict[str, Any]:
    return asdict(Config())


def load_config(
    yaml_path: str | Path | None,
    cli_overrides: dict[str, Any] | None = None,
    dotted_overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configuration.

    Args:
        yaml_path: Path to a YAML config file. If None, only defaults + overrides apply.
        cli_overrides: Nested dict of overrides (e.g. {"model": {"arch": "resnet50"}}).
            Only keys that are explicitly set should be included (omit None/unset flags).
        dotted_overrides: Flat dict of dotted-key overrides (e.g. {"train.warmup_epochs": 10}),
            used for the generic --set key=value CLI mechanism.
    """
    merged = _dataclass_to_default_dict()

    if yaml_path is not None:
        yaml_path = Path(yaml_path)
        with open(yaml_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, loaded)

    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    if dotted_overrides:
        for dotted_key, value in dotted_overrides.items():
            _set_dotted(merged, dotted_key, value)

    config = Config(
        data=DataConfig(**merged["data"]),
        cross_validation=CrossValidationConfig(**merged["cross_validation"]),
        model=ModelConfig(**merged["model"]),
        train=TrainConfig(**merged["train"]),
        augment=AugmentConfig(**merged["augment"]),
        output=OutputConfig(**merged["output"]),
    )
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Validate numeric ranges and cross-field constraints before expensive work."""
    if config.data.batch_size < 1:
        raise ValueError("data.batch_size must be >= 1")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers must be >= 0")
    if config.cross_validation.num_folds < 3:
        raise ValueError("cross_validation.num_folds must be >= 3")
    if config.model.num_classes != 2:
        raise ValueError("This binary classifier requires model.num_classes == 2")
    if config.model.hidden_dim < 1:
        raise ValueError("model.hidden_dim must be >= 1")
    if not 0 <= config.model.dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if config.train.max_epochs < 1 or config.train.min_epochs < 0:
        raise ValueError("train epochs must be non-negative and max_epochs >= 1")
    if config.train.min_epochs > config.train.max_epochs:
        raise ValueError("train.min_epochs cannot exceed train.max_epochs")
    if config.train.lr <= 0 or config.train.weight_decay < 0:
        raise ValueError("train.lr must be positive and weight_decay non-negative")
    allowed_precisions = {"bf16-mixed", "16-mixed", "32-true"}
    if config.train.precision not in allowed_precisions:
        raise ValueError(
            f"train.precision must be one of {sorted(allowed_precisions)}, "
            f"got '{config.train.precision}'."
        )
    if not 0 <= config.train.warmup_epochs <= config.train.max_epochs:
        raise ValueError("train.warmup_epochs must be between 0 and max_epochs")
    for name, values in (("rrc_scale", config.augment.rrc_scale), ("rrc_ratio", config.augment.rrc_ratio)):
        if len(values) != 2 or values[0] <= 0 or values[0] > values[1]:
            raise ValueError(f"augment.{name} must be two positive ascending values")


def save_config(config: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False, allow_unicode=True)


def parse_set_args(set_args: list[str] | None) -> dict[str, Any]:
    """Parse a list of "key=value" strings (from --set) into a dotted-key dict.

    Values are parsed with yaml.safe_load so ints/floats/bools/lists work naturally.
    """
    result: dict[str, Any] = {}
    for item in set_args or []:
        if "=" not in item:
            raise ValueError(f"Invalid --set argument (expected key=value): '{item}'")
        key, raw_value = item.split("=", 1)
        result[key.strip()] = yaml.safe_load(raw_value)
    return result
