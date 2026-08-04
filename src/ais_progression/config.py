"""Configuration: YAML defaults + CLI overrides -> typed dataclasses.

Override priority: ``--set`` dotted overrides > explicit CLI flags > YAML file >
the dataclass defaults below. The defaults are the reference settings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Backbones of the reference configuration, keyed by the short name used on the CLI.
REFERENCE_ARCHS: dict[str, str] = {
    "vit": "vit_base_patch16_384.augreg_in21k_ft_in1k",
    "swint": "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
    "convnextv2": "convnextv2_base.fcmae_ft_in22k_in1k_384",
}

IMAGE_MODELS = tuple(REFERENCE_ARCHS)
CLINICAL_MODELS = ("logreg", "svm", "rf")

# The single source of truth for modality names; imported rather than re-spelled.
CLINICAL_MODALITY = "clinical"
IMAGE_MODALITIES = ("front", "lateral")
MODALITIES = (*IMAGE_MODALITIES, CLINICAL_MODALITY)

ENSEMBLE_METHODS = ("weighted", "average", "logreg", "svm", "rf")


def is_image_modality(modality: str) -> bool:
    return modality != CLINICAL_MODALITY


def models_for_modality(config: Config, modality: str) -> tuple[str, ...]:
    """Model names valid for a modality, given the configured architectures."""
    return CLINICAL_MODELS if modality == CLINICAL_MODALITY else tuple(config.image.archs)


@dataclass
class DataConfig:
    csv_path: str | None = None
    num_workers: int = 4
    batch_size: int = 32


@dataclass
class CrossValidationConfig:
    """Repeated stratified K-fold, as used for every modality."""

    num_reps: int = 10
    num_folds: int = 10
    seed: int = 42


@dataclass
class ImageConfig:
    archs: dict[str, str] = field(default_factory=lambda: dict(REFERENCE_ARCHS))
    num_classes: int = 2
    hidden_dim: int = 512
    dropout: float = 0.5
    pretrained: bool = True
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
    precision: str = "16-mixed"
    matmul_precision: str = "high"
    deterministic: bool = True


@dataclass
class AugmentConfig:
    horizontal_flip: bool = True
    random_resized_crop: bool = True
    rrc_scale: list[float] = field(default_factory=lambda: [0.5, 1.0])
    rrc_ratio: list[float] = field(default_factory=lambda: [1.0, 1.0])


@dataclass
class ClinicalConfig:
    """Four a priori clinical predictors plus the Optuna search budget."""

    numeric_features: list[str] = field(default_factory=lambda: ["age", "cobb_baseline"])
    binary_features: list[str] = field(default_factory=lambda: ["sex"])
    ordinal_features: list[str] = field(default_factory=lambda: ["risser"])
    n_trials: int = 30
    inner_folds: int = 10

    @property
    def features(self) -> list[str]:
        return [*self.numeric_features, *self.binary_features, *self.ordinal_features]


@dataclass
class EnsembleConfig:
    n_trials: int = 30
    inner_folds: int = 10


@dataclass
class FinalModelConfig:
    """The deployment model, trained outside the cross-validation protocol.

    There is no validation fraction: the final model trains on the whole cohort
    for an epoch count taken from cross-validation, and every performance figure
    attached to it comes from cross-validation too.
    """

    seed: int = 42


@dataclass
class OutputConfig:
    dir: str = "outputs"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    clinical: ClinicalConfig = field(default_factory=ClinicalConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    final: FinalModelConfig = field(default_factory=FinalModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "cross_validation": CrossValidationConfig,
    "image": ImageConfig,
    "train": TrainConfig,
    "augment": AugmentConfig,
    "clinical": ClinicalConfig,
    "ensemble": EnsembleConfig,
    "final": FinalModelConfig,
    "output": OutputConfig,
}

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any], path: str = "") -> dict[str, Any]:
    """Merge ``override`` into ``base`` in place, raising ValueError on unknown keys."""
    for key, value in override.items():
        full_key = f"{path}.{key}" if path else key
        if key not in base:
            raise ValueError(f"Unknown config key: '{full_key}'")
        # image.archs is a free-form mapping of short name -> timm model name, so
        # it is replaced wholesale rather than merged key by key.
        if isinstance(value, dict) and isinstance(base[key], dict) and full_key != "image.archs":
            _deep_merge(base[key], value, full_key)
        else:
            base[key] = value
    return base


def _set_dotted(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            raise ValueError(f"Unknown config key: '{dotted_key}'")
        cursor = cursor[part]
    if parts[-1] not in cursor:
        raise ValueError(f"Unknown config key: '{dotted_key}'")
    cursor[parts[-1]] = value


def load_config(
    yaml_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
    dotted_overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configuration from YAML plus overrides.

    Args:
        yaml_path: YAML file to read. When None, ``configs/default.yaml`` is used
            if it exists; otherwise the dataclass defaults apply.
        cli_overrides: Nested dict of overrides, e.g. ``{"data": {"batch_size": 8}}``.
            Include only keys the user actually set.
        dotted_overrides: Flat dict from the generic ``--set key=value`` mechanism.
    """
    merged = asdict(Config())

    if yaml_path is None and DEFAULT_CONFIG_PATH.exists():
        yaml_path = DEFAULT_CONFIG_PATH
    if yaml_path is not None:
        with open(yaml_path, encoding="utf-8") as handle:
            merged = _deep_merge(merged, yaml.safe_load(handle) or {})

    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)
    for dotted_key, value in (dotted_overrides or {}).items():
        _set_dotted(merged, dotted_key, value)

    config = Config(**{name: cls(**merged[name]) for name, cls in _SECTIONS.items()})
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Check numeric ranges and cross-field constraints before any expensive work.

    ``train.min_epochs`` is clamped to ``train.max_epochs`` rather than rejected:
    it only stops early stopping from firing too soon, so capping it changes
    nothing, and it keeps short smoke runs (``--set train.max_epochs=2``) working
    without a second override.
    """
    if config.train.min_epochs > config.train.max_epochs:
        config.train.min_epochs = config.train.max_epochs
    if config.train.warmup_epochs > config.train.max_epochs:
        config.train.warmup_epochs = config.train.max_epochs

    if config.data.batch_size < 1:
        raise ValueError("data.batch_size must be >= 1")
    if config.data.num_workers < 0:
        raise ValueError("data.num_workers must be >= 0")
    if config.cross_validation.num_folds < 3:
        raise ValueError("cross_validation.num_folds must be >= 3")
    if config.cross_validation.num_reps < 1:
        raise ValueError("cross_validation.num_reps must be >= 1")
    if config.image.num_classes != 2:
        raise ValueError("This binary classifier requires image.num_classes == 2")
    if config.image.hidden_dim < 1:
        raise ValueError("image.hidden_dim must be >= 1")
    if not 0 <= config.image.dropout < 1:
        raise ValueError("image.dropout must be in [0, 1)")
    if config.train.max_epochs < 1 or config.train.min_epochs < 0:
        raise ValueError("train epochs must be non-negative and max_epochs >= 1")
    if config.train.lr <= 0 or config.train.weight_decay < 0:
        raise ValueError("train.lr must be positive and weight_decay non-negative")
    if config.train.warmup_epochs < 1:
        # The warmup schedule divides by warmup_epochs.
        raise ValueError("train.warmup_epochs must be >= 1")
    allowed_precisions = {"bf16-mixed", "16-mixed", "32-true"}
    if config.train.precision not in allowed_precisions:
        raise ValueError(
            f"train.precision must be one of {sorted(allowed_precisions)}, "
            f"got '{config.train.precision}'."
        )
    allowed_matmul = {"highest", "high", "medium"}
    if config.train.matmul_precision not in allowed_matmul:
        raise ValueError(
            f"train.matmul_precision must be one of {sorted(allowed_matmul)}, "
            f"got '{config.train.matmul_precision}'."
        )
    for name, values in (
        ("rrc_scale", config.augment.rrc_scale),
        ("rrc_ratio", config.augment.rrc_ratio),
    ):
        if len(values) != 2 or values[0] <= 0 or values[0] > values[1]:
            raise ValueError(f"augment.{name} must be two positive ascending values")
    if not config.clinical.features:
        raise ValueError("clinical must declare at least one feature")
    for section in (config.clinical, config.ensemble):
        if section.n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        if section.inner_folds < 2:
            raise ValueError("inner_folds must be >= 2")


def resolve_arch(config: Config, model: str) -> str:
    """Map a short image-model name (e.g. 'vit') to its timm architecture string."""
    try:
        return config.image.archs[model]
    except KeyError:
        raise ValueError(
            f"Unknown image model '{model}'. Available: {sorted(config.image.archs)}"
        ) from None


def save_config(config: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False, allow_unicode=True)


def _parse_value(raw_value: str) -> Any:
    """Parse one ``--set`` value: ints, floats, bools, lists, and strings.

    YAML 1.1 does not recognise unpunctuated scientific notation, so a plain
    ``1e-4`` would come back as the string "1e-4" and later fail a numeric
    comparison. Retry the float conversion when that happens.
    """
    value = yaml.safe_load(raw_value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def parse_set_args(set_args: list[str] | None) -> dict[str, Any]:
    """Parse ``--set key=value`` strings into a dotted-key dict."""
    result: dict[str, Any] = {}
    for item in set_args or []:
        if "=" not in item:
            raise ValueError(f"Invalid --set argument (expected key=value): '{item}'")
        key, raw_value = item.split("=", 1)
        result[key.strip()] = _parse_value(raw_value)
    return result
