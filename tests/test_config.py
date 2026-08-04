import pytest

from ais_progression.config import (
    PAPER_ARCHS,
    Config,
    load_config,
    parse_set_args,
    resolve_arch,
    save_config,
    validate_config,
)


def test_defaults_match_the_published_settings():
    config = load_config()
    assert config.cross_validation.num_reps == 10
    assert config.cross_validation.num_folds == 10
    assert config.cross_validation.seed == 42
    assert config.data.batch_size == 32
    assert config.train.lr == pytest.approx(1e-5)
    assert config.train.weight_decay == pytest.approx(1e-3)
    assert config.train.warmup_epochs == 5
    assert config.train.early_stopping_patience == 5
    assert config.train.max_epochs == 100
    assert config.image.hidden_dim == 512
    assert config.image.dropout == pytest.approx(0.5)
    assert config.augment.rrc_scale == [0.5, 1.0]
    assert config.augment.rrc_ratio == [1.0, 1.0]
    assert config.clinical.features == ["age", "cobb_baseline", "sex", "risser"]
    assert set(config.image.archs) == set(PAPER_ARCHS)


def test_all_three_backbones_take_384_pixel_inputs():
    for arch in PAPER_ARCHS.values():
        assert "384" in arch


def test_dotted_overrides_apply():
    config = load_config(dotted_overrides={"train.max_epochs": 3, "data.batch_size": 8})
    assert config.train.max_epochs == 3
    assert config.data.batch_size == 8


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(dotted_overrides={"train.nope": 1})


def test_archs_mapping_is_replaced_not_merged():
    config = load_config(cli_overrides={"image": {"archs": {"tiny": "resnet18"}}})
    assert config.image.archs == {"tiny": "resnet18"}
    assert resolve_arch(config, "tiny") == "resnet18"


def test_resolve_arch_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown image model"):
        resolve_arch(load_config(), "not_a_model")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: setattr(c.data, "batch_size", 0),
        lambda c: setattr(c.train, "precision", "fp8"),
        lambda c: setattr(c.image, "dropout", 1.0),
        lambda c: setattr(c.train, "matmul_precision", "turbo"),
        lambda c: setattr(c.cross_validation, "num_folds", 2),
        lambda c: setattr(c.train, "warmup_epochs", 0),
    ],
)
def test_validation_rejects_bad_values(mutate):
    config = Config()
    mutate(config)
    with pytest.raises(ValueError):
        validate_config(config)


def test_short_smoke_runs_clamp_min_and_warmup_epochs():
    # --set train.max_epochs=2 must not trip over the defaults of 10 and 5.
    config = load_config(dotted_overrides={"train.max_epochs": 2})
    assert config.train.max_epochs == 2
    assert config.train.min_epochs == 2
    assert config.train.warmup_epochs == 2


def test_parse_set_args_types():
    parsed = parse_set_args(
        ["train.lr=1e-4", "train.deterministic=false", "augment.rrc_scale=[0.2,1.0]"]
    )
    assert parsed["train.lr"] == pytest.approx(1e-4)
    assert parsed["train.deterministic"] is False
    assert parsed["augment.rrc_scale"] == [0.2, 1.0]


def test_parse_set_args_rejects_missing_equals():
    with pytest.raises(ValueError, match="expected key=value"):
        parse_set_args(["train.lr"])


def test_config_round_trips_through_yaml(tmp_path):
    original = load_config(dotted_overrides={"train.max_epochs": 7})
    path = tmp_path / "config.yaml"
    save_config(original, path)
    assert load_config(path).train.max_epochs == 7
