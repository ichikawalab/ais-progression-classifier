import pytest

from ais_progression.config import load_config, parse_set_args, save_config


def test_load_config_defaults_only():
    cfg = load_config(None)
    assert cfg.model.arch == "resnet50"
    assert cfg.cross_validation.num_folds == 10
    assert cfg.train.max_epochs == 100
    assert cfg.augment.horizontal_flip is True
    assert cfg.train.deterministic is True


def test_invalid_batch_size_raises():
    with pytest.raises(ValueError, match="batch_size"):
        load_config(None, cli_overrides={"data": {"batch_size": 0}})


def test_load_config_yaml_and_cli_priority(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("model:\n  arch: densenet121\ntrain:\n  lr: 0.001\n", encoding="utf-8")

    cfg = load_config(yaml_path, cli_overrides={"train": {"lr": 0.005}})
    assert cfg.model.arch == "densenet121"  # from YAML
    assert cfg.train.lr == 0.005  # CLI overrides YAML


def test_load_config_unknown_key_raises(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("model:\n  not_a_real_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(yaml_path)


def test_dotted_overrides():
    cfg = load_config(None, dotted_overrides={"train.warmup_epochs": 3})
    assert cfg.train.warmup_epochs == 3


def test_parse_set_args_types():
    parsed = parse_set_args(["train.lr=0.5", "model.freeze_backbone=true", "output.run_name=exp1"])
    assert parsed == {"train.lr": 0.5, "model.freeze_backbone": True, "output.run_name": "exp1"}


def test_save_and_reload_config_roundtrip(tmp_path):
    cfg = load_config(None, cli_overrides={"model": {"arch": "vit_base_patch16_384"}})
    out_path = tmp_path / "saved.yaml"
    save_config(cfg, out_path)

    reloaded = load_config(out_path)
    assert reloaded.model.arch == "vit_base_patch16_384"
