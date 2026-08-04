import numpy as np
import pytest
import torch

from ais_progression.config import ImageConfig, TrainConfig
from ais_progression.models.clinical_model import (
    build_preprocessor,
    fit_clinical_model,
    predict_clinical_model,
)
from ais_progression.models.image_model import compute_balanced_class_weights
from ais_progression.models.lightning import ImageClassifier

TINY_ARCH = "resnet18"


def test_classification_head_matches_the_published_architecture():
    model = ImageClassifier(
        ImageConfig(pretrained=False), TrainConfig(), TINY_ARCH, initialize_pretrained=False
    ).model
    layers = list(model.classifier)
    assert isinstance(layers[0], torch.nn.LayerNorm)
    assert isinstance(layers[1], torch.nn.Linear) and layers[1].out_features == 512
    assert isinstance(layers[2], torch.nn.GELU)
    assert isinstance(layers[3], torch.nn.Dropout) and layers[3].p == pytest.approx(0.5)
    assert isinstance(layers[4], torch.nn.Linear) and layers[4].out_features == 2


def test_forward_produces_two_logits_per_image():
    module = ImageClassifier(
        ImageConfig(pretrained=False), TrainConfig(), TINY_ARCH, initialize_pretrained=False
    )
    assert module(torch.randn(2, 3, 64, 64)).shape == (2, 2)


def test_optimizer_is_adamw_with_the_published_hyperparameters():
    train_cfg = TrainConfig()
    module = ImageClassifier(
        ImageConfig(pretrained=False), train_cfg, TINY_ARCH, initialize_pretrained=False
    )
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == pytest.approx(train_cfg.lr)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(train_cfg.weight_decay)
    # LambdaLR applies the first warmup factor (0.1) as soon as it is attached.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * train_cfg.lr)


def test_learning_rate_warms_up_then_decays_to_zero():
    train_cfg = TrainConfig(warmup_epochs=5, max_epochs=100)
    module = ImageClassifier(
        ImageConfig(pretrained=False), train_cfg, TINY_ARCH, initialize_pretrained=False
    )
    scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
    factors = [scheduler.lr_lambdas[0](epoch) for epoch in range(train_cfg.max_epochs + 1)]
    assert factors[0] == pytest.approx(0.1)
    assert factors[:5] == sorted(factors[:5])
    assert factors[5] == pytest.approx(1.0)
    assert factors[-1] == pytest.approx(0.0, abs=1e-9)


def test_class_weights_are_inverse_frequency():
    import pandas as pd

    labels = pd.Series([0] * 30 + [1] * 70)
    weights = compute_balanced_class_weights(labels)
    # sklearn's "balanced" weight is n / (2 * n_c).
    assert weights[0] == pytest.approx(100 / (2 * 30))
    assert weights[1] == pytest.approx(100 / (2 * 70))


def test_class_weights_require_both_classes():
    import pandas as pd

    with pytest.raises(ValueError, match="both classes"):
        compute_balanced_class_weights(pd.Series([1, 1, 1]))


def test_clinical_preprocessor_encodes_each_variable_type(small_config, synthetic_cohort):
    _, frame = synthetic_cohort
    preprocessor = build_preprocessor(small_config.clinical, use_scaler=True)
    transformed = preprocessor.fit_transform(frame[small_config.clinical.features])
    # age + cobb (z-scored) + sex (one-hot, 2 levels) + risser (ordinal) = 5 columns.
    assert transformed.shape == (len(frame), 5)
    assert np.abs(transformed[:, 0].mean()) < 1e-9
    assert set(np.unique(transformed[:, 2:4])) <= {0.0, 1.0}


@pytest.mark.parametrize("model", ["logreg", "svm", "rf"])
def test_clinical_models_fit_and_score(model, small_config, synthetic_cohort):
    _, frame = synthetic_cohort
    features = frame[small_config.clinical.features]
    labels = frame["label"].astype(int)
    result = fit_clinical_model(features, labels, model, small_config.clinical, seed=42)
    probabilities = predict_clinical_model(result.pipeline, features)
    assert probabilities.shape == (len(frame),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert 0 <= result.inner_cv_auc <= 1
    assert result.best_params


def test_unknown_clinical_model_is_rejected(small_config, synthetic_cohort):
    _, frame = synthetic_cohort
    with pytest.raises(ValueError, match="Unknown clinical model"):
        fit_clinical_model(
            frame[small_config.clinical.features],
            frame["label"],
            "xgboost",
            small_config.clinical,
            seed=42,
        )


def test_checkpoint_callback_saves_weights_only(small_config, tmp_path):
    from ais_progression.models.image_model import build_callbacks

    checkpoint, _ = build_callbacks(small_config, tmp_path)
    assert checkpoint.save_weights_only is True
    assert checkpoint.monitor == "val_loss"
    assert checkpoint.mode == "min"
    assert checkpoint.save_top_k == 1


def test_early_stopping_leaves_the_hook_choice_to_lightning(small_config, tmp_path):
    """check_on_train_epoch_end must stay unpinned.

    Lightning resolves None to True whenever validation runs once per epoch --
    the published behaviour. Pinning True would keep checking at train-epoch end
    against a stale val_loss if the validation cadence ever changed.
    """
    from ais_progression.models.image_model import build_callbacks

    _, early_stopping = build_callbacks(small_config, tmp_path)
    assert early_stopping.monitor == "val_loss"
    assert early_stopping.patience == small_config.train.early_stopping_patience
    assert early_stopping.strict is True
    assert early_stopping._check_on_train_epoch_end is None


def test_early_stopping_resolves_to_the_published_hook(small_config, tmp_path):
    """With one validation pass per epoch, Lightning resolves the hook to True."""
    import pytorch_lightning as pl

    from ais_progression.models.image_model import build_callbacks

    _, early_stopping = build_callbacks(small_config, tmp_path)
    trainer = pl.Trainer(logger=False, enable_progress_bar=False, accelerator="cpu")
    early_stopping.setup(trainer, None, "fit")
    assert early_stopping._check_on_train_epoch_end is True


def test_set_matmul_precision_changes_the_global_mode():
    from ais_progression.utils import set_matmul_precision

    original = torch.get_float32_matmul_precision()
    try:
        set_matmul_precision("high")
        assert torch.get_float32_matmul_precision() == "high"
        set_matmul_precision("highest")
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(original)


def test_full_cohort_training_writes_only_into_its_work_dir(
    tiny_arch_config, synthetic_cohort, tmp_path, monkeypatch
):
    """Lightning must not install a default checkpointer.

    Without enable_checkpointing=False it writes into <cwd>/checkpoints, which
    for the real 384px backbones is hundreds of megabytes of stray weights per
    model, dumped wherever the command happened to be run.
    """
    from ais_progression.models.image_model import fit_image_model

    _, frame = synthetic_cohort
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    work_dir = tmp_path / "work"
    result = fit_image_model(
        tiny_arch_config, TINY_ARCH, "front", frame, val_df=None,
        work_dir=work_dir, fixed_epochs=1,
    )

    assert result.checkpoint_path.exists()
    assert result.best_epoch == 0
    assert not (cwd / "checkpoints").exists()
    assert not list(cwd.iterdir())


def test_fit_image_model_requires_exactly_one_stopping_rule(
    tiny_arch_config, synthetic_cohort, tmp_path
):
    from ais_progression.models.image_model import fit_image_model

    _, frame = synthetic_cohort
    with pytest.raises(ValueError, match="not both and not neither"):
        fit_image_model(
            tiny_arch_config, TINY_ARCH, "front", frame, val_df=frame,
            work_dir=tmp_path, fixed_epochs=3,
        )
    with pytest.raises(ValueError, match="not both and not neither"):
        fit_image_model(
            tiny_arch_config, TINY_ARCH, "front", frame, val_df=None, work_dir=tmp_path
        )
