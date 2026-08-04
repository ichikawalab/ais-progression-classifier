"""End-to-end runs of the whole protocol on a tiny synthetic cohort.

These use a small CNN rather than the 384-pixel transformers so they finish in
seconds on CPU, but they exercise the real code paths: cross-validation,
ensembling, profile construction, full-cohort training, bundling, and inference.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from ais_progression.data.schema import load_dataset
from ais_progression.experiments.ensemble_cv import (
    build_oof_matrix,
    load_base_predictions,
    run_ensemble_cv,
)
from ais_progression.experiments.modality_cv import run_modality_cv
from ais_progression.final.bundle import BUNDLE_FORMAT_VERSION, ModelBundle
from ais_progression.final.profiles import build_profiles
from ais_progression.final.train import (
    resolve_epoch_plan,
    train_final_model,
    validate_cv_compatibility,
)

TINY_ARCH = "resnet18"


@pytest.fixture
def tiny_image_config(small_config):
    """Swap the 384-pixel backbones for one small CNN."""
    small_config.image.archs = {"tiny": TINY_ARCH}
    small_config.image.pretrained = False
    return small_config


# --------------------------------------------------------------- cross-validation


def test_clinical_cross_validation_records_selection_and_test_auc(
    small_config, synthetic_cohort, tmp_path
):
    csv_path, _ = synthetic_cohort
    dataset = load_dataset(csv_path)
    run_dir = tmp_path / "cv" / "clinical_logreg"

    summary = run_modality_cv(small_config, dataset, "clinical", "logreg", run_dir)

    predictions = pd.read_csv(run_dir / "predictions.csv")
    fold_metrics = pd.read_csv(run_dir / "fold_metrics.csv")
    reps, folds = small_config.cross_validation.num_reps, small_config.cross_validation.num_folds

    assert len(fold_metrics) == reps * folds
    assert fold_metrics["selection_auc"].notna().all()
    assert (fold_metrics["selection_auc_source"] == "inner_cv").all()
    assert len(predictions) == reps * len(dataset)
    assert summary["test_auc_pooled_per_rep"]["n"] == reps
    # Selection AUCs are grouped by source and never pooled across kinds.
    assert set(summary["selection_auc_by_source"]) == {"inner_cv"}


def test_image_cross_validation_records_the_best_epoch(
    tiny_image_config, synthetic_cohort, tmp_path
):
    csv_path, _ = synthetic_cohort
    dataset = load_dataset(csv_path)
    run_dir = tmp_path / "cv" / "front_tiny"

    run_modality_cv(tiny_image_config, dataset, "front", "tiny", run_dir)

    fold_metrics = pd.read_csv(run_dir / "fold_metrics.csv")
    assert (fold_metrics["selection_auc_source"] == "holdout").all()
    # best_epoch is what the final model needs; stopped_epoch is diagnostic only.
    assert fold_metrics["best_epoch"].notna().all()
    assert (fold_metrics["best_epoch"] >= 0).all()
    assert (fold_metrics["best_epoch"] <= fold_metrics["stopped_epoch"]).all()
    # Fold working directories are removed entirely, not just their checkpoints.
    assert not list((run_dir / "folds").glob("work_*"))


def test_cross_validation_resumes_completed_folds(small_config, synthetic_cohort, tmp_path):
    csv_path, _ = synthetic_cohort
    dataset = load_dataset(csv_path)
    run_dir = tmp_path / "cv" / "clinical_rf"

    run_modality_cv(small_config, dataset, "clinical", "rf", run_dir)
    first = pd.read_csv(run_dir / "predictions.csv")
    stamps = {path: path.stat().st_mtime_ns for path in (run_dir / "folds").glob("*.csv")}

    run_modality_cv(small_config, dataset, "clinical", "rf", run_dir, resume=True)
    second = pd.read_csv(run_dir / "predictions.csv")

    assert {p: p.stat().st_mtime_ns for p in (run_dir / "folds").glob("*.csv")} == stamps
    pd.testing.assert_frame_equal(first, second)


def test_modality_resume_rejects_changed_algorithm_config(
    small_config, synthetic_cohort, tmp_path
):
    csv_path, _ = synthetic_cohort
    dataset = load_dataset(csv_path)
    run_dir = tmp_path / "cv" / "clinical_logreg"
    run_modality_cv(small_config, dataset, "clinical", "logreg", run_dir)

    small_config.clinical.n_trials += 1
    with pytest.raises(ValueError, match="different configuration"):
        run_modality_cv(
            small_config, dataset, "clinical", "logreg", run_dir, resume=True
        )


def _run_clinical_base_models(config, dataset, root):
    sources = {}
    for model in ("logreg", "rf", "svm"):
        name = f"clinical_{model}"
        run_modality_cv(config, dataset, "clinical", model, root / name)
        sources[name] = root / name
    return sources


def test_oof_matrix_aligns_base_models_to_the_cohort(small_config, synthetic_cohort, tmp_path):
    csv_path, _ = synthetic_cohort
    dataset = load_dataset(csv_path)
    sources = _run_clinical_base_models(small_config, dataset, tmp_path / "cv")

    matrix = build_oof_matrix(load_base_predictions(sources), dataset, rep=1)
    assert list(matrix.index) == list(dataset["patient_id"])
    assert set(matrix.columns) == set(sources)
    assert matrix.notna().all().all()


@pytest.mark.parametrize("method", ["weighted", "average", "logreg"])
def test_ensemble_cross_validation_runs(method, small_config, synthetic_cohort, tmp_path):
    csv_path, _ = synthetic_cohort
    dataset = load_dataset(csv_path)
    sources = _run_clinical_base_models(small_config, dataset, tmp_path / "cv")
    run_dir = tmp_path / "ensemble" / method

    summary = run_ensemble_cv(
        small_config, dataset, load_base_predictions(sources), method, run_dir
    )

    reps, folds = small_config.cross_validation.num_reps, small_config.cross_validation.num_folds
    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert len(predictions) == reps * len(dataset)
    assert summary["test_auc_pooled_per_rep"]["n"] == reps
    # The selection-bias caveat travels with the result.
    assert "ensemble_method_selection_warning" in summary
    assert "stacking_leakage_warning" in summary
    assert "unbiased" not in summary["ensemble_method_selection_warning"]
    if method == "weighted":
        weights = pd.read_csv(run_dir / "weights_by_fold.csv")
        assert len(weights) == reps * folds
        weight_summary = json.loads((run_dir / "weights_summary.json").read_text())
        assert set(weight_summary["by_modality"]) == {"clinical"}


def test_ensemble_test_folds_match_the_base_model_folds(small_config, synthetic_cohort, tmp_path):
    csv_path, _ = synthetic_cohort
    dataset = load_dataset(csv_path)
    sources = _run_clinical_base_models(small_config, dataset, tmp_path / "cv")
    run_dir = tmp_path / "ensemble" / "average"
    run_ensemble_cv(small_config, dataset, load_base_predictions(sources), "average", run_dir)

    base = pd.read_csv(next(iter(sources.values())) / "predictions.csv")
    base = base[base["split"] == "test"]
    ensemble = pd.read_csv(run_dir / "predictions.csv")
    key = ["patient_id", "rep", "fold"]
    assert len(base[key].merge(ensemble[key], on=key, how="inner")) == len(base)


# ------------------------------------------------------------------- final model


def _prepare_cv(config, dataset, cv_root):
    """One image model per modality plus one clinical model."""
    sources = {}
    for modality in ("front", "lateral"):
        name = f"{modality}_tiny"
        run_modality_cv(config, dataset, modality, "tiny", cv_root / name)
        sources[name] = cv_root / name
    run_modality_cv(config, dataset, "clinical", "logreg", cv_root / "clinical_logreg")
    sources["clinical_logreg"] = cv_root / "clinical_logreg"
    return sources


def test_epoch_plan_comes_from_the_cross_validated_best_epochs(
    tiny_image_config, synthetic_cohort, tmp_path
):
    csv_path, _ = synthetic_cohort
    dataset = load_dataset(csv_path)
    cv_root = tmp_path / "cv"
    run_modality_cv(tiny_image_config, dataset, "front", "tiny", cv_root / "front_tiny")

    plan = resolve_epoch_plan(cv_root, ["front_tiny"])
    fold_metrics = pd.read_csv(cv_root / "front_tiny" / "fold_metrics.csv")
    assert plan["front_tiny"] == int(np.median(fold_metrics["best_epoch"])) + 1
    assert plan["front_tiny"] >= 1

    # A model with no cross-validation record is an error, not a guess: the
    # bundle would otherwise carry cross-validated numbers for a model trained
    # for an epoch count nobody chose.
    with pytest.raises(ValueError, match="never_trained"):
        resolve_epoch_plan(cv_root, ["never_trained"])


def test_final_training_rejects_config_that_differs_from_image_cv(
    tiny_image_config, synthetic_cohort, tmp_path
):
    csv_path, _ = synthetic_cohort
    dataset = load_dataset(csv_path)
    source = tmp_path / "cv" / "front_tiny"
    run_modality_cv(tiny_image_config, dataset, "front", "tiny", source)

    tiny_image_config.train.lr *= 2
    with pytest.raises(ValueError, match="algorithm differs"):
        validate_cv_compatibility(
            tiny_image_config, dataset, {"front_tiny": source}
        )


def test_final_model_trains_on_everything_and_serves_profiles(
    tiny_image_config, synthetic_cohort, tmp_path
):
    csv_path, _ = synthetic_cohort
    dataset = load_dataset(csv_path)
    config = tiny_image_config
    cv_root = tmp_path / "cv"
    sources = _prepare_cv(config, dataset, cv_root)

    profiles = build_profiles(
        requested={"full": None, "clinical_only": ("clinical",)},
        config=config,
        dataset=dataset,
        base_predictions=load_base_predictions(sources),
        ensemble_root=tmp_path / "ensemble",
        calibration="isotonic",
    )
    assert {p.name for p in profiles} == {"full", "clinical_only"}
    for profile in profiles:
        assert profile.weights.sum() == pytest.approx(1.0)
        assert 0.0 <= profile.operating_point.threshold <= 1.0
        assert profile.operating_point.n_reps == config.cross_validation.num_reps
        assert profile.cv_metrics["serving_weight_n_trials"] == config.ensemble.n_trials
        assert "all base-model OOF probabilities" in profile.cv_metrics[
            "serving_weight_source"
        ]
    # A profile's threshold is its own, not inherited from the full ensemble.
    assert len([p for p in profiles if p.name == "clinical_only"][0].members) == 1

    bundle_dir = tmp_path / "final"
    metrics = train_final_model(
        config=config,
        dataset=dataset,
        bundle_dir=bundle_dir,
        profiles=profiles,
        cv_identities=validate_cv_compatibility(config, dataset, sources),
        environment={"test": True},
        epoch_plan=resolve_epoch_plan(cv_root, ["front_tiny", "lateral_tiny"]),
        image_models=("tiny",),
        clinical_models=("logreg",),
    )

    # Trained on the whole cohort: no holdout was carved out.
    assert metrics["n_train"] == len(dataset)
    assert all(
        "final_train_loss" not in report for report in metrics["training"].values()
    )
    assert not (bundle_dir / "_work").exists()

    weight_file = bundle_dir / "models" / "front_tiny.pt"
    state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
    assert all(key.startswith("model.") for key in state_dict)
    assert not any(key.startswith("loss_fn.") for key in state_dict)

    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["format_version"] == BUNDLE_FORMAT_VERSION
    assert manifest["default_profile"] == "full"


def test_bundle_serves_each_profile_and_caches_models(
    tiny_image_config, synthetic_cohort, tmp_path
):
    csv_path, _ = synthetic_cohort
    dataset = load_dataset(csv_path)
    config = tiny_image_config
    cv_root = tmp_path / "cv"
    sources = _prepare_cv(config, dataset, cv_root)
    profiles = build_profiles(
        requested={"full": None, "clinical_only": ("clinical",)},
        config=config,
        dataset=dataset,
        base_predictions=load_base_predictions(sources),
        ensemble_root=tmp_path / "ensemble",
    )
    bundle_dir = tmp_path / "final"
    train_final_model(
        config=config,
        dataset=dataset,
        bundle_dir=bundle_dir,
        profiles=profiles,
        cv_identities=validate_cv_compatibility(config, dataset, sources),
        environment={"test": True},
        epoch_plan=resolve_epoch_plan(cv_root, ["front_tiny", "lateral_tiny"]),
        image_models=("tiny",),
        clinical_models=("logreg",),
    )

    bundle = ModelBundle.load(bundle_dir)
    assert set(bundle.profiles) == {"full", "clinical_only"}

    full = bundle.predict(dataset.head(4), "full")
    assert set(full.columns) >= {
        "probability",
        "calibrated_probability",
        "predicted_label",
        "threshold",
    }
    assert full["probability"].between(0, 1).all()
    assert full["calibrated_probability"].between(0, 1).all()
    assert (full["threshold"] == bundle.profiles["full"].operating_point.threshold).all()

    # The cheap profile runs only its own models and uses its own threshold.
    cheap = bundle.predict(dataset.head(4), "clinical_only")
    assert "front_tiny" not in cheap.columns
    assert cheap["threshold"].iloc[0] == bundle.profiles["clinical_only"].operating_point.threshold

    # Models are loaded once and kept, rather than re-read per call.
    bundle.release()
    bundle.warmup("full")
    cached = dict(bundle._model_cache)
    bundle.predict(dataset.head(2), "full")
    assert bundle._model_cache == cached

    with pytest.raises(ValueError, match="Unknown profile"):
        bundle.predict(dataset.head(1), "does_not_exist")


def test_bundle_rejects_an_incompatible_format_version(
    tiny_image_config, synthetic_cohort, tmp_path
):
    csv_path, _ = synthetic_cohort
    dataset = load_dataset(csv_path)
    config = tiny_image_config
    cv_root = tmp_path / "cv"
    source = cv_root / "clinical_logreg"
    run_modality_cv(config, dataset, "clinical", "logreg", source)
    profiles = build_profiles(
        requested={"full": None},
        config=config,
        dataset=dataset,
        base_predictions=load_base_predictions({"clinical_logreg": source}),
        ensemble_root=tmp_path / "ensemble",
    )
    bundle_dir = tmp_path / "final"
    train_final_model(
        config=config,
        dataset=dataset,
        bundle_dir=bundle_dir,
        profiles=profiles,
        cv_identities=validate_cv_compatibility(
            config, dataset, {"clinical_logreg": source}
        ),
        environment={"test": True},
        epoch_plan={},
        image_models=(),
        clinical_models=("logreg",),
        image_modalities=(),
    )

    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = BUNDLE_FORMAT_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="format"):
        ModelBundle.load(bundle_dir)


def test_profiles_are_validated_before_any_training(tiny_image_config, synthetic_cohort, tmp_path):
    """A profile naming an untrained model must fail immediately, not after hours."""
    from ais_progression.final.train import validate_plan

    with pytest.raises(ValueError, match="would not train"):
        validate_plan(
            tiny_image_config,
            profiles=[
                type("P", (), {"members": ["front_tiny", "lateral_absent"]})(),
            ],
            image_models=("tiny",),
            clinical_models=(),
            image_modalities=("front",),
        )


def test_failed_bundle_build_preserves_the_previous_bundle(
    tiny_image_config, synthetic_cohort, tmp_path, monkeypatch
):
    import ais_progression.final.train as final_train

    destination = tmp_path / "final"
    destination.mkdir()
    marker = destination / "previous.txt"
    marker.write_text("keep me", encoding="utf-8")

    def fail_build(**_kwargs):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(final_train, "_train_final_model_into", fail_build)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        final_train.train_final_model(
            config=tiny_image_config,
            dataset=synthetic_cohort[1],
            bundle_dir=destination,
            profiles=[],
            cv_identities={},
            environment={"test": True},
            epoch_plan={},
            image_models=(),
            clinical_models=(),
            image_modalities=(),
        )
    assert marker.read_text(encoding="utf-8") == "keep me"
    assert not list(tmp_path.glob(".final.staging-*"))
