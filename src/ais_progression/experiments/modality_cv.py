"""Repeated stratified cross-validation for one modality x one model.

Nine of these runs -- three backbones on frontal radiographs, three on lateral
radiographs, three learners on the clinical variables -- produce the
out-of-fold probability matrix the ensembles consume.

Image models hold out a stratified validation slice inside each outer training
fold for early stopping. Clinical models instead tune with an inner stratified
K-fold over the whole outer training fold (nested cross-validation), so their
reported validation AUC is that inner score.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ais_progression.config import CLINICAL_MODELS, Config, resolve_arch
from ais_progression.data.schema import ID_COLUMN, LABEL_COLUMN
from ais_progression.evaluation import format_auc, make_predictions_frame, safe_auc
from ais_progression.experiments.reporting import (
    assert_run_identity,
    finalize_run,
    fold_is_complete,
    write_fold,
)
from ais_progression.experiments.splits import Fold, iter_folds, rep_seed
from ais_progression.models.clinical_model import fit_clinical_model, predict_clinical_model
from ais_progression.models.image_model import (
    discard_checkpoints,
    fit_image_model,
    load_image_classifier,
    predict_image_model,
)
from ais_progression.provenance import modality_run_identity
from ais_progression.utils import progress_bar_enabled, set_seed

CLINICAL_MODALITY = "clinical"


def run_name(modality: str, model: str) -> str:
    return f"{modality}_{model}"


def _run_image_fold(
    config: Config, arch: str, modality: str, split: Fold, fold_dir: Path, keep_checkpoints: bool
) -> tuple[pd.DataFrame, dict]:
    fit = fit_image_model(
        config,
        arch,
        modality,
        split.train,
        split.val,
        fold_dir,
        enable_progress_bar=progress_bar_enabled(),
    )
    classifier = load_image_classifier(fit.checkpoint_path, config, arch)

    blocks = []
    aucs: dict[str, float | None] = {}
    for name, frame in (("val", split.val), ("test", split.test)):
        probabilities = predict_image_model(classifier, config, arch, modality, frame)
        blocks.append(
            make_predictions_frame(
                frame[ID_COLUMN], frame[LABEL_COLUMN], probabilities, split.rep, split.fold, name
            )
        )
        aucs[name] = safe_auc(frame[LABEL_COLUMN], probabilities)

    if not keep_checkpoints:
        discard_checkpoints(fold_dir)

    metrics = {
        "rep": split.rep,
        "fold": split.fold,
        "seed": split.seed,
        **split.sizes,
        # The AUC that drove model selection. For an image model that is the
        # held-out validation slice; see selection_auc_source.
        "selection_auc": aucs["val"],
        "selection_auc_source": "holdout",
        "test_auc": aucs["test"],
        "best_val_loss": fit.best_val_loss,
        "best_epoch": fit.best_epoch,
        "stopped_epoch": fit.stopped_epoch,
    }
    return pd.concat(blocks, ignore_index=True), metrics


def _run_clinical_fold(config: Config, model: str, split: Fold) -> tuple[pd.DataFrame, dict]:
    features = config.clinical.features
    fit = fit_clinical_model(
        split.train[features],
        split.train[LABEL_COLUMN].astype(int),
        model,
        config.clinical,
        split.seed,
    )
    probabilities = predict_clinical_model(fit.pipeline, split.test[features])
    predictions = make_predictions_frame(
        split.test[ID_COLUMN],
        split.test[LABEL_COLUMN],
        probabilities,
        split.rep,
        split.fold,
        "test",
    )
    metrics = {
        "rep": split.rep,
        "fold": split.fold,
        "seed": split.seed,
        **split.sizes,
        # Clinical models tune with an inner cross-validation over the whole
        # outer training fold, so their selection AUC is that inner score --
        # not comparable with an image model's held-out AUC.
        "selection_auc": fit.inner_cv_auc,
        "selection_auc_source": "inner_cv",
        "test_auc": safe_auc(split.test[LABEL_COLUMN], probabilities),
        "best_params": fit.best_params,
    }
    return predictions, metrics


def run_modality_cv(
    config: Config,
    dataset: pd.DataFrame,
    modality: str,
    model: str,
    run_dir: str | Path,
    resume: bool = True,
    keep_checkpoints: bool = False,
) -> dict:
    """Run the full repeated cross-validation for one modality/model pair.

    Completed folds are skipped when ``resume`` is set, so an interrupted run can
    be restarted with the same command.

    Note on seeding: the published code seeded once per repetition and let the
    RNG carry across that repetition's folds, so a fold's result depended on
    every fold before it. Here each fold is re-seeded with its repetition's seed,
    which makes a fold reproducible on its own -- the precondition for resuming.
    Fold assignment is unaffected; only weight initialisation and augmentation
    draws differ from the original.
    """
    run_dir = Path(run_dir)
    folds_dir = run_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    assert_run_identity(run_dir, modality_run_identity(config, dataset, modality, model))

    is_clinical = modality == CLINICAL_MODALITY
    if is_clinical:
        if model not in CLINICAL_MODELS:
            raise ValueError(
                f"Unknown clinical model '{model}'. Available: {sorted(CLINICAL_MODELS)}"
            )
        arch = None
    else:
        arch = resolve_arch(config, model)

    cv = config.cross_validation
    total = cv.num_reps * cv.num_folds
    completed = 0
    for split in iter_folds(
        dataset,
        num_reps=cv.num_reps,
        num_folds=cv.num_folds,
        base_seed=cv.seed,
        with_validation=not is_clinical,
    ):
        completed += 1
        label = f"rep {split.rep}/{cv.num_reps} fold {split.fold}/{cv.num_folds}"
        if resume and fold_is_complete(folds_dir, split.rep, split.fold):
            print(f"[{run_name(modality, model)}] {label}: already done, skipping")
            continue

        # Announced before training, not only after: the progress bar below
        # counts epochs within a fold and says nothing about which of the
        # hundred folds is on screen.
        print(
            f"[{run_name(modality, model)}] {label} ({completed}/{total}): training",
            flush=True,
        )
        set_seed(rep_seed(cv.seed, split.rep), deterministic=config.train.deterministic)
        if is_clinical:
            predictions, metrics = _run_clinical_fold(config, model, split)
        else:
            predictions, metrics = _run_image_fold(
                config,
                arch,
                modality,
                split,
                folds_dir / f"work_rep{split.rep:02d}_fold{split.fold:02d}",
                keep_checkpoints,
            )
        write_fold(folds_dir, split.rep, split.fold, predictions, metrics)
        print(
            f"[{run_name(modality, model)}] {label} ({completed}/{total}): "
            f"selection AUC {format_auc(metrics['selection_auc'])}, "
            f"test AUC {format_auc(metrics['test_auc'])}",
            flush=True,
        )

    return finalize_run(run_dir)
