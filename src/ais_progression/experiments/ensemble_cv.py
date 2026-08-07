"""Nested cross-validation of the multimodal ensembles.

Inputs are the out-of-fold probabilities of the individual models. Within a
repetition the outer folds are regenerated with the same seed the base models
used -- ``assert_folds_match`` enforces it -- so an ensemble's test fold is
exactly the base models' test fold, and the probabilities it is *scored* on came
from base models that never saw those patients. The ensemble itself is fitted on
the outer training fold, with an inner stratified K-fold for weight or
hyperparameter search.

One caveat, inherent to stacking on a single out-of-fold matrix: a *training*
patient's probability came from a base model trained on every fold but that
patient's own -- including the current test fold. The fusion weights are
therefore chosen with indirect knowledge of the test fold, which biases the
reported ensemble AUC upward. Removing it would mean regenerating the base
models' out-of-fold probabilities separately inside each outer fold, i.e. ten
times the image training, and would no longer be the reference procedure. It is
recorded in ``STACKING_LEAKAGE_WARNING`` and in the README instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ais_progression.config import Config
from ais_progression.data.schema import ID_COLUMN, LABEL_COLUMN
from ais_progression.ensemble.meta import META_MODELS, fit_meta_ensemble, simple_average
from ais_progression.ensemble.weighted import fit_weighted_ensemble
from ais_progression.evaluation import (
    format_auc,
    load_predictions,
    make_predictions_frame,
    safe_auc,
)
from ais_progression.experiments.reporting import (
    assert_run_identity,
    collect_folds,
    finalize_run,
    fold_is_complete,
    write_fold,
)
from ais_progression.experiments.splits import Fold, iter_folds
from ais_progression.provenance import frame_sha256, software_identity
from ais_progression.utils import save_json, set_seed

AVERAGE_METHOD = "average"
WEIGHTED_METHOD = "weighted"

ENSEMBLE_METHOD_SELECTION_WARNING = (
    "Comparing several ensemble methods on these same test folds and keeping "
    "the best one adds selection bias to the stacking leakage described separately. "
    "It turns the test folds into a selection set and biases the reported "
    "performance upward. The choice of weighted averaging was made this "
    "way, so its AUC should be read as a selected-best value. Confirming it "
    "requires an independent external cohort or a nested selection design."
)

STACKING_LEAKAGE_WARNING = (
    "The stacking protocol reuses one base-model out-of-fold matrix. "
    "For an ensemble outer fold, its training patients' probabilities can therefore "
    "come from base models that were trained on the current test fold. The fusion "
    "weights have indirect knowledge of that fold, so the ensemble AUC may be "
    "optimistic. Removing this requires nested regeneration of the base predictions."
)


def ensemble_run_identity(
    config: Config,
    dataset: pd.DataFrame,
    base_predictions: dict[str, pd.DataFrame],
    method: str,
) -> dict:
    """Everything that can change a resumed ensemble run's fold outputs."""
    cv = config.cross_validation
    return {
        "method": method,
        "base_models": sorted(base_predictions),
        "base_prediction_sha256": {
            name: frame_sha256(base_predictions[name]) for name in sorted(base_predictions)
        },
        "dataset_sha256": frame_sha256(dataset),
        "num_reps": cv.num_reps,
        "num_folds": cv.num_folds,
        "seed": cv.seed,
        "ensemble_n_trials": config.ensemble.n_trials,
        "ensemble_inner_folds": config.ensemble.inner_folds,
        "software": software_identity(),
    }


def load_base_predictions(sources: dict[str, str | Path]) -> dict[str, pd.DataFrame]:
    """Load one base model's predictions per entry.

    Each value may be a modality-CV run directory or a predictions.csv path.
    """
    loaded: dict[str, pd.DataFrame] = {}
    for name, source in sources.items():
        path = Path(source)
        if path.is_dir():
            path = path / "predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"No predictions for base model '{name}': {path}")
        loaded[name] = load_predictions(path)
    return loaded


def build_oof_matrix(
    base_predictions: dict[str, pd.DataFrame], dataset: pd.DataFrame, rep: int
) -> pd.DataFrame:
    """Patients x base models matrix of out-of-fold probabilities for one repetition."""
    patient_ids = dataset[ID_COLUMN].astype(str)
    columns: dict[str, np.ndarray] = {}
    for name, predictions in base_predictions.items():
        subset = predictions[(predictions["split"] == "test") & (predictions["rep"] == rep)]
        if subset.empty:
            raise ValueError(f"Base model '{name}' has no test predictions for rep {rep}.")
        if subset["patient_id"].duplicated().any():
            raise ValueError(
                f"Base model '{name}' has more than one out-of-fold prediction per patient "
                f"in rep {rep}."
            )
        series = subset.set_index("patient_id")["prob"]
        missing = set(patient_ids) - set(series.index)
        if missing:
            raise ValueError(
                f"Base model '{name}' is missing rep {rep} predictions for "
                f"{len(missing)} patient(s), e.g. {sorted(missing)[:5]}"
            )
        columns[name] = series.reindex(patient_ids).to_numpy()
    return pd.DataFrame(columns, index=pd.Index(patient_ids, name=ID_COLUMN))


def assert_folds_match(
    base_predictions: dict[str, pd.DataFrame],
    dataset: pd.DataFrame,
    num_reps: int,
    num_folds: int,
    base_seed: int,
) -> None:
    """Require every base model's fold assignment to be the one used here.

    The whole design rests on the ensemble's test fold being the base models'
    test fold. Nothing enforced that: ``ais-cv-modality`` and ``ais-cv-ensemble``
    each take their own ``--seed``, and the matrix is assembled by patient id, so
    a mismatched seed silently pairs a patient's probability with the wrong fold
    -- and the model that produced it was then trained on other patients of this
    fold. No exception, just a quietly optimistic AUC.
    """
    expected = {
        (split.rep, patient_id): split.fold
        for split in iter_folds(
            dataset,
            num_reps=num_reps,
            num_folds=num_folds,
            base_seed=base_seed,
            with_validation=False,
        )
        for patient_id in split.test[ID_COLUMN].astype(str)
    }
    for name, predictions in base_predictions.items():
        test = predictions[predictions["split"] == "test"]
        mismatched = [
            (int(row.rep), row.patient_id, int(row.fold))
            for row in test.itertuples()
            if expected.get((int(row.rep), str(row.patient_id))) != int(row.fold)
        ]
        if mismatched:
            rep, patient_id, found = mismatched[0]
            raise ValueError(
                f"Base model '{name}' was cross-validated with a different fold "
                f"assignment than this run: {len(mismatched)} prediction(s) disagree, "
                f"e.g. patient {patient_id} is in fold {found} of rep {rep} there but "
                f"fold {expected.get((rep, str(patient_id)))} here. Re-run it with the "
                f"same --reps/--folds/--seed ({num_reps}/{num_folds}/{base_seed})."
            )


def _fit_and_predict(
    method: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    config: Config,
    seed: int,
) -> tuple[pd.Series, float | None, str, dict]:
    """Returns (test probabilities, validation AUC, its source, extra metadata)."""
    ensemble_cfg = config.ensemble
    if method == AVERAGE_METHOD:
        # Unweighted mean: nothing is fitted, so the training-fold AUC is an
        # honest reference point rather than an in-sample optimum.
        train_auc = safe_auc(y_train, simple_average(X_train))
        return pd.Series(simple_average(X_test), index=X_test.index), train_auc, "train", {}

    if method == WEIGHTED_METHOD:
        ensemble = fit_weighted_ensemble(
            X_train, y_train, seed, ensemble_cfg.n_trials, ensemble_cfg.inner_folds
        )
        extra = {
            "weights": {
                column: float(weight)
                for column, weight in zip(ensemble.columns, ensemble.weights, strict=True)
            },
            "modality_weights": ensemble.modality_weights(),
        }
        return (
            pd.Series(ensemble.predict(X_test), index=X_test.index),
            ensemble.inner_cv_auc,
            "inner_cv",
            extra,
        )

    if method in META_MODELS:
        ensemble = fit_meta_ensemble(
            X_train, y_train, method, seed, ensemble_cfg.n_trials, ensemble_cfg.inner_folds
        )
        return (
            pd.Series(ensemble.predict(X_test), index=X_test.index),
            ensemble.inner_cv_auc,
            "inner_cv",
            {"best_params": ensemble.best_params},
        )

    raise ValueError(
        f"Unknown ensemble method '{method}'. Available: "
        f"{sorted({AVERAGE_METHOD, WEIGHTED_METHOD, *META_MODELS})}"
    )


def _run_ensemble_fold(
    config: Config, method: str, X: pd.DataFrame, split: Fold
) -> tuple[pd.DataFrame, dict]:
    train_ids = split.train[ID_COLUMN].astype(str)
    test_ids = split.test[ID_COLUMN].astype(str)
    X_train, X_test = X.loc[train_ids], X.loc[test_ids]
    y_train = split.train[LABEL_COLUMN].astype(int)
    y_test = split.test[LABEL_COLUMN].astype(int)

    probabilities, val_auc, val_source, extra = _fit_and_predict(
        method, X_train, y_train, X_test, config, split.model_seed
    )
    predictions = make_predictions_frame(
        test_ids, y_test, probabilities.to_numpy(), split.rep, split.fold, "test"
    )
    metrics = {
        "rep": split.rep,
        "fold": split.fold,
        "split_seed": split.split_seed,
        "model_seed": split.model_seed,
        **split.sizes,
        "selection_auc": val_auc,
        "selection_auc_source": val_source,
        "test_auc": safe_auc(y_test, probabilities.to_numpy()),
        **extra,
    }
    return predictions, metrics


def _write_weight_summary(run_dir: Path, fold_metrics: pd.DataFrame) -> None:
    """Per-model and per-modality ensemble weights, averaged over folds."""
    if "weights" not in fold_metrics.columns:
        return
    weights = pd.json_normalize(fold_metrics["weights"]).assign(
        rep=fold_metrics["rep"].to_numpy(), fold=fold_metrics["fold"].to_numpy()
    )
    weights.to_csv(run_dir / "weights_by_fold.csv", index=False)
    model_columns = [c for c in weights.columns if c not in ("rep", "fold")]
    summary = {
        "by_model": {
            column: {
                "mean": float(weights[column].mean()),
                "sd": float(weights[column].std(ddof=1)) if len(weights) > 1 else None,
            }
            for column in model_columns
        }
    }
    modality_weights = pd.json_normalize(fold_metrics["modality_weights"])
    summary["by_modality"] = {
        column: {
            "mean": float(modality_weights[column].mean()),
            "sd": float(modality_weights[column].std(ddof=1))
            if len(modality_weights) > 1
            else None,
        }
        for column in modality_weights.columns
    }
    save_json(summary, run_dir / "weights_summary.json")


def run_ensemble_cv(
    config: Config,
    dataset: pd.DataFrame,
    base_predictions: dict[str, pd.DataFrame],
    method: str,
    run_dir: str | Path,
    resume: bool = True,
) -> dict:
    """Run the nested cross-validation for one ensemble method."""
    run_dir = Path(run_dir)
    folds_dir = run_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    cv = config.cross_validation
    assert_run_identity(
        run_dir, ensemble_run_identity(config, dataset, base_predictions, method)
    )
    assert_folds_match(base_predictions, dataset, cv.num_reps, cv.num_folds, cv.seed)
    matrices = {
        rep: build_oof_matrix(base_predictions, dataset, rep)
        for rep in range(1, cv.num_reps + 1)
    }

    total = cv.num_reps * cv.num_folds
    completed = 0
    for split in iter_folds(
        dataset,
        num_reps=cv.num_reps,
        num_folds=cv.num_folds,
        base_seed=cv.seed,
        with_validation=False,
    ):
        completed += 1
        label = f"rep {split.rep}/{cv.num_reps} fold {split.fold}/{cv.num_folds}"
        if resume and fold_is_complete(folds_dir, split.rep, split.fold):
            print(f"[ensemble/{method}] {label}: already done, skipping")
            continue

        set_seed(split.model_seed, deterministic=False)
        predictions, metrics = _run_ensemble_fold(config, method, matrices[split.rep], split)
        write_fold(folds_dir, split.rep, split.fold, predictions, metrics)
        print(
            f"[ensemble/{method}] {label} ({completed}/{total}): "
            f"selection AUC {format_auc(metrics['selection_auc'])}, "
            f"test AUC {format_auc(metrics['test_auc'])}"
        )

    summary = finalize_run(
        run_dir,
        extra_summary={
            "method": method,
            "ensemble_method_selection_warning": ENSEMBLE_METHOD_SELECTION_WARNING,
            "stacking_leakage_warning": STACKING_LEAKAGE_WARNING,
        },
    )
    # Re-read the per-fold JSON rather than fold_metrics.csv: the weight columns
    # are nested dicts, which survive JSON but not a CSV round-trip.
    _write_weight_summary(run_dir, collect_folds(folds_dir)[1])
    return summary
