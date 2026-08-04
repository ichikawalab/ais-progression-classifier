"""Assemble per-fold artefacts into a run's predictions, metrics, and summary.

Every cross-validation run writes one small file pair per fold, so an
interrupted run resumes at fold granularity, and the run-level files are just a
concatenation of what already exists on disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ais_progression.evaluation import (
    PREDICTION_COLUMNS,
    auc_by_rep,
    binary_metrics,
    load_predictions,
)
from ais_progression.utils import load_json, save_json


def fold_stem(rep: int, fold: int) -> str:
    return f"rep{rep:02d}_fold{fold:02d}"


def fold_is_complete(folds_dir: Path, rep: int, fold: int) -> bool:
    stem = fold_stem(rep, fold)
    return (folds_dir / f"{stem}.csv").exists() and (folds_dir / f"{stem}.json").exists()


IDENTITY_NAME = "run_identity.json"


def assert_run_identity(run_dir: Path, identity: dict) -> None:
    """Pin what a run directory contains, so resuming cannot mix two runs.

    Completion is judged per fold by file existence, which says nothing about
    *what* produced those files. Re-running a differently composed ensemble into
    the same directory would therefore skip every fold and hand back the earlier
    run's predictions and weights -- and since the new members are a subset of
    the old columns, nothing would raise.
    """
    run_dir = Path(run_dir)
    path = run_dir / IDENTITY_NAME
    advice = "Use a different --run-dir, or delete this one to recompute it."

    if path.exists():
        recorded = load_json(path)
        if recorded != identity:
            changed = sorted(
                key
                for key in {*recorded, *identity}
                if recorded.get(key) != identity.get(key)
            )
            raise ValueError(
                f"{run_dir} holds a run with a different configuration ({changed} "
                f"differ: {[(key, recorded.get(key), identity.get(key)) for key in changed]}). "
                f"{advice}"
            )
        return

    # No identity, but folds are already here: a run from before this check, or
    # from somewhere else. Adopting it is the very thing being guarded against,
    # because resuming would then skip folds nobody can attribute.
    existing = sorted((run_dir / "folds").glob("rep*_fold*.json"))
    if existing:
        raise ValueError(
            f"{run_dir} already holds {len(existing)} completed fold(s) but no "
            f"{IDENTITY_NAME}, so what produced them cannot be established. {advice}"
        )
    save_json(identity, path)


def write_fold(
    folds_dir: Path, rep: int, fold: int, predictions: pd.DataFrame, metrics: dict
) -> None:
    folds_dir.mkdir(parents=True, exist_ok=True)
    stem = fold_stem(rep, fold)
    predictions[PREDICTION_COLUMNS].to_csv(folds_dir / f"{stem}.csv", index=False)
    save_json(metrics, folds_dir / f"{stem}.json")


def collect_folds(folds_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate every completed fold's predictions and metrics."""
    prediction_files = sorted(folds_dir.glob("rep*_fold*.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No completed folds found under {folds_dir}")
    predictions = pd.concat(
        [load_predictions(path) for path in prediction_files], ignore_index=True
    )
    metrics = pd.DataFrame(
        [load_json(path) for path in sorted(folds_dir.glob("rep*_fold*.json"))]
    ).sort_values(["rep", "fold"], ignore_index=True)
    return predictions, metrics


def _mean_sd(values: pd.Series | np.ndarray) -> dict:
    array = pd.Series(values).dropna().to_numpy(dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()) if array.size else None,
        # SD of a single observation is undefined, not zero.
        "sd": float(array.std(ddof=1)) if array.size > 1 else None,
    }


def summarize_run(predictions: pd.DataFrame, fold_metrics: pd.DataFrame) -> dict:
    """Validation and test AUC, per fold, per repetition, and pooled.

    ``test_auc_pooled_per_rep`` is the headline number: within a repetition every
    patient has exactly one out-of-fold prediction, so the ten folds combine into
    a single ROC over the whole cohort. Its mean and SD across repetitions are
    the figure to report.
    """
    summary: dict = {
        "n_folds_completed": int(len(fold_metrics)),
        "test_auc_per_fold": _mean_sd(fold_metrics.get("test_auc", pd.Series(dtype=float))),
    }

    # The selection AUC means different things per model family: a held-out slice
    # for image models, an inner cross-validation score for the clinical and
    # ensemble learners, the training fold for simple averaging. They are grouped
    # by source and never pooled, because averaging across them is meaningless.
    if {"selection_auc", "selection_auc_source"} <= set(fold_metrics.columns):
        summary["selection_auc_by_source"] = {
            str(source): _mean_sd(group["selection_auc"])
            for source, group in fold_metrics.groupby("selection_auc_source", dropna=True)
        }

    per_rep = auc_by_rep(predictions, "test")
    summary["test_auc_pooled_per_rep"] = _mean_sd(per_rep["auc"] if not per_rep.empty else [])
    summary["test_auc_by_rep"] = (
        {
            int(row.rep): (None if pd.isna(row.auc) else float(row.auc))
            for row in per_rep.itertuples()
        }
        if not per_rep.empty
        else {}
    )

    test = predictions[predictions["split"] == "test"]
    if not test.empty:
        # Threshold-dependent metrics are computed per repetition and then
        # averaged, which is how the reference tables were produced. Averaging the
        # probabilities first and thresholding once is a different quantity.
        per_rep_metrics = [
            binary_metrics(group["true_label"], group["prob"])
            for _, group in test.groupby("rep", sort=True)
        ]
        summary["test_metrics_per_rep_mean"] = {
            "threshold_source": "youden, per repetition",
            "n_reps": len(per_rep_metrics),
            **{
                name: _mean_sd([m[name] for m in per_rep_metrics])
                for name in ("sensitivity", "specificity", "ppv", "npv", "accuracy", "f1")
            },
        }
    return summary


def finalize_run(run_dir: Path, extra_summary: dict | None = None) -> dict:
    """Write predictions.csv, fold_metrics.csv, and summary.json for a run."""
    predictions, fold_metrics = collect_folds(run_dir / "folds")
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    summary = summarize_run(predictions, fold_metrics)
    summary.update(extra_summary or {})
    save_json(summary, run_dir / "summary.json")
    return summary
