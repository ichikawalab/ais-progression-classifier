"""Prediction bookkeeping and AUC-centred evaluation.

Every stage of the pipeline emits predictions in the same long format, so any
stage's output can be evaluated -- or fed to the ensemble -- with these helpers:

    patient_id, rep, fold, split, true_label, prob

``split`` is ``"val"`` or ``"test"``. ``rep``/``fold`` are 1-based, and are set
to 0 for the single-split final model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

PREDICTION_COLUMNS = ["patient_id", "rep", "fold", "split", "true_label", "prob"]


def format_auc(value: float | None, digits: int = 3) -> str:
    """Render an AUC for logs. AUC is undefined when a subset has one class."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def safe_auc(y_true, prob) -> float | None:
    """AUROC, or None when only one class is present in ``y_true``."""
    y_true = np.asarray(y_true, dtype=int)
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, np.asarray(prob, dtype=float)))


def youden_threshold(y_true, prob) -> float:
    """Threshold maximising sensitivity + specificity - 1, as used in the paper."""
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if np.unique(y_true).size < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    return float(thresholds[np.argmax(tpr - fpr)])


def binary_metrics(y_true, prob, threshold: float | None = None) -> dict:
    """AUC plus threshold-dependent metrics.

    ``threshold=None`` selects the Youden-index threshold, matching how the
    paper binarised predicted probabilities.
    """
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if threshold is None:
        threshold = youden_threshold(y_true, prob)
    predicted = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    def ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    # Reported with the clinical names only: sensitivity is recall and PPV is
    # precision, so emitting both spellings would just invite disagreement
    # between two copies of the same number.
    return {
        "n": int(y_true.size),
        "auc": safe_auc(y_true, prob),
        "threshold": float(threshold),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def make_predictions_frame(
    patient_ids,
    y_true,
    prob,
    rep: int,
    fold: int,
    split: str,
) -> pd.DataFrame:
    """Build one long-format prediction block."""
    if split not in {"val", "test"}:
        raise ValueError(f"split must be 'val' or 'test', got '{split}'")
    return pd.DataFrame(
        {
            "patient_id": np.asarray(patient_ids),
            "rep": int(rep),
            "fold": int(fold),
            "split": split,
            "true_label": np.asarray(y_true, dtype=int),
            "prob": np.asarray(prob, dtype=float),
        }
    )[PREDICTION_COLUMNS]


def load_predictions(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(PREDICTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file {path} is missing column(s): {sorted(missing)}")
    frame["patient_id"] = frame["patient_id"].astype(str)
    return frame


def auc_by_rep(predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    """AUC per repetition.

    For ``split="test"`` this pools the out-of-fold predictions of a repetition
    into a single ROC over the whole cohort -- the paper's ``trial_n`` column.
    For ``split="val"`` each fold has its own validation subset, so the AUC is
    computed per fold and then averaged within the repetition.
    """
    subset = predictions[predictions["split"] == split]
    if subset.empty:
        return pd.DataFrame(columns=["rep", "auc", "n"])

    if split == "test":
        rows = [
            {"rep": int(rep), "auc": safe_auc(group["true_label"], group["prob"]), "n": len(group)}
            for rep, group in subset.groupby("rep", sort=True)
        ]
        return pd.DataFrame(rows)

    per_fold = [
        {
            "rep": int(rep),
            "fold": int(fold),
            "auc": safe_auc(group["true_label"], group["prob"]),
            "n": len(group),
        }
        for (rep, fold), group in subset.groupby(["rep", "fold"], sort=True)
    ]
    frame = pd.DataFrame(per_fold)
    return (
        frame.groupby("rep", as_index=False)
        .agg(auc=("auc", "mean"), n=("n", "sum"))
        .astype({"rep": int})
    )


