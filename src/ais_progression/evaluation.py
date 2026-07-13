"""Out-of-fold prediction and binary classification metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
)


# A single metrics record: numeric scalars, an optional value (None when a
# metric is undefined for the subset), or the confusion-matrix nested list.
MetricValue = float | int | list | None
MetricDict = dict[str, MetricValue]


def binary_metrics(y_true, probability, threshold: float = 0.5) -> MetricDict:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    def safe_div(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    has_both_classes = np.unique(y_true).size == 2
    return {
        "n": int(y_true.size),
        "threshold": threshold,
        "loss": float(log_loss(y_true, np.column_stack([1 - probability, probability]), labels=[0, 1])),
        "auroc": float(roc_auc_score(y_true, probability)) if has_both_classes else None,
        "auprc": float(average_precision_score(y_true, probability)) if has_both_classes else None,
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, predicted)) if has_both_classes else None
        ),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "negative_predictive_value": safe_div(tn, tn + fn),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def predict_dataframe(
    model: torch.nn.Module,
    loader,
    source_df: pd.DataFrame,
    device: torch.device,
    fold: int,
    threshold: float = 0.5,
) -> pd.DataFrame:
    probabilities: list[np.ndarray] = []
    model.eval()
    model.to(device)
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            probabilities.append(torch.softmax(model(images), dim=1).cpu().numpy())
    probs = np.concatenate(probabilities, axis=0)
    if len(probs) != len(source_df):
        raise RuntimeError("Prediction count does not match test manifest.")
    return pd.DataFrame(
        {
            "sample_id": source_df["sample_id"].to_numpy(),
            "patient_id": source_df["patient_id"].to_numpy(),
            "true_label": source_df["label"].astype(int).to_numpy(),
            "fold": fold,
            "prob_class0": probs[:, 0],
            "prob_class1": probs[:, 1],
            "predicted_label": (probs[:, 1] >= threshold).astype(int),
        }
    )


def aggregate_patient_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average sample probabilities within patient for primary patient-level metrics."""
    label_counts = predictions.groupby("patient_id")["true_label"].nunique()
    if (label_counts != 1).any():
        raise ValueError("A patient has inconsistent true labels in predictions.")
    return (
        predictions.groupby("patient_id", as_index=False)
        .agg(true_label=("true_label", "first"), prob_class1=("prob_class1", "mean"))
        .assign(
            prob_class0=lambda frame: 1.0 - frame["prob_class1"],
            predicted_label=lambda frame: (frame["prob_class1"] >= 0.5).astype(int),
        )
    )


def bootstrap_confidence_intervals(
    patient_predictions: pd.DataFrame,
    seed: int,
    n_resamples: int = 2000,
) -> dict[str, list[float] | None]:
    """Percentile 95% CIs from a *class-stratified* patient-level bootstrap.

    Resampling each class with replacement to its original size keeps both
    classes present in every replicate, so threshold-free metrics (AUROC/AUPRC)
    are always defined and no replicate is silently dropped -- an unstratified
    bootstrap can draw single-class replicates that bias the interval.
    """
    rng = np.random.default_rng(seed)
    y_true = patient_predictions["true_label"].to_numpy().astype(int)
    indices_by_class = [np.flatnonzero(y_true == cls) for cls in (0, 1)]
    if any(idx.size == 0 for idx in indices_by_class):
        raise ValueError("Both classes must be present to bootstrap patient-level CIs.")

    metric_names = ("auroc", "auprc", "sensitivity", "specificity", "accuracy", "brier_score")
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(n_resamples):
        sampled_idx = np.concatenate(
            [rng.choice(idx, size=idx.size, replace=True) for idx in indices_by_class]
        )
        sampled = patient_predictions.iloc[sampled_idx]
        metrics = binary_metrics(sampled["true_label"], sampled["prob_class1"])
        for name in metric_names:
            value = metrics[name]
            if value is not None:
                values[name].append(float(value))
    return {
        name: [float(np.percentile(items, 2.5)), float(np.percentile(items, 97.5))]
        if items else None
        for name, items in values.items()
    }


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
