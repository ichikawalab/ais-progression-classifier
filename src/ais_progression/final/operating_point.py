"""Decision threshold and probability calibration, both derived from cross-validation.

A probability alone is not a decision. Turning one into "progression" or
"non-progression" needs a threshold, and quoting it as a risk needs the
probability to mean what it says. Both are properties of a *configuration* --
change which models are in the ensemble and both change -- so they are fitted
here from that configuration's out-of-fold predictions and stored with it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

from ais_progression.evaluation import binary_metrics, youden_threshold

CALIBRATION_METHODS = ("isotonic", "platt", "none")
DEFAULT_TARGET_SENSITIVITY = 0.90


@dataclass
class OperatingPoint:
    """A decision threshold plus what it achieved across the cross-validation."""

    threshold: float
    policy: str
    sensitivity_mean: float | None
    sensitivity_sd: float | None
    specificity_mean: float | None
    specificity_sd: float | None
    n_reps: int

    def as_dict(self) -> dict:
        return {
            "threshold": float(self.threshold),
            "policy": self.policy,
            "sensitivity_mean": self.sensitivity_mean,
            "sensitivity_sd": self.sensitivity_sd,
            "specificity_mean": self.specificity_mean,
            "specificity_sd": self.specificity_sd,
            "n_reps": self.n_reps,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> OperatingPoint:
        return cls(**payload)


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    array = np.asarray([v for v in values if v is not None], dtype=float)
    if array.size == 0:
        return None, None
    return float(array.mean()), (float(array.std(ddof=1)) if array.size > 1 else None)


def choose_operating_point(
    predictions: pd.DataFrame,
    policy: str = "youden",
    target_sensitivity: float = DEFAULT_TARGET_SENSITIVITY,
) -> OperatingPoint:
    """Pick a threshold from out-of-fold predictions.

    Youden's policy takes one candidate per repetition and uses their median.
    The target-sensitivity policy instead chooses the highest observed positive
    score whose mean sensitivity across repetitions reaches the requested
    target. Sensitivity and specificity are then measured per repetition at that
    fixed threshold.
    These values describe the cross-validated outer-fold predictions. The final
    serving weights are fitted separately on all base-model OOF probabilities,
    so the deployed refit is not itself evaluated by these values.
    """
    test = predictions[predictions["split"] == "test"]
    if test.empty:
        raise ValueError("No test predictions to choose an operating point from.")

    groups = [group for _, group in test.groupby("rep", sort=True)]
    if policy == "youden":
        candidates = []
        for group in groups:
            candidates.append(youden_threshold(group["true_label"], group["prob"]))
        threshold = float(np.median(candidates))
    elif policy == "target_sensitivity":
        if not 0 < target_sensitivity <= 1:
            raise ValueError("target_sensitivity must be in (0, 1].")
        candidates = np.unique(test.loc[test["true_label"] == 1, "prob"])
        qualifying = [
            float(candidate)
            for candidate in candidates
            if np.mean(
                [
                    np.mean(
                        group.loc[group["true_label"] == 1, "prob"].to_numpy()
                        >= candidate
                    )
                    for group in groups
                ]
            )
            >= target_sensitivity
        ]
        threshold = max(qualifying)
    else:
        raise ValueError(
            f"Unknown threshold policy '{policy}'. "
            "Available: 'youden', 'target_sensitivity'."
        )

    sensitivities, specificities = [], []
    for group in groups:
        metrics = binary_metrics(group["true_label"], group["prob"], threshold=threshold)
        sensitivities.append(metrics["sensitivity"])
        specificities.append(metrics["specificity"])
    sensitivity_mean, sensitivity_sd = _mean_sd(sensitivities)
    specificity_mean, specificity_sd = _mean_sd(specificities)

    label = (
        policy
        if policy == "youden"
        else f"mean_sensitivity>={target_sensitivity:g}"
    )
    return OperatingPoint(
        threshold=threshold,
        policy=label,
        sensitivity_mean=sensitivity_mean,
        sensitivity_sd=sensitivity_sd,
        specificity_mean=specificity_mean,
        specificity_sd=specificity_sd,
        n_reps=len(groups),
    )


class Calibrator:
    """Maps a raw ensemble score onto a calibrated probability.

    Fitted on out-of-fold predictions, so the mapping is learned from
    probabilities the base models produced for unseen patients. Calibration is
    monotonic, so AUC is unchanged; what improves is whether "0.7" means roughly
    a 70% chance of progression.
    """

    def __init__(self, method: str, model: Any):
        self.method = method
        self.model = model

    def transform(self, prob) -> np.ndarray:
        prob = np.asarray(prob, dtype=float)
        if self.method == "isotonic":
            return np.clip(self.model.predict(prob), 0.0, 1.0)
        if self.method == "platt":
            return self.model.predict_proba(prob.reshape(-1, 1))[:, 1]
        raise ValueError(f"Unknown calibration method '{self.method}'")


def _fit_calibrator(method: str, prob: np.ndarray, y: np.ndarray) -> Calibrator:
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(prob, y)
    elif method == "platt":
        model = LogisticRegression(C=1e10, solver="lbfgs")
        model.fit(prob.reshape(-1, 1), y)
    else:
        raise ValueError(
            f"Unknown calibration method '{method}'. Available: {CALIBRATION_METHODS}"
        )
    return Calibrator(method, model)


def fit_calibrator(
    predictions: pd.DataFrame, method: str = "isotonic", seed: int = 42, n_folds: int = 10
) -> tuple[Calibrator | None, dict]:
    """Fit a calibrator on the out-of-fold predictions.

    Returns the calibrator plus a report. The reported calibrated Brier score
    comes from an internal cross-validation over the same predictions, because
    scoring a calibrator on the data it was fitted to would flatter it. That
    internal split is grouped by patient: with ten repetitions every patient
    contributes ten rows, and splitting rows would leave the same patient's
    other repetitions -- carrying the same label -- in the calibrator's own
    training set.
    """
    if method == "none":
        return None, {"method": "none"}

    test = predictions[predictions["split"] == "test"]
    if test.empty:
        raise ValueError("No test predictions to calibrate against.")
    prob = test["prob"].to_numpy(dtype=float)
    y = test["true_label"].to_numpy(dtype=int)
    patients = test["patient_id"].to_numpy()

    honest = np.empty_like(prob)
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(prob.reshape(-1, 1), y, groups=patients):
        fold_calibrator = _fit_calibrator(method, prob[train_idx], y[train_idx])
        honest[test_idx] = fold_calibrator.transform(prob[test_idx])

    report = {
        "method": method,
        "n_observations": int(prob.size),
        "n_patients": int(pd.unique(patients).size),
        "brier_raw": float(brier_score_loss(y, prob)),
        "brier_calibrated": float(brier_score_loss(y, honest)),
        "brier_calibrated_estimate": (
            f"internal {n_folds}-fold over the out-of-fold predictions, grouped by patient"
        ),
    }
    return _fit_calibrator(method, prob, y), report
