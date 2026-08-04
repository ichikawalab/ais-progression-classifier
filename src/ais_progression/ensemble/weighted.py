"""Weighted averaging of the individual models' predicted probabilities.

Weights are searched with Optuna TPE: each model gets a weight in [0, 1], the
vector is normalised to sum to one, and the objective is the mean AUC of the
weighted average across an inner stratified K-fold. This is the ensemble method
that performed best.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class WeightedEnsemble:
    """A fitted weighted-average ensemble over named model columns."""

    columns: list[str]
    weights: np.ndarray
    # None when the weights were fitted by a different objective, as the final
    # serving profiles are.
    inner_cv_auc: float | None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.columns if column not in X.columns]
        if missing:
            raise ValueError(f"Input is missing model column(s): {missing}")
        return X[self.columns].to_numpy(dtype=float) @ self.weights

    def modality_weights(self) -> dict[str, float]:
        """Weight totalled per modality, using the ``<modality>_<model>`` naming."""
        totals: dict[str, float] = {}
        for column, weight in zip(self.columns, self.weights, strict=True):
            modality = column.split("_", 1)[0]
            totals[modality] = totals.get(modality, 0.0) + float(weight)
        return totals


def fit_weighted_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_trials: int,
    inner_folds: int,
) -> WeightedEnsemble:
    """Search weights on ``X`` (one column per base model)."""
    columns = list(X.columns)
    values = X.to_numpy(dtype=float)
    y = np.asarray(y, dtype=int)
    splits = list(
        StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed).split(values, y)
    )

    def objective(trial: optuna.Trial) -> float:
        weights = np.array(
            [trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(len(columns))]
        )
        weights = weights / (weights.sum() + 1e-12)
        scores = [
            roc_auc_score(y[val_idx], values[val_idx] @ weights)
            for _, val_idx in splits
            if np.unique(y[val_idx]).size == 2
        ]
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    weights = np.array([study.best_trial.params[f"w{i}"] for i in range(len(columns))])
    total = weights.sum()
    # Degenerate all-zero draw: fall back to equal weighting rather than dividing by zero.
    weights = weights / total if total > 0 else np.full(len(columns), 1.0 / len(columns))
    return WeightedEnsemble(
        columns=columns, weights=weights, inner_cv_auc=float(study.best_value)
    )


def fit_repeated_oof_ensemble(
    matrices: dict[int, pd.DataFrame],
    y,
    seed: int,
    n_trials: int,
) -> WeightedEnsemble:
    """Fit one deployable weight vector on all repeated OOF predictions.

    A trial is scored by computing one full-cohort AUC per repetition and
    averaging those AUCs. Patients therefore keep the same influence regardless
    of the number of repetitions, and the objective matches the repository's
    headline reporting unit. The selected-best objective value is deliberately
    not exposed as performance; outer-fold predictions remain the source of the
    profile's AUC, operating point, and calibration report.
    """
    repetitions = sorted(matrices)
    columns = list(matrices[repetitions[0]].columns)
    values = []
    for rep in repetitions:
        matrix = matrices[rep]
        if list(matrix.columns) != columns:
            raise ValueError(f"Base-model columns differ in repetition {rep}.")
        values.append(matrix.to_numpy(dtype=float))
    y = np.asarray(y, dtype=int)

    def objective(trial: optuna.Trial) -> float:
        weights = np.array(
            [trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(len(columns))]
        )
        weights = weights / (weights.sum() + 1e-12)
        return float(np.mean([roc_auc_score(y, matrix @ weights) for matrix in values]))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    weights = np.array([study.best_trial.params[f"w{i}"] for i in range(len(columns))])
    total = weights.sum()
    weights = weights / total if total > 0 else np.full(len(columns), 1.0 / len(columns))
    return WeightedEnsemble(columns=columns, weights=weights, inner_cv_auc=None)
