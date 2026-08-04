"""The comparison ensembles: simple averaging and stacked meta-learners.

The meta-learners (logistic regression, SVM, random forest) are trained on the
nine base models' predicted probabilities. No feature preprocessing is applied:
the inputs are already probabilities on a common scale.

The search spaces here are deliberately *not* the clinical ones: the published
code widened the SVM's gamma range and narrowed the random forest's depth for
this stage. Those two bounds are the whole difference, so the spaces themselves
are shared with the clinical modality and only the ranges are rebound.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import optuna
import pandas as pd

from ais_progression.models.clinical_model import (
    build_estimator,
    suggest_logreg,
    suggest_rf,
    suggest_svm,
    tune_with_search_space,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# The published code widened the SVM's gamma range and narrowed the forest's
# depth for this stage; logistic regression was shared unchanged.
ENSEMBLE_GAMMA_RANGE = (1e-3, 10.0)
ENSEMBLE_MAX_DEPTH_RANGE = (3, 10)

ENSEMBLE_SEARCH_SPACES: dict[str, Callable[[optuna.Trial, int], dict[str, Any]]] = {
    "logreg": suggest_logreg,
    "svm": partial(suggest_svm, gamma_range=ENSEMBLE_GAMMA_RANGE),
    "rf": partial(suggest_rf, max_depth_range=ENSEMBLE_MAX_DEPTH_RANGE),
}
META_MODELS = tuple(ENSEMBLE_SEARCH_SPACES)


@dataclass
class MetaEnsemble:
    """A fitted stacked ensemble over named model columns."""

    columns: list[str]
    model: str
    estimator: Any
    best_params: dict[str, Any]
    inner_cv_auc: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.columns if column not in X.columns]
        if missing:
            raise ValueError(f"Input is missing model column(s): {missing}")
        return self.estimator.predict_proba(X[self.columns])[:, 1]


def fit_meta_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    model: str,
    seed: int,
    n_trials: int,
    inner_folds: int,
) -> MetaEnsemble:
    """Tune a meta-learner on the base models' probabilities, then refit on all of ``X``."""
    if model not in ENSEMBLE_SEARCH_SPACES:
        raise ValueError(
            f"Unknown meta model '{model}'. Available: {sorted(ENSEMBLE_SEARCH_SPACES)}"
        )
    # X is indexed by patient_id while y is not; scikit-learn indexes positionally
    # so this works, but converting makes the positional contract explicit.
    y = np.asarray(y, dtype=int)
    best_params, inner_cv_auc = tune_with_search_space(
        X,
        y,
        model,
        ENSEMBLE_SEARCH_SPACES[model],
        seed=seed,
        n_trials=n_trials,
        inner_folds=inner_folds,
        preprocessor_factory=None,
    )
    estimator = build_estimator(model, best_params)
    estimator.fit(X, y)
    return MetaEnsemble(
        columns=list(X.columns),
        model=model,
        estimator=estimator,
        best_params=best_params,
        inner_cv_auc=inner_cv_auc,
    )


def simple_average(X: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    """Unweighted mean of the given model columns."""
    columns = columns or list(X.columns)
    return X[columns].to_numpy(dtype=float).mean(axis=1)
