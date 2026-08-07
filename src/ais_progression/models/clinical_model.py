"""Clinical-modality models: logistic regression, SVM, and random forest.

Four a priori predictors (age, sex, Risser sign, baseline Cobb angle) feed a
scikit-learn pipeline. Continuous variables are z-scored, sex is one-hot
encoded, and the Risser sign is ordinal encoded. Hyperparameters come from an
Optuna TPE search over an inner stratified K-fold, scored by AUC.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from joblib import parallel_backend
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

from ais_progression.config import ClinicalConfig

# Distance-based models need scaled inputs; trees do not.
SCALED_MODELS = frozenset({"logreg", "svm"})
SEX_CATEGORIES = [[1.0, 2.0]]

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class ClinicalFitResult:
    pipeline: Pipeline
    best_params: dict[str, Any]
    inner_cv_auc: float


def build_preprocessor(clinical_cfg: ClinicalConfig, use_scaler: bool) -> ColumnTransformer:
    """Impute, then scale / one-hot / ordinal encode the clinical variables."""
    numeric_steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), clinical_cfg.numeric_features),
            (
                "binary_cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                categories=SEX_CATEGORIES * len(clinical_cfg.binary_features),
                                sparse_output=False,
                                handle_unknown="ignore",
                            ),
                        ),
                    ]
                ),
                clinical_cfg.binary_features,
            ),
            (
                "ordinal_cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                clinical_cfg.ordinal_features,
            ),
        ]
    )


def suggest_logreg(trial: optuna.Trial, seed: int) -> dict[str, Any]:
    pair = trial.suggest_categorical(
        "penalty_solver",
        [
            "l1/liblinear",
            "l1/saga",
            "l2/lbfgs",
            "l2/liblinear",
            "l2/saga",
            "elasticnet/saga",
        ],
    )
    penalty, solver = pair.split("/", 1)

    params: dict[str, Any] = {
        "penalty": penalty,
        "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
        "solver": solver,
        "fit_intercept": True,
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "random_state": seed,
        "max_iter": 10000,
    }
    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    return params


# The SVM's gamma range and the forest's depth range are the only things that
# differ between the clinical and the ensemble stage, so they are parameters
# rather than a second copy of each search space. Every other bound is shared.
CLINICAL_GAMMA_RANGE = (1e-4, 1.0)
CLINICAL_MAX_DEPTH_RANGE = (3, 20)


def suggest_svm(
    trial: optuna.Trial, seed: int, gamma_range: tuple[float, float] = CLINICAL_GAMMA_RANGE
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
        "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
        "probability": True,
        "random_state": seed,
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "shrinking": trial.suggest_categorical("shrinking", [True, False]),
    }
    if params["kernel"] == "rbf":
        params["gamma"] = trial.suggest_float("gamma", *gamma_range, log=True)
    return params


def suggest_rf(
    trial: optuna.Trial, seed: int, max_depth_range: tuple[int, int] = CLINICAL_MAX_DEPTH_RANGE
) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", *max_depth_range),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "random_state": seed,
        "n_jobs": -1,
    }


# Search spaces for the clinical modality. The ensemble stage widens gamma and
# narrows max_depth -- see ais_progression.ensemble.meta.
CLINICAL_SEARCH_SPACES: dict[str, Callable[[optuna.Trial, int], dict[str, Any]]] = {
    "logreg": suggest_logreg,
    "svm": suggest_svm,
    "rf": suggest_rf,
}
_BUILDERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "logreg": lambda params: LogisticRegression(**params),
    "svm": lambda params: SVC(**params),
    "rf": lambda params: RandomForestClassifier(**params),
}


def tune_with_search_space(
    X: pd.DataFrame,
    y: pd.Series,
    model: str,
    suggest: Callable[[optuna.Trial, int], dict[str, Any]],
    seed: int,
    n_trials: int,
    inner_folds: int,
    preprocessor_factory: Callable[[], Any] | None = None,
) -> tuple[dict[str, Any], float]:
    """Optuna TPE search over an inner stratified K-fold, maximising AUC.

    ``preprocessor_factory`` is None for inputs that are already on a common
    scale, such as the ensemble stage's predicted probabilities.
    """

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, seed)
        estimator = build_estimator(model, params)
        pipeline = (
            Pipeline([("preproc", preprocessor_factory()), ("est", estimator)])
            if preprocessor_factory is not None
            else Pipeline([("est", estimator)])
        )
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        with parallel_backend("threading"):
            scores = cross_val_score(pipeline, X, y, cv=inner_cv, scoring="roc_auc", n_jobs=-1)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    # Replay the winning trial so conditional parameters (e.g. gamma, l1_ratio)
    # are filled in exactly as they were during the search.
    return suggest(optuna.trial.FixedTrial(study.best_trial.params), seed), float(study.best_value)


def build_estimator(model: str, params: dict[str, Any]):
    try:
        return _BUILDERS[model](params)
    except KeyError:
        raise ValueError(
            f"Unknown clinical model '{model}'. Available: {sorted(_BUILDERS)}"
        ) from None


def fit_clinical_model(
    X: pd.DataFrame,
    y: pd.Series,
    model: str,
    clinical_cfg: ClinicalConfig,
    seed: int,
) -> ClinicalFitResult:
    """Tune on the training subset, then refit the winning pipeline on all of it."""
    if model not in CLINICAL_SEARCH_SPACES:
        raise ValueError(
            f"Unknown clinical model '{model}'. Available: {sorted(CLINICAL_SEARCH_SPACES)}"
        )
    use_scaler = model in SCALED_MODELS
    best_params, inner_cv_auc = tune_with_search_space(
        X,
        y,
        model,
        CLINICAL_SEARCH_SPACES[model],
        seed=seed,
        n_trials=clinical_cfg.n_trials,
        inner_folds=clinical_cfg.inner_folds,
        preprocessor_factory=lambda: build_preprocessor(clinical_cfg, use_scaler),
    )
    pipeline = Pipeline(
        [
            ("preproc", build_preprocessor(clinical_cfg, use_scaler)),
            ("est", build_estimator(model, best_params)),
        ]
    )
    pipeline.fit(X, y)
    return ClinicalFitResult(
        pipeline=pipeline, best_params=best_params, inner_cv_auc=inner_cv_auc
    )


def predict_clinical_model(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Progression probabilities for ``X``, in row order."""
    return pipeline.predict_proba(X)[:, 1]
