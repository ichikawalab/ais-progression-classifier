"""Serving profiles: a model subset with its own weights, threshold, and calibrator.

The deployed ensemble does not have to be all nine models. A caller may want a
cheaper configuration -- clinical variables only, or frontal plus clinical --
and each one is a different model with a different operating point. Reusing the
nine-model threshold for a three-model ensemble would silently invalidate the
reported sensitivity and specificity.

Every profile is derived from cross-validation predictions, so building extra
profiles costs no image training: only the ensemble stage is recomputed, over
probabilities that already exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ais_progression.config import CLINICAL_MODALITY, Config
from ais_progression.data.schema import LABEL_COLUMN
from ais_progression.ensemble.weighted import WeightedEnsemble, fit_repeated_oof_ensemble
from ais_progression.evaluation import load_predictions
from ais_progression.experiments.ensemble_cv import (
    WEIGHTED_METHOD,
    build_oof_matrix,
    run_ensemble_cv,
)
from ais_progression.final.operating_point import (
    Calibrator,
    OperatingPoint,
    choose_operating_point,
    fit_calibrator,
)
from ais_progression.utils import load_json

FULL_PROFILE = "full"

# Named subsets offered out of the box. A profile is kept only if every member
# it names has a completed cross-validation run.
DEFAULT_PROFILE_MEMBERS: dict[str, tuple[str, ...] | None] = {
    FULL_PROFILE: None,  # every available base model
    "front_clinical": ("front", CLINICAL_MODALITY),
    "clinical_only": (CLINICAL_MODALITY,),
}


@dataclass
class Profile:
    """One servable configuration."""

    name: str
    members: list[str]
    weights: np.ndarray
    operating_point: OperatingPoint
    calibration: dict
    cv_metrics: dict
    calibrator: Calibrator | None = None

    @property
    def ensemble(self) -> WeightedEnsemble:
        return WeightedEnsemble(
            columns=list(self.members),
            weights=self.weights,
            inner_cv_auc=None,
        )

    def as_dict(self, calibrator_artifact: str | None) -> dict:
        return {
            "name": self.name,
            "members": list(self.members),
            "weights": [float(w) for w in self.weights],
            "operating_point": self.operating_point.as_dict(),
            "calibration": {**self.calibration, "artifact": calibrator_artifact},
            "cv_metrics": self.cv_metrics,
        }


def select_members(available: list[str], modalities: tuple[str, ...] | None) -> list[str]:
    """Base models belonging to the given modalities, in a stable order."""
    if modalities is None:
        return sorted(available)
    return sorted(
        name for name in available if name.split("_", 1)[0] in set(modalities)
    )


def build_profile(
    name: str,
    members: list[str],
    config: Config,
    dataset: pd.DataFrame,
    base_predictions: dict[str, pd.DataFrame],
    ensemble_root: Path,
    threshold_policy: str = "youden",
    target_sensitivity: float = 0.90,
    calibration: str = "isotonic",
    resume: bool = True,
) -> Profile:
    """Run (or reuse) the weighted ensemble CV for ``members`` and derive its operating point."""
    run_dir = ensemble_root / f"weighted_{name}"
    subset = {member: base_predictions[member] for member in members}
    run_ensemble_cv(
        config=config,
        dataset=dataset,
        base_predictions=subset,
        method=WEIGHTED_METHOD,
        run_dir=run_dir,
        resume=resume,
    )

    predictions = load_predictions(run_dir / "predictions.csv")
    summary = load_json(run_dir / "summary.json")
    operating_point = choose_operating_point(
        predictions, policy=threshold_policy, target_sensitivity=target_sensitivity
    )
    calibrator, calibration_report = fit_calibrator(
        predictions, method=calibration, seed=config.cross_validation.seed
    )
    matrices = {
        rep: build_oof_matrix(subset, dataset, rep)
        for rep in range(1, config.cross_validation.num_reps + 1)
    }
    serving_ensemble = fit_repeated_oof_ensemble(
        matrices,
        dataset[LABEL_COLUMN].astype(int),
        seed=config.final.seed,
        n_trials=config.ensemble.n_trials,
    )
    pooled = summary.get("test_auc_pooled_per_rep", {})
    selection = summary.get("selection_auc_by_source", {}).get("inner_cv", {})

    return Profile(
        name=name,
        members=members,
        weights=serving_ensemble.weights,
        operating_point=operating_point,
        calibration=calibration_report,
        cv_metrics={
            "auc_mean": pooled.get("mean"),
            "auc_sd": pooled.get("sd"),
            "n_reps": pooled.get("n"),
            "selection_auc_mean": selection.get("mean"),
            "ensemble_cv_dir": str(run_dir),
            "serving_weight_source": (
                "one Optuna fit on all base-model OOF probabilities; objective is "
                "mean full-cohort AUC across repetitions"
            ),
            "serving_weight_seed": config.final.seed,
            "serving_weight_n_trials": config.ensemble.n_trials,
        },
        calibrator=calibrator,
    )


def build_profiles(
    requested: dict[str, tuple[str, ...] | None],
    config: Config,
    dataset: pd.DataFrame,
    base_predictions: dict[str, pd.DataFrame],
    ensemble_root: Path,
    threshold_policy: str = "youden",
    target_sensitivity: float = 0.90,
    calibration: str = "isotonic",
    resume: bool = True,
) -> list[Profile]:
    """Build every requested profile, skipping those with no available members."""
    available = list(base_predictions)
    profiles: list[Profile] = []
    for name, modalities in requested.items():
        members = select_members(available, modalities)
        if not members:
            print(f"Skipping profile '{name}': none of its models were cross-validated")
            continue
        print(f"Profile '{name}': {len(members)} model(s) -- {', '.join(members)}")
        profiles.append(
            build_profile(
                name=name,
                members=members,
                config=config,
                dataset=dataset,
                base_predictions=base_predictions,
                ensemble_root=ensemble_root,
                threshold_policy=threshold_policy,
                target_sensitivity=target_sensitivity,
                calibration=calibration,
                resume=resume,
            )
        )
    if not profiles:
        raise ValueError("No profile could be built from the available base models.")
    return profiles
