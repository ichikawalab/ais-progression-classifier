import numpy as np
import pandas as pd
import pytest

from ais_progression.ensemble.meta import fit_meta_ensemble, simple_average
from ais_progression.ensemble.weighted import (
    WeightedEnsemble,
    fit_repeated_oof_ensemble,
    fit_weighted_ensemble,
)
from ais_progression.evaluation import make_predictions_frame
from ais_progression.experiments.ensemble_cv import (
    assert_folds_match,
    ensemble_run_identity,
)
from ais_progression.experiments.splits import iter_folds


@pytest.fixture
def base_probabilities():
    """Three base models: one informative, two noisy."""
    rng = np.random.default_rng(0)
    n = 120
    y = pd.Series(rng.integers(0, 2, size=n))
    X = pd.DataFrame(
        {
            "front_vit": np.clip(y * 0.5 + rng.normal(0.25, 0.10, n), 0, 1),
            "lateral_vit": np.clip(y * 0.1 + rng.normal(0.45, 0.25, n), 0, 1),
            "clinical_logreg": np.clip(rng.normal(0.5, 0.25, n), 0, 1),
        }
    )
    return X, y


def test_weighted_ensemble_weights_are_a_simplex(base_probabilities):
    X, y = base_probabilities
    ensemble = fit_weighted_ensemble(X, y, seed=42, n_trials=15, inner_folds=3)
    assert ensemble.columns == list(X.columns)
    assert ensemble.weights.sum() == pytest.approx(1.0)
    assert (ensemble.weights >= 0).all()


def test_weighted_ensemble_favours_the_informative_model(base_probabilities):
    X, y = base_probabilities
    ensemble = fit_weighted_ensemble(X, y, seed=42, n_trials=40, inner_folds=3)
    assert ensemble.columns[int(np.argmax(ensemble.weights))] == "front_vit"


def test_weighted_ensemble_prediction_is_the_weighted_sum(base_probabilities):
    X, y = base_probabilities
    ensemble = fit_weighted_ensemble(X, y, seed=42, n_trials=10, inner_folds=3)
    expected = X.to_numpy() @ ensemble.weights
    assert np.allclose(ensemble.predict(X), expected)


def test_weighted_ensemble_rejects_a_missing_column(base_probabilities):
    X, y = base_probabilities
    ensemble = fit_weighted_ensemble(X, y, seed=42, n_trials=5, inner_folds=3)
    with pytest.raises(ValueError, match="missing model column"):
        ensemble.predict(X.drop(columns=["front_vit"]))


def test_final_weights_optimize_mean_full_cohort_auc_across_repetitions():
    y = pd.Series([0, 0, 0, 1, 1, 1] * 10)
    rng = np.random.default_rng(7)
    matrices = {
        rep: pd.DataFrame(
            {
                "reliable": np.clip(y * 0.8 + rng.normal(0.1, 0.02, len(y)), 0, 1),
                "noise": rng.random(len(y)),
            }
        )
        for rep in range(1, 4)
    }

    ensemble = fit_repeated_oof_ensemble(matrices, y, seed=42, n_trials=40)

    assert ensemble.columns == ["reliable", "noise"]
    assert ensemble.weights.sum() == pytest.approx(1.0)
    assert ensemble.weights[0] > ensemble.weights[1]
    assert ensemble.inner_cv_auc is None


def test_modality_weights_sum_per_modality():
    ensemble = WeightedEnsemble(
        columns=["front_vit", "front_swint", "lateral_vit", "clinical_rf"],
        weights=np.array([0.3, 0.2, 0.4, 0.1]),
        inner_cv_auc=0.8,
    )
    totals = ensemble.modality_weights()
    assert totals == pytest.approx({"front": 0.5, "lateral": 0.4, "clinical": 0.1})
    assert sum(totals.values()) == pytest.approx(1.0)


def test_simple_average_is_the_unweighted_mean(base_probabilities):
    X, _ = base_probabilities
    assert np.allclose(simple_average(X), X.mean(axis=1).to_numpy())


@pytest.mark.parametrize("model", ["logreg", "svm", "rf"])
def test_meta_ensembles_fit_and_score(model, base_probabilities):
    X, y = base_probabilities
    ensemble = fit_meta_ensemble(X, y, model, seed=42, n_trials=3, inner_folds=3)
    probabilities = ensemble.predict(X)
    assert probabilities.shape == (len(X),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert 0 <= ensemble.inner_cv_auc <= 1


def test_unknown_meta_model_is_rejected(base_probabilities):
    X, y = base_probabilities
    with pytest.raises(ValueError, match="Unknown meta model"):
        fit_meta_ensemble(X, y, "knn", seed=42, n_trials=2, inner_folds=3)


def test_ensemble_search_spaces_differ_from_the_clinical_ones():
    """The ensemble stage widens SVM gamma and narrows RF depth."""
    import optuna

    from ais_progression.ensemble.meta import ENSEMBLE_SEARCH_SPACES
    from ais_progression.models.clinical_model import CLINICAL_SEARCH_SPACES

    def distribution(spaces, model, name):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        for _ in range(60):
            trial = study.ask()
            try:
                spaces[model](trial, 42)
            except Exception:
                continue
            if name in trial.distributions:
                return trial.distributions[name]
        raise AssertionError(f"{name} never sampled for {model}")

    ensemble_gamma = distribution(ENSEMBLE_SEARCH_SPACES, "svm", "gamma")
    clinical_gamma = distribution(CLINICAL_SEARCH_SPACES, "svm", "gamma")
    assert (ensemble_gamma.low, ensemble_gamma.high) == pytest.approx((1e-3, 10.0))
    assert (clinical_gamma.low, clinical_gamma.high) == pytest.approx((1e-4, 1.0))

    ensemble_depth = distribution(ENSEMBLE_SEARCH_SPACES, "rf", "max_depth")
    clinical_depth = distribution(CLINICAL_SEARCH_SPACES, "rf", "max_depth")
    assert (ensemble_depth.low, ensemble_depth.high) == (3, 10)
    assert (clinical_depth.low, clinical_depth.high) == (3, 20)


def _fold_predictions(dataset, reps: int, folds: int, seed: int) -> pd.DataFrame:
    blocks = []
    for split in iter_folds(dataset, reps, folds, seed, with_validation=False):
        blocks.append(
            make_predictions_frame(
                split.test["patient_id"],
                split.test["label"],
                np.full(len(split.test), 0.5),
                split.rep,
                split.fold,
                "test",
            )
        )
    return pd.concat(blocks, ignore_index=True)


def test_fold_assignment_mismatch_is_rejected(small_config, synthetic_cohort):
    _, dataset = synthetic_cohort
    cv = small_config.cross_validation
    predictions = _fold_predictions(dataset, cv.num_reps, cv.num_folds, cv.seed)

    with pytest.raises(ValueError, match="different fold assignment"):
        assert_folds_match(
            {"clinical_logreg": predictions},
            dataset,
            cv.num_reps,
            cv.num_folds,
            cv.seed + 1,
        )


def test_ensemble_identity_tracks_inputs_and_search_settings(
    small_config, synthetic_cohort
):
    _, dataset = synthetic_cohort
    cv = small_config.cross_validation
    predictions = _fold_predictions(dataset, cv.num_reps, cv.num_folds, cv.seed)
    first = ensemble_run_identity(
        small_config, dataset, {"clinical_logreg": predictions}, "weighted"
    )

    changed_predictions = predictions.copy()
    changed_predictions.loc[0, "prob"] = 0.6
    changed_input = ensemble_run_identity(
        small_config,
        dataset,
        {"clinical_logreg": changed_predictions},
        "weighted",
    )
    assert changed_input != first

    small_config.ensemble.n_trials += 1
    changed_search = ensemble_run_identity(
        small_config, dataset, {"clinical_logreg": predictions}, "weighted"
    )
    assert changed_search != first
