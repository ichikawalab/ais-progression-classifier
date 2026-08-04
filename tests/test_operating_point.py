import numpy as np
import pandas as pd
import pytest

import ais_progression.final.operating_point as operating_point
from ais_progression.evaluation import make_predictions_frame


@pytest.mark.parametrize("n_positives", [10, 20, 30, 47])
@pytest.mark.parametrize("target", [0.8, 0.9, 0.95, 1.0])
def test_target_sensitivity_uses_the_highest_qualifying_threshold(
    n_positives, target
):
    probabilities = np.linspace(0.01, 0.99, n_positives)
    predictions = make_predictions_frame(
        [f"positive-{i}" for i in range(n_positives)] + ["negative"],
        np.r_[np.ones(n_positives, dtype=int), 0],
        np.r_[probabilities, 0.0],
        1,
        1,
        "test",
    )
    point = operating_point.choose_operating_point(
        predictions, policy="target_sensitivity", target_sensitivity=target
    )
    threshold = point.threshold
    achieved = float(np.mean(probabilities >= threshold))

    assert achieved >= target
    assert achieved - target < 1 / n_positives
    higher = probabilities[probabilities > threshold]
    if higher.size:
        assert float(np.mean(probabilities >= higher[0])) < target


def test_target_sensitivity_is_enforced_on_the_mean_across_repetitions():
    blocks = []
    for rep, positive_probabilities in ((1, [0.9, 0.8]), (2, [0.7, 0.1])):
        blocks.append(
            make_predictions_frame(
                [f"p{i}" for i in range(3)],
                [1, 1, 0],
                [*positive_probabilities, 0.0],
                rep,
                1,
                "test",
            )
        )
    point = operating_point.choose_operating_point(
        pd.concat(blocks), policy="target_sensitivity", target_sensitivity=0.75
    )
    assert point.threshold == pytest.approx(0.7)
    assert point.sensitivity_mean == pytest.approx(0.75)


@pytest.mark.parametrize("target", [0, -0.1, 1.1])
def test_target_sensitivity_rejects_an_invalid_target(target):
    with pytest.raises(ValueError, match="must be in"):
        operating_point.choose_operating_point(
            _repeated_predictions(),
            policy="target_sensitivity",
            target_sensitivity=target,
        )


def _repeated_predictions(n_patients=20, n_reps=2):
    labels = np.arange(n_patients) % 2
    blocks = []
    for rep in range(1, n_reps + 1):
        probabilities = np.clip(0.2 + 0.6 * labels + rep * 0.01, 0, 1)
        blocks.append(
            make_predictions_frame(
                [f"p{i:03d}" for i in range(n_patients)],
                labels,
                probabilities,
                rep,
                1,
                "test",
            )
        )
    return pd.concat(blocks, ignore_index=True)


def test_calibration_cv_never_splits_one_patient_across_train_and_test(monkeypatch):
    real_splitter = operating_point.StratifiedGroupKFold
    checked = []

    class CheckingSplitter:
        def __init__(self, *args, **kwargs):
            self.inner = real_splitter(*args, **kwargs)

        def split(self, X, y, groups):
            groups = np.asarray(groups)
            for train_idx, test_idx in self.inner.split(X, y, groups):
                assert set(groups[train_idx]).isdisjoint(groups[test_idx])
                checked.append(True)
                yield train_idx, test_idx

    monkeypatch.setattr(operating_point, "StratifiedGroupKFold", CheckingSplitter)
    predictions = _repeated_predictions()
    _, report = operating_point.fit_calibrator(
        predictions, method="platt", n_folds=2
    )

    assert len(checked) == 2
    assert report["n_patients"] == 20
    assert report["n_observations"] == 40
