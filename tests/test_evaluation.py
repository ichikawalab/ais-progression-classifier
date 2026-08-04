import numpy as np
import pandas as pd
import pytest

from ais_progression.evaluation import (
    PREDICTION_COLUMNS,
    auc_by_rep,
    binary_metrics,
    format_auc,
    make_predictions_frame,
    safe_auc,
    youden_threshold,
)


def test_safe_auc_is_one_for_a_perfect_ranking():
    assert safe_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_safe_auc_returns_none_for_a_single_class():
    assert safe_auc([1, 1, 1], [0.2, 0.5, 0.9]) is None


def test_youden_threshold_separates_a_clean_split():
    threshold = youden_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.2 < threshold <= 0.8


def test_binary_metrics_defaults_to_the_youden_threshold():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["n"] == 4


def test_binary_metrics_honours_an_explicit_threshold():
    metrics = binary_metrics([0, 1], [0.4, 0.6], threshold=0.9)
    assert metrics["threshold"] == pytest.approx(0.9)
    assert metrics["sensitivity"] == pytest.approx(0.0)


def test_make_predictions_frame_has_the_shared_columns():
    frame = make_predictions_frame(["a", "b"], [0, 1], [0.2, 0.7], rep=1, fold=3, split="test")
    assert list(frame.columns) == PREDICTION_COLUMNS
    assert frame["rep"].unique().tolist() == [1]
    assert frame["fold"].unique().tolist() == [3]


def test_make_predictions_frame_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="split must be"):
        make_predictions_frame(["a"], [0], [0.1], rep=1, fold=1, split="train")


def _fake_run(n_reps=3, n_folds=4, n=40, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array([i % 2 for i in range(n)])
    blocks = []
    for rep in range(1, n_reps + 1):
        order = rng.permutation(n)
        for fold, chunk in enumerate(np.array_split(order, n_folds), start=1):
            probability = np.clip(labels[chunk] * 0.4 + rng.normal(0.3, 0.1, size=len(chunk)), 0, 1)
            blocks.append(
                make_predictions_frame(
                    [f"p{i:03d}" for i in chunk], labels[chunk], probability, rep, fold, "test"
                )
            )
            blocks.append(
                make_predictions_frame(
                    [f"p{i:03d}" for i in chunk], labels[chunk], probability, rep, fold, "val"
                )
            )
    return pd.concat(blocks, ignore_index=True)


def test_auc_by_rep_pools_the_test_folds():
    predictions = _fake_run()
    per_rep = auc_by_rep(predictions, "test")
    assert len(per_rep) == 3
    # Every patient appears exactly once per repetition, so the pooled ROC covers the cohort.
    assert set(per_rep["n"]) == {40}


def test_auc_by_rep_averages_validation_folds():
    per_rep = auc_by_rep(_fake_run(), "val")
    assert len(per_rep) == 3
    assert per_rep["auc"].between(0, 1).all()


def test_format_auc_renders_undefined_values():
    assert format_auc(0.8194) == "0.819"
    assert format_auc(None) == "n/a"
    assert format_auc(float("nan")) == "n/a"


def test_binary_metrics_reports_clinical_names_only():
    """sensitivity is recall and ppv is precision; only one spelling is emitted."""
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert "sensitivity" in metrics and "ppv" in metrics
    assert "recall" not in metrics and "precision" not in metrics
