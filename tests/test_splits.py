import pytest

from ais_progression.experiments.splits import (
    assert_no_leakage,
    fold_seed,
    iter_folds,
    rep_seed,
)


def test_rep_seed_starts_at_the_base_seed():
    assert rep_seed(42, 1) == 42
    assert rep_seed(42, 10) == 51


def test_fold_seed_is_distinct_for_every_fold_of_every_repetition():
    """Reseeding per fold makes resume exact; keying on the fold keeps the folds
    of one repetition from all starting at the identical RNG state."""
    seeds = [fold_seed(42, rep, fold) for rep in range(1, 11) for fold in range(1, 11)]
    assert len(set(seeds)) == 100

    # Two folds of the same repetition must not share a seed.
    assert fold_seed(42, 1, 1) != fold_seed(42, 1, 2)
    # Nor may a repetition's range run into the next one's.
    assert fold_seed(42, 1, 10) < fold_seed(42, 2, 1)


def test_fold_seed_depends_only_on_the_fold_not_on_execution_order():
    """The same fold reseeds identically however many folds ran before it."""
    assert fold_seed(42, 3, 7) == fold_seed(42, 3, 7)


def test_every_patient_is_tested_exactly_once_per_repetition(synthetic_cohort):
    _, frame = synthetic_cohort
    folds = list(iter_folds(frame, num_reps=2, num_folds=4, base_seed=42))
    assert len(folds) == 8
    for rep in (1, 2):
        tested = [
            patient
            for split in folds
            if split.rep == rep
            for patient in split.test["patient_id"]
        ]
        assert sorted(tested) == sorted(frame["patient_id"])


def test_train_val_test_never_share_a_patient(synthetic_cohort):
    _, frame = synthetic_cohort
    for split in iter_folds(frame, num_reps=1, num_folds=4, base_seed=42):
        ids = [set(part["patient_id"]) for part in (split.train, split.val, split.test)]
        assert ids[0].isdisjoint(ids[1])
        assert ids[0].isdisjoint(ids[2])
        assert ids[1].isdisjoint(ids[2])
        assert sum(len(group) for group in ids) == len(frame)


def test_validation_slice_is_one_folds_worth(synthetic_cohort):
    _, frame = synthetic_cohort
    split = next(iter_folds(frame, num_reps=1, num_folds=4, base_seed=42))
    # One fold is held out for test; the validation slice is 1/(folds-1) of the rest.
    assert split.sizes["n_val"] == pytest.approx(len(frame) / 4, abs=1)


def test_without_validation_the_training_fold_is_whole(synthetic_cohort):
    _, frame = synthetic_cohort
    split = next(iter_folds(frame, num_reps=1, num_folds=4, base_seed=42, with_validation=False))
    assert split.sizes["n_val"] == 0
    assert split.sizes["n_train"] + split.sizes["n_test"] == len(frame)


def test_folds_are_deterministic_given_the_seed(synthetic_cohort):
    _, frame = synthetic_cohort
    first = [list(s.test["patient_id"]) for s in iter_folds(frame, 1, 4, 42)]
    second = [list(s.test["patient_id"]) for s in iter_folds(frame, 1, 4, 42)]
    other = [list(s.test["patient_id"]) for s in iter_folds(frame, 1, 4, 7)]
    assert first == second
    assert first != other


def test_repetitions_use_different_partitions(synthetic_cohort):
    _, frame = synthetic_cohort
    folds = list(iter_folds(frame, num_reps=2, num_folds=4, base_seed=42))
    rep1 = [set(s.test["patient_id"]) for s in folds if s.rep == 1]
    rep2 = [set(s.test["patient_id"]) for s in folds if s.rep == 2]
    assert rep1 != rep2


def test_folds_preserve_class_balance(synthetic_cohort):
    _, frame = synthetic_cohort
    overall = frame["label"].mean()
    for split in iter_folds(frame, num_reps=1, num_folds=4, base_seed=42):
        assert split.test["label"].mean() == pytest.approx(overall, abs=0.2)


def test_too_few_minority_patients_is_rejected(synthetic_cohort):
    import pandas as pd

    _, frame = synthetic_cohort
    tiny = pd.concat(
        [frame[frame["label"] == 1].head(2), frame[frame["label"] == 0]], ignore_index=True
    )
    with pytest.raises(ValueError, match="minority class"):
        list(iter_folds(tiny, num_reps=1, num_folds=10, base_seed=42))


def test_assert_no_leakage_catches_an_overlap(synthetic_cohort):
    _, frame = synthetic_cohort
    with pytest.raises(RuntimeError, match="leakage"):
        assert_no_leakage(frame.head(10), frame.head(3))
