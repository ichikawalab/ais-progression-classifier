"""Repeated stratified K-fold splitting, with leakage checks.

The dataset holds one row per patient, so stratifying on the label already
splits at the patient level. ``assert_no_leakage`` re-checks that invariant
because it is the assumption the whole evaluation rests on.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ais_progression.data.schema import ID_COLUMN, LABEL_COLUMN


@dataclass
class Fold:
    """One outer fold: train / validation / test frames plus their indices.

    ``seed`` is this fold's own seed, for whatever a model fitted on it needs to
    draw -- an Optuna sampler, an inner K-fold. It is deliberately *not* the seed
    that produced the split: that one is keyed on the repetition alone, because
    every fold of a repetition has to come from the same partition of the cohort.
    """

    rep: int
    fold: int
    seed: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict[str, int]:
        return {"n_train": len(self.train), "n_val": len(self.val), "n_test": len(self.test)}


def rep_seed(base_seed: int, rep: int) -> int:
    """Seed for a repetition. ``rep`` is 1-based, so rep 1 reuses the base seed."""
    return base_seed + rep - 1


def fold_seed(base_seed: int, rep: int, fold: int) -> int:
    """Seed for one fold, independent of both resume history and its sibling folds.

    Weight initialisation and augmentation draws are reseeded before every fold
    so a fold's result does not depend on which folds ran before it in the same
    process -- the precondition for resuming a run fold by fold. Keying only on
    ``rep`` met that requirement but left every fold of a repetition starting
    from the identical RNG state, correlating them; keying on both ``rep`` and
    ``fold`` removes that correlation as well. ``fold`` is 1-based and bounded
    by ``num_folds``, so 1000 leaves no realistic collision with the next rep's
    seed range.
    """
    return rep_seed(base_seed, rep) * 1000 + fold


def assert_no_leakage(*frames: pd.DataFrame) -> None:
    """Fail if any patient appears in more than one subset."""
    named = list(enumerate(frames))
    for left_index, left in named:
        for right_index, right in named[left_index + 1 :]:
            overlap = set(left[ID_COLUMN]) & set(right[ID_COLUMN])
            if overlap:
                raise RuntimeError(
                    f"Patient leakage between subset {left_index} and {right_index}: "
                    f"{sorted(overlap)[:10]}"
                )


def check_splittable(df: pd.DataFrame, num_folds: int) -> None:
    counts = df[LABEL_COLUMN].value_counts()
    missing = {0, 1} - set(counts.index)
    if missing:
        raise ValueError(f"Dataset is missing class(es): {sorted(missing)}")
    if int(counts.min()) < num_folds:
        raise ValueError(
            f"The minority class has {int(counts.min())} patients, "
            f"but {num_folds} folds were requested."
        )


def iter_folds(
    df: pd.DataFrame,
    num_reps: int,
    num_folds: int,
    base_seed: int,
    with_validation: bool = True,
):
    """Yield every (repetition, fold) split of the repeated stratified K-fold.

    One fold is held out for test. When ``with_validation`` is set, a stratified
    slice of size ``1/(num_folds-1)`` is carved out of the remaining folds for
    early stopping, leaving eight folds' worth of training data at the
    ten-fold setting. Models that tune with an inner cross-validation instead
    (the clinical and ensemble learners) pass ``with_validation=False`` and get
    an empty validation frame.
    """
    check_splittable(df, num_folds)
    df = df.reset_index(drop=True)
    labels = df[LABEL_COLUMN].astype(int)
    val_fraction = 1 / (num_folds - 1)

    for rep in range(1, num_reps + 1):
        # Keyed on the repetition: all of its folds must partition the same
        # cohort the same way, so this seed cannot vary within a repetition.
        split_seed = rep_seed(base_seed, rep)
        splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=split_seed)
        for fold, (train_val_idx, test_idx) in enumerate(splitter.split(df, labels), start=1):
            if with_validation:
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_fraction,
                    stratify=labels.iloc[train_val_idx],
                    random_state=split_seed,
                )
            else:
                train_idx, val_idx = train_val_idx, []

            train = df.iloc[train_idx].reset_index(drop=True)
            val = df.iloc[val_idx].reset_index(drop=True)
            test = df.iloc[test_idx].reset_index(drop=True)
            assert_no_leakage(train, val, test)
            yield Fold(
                rep=rep,
                fold=fold,
                seed=fold_seed(base_seed, rep, fold),
                train=train,
                val=val,
                test=test,
            )
