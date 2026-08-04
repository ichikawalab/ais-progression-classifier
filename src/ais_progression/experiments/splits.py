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
    """One outer fold: train / validation / test frames plus their indices."""

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
    early stopping, leaving eight folds' worth of training data at the paper's
    ten-fold setting. Models that tune with an inner cross-validation instead
    (the clinical and ensemble learners) pass ``with_validation=False`` and get
    an empty validation frame.
    """
    check_splittable(df, num_folds)
    df = df.reset_index(drop=True)
    labels = df[LABEL_COLUMN].astype(int)
    val_fraction = 1 / (num_folds - 1)

    for rep in range(1, num_reps + 1):
        seed = rep_seed(base_seed, rep)
        splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        for fold, (train_val_idx, test_idx) in enumerate(splitter.split(df, labels), start=1):
            if with_validation:
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_fraction,
                    stratify=labels.iloc[train_val_idx],
                    random_state=seed,
                )
            else:
                train_idx, val_idx = train_val_idx, []

            train = df.iloc[train_idx].reset_index(drop=True)
            val = df.iloc[val_idx].reset_index(drop=True)
            test = df.iloc[test_idx].reset_index(drop=True)
            assert_no_leakage(train, val, test)
            yield Fold(rep=rep, fold=fold, seed=seed, train=train, val=val, test=test)
