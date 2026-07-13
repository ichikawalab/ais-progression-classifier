"""Patient-level stratified fold assignment and leakage checks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_stratified_group_folds(
    df: pd.DataFrame, num_folds: int, seed: int
) -> pd.DataFrame:
    """Return a copy with a deterministic patient-level `fold` column."""
    if "image_sha256" in df.columns:
        digest_patients = df.groupby("image_sha256")["patient_id"].nunique()
        conflicting_digests = digest_patients[digest_patients > 1].index.tolist()
        if conflicting_digests:
            raise ValueError(
                "Identical image content is assigned to different patients; digests include: "
                f"{conflicting_digests[:3]}"
            )
    patient_labels = df.groupby("patient_id", sort=False)["label"].nunique()
    inconsistent = patient_labels[patient_labels != 1].index.tolist()
    if inconsistent:
        raise ValueError(
            "Each patient must have one label; inconsistent patient_id values include: "
            f"{inconsistent[:10]}"
        )

    counts = df.drop_duplicates("patient_id")["label"].value_counts()
    missing_classes = {0, 1} - set(counts.index)
    if missing_classes:
        raise ValueError(f"Dataset is missing class(es): {sorted(missing_classes)}")
    if int(counts.min()) < num_folds:
        raise ValueError(
            f"The minority class has {int(counts.min())} patients, but {num_folds} folds were requested."
        )

    patients = (
        df[["patient_id", "label"]]
        .drop_duplicates("patient_id")
        .reset_index(drop=True)
    )
    splitter = StratifiedKFold(
        n_splits=num_folds, shuffle=True, random_state=seed
    )
    patient_to_fold: dict[str, int] = {}
    for fold, (_, test_idx) in enumerate(splitter.split(patients, patients["label"])):
        for patient_id in patients.iloc[test_idx]["patient_id"]:
            patient_to_fold[patient_id] = fold

    result = df.copy()
    result["fold"] = result["patient_id"].map(patient_to_fold)
    if result["fold"].isna().any():
        raise RuntimeError("Failed to map every patient to a fold.")
    result["fold"] = result["fold"].astype(int)
    if (result["fold"] < 0).any():
        raise RuntimeError("Failed to assign every sample to a fold.")
    return result


def split_for_outer_fold(
    folded_df: pd.DataFrame, outer_fold: int, num_folds: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use one fold for test, the next for validation, and all others for train."""
    if outer_fold not in range(num_folds):
        raise ValueError(f"outer_fold must be in [0, {num_folds - 1}]")
    test_fold = outer_fold
    val_fold = (outer_fold + 1) % num_folds
    test_df = folded_df[folded_df["fold"] == test_fold].copy()
    val_df = folded_df[folded_df["fold"] == val_fold].copy()
    train_df = folded_df[~folded_df["fold"].isin([test_fold, val_fold])].copy()
    assert_no_leakage(train_df, val_df, test_df)
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def assert_no_leakage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Fail if patient IDs or known content digests overlap across subsets."""
    subsets = {"train": train_df, "val": val_df, "test": test_df}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        patient_overlap = set(subsets[left]["patient_id"]) & set(subsets[right]["patient_id"])
        if patient_overlap:
            raise RuntimeError(
                f"Patient leakage between {left} and {right}: {sorted(patient_overlap)[:10]}"
            )
        if "image_sha256" in train_df.columns:
            digest_overlap = set(subsets[left]["image_sha256"]) & set(
                subsets[right]["image_sha256"]
            )
            if digest_overlap:
                raise RuntimeError(
                    f"Duplicate image content between {left} and {right}: "
                    f"{sorted(digest_overlap)[:3]}"
                )


def build_split_manifest(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    frames = []
    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        frames.append(frame.assign(split=name))
    return pd.concat(frames, ignore_index=True)
