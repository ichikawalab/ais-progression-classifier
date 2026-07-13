import pandas as pd
import pytest

from ais_progression.splitting import assign_stratified_group_folds, split_for_outer_fold


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"sample_{i:02d}" for i in range(40)],
            "patient_id": [f"patient_{i:02d}" for i in range(40)],
            "image_path": [f"image_{i:02d}.png" for i in range(40)],
            "image_sha256": [f"digest_{i:02d}" for i in range(40)],
            "label": [i % 2 for i in range(40)],
        }
    )


def test_fold_assignment_is_reproducible():
    frame = make_frame()
    first = assign_stratified_group_folds(frame, num_folds=10, seed=42)
    second = assign_stratified_group_folds(frame, num_folds=10, seed=42)
    assert first["fold"].tolist() == second["fold"].tolist()


def test_each_patient_has_8_train_1_val_1_test_uses():
    folded = assign_stratified_group_folds(make_frame(), num_folds=10, seed=42)
    uses = {patient: {"train": 0, "val": 0, "test": 0} for patient in folded["patient_id"]}
    for fold in range(10):
        train, val, test = split_for_outer_fold(folded, fold, 10)
        for split_name, subset in (("train", train), ("val", val), ("test", test)):
            for patient in subset["patient_id"]:
                uses[patient][split_name] += 1
    assert all(counts == {"train": 8, "val": 1, "test": 1} for counts in uses.values())


def test_patient_sets_are_disjoint():
    folded = assign_stratified_group_folds(make_frame(), num_folds=10, seed=42)
    train, val, test = split_for_outer_fold(folded, 0, 10)
    assert set(train.patient_id).isdisjoint(val.patient_id)
    assert set(train.patient_id).isdisjoint(test.patient_id)
    assert set(val.patient_id).isdisjoint(test.patient_id)


def test_identical_content_with_different_patient_ids_is_rejected():
    frame = make_frame()
    frame.loc[1, "image_sha256"] = frame.loc[0, "image_sha256"]
    with pytest.raises(ValueError, match="Identical image content"):
        assign_stratified_group_folds(frame, num_folds=10, seed=42)


def test_stratification_is_patient_weighted_with_multiple_images():
    frame = make_frame()
    repeated = pd.concat([frame, frame.iloc[[0]].copy().loc[lambda x: x.index.repeat(20)]])
    repeated = repeated.reset_index(drop=True)
    repeated["sample_id"] = [f"sample_{i:03d}" for i in range(len(repeated))]
    repeated["image_sha256"] = [f"digest_{i:03d}" for i in range(len(repeated))]
    folded = assign_stratified_group_folds(repeated, num_folds=10, seed=42)
    patient_folds = folded.drop_duplicates("patient_id")
    counts = patient_folds.groupby(["fold", "label"]).size().unstack(fill_value=0)
    assert (counts[0] == counts[1]).all()
