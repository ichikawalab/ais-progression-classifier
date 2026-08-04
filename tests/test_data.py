from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ais_progression.config import AugmentConfig
from ais_progression.data.build import reroot_path
from ais_progression.data.images import RadiographDataset, build_transforms
from ais_progression.data.preprocess import apply_clahe, pad_to_square, preprocess_dataset
from ais_progression.data.schema import (
    DATASET_COLUMNS,
    describe_dataset,
    image_column,
    load_dataset,
)


def test_load_dataset_reads_the_unified_schema(synthetic_cohort):
    csv_path, frame = synthetic_cohort
    loaded = load_dataset(csv_path)
    assert list(loaded.columns) == DATASET_COLUMNS
    assert len(loaded) == len(frame)
    assert loaded["label"].isin([0, 1]).all()


def test_relative_image_paths_resolve_against_the_csv(synthetic_cohort, tmp_path):
    csv_path, frame = synthetic_cohort
    relative = frame.copy()
    for column in ("front_path", "lateral_path"):
        relative[column] = [str(pd.Series([p]).iloc[0]).split("\\")[-1] for p in relative[column]]
    # Put the CSV next to the images so the bare filenames resolve.
    nested = tmp_path / "front" / "relative.csv"
    relative["lateral_path"] = frame["lateral_path"]
    relative.to_csv(nested, index=False)
    loaded = load_dataset(nested)
    assert all(str(p).endswith(".png") for p in loaded["front_path"])


def test_duplicate_patient_ids_are_rejected(synthetic_cohort, tmp_path):
    _, frame = synthetic_cohort
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "dup.csv"
    duplicated.to_csv(path, index=False)
    with pytest.raises(ValueError, match="must be unique"):
        load_dataset(path)


def test_non_binary_labels_are_rejected(synthetic_cohort, tmp_path):
    _, frame = synthetic_cohort
    bad = frame.copy()
    bad.loc[0, "label"] = 2
    path = tmp_path / "bad_label.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="must contain only 0 or 1"):
        load_dataset(path)


def test_invalid_sex_coding_is_rejected(synthetic_cohort, tmp_path):
    _, frame = synthetic_cohort
    bad = frame.copy()
    bad.loc[0, "sex"] = 0
    path = tmp_path / "bad_sex.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="coded 1"):
        load_dataset(path)


def test_missing_image_file_is_reported(synthetic_cohort, tmp_path):
    _, frame = synthetic_cohort
    bad = frame.copy()
    bad.loc[0, "lateral_path"] = str(tmp_path / "nope.png")
    path = tmp_path / "missing.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(FileNotFoundError, match="lateral_path"):
        load_dataset(path)


def test_check_files_can_be_disabled(synthetic_cohort, tmp_path):
    _, frame = synthetic_cohort
    bad = frame.copy()
    bad.loc[0, "front_path"] = str(tmp_path / "nope.png")
    path = tmp_path / "unchecked.csv"
    bad.to_csv(path, index=False)
    assert len(load_dataset(path, check_files=False)) == len(frame)


def test_clinical_only_input_does_not_require_image_columns(synthetic_cohort, tmp_path):
    _, frame = synthetic_cohort
    clinical = frame[["patient_id", "age", "sex", "risser", "cobb_baseline", "label"]]
    path = tmp_path / "clinical.csv"
    clinical.to_csv(path, index=False)

    loaded = load_dataset(
        path,
        required_modalities=(),
        required_clinical_features=("age", "sex", "risser", "cobb_baseline"),
    )
    assert len(loaded) == len(frame)
    assert "front_path" not in loaded


def test_front_only_input_does_not_require_lateral_or_clinical_columns(
    synthetic_cohort, tmp_path
):
    _, frame = synthetic_cohort
    front = frame[["patient_id", "front_path", "label"]]
    path = tmp_path / "front.csv"
    front.to_csv(path, index=False)

    loaded = load_dataset(
        path, required_modalities=("front",), required_clinical_features=()
    )
    assert len(loaded) == len(frame)
    assert "lateral_path" not in loaded


def test_describe_dataset_counts_the_cohort(synthetic_cohort):
    _, frame = synthetic_cohort
    summary = describe_dataset(frame)
    assert summary["n_patients"] == len(frame)
    assert summary["n_progression"] + summary["n_non_progression"] == len(frame)
    assert summary["n_female"] + summary["n_male"] == len(frame)


def test_image_column_lookup():
    assert image_column("front") == "front_path"
    assert image_column("lateral") == "lateral_path"
    with pytest.raises(ValueError, match="Unknown image modality"):
        image_column("oblique")


def test_pad_to_square_centres_and_preserves_content():
    image = np.full((10, 4), 200, dtype=np.uint8)
    padded = pad_to_square(image)
    assert padded.shape == (10, 10)
    assert (padded[:, 3:7] == 200).all()
    assert (padded[:, :3] == 0).all()


def test_clahe_preserves_shape_and_dtype():
    image = np.random.default_rng(0).integers(0, 255, size=(32, 24), dtype=np.uint8)
    enhanced = apply_clahe(image)
    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8


def test_preprocess_dataset_writes_square_images(synthetic_cohort, tmp_path):
    from PIL import Image

    _, frame = synthetic_cohort
    processed = preprocess_dataset(frame.head(3), tmp_path / "processed")
    for column in ("front_path", "lateral_path"):
        for path in processed[column]:
            with Image.open(path) as image:
                assert image.width == image.height


def test_reroot_path_matches_the_longest_existing_suffix(tmp_path):
    target = tmp_path / "Input_Original_n471" / "Toyama" / "toyama1_Front.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    recorded = r"D:\elsewhere\ROI1_Original\Input_Original_n471\Toyama\toyama1_Front.png"
    assert reroot_path(recorded, tmp_path) == str(target.resolve())
    assert reroot_path(r"D:\elsewhere\missing.png", tmp_path) is None


def test_train_transform_augments_and_eval_transform_does_not(synthetic_cohort):
    from PIL import Image

    _, frame = synthetic_cohort
    timm_config = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5), "input_size": (3, 32, 32)}
    augment = AugmentConfig()
    with Image.open(frame.loc[0, "front_path"]) as source:
        image = source.convert("RGB")

    eval_transform = build_transforms(timm_config, augment, is_training=False)
    assert np.allclose(eval_transform(image).numpy(), eval_transform(image).numpy())

    train_transform = build_transforms(timm_config, augment, is_training=True)
    draws = [train_transform(image).numpy() for _ in range(8)]
    assert any(not np.allclose(draws[0], other) for other in draws[1:])
    assert draws[0].shape == (3, 32, 32)


def test_dataset_returns_labels_or_patient_ids(synthetic_cohort):
    _, frame = synthetic_cohort
    timm_config = {"mean": (0.5,) * 3, "std": (0.5,) * 3, "input_size": (3, 16, 16)}
    transform = build_transforms(timm_config, AugmentConfig(), is_training=False)

    labelled = RadiographDataset(frame, "front", transform, has_labels=True)
    image, label = labelled[0]
    assert image.shape == (3, 16, 16)
    assert int(label) in (0, 1)

    unlabelled = RadiographDataset(frame, "lateral", transform, has_labels=False)
    _, patient_id = unlabelled[0]
    assert patient_id == frame.loc[0, "patient_id"]


def test_reroot_path_cannot_escape_the_image_root(tmp_path):
    """A recorded absolute path must not resolve outside image_root.

    Path.joinpath discards everything to the left of an absolute component, so
    an unfiltered drive letter or leading slash would silently match the
    original acquisition-machine location.
    """
    outside = tmp_path / "outside" / "secret.png"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"x")
    image_root = tmp_path / "root"
    (image_root / "Toyama").mkdir(parents=True)

    assert reroot_path(str(outside), image_root) is None
    assert reroot_path(r"..\outside\secret.png", image_root) is None


def test_reroot_path_prefers_the_longest_match(tmp_path):
    image_root = tmp_path / "root"
    deep = image_root / "Input_Original_n471" / "Toyama" / "a.png"
    deep.parent.mkdir(parents=True)
    deep.write_bytes(b"x")
    shallow = image_root / "a.png"
    shallow.write_bytes(b"y")

    recorded = r"D:\acq\Input_Original_n471\Toyama\a.png"
    assert reroot_path(recorded, image_root) == str(deep.resolve())


def test_duplicate_ids_in_a_path_workbook_are_named(tmp_path):
    import pandas as pd

    from ais_progression.data.build import _path_lookup

    workbook = tmp_path / "paths.xlsx"
    pd.DataFrame(
        {"ID": ["a", "a", "b"], "Front_paths": ["x.png", "y.png", "z.png"]}
    ).to_excel(workbook, index=False)
    with pytest.raises(ValueError, match="duplicate patient IDs"):
        _path_lookup(workbook, "Front_paths", tmp_path)


def test_only_the_training_loader_persists_workers(synthetic_cohort):
    from ais_progression.config import DataConfig
    from ais_progression.data.images import build_loader

    _, frame = synthetic_cohort
    timm_config = {"mean": (0.5,) * 3, "std": (0.5,) * 3, "input_size": (3, 16, 16)}
    transform = build_transforms(timm_config, AugmentConfig(), is_training=False)
    data_cfg = DataConfig(num_workers=2, batch_size=4)

    train = build_loader(frame, "front", transform, data_cfg, shuffle=True)
    evaluation = build_loader(frame, "front", transform, data_cfg, shuffle=False)
    assert train.persistent_workers is True
    # Eval loaders are rebuilt every fold; persisting them would leak a pool per fold.
    assert evaluation.persistent_workers is False


def test_relativize_paths_round_trips_through_load_dataset(synthetic_cohort, tmp_path):
    """A relative CSV must resolve back to the same files, and survive a move."""
    from ais_progression.data.schema import relativize_paths

    _, frame = synthetic_cohort
    csv_path = tmp_path / "nested" / "dataset.csv"
    csv_path.parent.mkdir(parents=True)
    relative = relativize_paths(frame, csv_path)

    assert not any(Path(p).is_absolute() for p in relative["front_path"])
    # Forward slashes only, so a CSV written on Windows still resolves elsewhere.
    assert not any("\\" in p for p in relative["front_path"])

    relative.to_csv(csv_path, index=False)
    loaded = load_dataset(csv_path)
    assert [Path(p).resolve() for p in loaded["front_path"]] == [
        Path(p).resolve() for p in frame["front_path"]
    ]
    assert [Path(p).resolve() for p in loaded["lateral_path"]] == [
        Path(p).resolve() for p in frame["lateral_path"]
    ]


def test_relativize_paths_keeps_absolute_when_no_relative_exists(synthetic_cohort):
    from ais_progression.data.schema import relativize_paths

    _, frame = synthetic_cohort
    # A CSV on another drive has no relative path to the images on Windows;
    # elsewhere relpath succeeds and simply produces a "../.." chain.
    result = relativize_paths(frame.head(1), Path("Z:/elsewhere/dataset.csv"))
    assert Path(result.loc[0, "front_path"]).name.endswith(".png")


def test_missing_clinical_values_are_rejected_by_default(synthetic_cohort, tmp_path):
    """Silently imputing a blank Cobb angle would return a confident guess."""
    _, frame = synthetic_cohort
    gapped = frame.copy()
    gapped.loc[0, "cobb_baseline"] = None
    path = tmp_path / "gapped.csv"
    gapped.to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing clinical variables"):
        load_dataset(path)

    allowed = load_dataset(path, allow_missing_features=True)
    assert allowed["cobb_baseline"].isna().sum() == 1


def test_missing_feature_report_names_the_gaps(synthetic_cohort):
    from ais_progression.data.schema import missing_feature_report

    _, frame = synthetic_cohort
    gapped = frame.copy()
    gapped.loc[0, "age"] = None
    gapped.loc[0, "risser"] = None
    report = missing_feature_report(gapped)
    assert report.iloc[0] == "age;risser"
    assert (report.iloc[1:] == "").all()


@pytest.mark.parametrize(
    "column,value",
    [("age", 200.0), ("cobb_baseline", -5.0), ("risser", 9), ("age", 0.5)],
)
def test_out_of_range_clinical_values_are_rejected(column, value, synthetic_cohort, tmp_path):
    _, frame = synthetic_cohort
    bad = frame.copy()
    bad.loc[0, column] = value
    path = tmp_path / "bad_range.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match=column):
        load_dataset(path)


def test_non_integer_risser_is_rejected(synthetic_cohort, tmp_path):
    """A fractional Risser sign would hit the ordinal encoder's unknown fallback."""
    _, frame = synthetic_cohort
    bad = frame.copy()
    bad["risser"] = bad["risser"].astype(float)
    bad.loc[0, "risser"] = 2.5
    path = tmp_path / "fractional.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="whole number"):
        load_dataset(path)


def test_range_validation_can_be_disabled(synthetic_cohort, tmp_path):
    _, frame = synthetic_cohort
    bad = frame.copy()
    bad.loc[0, "cobb_baseline"] = 999.0
    path = tmp_path / "unchecked_range.csv"
    bad.to_csv(path, index=False)
    assert len(load_dataset(path, validate_ranges=False)) == len(frame)
