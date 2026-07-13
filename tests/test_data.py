import pandas as pd
import pytest
from PIL import Image

from ais_progression.data import load_and_validate_train_csv


def test_training_csv_contract_and_relative_path(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("L", (8, 8), color=128).save(image_path)
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "sample_id": ["sample-1"],
            "patient_id": ["patient-1"],
            "image_path": ["image.png"],
            "label": [1],
        }
    ).to_csv(csv_path, index=False)
    frame = load_and_validate_train_csv(csv_path)
    assert frame.loc[0, "image_path"] == str(image_path)
    assert len(frame.loc[0, "image_sha256"]) == 64


def test_training_csv_requires_patient_id(tmp_path):
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"sample_id": ["s"], "image_path": ["missing.png"], "label": [0]}).to_csv(
        csv_path, index=False
    )
    with pytest.raises(ValueError, match="patient_id"):
        load_and_validate_train_csv(csv_path)


def test_training_csv_rejects_fractional_labels(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("L", (8, 8), color=128).save(image_path)
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "sample_id": ["sample-1"],
            "patient_id": ["patient-1"],
            "image_path": ["image.png"],
            "label": [0.5],
        }
    ).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="only 0 or 1"):
        load_and_validate_train_csv(csv_path)


def test_training_csv_rejects_unreadable_images(tmp_path):
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"not an image")
    csv_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "sample_id": ["sample-1"],
            "patient_id": ["patient-1"],
            "image_path": ["broken.png"],
            "label": [0],
        }
    ).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Unreadable image"):
        load_and_validate_train_csv(csv_path)
