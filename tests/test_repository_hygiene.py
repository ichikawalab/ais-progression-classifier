from pathlib import Path
import subprocess

import pytest


FORBIDDEN_ROOTS = {
    "data",
    "outputs",
    "models",
    "weights",
    "checkpoints",
    "artifacts",
    "runs",
    "logs",
}

FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".pth",
    ".pt",
    ".safetensors",
    ".onnx",
    ".dcm",
    ".dicom",
    ".nii",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".tsv",
    ".xls",
    ".xlsx",
    ".parquet",
}


def test_repository_tracks_no_patient_data_or_model_artifacts():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("Repository hygiene check requires a Git checkout.")
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    violations = []
    for item in tracked:
        path = Path(item)
        if path.parts and path.parts[0].lower() in FORBIDDEN_ROOTS:
            violations.append(item)
            continue
        lower_name = path.name.lower()
        if lower_name.endswith(".nii.gz") or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(item)
            continue
        if path.suffix.lower() == ".csv" and path.parts[:1] != ("examples",):
            violations.append(item)

    assert not violations, (
        "Patient data, medical images, cohort tables, or model artifacts are tracked: "
        f"{violations}"
    )
