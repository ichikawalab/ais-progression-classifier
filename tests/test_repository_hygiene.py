"""Guardrails: no patient data in the repository, and every CLI is wired up."""
from __future__ import annotations

import importlib
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".dcm", ".nii", ".xlsx", ".xls", ".ckpt", ".pt", ".joblib",
}
ALLOWED_CSV_DIRS = {"examples"}
README_EXEMPT_ENTRY_POINTS = {
    # Converts the study's original workbooks and is not part of the public workflow.
    "ais-build-dataset",
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def test_no_patient_data_or_weights_are_tracked():
    offenders = [
        path for path in tracked_files() if path.suffix.lower() in DATA_EXTENSIONS
    ]
    assert not offenders, f"Patient data or model weights are tracked: {offenders}"


def test_only_synthetic_csvs_are_tracked():
    offenders = [
        path
        for path in tracked_files()
        if path.suffix.lower() == ".csv" and path.parts[0] not in ALLOWED_CSV_DIRS
    ]
    assert not offenders, f"Unexpected CSV files are tracked: {offenders}"


def test_gitignore_anchors_data_directories_to_the_repository_root():
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    # A bare "models/" would also ignore src/ais_progression/models/.
    assert "/models/" in lines
    assert "/data/" in lines
    assert "models/" not in lines


def _entry_points() -> dict[str, str]:
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


@pytest.mark.parametrize("command,target", sorted(_entry_points().items()))
def test_every_entry_point_imports_and_parses_arguments(command, target):
    module_name, function_name = target.split(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, function_name))
    parser = module.build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_documented_commands_exist():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for command in _entry_points().keys() - README_EXEMPT_ENTRY_POINTS:
        assert command in readme, f"{command} is not documented in the README"


def test_ci_verifies_every_installed_entry_point():
    """CI once invoked a command that does not exist, and stayed red.

    The workflow names each console script explicitly, so a renamed or added
    entry point has to be reflected there too.
    """
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    missing = [command for command in _entry_points() if f"uv run {command} " not in workflow]
    assert not missing, f"CI does not verify these entry points: {missing}"


def test_no_training_artifacts_were_left_in_the_repository_root():
    """Training must write only where it was told to.

    Lightning silently creates <cwd>/checkpoints and <cwd>/lightning_logs unless
    checkpointing and logging are switched off, which dumps hundreds of
    megabytes wherever the command happened to be run.

    ``logs/`` is not on this list: .gitignore reserves it, and it is where a
    caller redirecting a long run's stderr would reasonably put it. The point
    here is what training writes *without being asked*.
    """
    strays = [
        name for name in ("checkpoints", "lightning_logs") if (REPO_ROOT / name).exists()
    ]
    assert not strays, f"Training left artefacts in the repository root: {strays}"


def test_save_json_refuses_to_write_invalid_json(tmp_path):
    """NaN is not valid JSON; a bundle containing one breaks non-Python readers."""
    import json
    import math

    from ais_progression.utils import save_json

    path = tmp_path / "out.json"
    save_json({"auc": float("nan"), "weights": [1.0, float("inf")]}, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["auc"] is None
    assert payload["weights"] == [1.0, None]
    assert not math.isnan(payload["weights"][0])
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_build_dataset_discovers_relocated_workbooks(tmp_path):
    """The cohort layout changes; discovery must survive a moved workbook."""
    import pandas as pd

    from ais_progression.cli.build_dataset import build_arg_parser, resolve_sources

    data_dir = tmp_path / "data"
    # Clinical workbook nested one level deeper than the flat default.
    (data_dir / "Clinical_Info").mkdir(parents=True)
    (data_dir / "Front" / "Input_Original_n471").mkdir(parents=True)
    (data_dir / "Lateral" / "Input_Original_n471").mkdir(parents=True)
    frame = pd.DataFrame({"a": [1]})
    frame.to_excel(data_dir / "Clinical_Info" / "Clinical_Info_n471.xlsx", index=False)
    frame.to_excel(data_dir / "Front" / "Input_Original_n471" / "InputPath_n471.xlsx", index=False)
    frame.to_excel(
        data_dir / "Lateral" / "Input_Original_n471" / "InputPath_n471.xlsx", index=False
    )

    args = build_arg_parser().parse_args(["--data-dir", str(data_dir)])
    resolved = resolve_sources(args)
    assert resolved["clinical_xlsx"].name == "Clinical_Info_n471.xlsx"
    assert resolved["front_xlsx"].parent.parent.name == "Front"
    assert resolved["lateral_xlsx"].parent.parent.name == "Lateral"


def test_build_dataset_reports_what_it_could_not_find(tmp_path):
    from ais_progression.cli.build_dataset import build_arg_parser, resolve_sources

    args = build_arg_parser().parse_args(["--data-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="Could not locate"):
        resolve_sources(args)
