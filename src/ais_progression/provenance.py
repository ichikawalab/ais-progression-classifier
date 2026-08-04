"""Content identities for reproducible runs and safe resume."""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from ais_progression.config import CLINICAL_MODALITY, Config, resolve_arch
from ais_progression.data.schema import ID_COLUMN, image_column
from ais_progression.utils import file_sha256

SOURCE_ROOT = Path(__file__).resolve().parents[2]
SOURCE_INPUTS = ("src", "configs", "pyproject.toml", "uv.lock")
RUNTIME_PACKAGES = (
    "torch",
    "torchvision",
    "pytorch-lightning",
    "timm",
    "opencv-python-headless",
    "pandas",
    "numpy",
    "scikit-learn",
    "optuna",
    "joblib",
    "openpyxl",
    "grad-cam",
    "PyYAML",
    "pillow",
    "tqdm",
)


def frame_sha256(frame: pd.DataFrame) -> str:
    """Digest the exact table consumed, including column and row order."""
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_tree_sha256(root: Path = SOURCE_ROOT) -> str:
    """Digest tracked and untracked source inputs that can change execution."""
    digest = hashlib.sha256()
    candidates: list[Path] = []
    for name in SOURCE_INPUTS:
        path = root / name
        if path.is_dir():
            candidates.extend(
                child
                for child in path.rglob("*")
                if child.is_file() and "__pycache__" not in child.parts
            )
        elif path.is_file():
            candidates.append(path)
    if not candidates:
        candidates = [
            child
            for child in Path(__file__).resolve().parent.rglob("*")
            if child.is_file() and "__pycache__" not in child.parts
        ]
        root = Path(__file__).resolve().parent
    for path in sorted(candidates, key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def software_identity(root: Path = SOURCE_ROOT) -> dict:
    """Code and dependency identity used to decide whether resume is valid."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    packages = {}
    for package in RUNTIME_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "source_tree_sha256": source_tree_sha256(root),
        "packages": packages,
    }


def image_content_sha256(dataset: pd.DataFrame, modalities: tuple[str, ...]) -> str:
    """Aggregate the exact radiograph bytes used, in patient/modality order."""
    digest = hashlib.sha256()
    rows = dataset.sort_values(ID_COLUMN)
    for row in rows.itertuples(index=False):
        patient_id = str(getattr(row, ID_COLUMN))
        for modality in sorted(modalities):
            path = Path(getattr(row, image_column(modality)))
            record = f"{patient_id}\0{modality}\0{file_sha256(path)}".encode()
            digest.update(len(record).to_bytes(8, "big"))
            digest.update(record)
    return digest.hexdigest()


def modality_run_identity(
    config: Config, dataset: pd.DataFrame, modality: str, model: str
) -> dict:
    """Everything that can change one base-model CV run's fold outputs."""
    common = {
        "kind": "modality_cv",
        "modality": modality,
        "model": model,
        "dataset_sha256": frame_sha256(dataset),
        "cross_validation": asdict(config.cross_validation),
        "software": software_identity(),
    }
    if modality == CLINICAL_MODALITY:
        common["algorithm"] = {"clinical": asdict(config.clinical)}
    else:
        common["algorithm"] = {
            "arch": resolve_arch(config, model),
            "image": asdict(config.image),
            "train": asdict(config.train),
            "augment": asdict(config.augment),
            "data": {
                "batch_size": config.data.batch_size,
                "num_workers": config.data.num_workers,
            },
        }
        common["image_content_sha256"] = image_content_sha256(dataset, (modality,))
    return common


def final_compatible_algorithm(config: Config, modality: str, model: str) -> dict:
    """Current algorithm settings that must match a base model's CV identity."""
    if modality == CLINICAL_MODALITY:
        return {"clinical": asdict(config.clinical)}
    return {
        "arch": resolve_arch(config, model),
        "image": asdict(config.image),
        "train": asdict(config.train),
        "augment": asdict(config.augment),
        "data": {
            "batch_size": config.data.batch_size,
            "num_workers": config.data.num_workers,
        },
    }
