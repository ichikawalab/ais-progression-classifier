"""Shared fixtures: a tiny synthetic cohort with real (random) PNG images."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ais_progression.config import load_config
from ais_progression.data.schema import DATASET_COLUMNS


def _write_image(path, size=(40, 30), seed=0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, size=size, dtype=np.uint8), mode="L").save(path)


@pytest.fixture
def synthetic_cohort(tmp_path):
    """40 patients, balanced-ish labels, with a frontal and lateral image each."""
    n = 40
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        patient_id = f"p{i:03d}"
        label = int(i % 2 == 0)
        front = tmp_path / "front" / f"{patient_id}.png"
        lateral = tmp_path / "lateral" / f"{patient_id}.png"
        _write_image(front, seed=i)
        _write_image(lateral, seed=i + 1000)
        rows.append(
            {
                "patient_id": patient_id,
                "front_path": str(front),
                "lateral_path": str(lateral),
                "age": float(rng.normal(12.7, 1.8)),
                "sex": 2 if i % 10 else 1,
                "risser": int(rng.integers(0, 6)),
                "cobb_baseline": float(rng.normal(28, 10)),
                "label": label,
            }
        )
    frame = pd.DataFrame(rows)[DATASET_COLUMNS]
    csv_path = tmp_path / "dataset.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path, frame


@pytest.fixture
def tiny_arch_config(small_config):
    """small_config with the 384-pixel backbones swapped for one small CNN."""
    small_config.image.archs = {"tiny": "resnet18"}
    small_config.image.pretrained = False
    return small_config


@pytest.fixture
def small_config(tmp_path):
    """Config scaled down so the whole protocol runs in seconds."""
    return load_config(
        dotted_overrides={
            "data.csv_path": None,
            "data.num_workers": 0,
            "data.batch_size": 4,
            "cross_validation.num_reps": 2,
            "cross_validation.num_folds": 4,
            # Enough trials that a fold cannot see every one of them pruned.
            # The logistic-regression space prunes incompatible penalty/solver
            # pairs -- a third of the grid -- so two trials fail outright about
            # one seed in nine, and folds now each draw their own seed.
            "clinical.n_trials": 6,
            "clinical.inner_folds": 3,
            "ensemble.n_trials": 6,
            "ensemble.inner_folds": 3,
            "train.max_epochs": 1,
            "train.min_epochs": 1,
            "train.warmup_epochs": 1,
            "train.precision": "32-true",
            "train.deterministic": False,
            "output.dir": str(tmp_path / "outputs"),
        }
    )
