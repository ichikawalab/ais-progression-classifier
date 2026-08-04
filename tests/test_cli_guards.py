"""The command-line refusals that stop a run before it wastes days."""
from __future__ import annotations

import pytest

from ais_progression.cli import cv_modality, train_final
from ais_progression.cli._common import require_gpu


def test_require_gpu_refuses_when_torch_sees_no_device(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as excinfo:
        require_gpu(allow_cpu=False)

    message = str(excinfo.value)
    # The message has to carry the fix: the usual cause is an environment that
    # reverted to the CPU wheels, and the reader is mid-run, not reading docs.
    assert "--allow-cpu" in message
    assert "download.pytorch.org" in message
    assert "--no-sync" in message


def test_require_gpu_passes_when_explicitly_allowed(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    require_gpu(allow_cpu=True)


def test_require_gpu_passes_when_a_device_is_present(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    require_gpu(allow_cpu=False)


def test_image_cross_validation_refuses_a_cpu_environment(monkeypatch, synthetic_cohort):
    """Torch reports the missing GPU and trains anyway; the CLI must not."""
    import torch

    csv_path, _ = synthetic_cohort
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="No GPU available"):
        cv_modality.main(
            ["--modality", "front", "--model", "vit", "--dataset-csv", str(csv_path)]
        )


def test_clinical_cross_validation_does_not_need_a_gpu(monkeypatch, synthetic_cohort, tmp_path):
    """The clinical models are scikit-learn; refusing them would be wrong."""
    import torch

    csv_path, _ = synthetic_cohort
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cv_modality.main(
        [
            "--modality", "clinical", "--model", "logreg",
            "--dataset-csv", str(csv_path),
            "--run-dir", str(tmp_path / "clinical_logreg"),
            "--reps", "1", "--folds", "3",
            "--set", "clinical.n_trials=2", "--set", "clinical.inner_folds=2",
        ]
    )
    assert (tmp_path / "clinical_logreg" / "predictions.csv").exists()


def test_final_training_refuses_a_cpu_environment(monkeypatch, synthetic_cohort):
    import torch

    csv_path, _ = synthetic_cohort
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="No GPU available"):
        train_final.main(["--dataset-csv", str(csv_path)])


def test_clinical_only_bundle_does_not_need_a_gpu(monkeypatch, synthetic_cohort):
    """A clinical-only profile trains no image model, so the guard must not fire.

    It still fails -- there are no cross-validation runs in this tmp dir -- but
    on that, not on the absent GPU.
    """
    import torch

    csv_path, _ = synthetic_cohort
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as excinfo:
        train_final.main(
            [
                "--dataset-csv", str(csv_path),
                "--profile", "cheap=clinical",
                "--default-profile", "cheap",
            ]
        )
    assert "No GPU available" not in str(excinfo.value)
