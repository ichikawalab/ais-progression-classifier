"""Progress reporting: a bar on a terminal, plain lines in a log."""
from __future__ import annotations

import sys

from ais_progression.experiments.modality_cv import run_modality_cv
from ais_progression.utils import progress_bar_enabled


class _Stream:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_progress_bar_follows_whether_stdout_is_a_terminal(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _Stream(tty=True))
    assert progress_bar_enabled() is True

    # Redirected to a file: the bar would bury the fold results under redraws.
    monkeypatch.setattr(sys, "stdout", _Stream(tty=False))
    assert progress_bar_enabled() is False


def test_each_fold_is_announced_before_it_starts(
    tiny_arch_config, synthetic_cohort, tmp_path, capsys
):
    """A bar counts epochs, not folds, so the fold has to name itself first."""
    _, frame = synthetic_cohort
    run_modality_cv(tiny_arch_config, frame, "front", "tiny", tmp_path / "front_tiny")

    lines = capsys.readouterr().out.splitlines()
    started = [line for line in lines if line.endswith(": training")]
    finished = [line for line in lines if "test AUC" in line]

    cv = tiny_arch_config.cross_validation
    assert len(started) == cv.num_reps * cv.num_folds
    assert len(finished) == len(started)
    assert "rep 1/2 fold 1/4 (1/8): training" in started[0]
    # The announcement precedes the result for every fold.
    assert lines.index(started[0]) < lines.index(finished[0])


def test_resumed_folds_are_not_announced_as_training(
    tiny_arch_config, synthetic_cohort, tmp_path, capsys
):
    _, frame = synthetic_cohort
    run_dir = tmp_path / "front_tiny"
    run_modality_cv(tiny_arch_config, frame, "front", "tiny", run_dir)
    capsys.readouterr()

    run_modality_cv(tiny_arch_config, frame, "front", "tiny", run_dir)
    lines = capsys.readouterr().out.splitlines()
    assert not [line for line in lines if line.endswith(": training")]
    assert all("already done, skipping" in line for line in lines if "fold" in line)
