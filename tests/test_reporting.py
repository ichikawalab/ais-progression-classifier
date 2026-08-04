import pytest

from ais_progression.experiments.reporting import assert_run_identity


def test_run_identity_rejects_a_changed_configuration(tmp_path):
    run_dir = tmp_path / "ensemble"
    assert_run_identity(run_dir, {"method": "weighted", "seed": 42})

    with pytest.raises(ValueError, match="different configuration"):
        assert_run_identity(run_dir, {"method": "weighted", "seed": 43})


def test_existing_folds_without_an_identity_are_rejected(tmp_path):
    run_dir = tmp_path / "ensemble"
    folds_dir = run_dir / "folds"
    folds_dir.mkdir(parents=True)
    (folds_dir / "rep01_fold01.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="no run_identity.json"):
        assert_run_identity(run_dir, {"method": "weighted", "seed": 42})
