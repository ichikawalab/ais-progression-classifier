from __future__ import annotations

import json

import pytest

from ais_progression.cli.predict import main


def _bundle(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = {
        "default_profile": "full",
        "profiles": [
            {
                "name": "full",
                "members": ["front_vit", "clinical_logreg"],
                "operating_point": {"threshold": 0.54},
                "cv_metrics": {"auc_mean": 0.825},
            },
            {
                "name": "clinical_only",
                "members": ["clinical_logreg"],
                "operating_point": {"threshold": 0.55},
                "cv_metrics": {"auc_mean": 0.719},
            },
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle


def test_list_profiles_needs_no_input_or_output_csv(tmp_path, capsys):
    bundle = _bundle(tmp_path)

    main(["--bundle-dir", str(bundle), "--list-profiles"])

    output = capsys.readouterr().out
    assert "full (default): 2 model(s), CV AUC 0.825, threshold 0.540" in output
    assert "clinical_only: 1 model(s), CV AUC 0.719, threshold 0.550" in output


def test_prediction_still_requires_input_and_output_csv(tmp_path):
    bundle = _bundle(tmp_path)

    with pytest.raises(SystemExit, match="2"):
        main(["--bundle-dir", str(bundle)])
