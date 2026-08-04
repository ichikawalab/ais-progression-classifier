from pathlib import Path

from ais_progression.data.schema import load_dataset
from ais_progression.provenance import (
    modality_run_identity,
    source_tree_sha256,
)
from ais_progression.utils import environment_report


def test_source_tree_digest_includes_untracked_source_files(tmp_path):
    source = tmp_path / "src" / "package"
    source.mkdir(parents=True)
    module = source / "module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    first = source_tree_sha256(tmp_path)

    module.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_sha256(tmp_path) != first

    (source / "untracked.py").write_text("NEW = True\n", encoding="utf-8")
    assert source_tree_sha256(tmp_path) != first


def test_image_run_identity_changes_when_radiograph_bytes_change(
    tiny_arch_config, synthetic_cohort
):
    csv_path, frame = synthetic_cohort
    dataset = load_dataset(csv_path)
    first = modality_run_identity(tiny_arch_config, dataset, "front", "tiny")

    Path(frame.loc[0, "front_path"]).write_bytes(b"different image bytes")
    second = modality_run_identity(tiny_arch_config, dataset, "front", "tiny")
    assert second["image_content_sha256"] != first["image_content_sha256"]


def test_environment_records_commit_dirty_state_and_source_digest():
    report = environment_report()
    assert {"git_commit", "git_dirty", "source_tree_sha256"} <= set(report)
    assert len(report["source_tree_sha256"]) == 64
