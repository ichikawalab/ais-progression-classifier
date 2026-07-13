import pytest

from ais_progression.gradcam_utils import SUPPORTED_CAM_FAMILIES, resolve_cam_targets


class FakeModel:
    backbone = object()


def test_gradcam_supported_families_are_explicit():
    assert SUPPORTED_CAM_FAMILIES == (
        "resnet",
        "densenet",
        "inception",
        "convnext",
        "efficientnet",
        "swin",
        "vit",
    )


def test_gradcam_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Supported families"):
        resolve_cam_targets(FakeModel(), "unknown_architecture")


def test_gradcam_reports_changed_timm_layout():
    with pytest.raises(ValueError, match="module layout is unsupported"):
        resolve_cam_targets(FakeModel(), "resnet_custom")
