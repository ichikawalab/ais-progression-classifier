"""Grad-CAM must work for all three configured backbone families.

ViT and Swin need a reshape from tokens back to a spatial grid, and that reshape
is the part most likely to break on a timm upgrade. These tests use small
variants of the same families so they run on CPU without downloading weights.
"""
from __future__ import annotations

import pytest
import torch

from ais_progression.config import ImageConfig, TrainConfig, load_config
from ais_progression.gradcam import generate_gradcam, resolve_target_layer
from ais_progression.models.lightning import ImageClassifier

# One small stand-in per family in configs/default.yaml.
FAMILY_ARCHS = {
    "vit": "vit_tiny_patch16_384",
    "swint": "swin_tiny_patch4_window7_224",
    "convnextv2": "convnextv2_atto",
}


def _classifier(arch: str) -> ImageClassifier:
    return ImageClassifier(
        ImageConfig(pretrained=False), TrainConfig(), arch, initialize_pretrained=False
    )


@pytest.mark.parametrize("family,arch", sorted(FAMILY_ARCHS.items()))
def test_target_layer_resolves_for_each_family(family, arch):
    import timm

    input_size = timm.data.resolve_data_config(
        {}, model=timm.create_model(arch, pretrained=False, num_classes=0)
    )["input_size"][1:]
    layer, reshape = resolve_target_layer(_classifier(arch).model, arch, input_size)
    assert isinstance(layer, torch.nn.Module)
    assert (reshape is None) == (family == "convnextv2")


def test_unsupported_architecture_is_reported():
    with pytest.raises(ValueError, match="does not know which layer"):
        resolve_target_layer(_classifier("resnet18").model, "mlp_mixer_b16", (224, 224))


@pytest.mark.parametrize("arch", sorted(FAMILY_ARCHS.values()))
def test_gradcam_writes_overlays_for_each_family(arch, synthetic_cohort, tmp_path):
    _, frame = synthetic_cohort
    config = load_config(dotted_overrides={"data.batch_size": 1, "data.num_workers": 0})
    written = generate_gradcam(
        _classifier(arch), config, arch, "front", frame.head(1), tmp_path / arch
    )
    assert len(written) == 1
    assert written[0].exists()
    assert (tmp_path / arch / f"heatmap_{frame.loc[0, 'patient_id']}.png").exists()
