"""Grad-CAM overlays for the image models.

Exploratory only: a saliency map shows where activation was high, not why a
prediction was made, and it does not establish a causal explanation.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from ais_progression.config import Config
from ais_progression.data.images import (
    RadiographDataset,
    build_transforms,
    denormalize,
    resolve_timm_data_config,
)
from ais_progression.models.lightning import ImageClassifier
from ais_progression.utils import get_device


def resolve_target_layer(
    model: torch.nn.Module, arch: str, input_size: tuple[int, int]
) -> tuple[torch.nn.Module, Callable | None]:
    """Pick the layer to visualise, plus a reshape for token-based backbones."""
    backbone = model.backbone
    arch = arch.lower()

    if "convnext" in arch:
        return backbone.stages[-1].blocks[-1].conv_dw, None
    if "resnet" in arch:
        return backbone.layer4[-1], None
    if "densenet" in arch:
        return backbone.features[-1], None
    if "inception" in arch:
        return backbone.Mixed_7c, None
    if "efficientnet" in arch:
        return backbone.conv_head, None
    if "swin" in arch:
        height, width = input_size[0] // 32, input_size[1] // 32

        def reshape_swin(tensor: torch.Tensor) -> torch.Tensor:
            # Swin already emits (B, H, W, C) at the final stage in recent timm.
            if tensor.dim() == 4:
                return tensor.permute(0, 3, 1, 2)
            x = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
            return x.permute(0, 3, 1, 2)

        return backbone.layers[-1].blocks[-1].norm2, reshape_swin
    if "vit" in arch:
        height, width = input_size[0] // 16, input_size[1] // 16
        num_prefix = getattr(backbone, "num_prefix_tokens", 1)

        def reshape_vit(tensor: torch.Tensor) -> torch.Tensor:
            x = tensor[:, num_prefix:, :]
            x = x.reshape(x.size(0), height, width, x.size(2))
            return x.permute(0, 3, 1, 2)

        return backbone.blocks[-1].norm1, reshape_vit

    raise ValueError(
        f"Grad-CAM does not know which layer to target for '{arch}'. Supported families: "
        "ResNet, DenseNet, Inception, EfficientNet, ConvNeXt, Swin, ViT."
    )


def generate_gradcam(
    module: ImageClassifier,
    config: Config,
    arch: str,
    modality: str,
    df: pd.DataFrame,
    output_dir: str | Path,
    target_class: str = "1",
    alpha: float = 0.5,
) -> list[Path]:
    """Write a heatmap and an overlay per row of ``df``. Returns the overlay paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    module = module.to(device).eval()

    timm_data_config = resolve_timm_data_config(arch)
    mean, std = timm_data_config["mean"], timm_data_config["std"]
    input_size = timm_data_config["input_size"][1:]
    transform = build_transforms(timm_data_config, config.augment, is_training=False)
    dataset = RadiographDataset(df, modality, transform, has_labels=False)

    target_layer, reshape = resolve_target_layer(module.model, arch, input_size)
    cam = GradCAM(
        model=module.model, target_layers=[target_layer], reshape_transform=reshape
    )

    written: list[Path] = []
    for index in range(len(dataset)):
        image, patient_id = dataset[index]
        batch = image.unsqueeze(0).to(device)

        if target_class == "pred":
            with torch.inference_mode():
                class_index = int(torch.argmax(module(batch), dim=1).item())
        else:
            class_index = int(target_class)

        grayscale = cam(input_tensor=batch, targets=[ClassifierOutputTarget(class_index)])[0]
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale), cv2.COLORMAP_JET)
        overlay = show_cam_on_image(
            denormalize(image, mean, std), grayscale, use_rgb=True, image_weight=1 - alpha
        )

        heatmap_path = output_dir / f"heatmap_{patient_id}.png"
        overlay_path = output_dir / f"overlay_{patient_id}.png"
        cv2.imwrite(str(heatmap_path), heatmap)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        written.append(overlay_path)
    return written
