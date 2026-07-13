"""Architecture-aware Grad-CAM target-layer resolution and CAM generation."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import cv2
import pandas as pd
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from ais_progression.data import denormalize
from ais_progression.model import TransferModel

SUPPORTED_CAM_FAMILIES = (
    "resnet",
    "densenet",
    "inception",
    "convnext",
    "efficientnet",
    "swin",
    "vit",
)


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """[B, 1+HW, C] (cls token + patch tokens) -> [B, C, H, W]."""
    tokens = tensor[:, 1:, :]
    num_tokens = tokens.shape[1]
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(
            f"Expected a square patch grid for ViT reshape, got {num_tokens} tokens."
        )
    result = tokens.reshape(tokens.size(0), side, side, tokens.size(2))
    return result.permute(0, 3, 1, 2)


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """timm Swin block output -> [B, C, H, W].

    Depending on the timm version, the block output is either [B, H, W, C]
    (4D, spatial layout preserved) or [B, L, C] (3D, flattened, requiring a
    square-grid assumption to reshape).
    """
    if tensor.dim() == 4:
        return tensor.permute(0, 3, 1, 2)
    if tensor.dim() == 3:
        num_tokens = tensor.shape[1]
        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(
                f"Expected a square patch grid for Swin reshape, got {num_tokens} tokens."
            )
        result = tensor.reshape(tensor.size(0), side, side, tensor.size(2))
        return result.permute(0, 3, 1, 2)
    raise ValueError(f"Unexpected Swin block output shape: {tuple(tensor.shape)}")


def resolve_cam_targets(
    model: TransferModel, arch: str
) -> tuple[list[nn.Module], Callable | None]:
    """Resolve the Grad-CAM target layer(s) and optional reshape_transform for a
    given timm architecture family, based on its name prefix."""
    backbone = model.backbone
    arch_low = arch.lower()

    resolvers = {
        "resnet": lambda: ([backbone.layer4[-1]], None),
        "densenet": lambda: ([backbone.features[-1]], None),
        "inception": lambda: ([backbone.Mixed_7c], None),
        "convnext": lambda: ([backbone.stages[-1].blocks[-1].conv_dw], None),
        "efficientnet": lambda: ([backbone.conv_head], None),
        "swin": lambda: (
            [backbone.layers[-1].blocks[-1].norm2],
            swin_reshape_transform,
        ),
        "vit": lambda: ([backbone.blocks[-1].norm1], vit_reshape_transform),
    }
    for family in SUPPORTED_CAM_FAMILIES:
        if family in arch_low:
            try:
                return resolvers[family]()
            except (AttributeError, IndexError) as exc:
                raise ValueError(
                    f"Architecture '{arch}' matched Grad-CAM family '{family}', "
                    "but its timm module layout is unsupported by this version."
                ) from exc

    raise ValueError(
        f"No Grad-CAM target layer configured for architecture '{arch}'. "
        f"Supported families: {', '.join(SUPPORTED_CAM_FAMILIES)}."
    )


def generate_gradcam_images(
    model: TransferModel,
    arch: str,
    df: pd.DataFrame,
    transform,
    output_dir: str | Path,
    target_class: str | int,
    alpha: float,
    device: torch.device,
    mean: list,
    std: list,
) -> pd.DataFrame:
    """Run Grad-CAM over every row of df (must have an `image_path` column) and
    save heatmap-overlay images. Returns a summary DataFrame that is also written
    to `output_dir/gradcam_summary.csv`."""
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_layers, reshape_transform = resolve_cam_targets(model, arch)
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        image_path = row["image_path"]
        with Image.open(image_path) as source:
            img = source.convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)[0]
        predicted_label = int(torch.argmax(probs).item())

        if target_class == "pred":
            cam_target_idx = predicted_label
        else:
            cam_target_idx = int(target_class)
        target = ClassifierOutputTarget(cam_target_idx)

        grayscale_cam = cam(input_tensor=input_tensor, targets=[target])[0]

        img_denorm = denormalize(input_tensor[0], mean, std)
        overlay = show_cam_on_image(img_denorm, grayscale_cam, use_rgb=True, image_weight=1 - alpha)

        stem = Path(image_path).stem
        out_name = f"{idx:04d}_{stem}_gradcam.png"
        out_path = output_dir / out_name
        encoded, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not encoded:
            raise RuntimeError(f"Could not encode Grad-CAM output: {out_path}")
        buffer.tofile(str(out_path))

        rows.append(
            {
                "image_path": image_path,
                "prob_class0": float(probs[0].item()),
                "prob_class1": float(probs[1].item()),
                "predicted_label": predicted_label,
                "cam_output_path": str(out_path),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "gradcam_summary.csv", index=False)
    return summary_df
