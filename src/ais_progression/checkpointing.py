"""Checkpoint loading shared by evaluation, prediction, and Grad-CAM."""
from __future__ import annotations

from pathlib import Path

import torch

from ais_progression.lit_module import TransferLightningModule


def load_trusted_checkpoint(
    path: str | Path, device: torch.device
) -> TransferLightningModule:
    """Load a checkpoint created locally by ais-train without downloading weights."""
    return TransferLightningModule.load_from_checkpoint(
        str(path),
        map_location=device,
        weights_only=True,
        initialize_pretrained=False,
        strict=True,
    )
