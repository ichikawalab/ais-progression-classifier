"""Transfer-learning model: a timm backbone + a small classification head."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class TransferModel(nn.Module):
    def __init__(
        self,
        arch: str,
        num_classes: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.arch = arch
        self.freeze_backbone = freeze_backbone

        # num_classes=0 strips the classification head, returning pooled features.
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.classifier(feats)
