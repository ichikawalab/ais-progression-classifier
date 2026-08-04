"""PyTorch Lightning wrapper: AdamW, warmup + cosine decay, weighted CE loss."""
from __future__ import annotations

import math
from dataclasses import asdict

import pytorch_lightning as pl
import torch
import torch.nn as nn

from ais_progression.config import ImageConfig, TrainConfig
from ais_progression.models.backbone import TransferModel


class ImageClassifier(pl.LightningModule):
    def __init__(
        self,
        image_cfg: ImageConfig | dict,
        train_cfg: TrainConfig | dict,
        arch: str,
        class_weights: list[float] | None = None,
        initialize_pretrained: bool = True,
    ):
        super().__init__()
        if isinstance(image_cfg, dict):
            image_cfg = ImageConfig(**image_cfg)
        if isinstance(train_cfg, dict):
            train_cfg = TrainConfig(**train_cfg)
        self.save_hyperparameters(
            {
                "image_cfg": asdict(image_cfg),
                "train_cfg": asdict(train_cfg),
                "arch": arch,
                "class_weights": class_weights,
            }
        )
        self.image_cfg = image_cfg
        self.train_cfg = train_cfg

        self.model = TransferModel(
            arch=arch,
            num_classes=image_cfg.num_classes,
            hidden_dim=image_cfg.hidden_dim,
            dropout=image_cfg.dropout,
            pretrained=image_cfg.pretrained and initialize_pretrained,
            freeze_backbone=image_cfg.freeze_backbone,
        )
        weights = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        if self.image_cfg.freeze_backbone:
            self.model.backbone.eval()

    def training_step(self, batch, batch_idx: int):
        images, labels = batch
        loss = self.loss_fn(self(images), labels)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        accuracy = (torch.argmax(logits, dim=1) == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        train_cfg = self.train_cfg
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(
            trainable, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )

        # Epoch-wise linear warmup from 0.1x to 1.0x, then cosine decay to 0 at
        # max_epochs. Matches the schedule used for the published results.
        def lr_lambda(epoch: int) -> float:
            if epoch < train_cfg.warmup_epochs:
                return 0.1 + 0.9 * epoch / train_cfg.warmup_epochs
            progress = (epoch - train_cfg.warmup_epochs) / max(
                1, train_cfg.max_epochs - train_cfg.warmup_epochs
            )
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
