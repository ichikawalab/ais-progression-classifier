"""PyTorch Lightning wrapper around TransferModel."""
from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import pytorch_lightning as pl

from ais_progression.config import ModelConfig, TrainConfig
from ais_progression.model import TransferModel


class TransferLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_cfg: ModelConfig | dict,
        train_cfg: TrainConfig | dict,
        class_weights: list[float] | None = None,
        initialize_pretrained: bool = True,
    ):
        super().__init__()
        if isinstance(model_cfg, dict):
            model_cfg = ModelConfig(**model_cfg)
        if isinstance(train_cfg, dict):
            train_cfg = TrainConfig(**train_cfg)
        self.save_hyperparameters(
            {
                "model_cfg": asdict(model_cfg),
                "train_cfg": asdict(train_cfg),
                "class_weights": class_weights,
            }
        )
        self.class_weights = class_weights
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

        self.model = TransferModel(
            arch=model_cfg.arch,
            num_classes=model_cfg.num_classes,
            hidden_dim=model_cfg.hidden_dim,
            dropout=model_cfg.dropout,
            pretrained=model_cfg.pretrained and initialize_pretrained,
            freeze_backbone=model_cfg.freeze_backbone,
        )

        cw = torch.tensor(class_weights, dtype=torch.float) if class_weights else None
        self.loss_fn = nn.CrossEntropyLoss(weight=cw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        if self.model_cfg.freeze_backbone:
            self.model.backbone.eval()

    def training_step(self, batch, batch_idx: int):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx: int):
        imgs, meta = batch
        logits = self(imgs)
        probs = torch.softmax(logits, dim=1)
        return probs, meta

    def configure_optimizers(self):
        train_cfg = self.train_cfg
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )

        # Epoch-wise warmup + cosine decay. LambdaLR is stepped once per epoch.
        # Linear warmup ramps to 1.0 at the end of the warmup window so it joins
        # the cosine branch (which also starts at 1.0) without a discontinuity.
        def lr_lambda(epoch: int) -> float:
            if epoch < train_cfg.warmup_epochs:
                return (epoch + 1) / train_cfg.warmup_epochs
            progress = (epoch - train_cfg.warmup_epochs) / max(
                1, train_cfg.max_epochs - train_cfg.warmup_epochs
            )
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
