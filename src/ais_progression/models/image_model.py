"""Train and score one image model (one modality x one backbone).

This is the unit of work shared by the cross-validation protocol and the final
model: fit on a training frame with early stopping on a validation frame, then
emit progression probabilities for any frame.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

from ais_progression.config import Config
from ais_progression.data.images import (
    build_loader,
    build_transforms,
    resolve_timm_data_config,
)
from ais_progression.data.schema import LABEL_COLUMN
from ais_progression.models.lightning import ImageClassifier
from ais_progression.utils import get_device, resolve_precision, set_matmul_precision

PROGRESSION_CLASS = 1


@dataclass
class ImageFitResult:
    """Outcome of one training run.

    ``best_epoch`` is the 0-based epoch whose weights were kept. It is what the
    final model needs in order to retrain on the whole cohort for the right
    number of epochs; ``stopped_epoch`` is later by roughly the early-stopping
    patience and is only useful for diagnosing runs.
    """

    checkpoint_path: Path
    best_val_loss: float | None
    best_epoch: int
    stopped_epoch: int
    class_weights: list[float] | None
    final_train_loss: float | None = None


# ModelCheckpoint does not expose the epoch it kept, so it is written into the
# filename and read back. Tying it to the actual saved file means the recorded
# epoch cannot drift away from the weights that were retained.
_EPOCH_IN_FILENAME = re.compile(r"epoch(\d+)")


def _best_epoch_from_path(checkpoint_path: Path) -> int:
    match = _EPOCH_IN_FILENAME.search(checkpoint_path.stem)
    if not match:
        raise RuntimeError(
            f"Could not read the epoch from checkpoint name '{checkpoint_path.name}'."
        )
    return int(match.group(1))


def compute_balanced_class_weights(labels: pd.Series) -> list[float]:
    """Inverse-frequency class weights from the training subset only."""
    y = labels.astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training subset must contain both classes to weight the loss.")
    return compute_class_weight("balanced", classes=np.array([0, 1]), y=y).tolist()


def build_callbacks(
    config: Config, work_dir: str | Path
) -> tuple[ModelCheckpoint, EarlyStopping]:
    """Checkpointing and early stopping for one training run.

    The published code also passed ``strict=True`` and
    ``check_on_train_epoch_end=True`` to EarlyStopping. Both are already
    Lightning's behaviour here -- ``strict`` defaults to True, and
    ``check_on_train_epoch_end`` defaults to None, which resolves to True
    whenever validation runs once per epoch. Pinning the latter would keep
    checking at train-epoch end, against a stale ``val_loss``, if the validation
    cadence ever changed.
    """
    checkpoint = ModelCheckpoint(
        dirpath=Path(work_dir) / "checkpoints",
        filename="best-epoch{epoch:03d}",
        auto_insert_metric_name=False,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=config.train.early_stopping_patience
    )
    return checkpoint, early_stopping


def fit_image_model(
    config: Config,
    arch: str,
    modality: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    work_dir: str | Path,
    enable_progress_bar: bool = False,
    fixed_epochs: int | None = None,
) -> ImageFitResult:
    """Train one backbone.

    With ``val_df``, training early-stops on validation loss and the epoch with
    the lowest loss is kept -- the cross-validation protocol.

    With ``val_df=None`` and ``fixed_epochs=n``, training runs for exactly n
    epochs on all of ``train_df`` and the final weights are kept. This is how the
    deployable model is built: the epoch count comes from the cross-validation
    runs, so no data has to be held back to discover it. The learning-rate
    schedule still spans ``config.train.max_epochs``, so the first n epochs
    follow exactly the same trajectory the cross-validated models saw.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    set_matmul_precision(config.train.matmul_precision)

    if (val_df is None) == (fixed_epochs is None):
        raise ValueError(
            "Provide either val_df (early stopping) or fixed_epochs (full-cohort "
            "training), not both and not neither."
        )

    timm_data_config = resolve_timm_data_config(arch)
    train_loader = build_loader(
        train_df,
        modality,
        build_transforms(timm_data_config, config.augment, is_training=True),
        config.data,
        shuffle=True,
    )
    class_weights = (
        compute_balanced_class_weights(train_df[LABEL_COLUMN])
        if config.train.use_class_weights
        else None
    )
    module = ImageClassifier(config.image, config.train, arch, class_weights)

    common = dict(
        accelerator="auto",
        devices=1,
        precision=resolve_precision(config.train.precision),
        deterministic="warn" if config.train.deterministic else False,
        logger=False,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=False,
    )

    if val_df is not None:
        val_loader = build_loader(
            val_df,
            modality,
            build_transforms(timm_data_config, config.augment, is_training=False),
            config.data,
            shuffle=False,
        )
        checkpoint, early_stopping = build_callbacks(config, work_dir)
        trainer = pl.Trainer(
            max_epochs=config.train.max_epochs,
            min_epochs=config.train.min_epochs,
            callbacks=[checkpoint, early_stopping],
            **common,
        )
        trainer.fit(module, train_loader, val_loader)
        if not checkpoint.best_model_path:
            raise RuntimeError(f"Training produced no checkpoint under {work_dir}.")
        checkpoint_path = Path(checkpoint.best_model_path)
        return ImageFitResult(
            checkpoint_path=checkpoint_path,
            best_val_loss=float(checkpoint.best_model_score),
            best_epoch=_best_epoch_from_path(checkpoint_path),
            stopped_epoch=int(trainer.current_epoch),
            class_weights=class_weights,
            final_train_loss=_metric(trainer, "train_loss"),
        )

    if fixed_epochs < 1:
        raise ValueError("fixed_epochs must be >= 1")
    # Without enable_checkpointing=False, Lightning installs a default
    # ModelCheckpoint that writes into <cwd>/checkpoints. The final weights are
    # saved explicitly below, so that would only litter the working directory
    # with hundreds of megabytes per model.
    trainer = pl.Trainer(max_epochs=fixed_epochs, enable_checkpointing=False, **common)
    trainer.fit(module, train_loader)

    checkpoint_path = work_dir / "checkpoints" / f"final-epoch{fixed_epochs - 1:03d}.ckpt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(checkpoint_path, weights_only=True)
    return ImageFitResult(
        checkpoint_path=checkpoint_path,
        best_val_loss=None,
        best_epoch=fixed_epochs - 1,
        stopped_epoch=fixed_epochs - 1,
        class_weights=class_weights,
        final_train_loss=_metric(trainer, "train_loss"),
    )


def _metric(trainer: pl.Trainer, name: str) -> float | None:
    value = trainer.callback_metrics.get(name)
    return float(value) if value is not None else None


def load_image_classifier(
    checkpoint_path: str | Path,
    config: Config,
    arch: str,
    device: torch.device | None = None,
) -> ImageClassifier:
    """Rebuild a trained classifier from a checkpoint.

    Every checkpoint this package writes -- fold checkpoints, bundle weights,
    and the full Lightning checkpoints from ``--save-full-checkpoints`` -- is
    saved weights-only, so it loads without unpickling arbitrary objects. A file
    that fails here did not come from this package; that is a reason to stop, not
    to retry with the unpickler.
    """
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # The loss weight buffer is training-only and depends on the fold's class balance.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss_fn.")}

    module = ImageClassifier(
        config.image, config.train, arch, class_weights=None, initialize_pretrained=False
    )
    module.load_state_dict(state_dict, strict=True)
    return module.to(device).eval()


def predict_image_model(
    module: ImageClassifier,
    config: Config,
    arch: str,
    modality: str,
    df: pd.DataFrame,
    device: torch.device | None = None,
) -> np.ndarray:
    """Progression probabilities for ``df``, in row order. Always fp32."""
    device = device or get_device()
    module = module.to(device).eval()
    loader = build_loader(
        df,
        modality,
        build_transforms(resolve_timm_data_config(arch), config.augment, is_training=False),
        config.data,
        shuffle=False,
        has_labels=False,
    )
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _ in loader:
            logits = module(images.to(device, non_blocking=True))
            batches.append(torch.softmax(logits, dim=1)[:, PROGRESSION_CLASS].cpu().numpy())
    probabilities = np.concatenate(batches) if batches else np.empty(0)
    if len(probabilities) != len(df):
        raise RuntimeError("Prediction count does not match the input frame.")
    return probabilities


def discard_checkpoints(work_dir: str | Path) -> None:
    """Delete a fold's working directory, weights and all.

    Cross-validation trains up to reps x folds models per backbone; keeping every
    checkpoint would cost hundreds of gigabytes, and only the predictions are
    needed downstream. The whole directory goes, so a full run does not leave
    hundreds of empty folders behind.
    """
    shutil.rmtree(Path(work_dir), ignore_errors=True)
