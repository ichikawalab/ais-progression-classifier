"""Torch Dataset, transforms, and loaders for the two radiograph modalities."""
from __future__ import annotations

from collections.abc import Callable
from functools import cache

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ais_progression.config import AugmentConfig, DataConfig
from ais_progression.data.schema import ID_COLUMN, LABEL_COLUMN, image_column


class RadiographDataset(Dataset):
    """Radiographs of one modality, indexed by the unified dataset's rows."""

    def __init__(
        self,
        df: pd.DataFrame,
        modality: str,
        transform: Callable,
        has_labels: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.path_column = image_column(modality)
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        with Image.open(row[self.path_column]) as source:
            image = source.convert("RGB")
        image = self.transform(image)
        if self.has_labels:
            return image, torch.tensor(int(row[LABEL_COLUMN]), dtype=torch.long)
        return image, str(row[ID_COLUMN])


@cache
def resolve_timm_data_config(arch: str) -> dict:
    """timm's resolved input size / mean / std for an architecture.

    Built from an un-pretrained throwaway model so no weights are downloaded,
    and cached because the cross-validation loop rebuilds transforms per fold.
    Callers must treat the returned dict as read-only.
    """
    dummy = timm.create_model(arch, pretrained=False, num_classes=0)
    return timm.data.resolve_data_config({}, model=dummy)


def build_transforms(
    timm_data_config: dict, augment: AugmentConfig, is_training: bool
) -> transforms.Compose:
    """Training or evaluation transform pipeline.

    Training augmentation is horizontal flipping (p=0.5) plus a random resized
    crop covering 50-100% of the image area at a fixed 1:1 aspect ratio.
    Validation and test data are only resized.
    """
    mean, std = timm_data_config["mean"], timm_data_config["std"]
    input_size = timm_data_config["input_size"][1:]  # (H, W)

    if not is_training:
        return transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    ops: list = []
    if augment.horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    if augment.random_resized_crop:
        ops.append(
            transforms.RandomResizedCrop(
                size=input_size,
                scale=tuple(augment.rrc_scale),
                ratio=tuple(augment.rrc_ratio),
            )
        )
    else:
        ops.append(transforms.Resize(input_size))
    ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(ops)


def build_loader(
    df: pd.DataFrame,
    modality: str,
    transform: Callable,
    data_cfg: DataConfig,
    *,
    shuffle: bool,
    has_labels: bool = True,
) -> DataLoader:
    dataset = RadiographDataset(df, modality, transform, has_labels=has_labels)
    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=shuffle,
        num_workers=data_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        # Only the training loader is iterated repeatedly, so it is the only one
        # worth keeping workers alive for. Evaluation and prediction loaders are
        # built fresh for every fold; persisting their workers would leak a
        # process pool per fold across a full cross-validation run.
        persistent_workers=shuffle and data_cfg.num_workers > 0,
    )


def denormalize(tensor: torch.Tensor, mean, std) -> np.ndarray:
    """Normalized (C, H, W) tensor -> NumPy (H, W, C) image in [0, 1]."""
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(array * np.array(std) + np.array(mean), 0, 1)
