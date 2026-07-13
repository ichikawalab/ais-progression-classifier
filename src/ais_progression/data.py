"""Dataset, DataModule, and transform construction shared by train / gradcam / predict."""
from __future__ import annotations

from pathlib import Path
from typing import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os

import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ais_progression.config import AugmentConfig, DataConfig
from ais_progression.splitting import file_sha256


def resolve_image_paths(df: pd.DataFrame, csv_path: str | Path) -> pd.DataFrame:
    """Resolve relative image_path entries against the directory containing csv_path."""
    base_dir = Path(csv_path).parent
    df = df.copy()

    def _resolve(p: str) -> str:
        path = Path(str(p).strip())
        if not path.is_absolute():
            path = base_dir / path
        return str(path)

    df["image_path"] = df["image_path"].map(_resolve)
    return df


def load_and_validate_train_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the public training CSV schema."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required = {"sample_id", "patient_id", "image_path", "label"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Training CSV is missing required column(s): {sorted(missing_cols)}")

    if df[list(required)].isna().any().any():
        raise ValueError(f"Required columns contain missing values: {sorted(required)}")
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    if df["sample_id"].duplicated().any():
        raise ValueError("'sample_id' values must be unique after trimming whitespace.")
    numeric_labels = pd.to_numeric(df["label"], errors="coerce")
    if numeric_labels.isna().any():
        raise ValueError("'label' column must contain numeric 0 or 1 values.")
    invalid_mask = ~numeric_labels.isin([0, 1])
    if invalid_mask.any():
        invalid_labels = sorted(numeric_labels[invalid_mask].unique().tolist())
        raise ValueError(
            f"'label' column must contain only 0 or 1, found: {invalid_labels}. "
            "Binarize your labels before training (e.g. df['label'] = (df['Label'] == 2).astype(int))."
        )
    df["label"] = numeric_labels.astype("int64")

    df = resolve_image_paths(df, csv_path)

    missing_files = [p for p in df["image_path"] if not Path(p).exists()]
    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(
            f"{len(missing_files)} image path(s) referenced in the CSV do not exist, e.g.:\n{preview}"
        )

    if (df["sample_id"] == "").any() or (df["patient_id"] == "").any():
        raise ValueError("sample_id and patient_id cannot be empty.")
    def validate_and_hash(path: str) -> str:
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unreadable image file '{path}': {exc}") from exc
        return file_sha256(path)

    workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        df["image_sha256"] = list(executor.map(validate_and_hash, df["image_path"]))
    return df


def load_and_validate_predict_input(
    input_csv: str | Path | None, input_dir: str | Path | None
) -> pd.DataFrame:
    """Load inference input from either a CSV (image_path column) or a directory (recursive glob)."""
    from ais_progression.preprocessing import IMAGE_EXTENSIONS

    if (input_csv is None) == (input_dir is None):
        raise ValueError("Exactly one of input_csv or input_dir must be provided.")

    if input_csv is not None:
        input_csv = Path(input_csv)
        df = pd.read_csv(input_csv)
        if "image_path" not in df.columns:
            raise ValueError(f"CSV must contain an 'image_path' column: {input_csv}")
        df = resolve_image_paths(df[["image_path"]], input_csv)
    else:
        input_dir = Path(input_dir)
        paths = sorted(
            str(p) for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        df = pd.DataFrame({"image_path": paths})

    missing_files = [p for p in df["image_path"] if not Path(p).exists()]
    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(f"{len(missing_files)} image path(s) do not exist, e.g.:\n{preview}")

    return df


class AISDataset(Dataset):
    """Dataset over a DataFrame with an `image_path` column and, when has_labels,
    a `label` column."""

    def __init__(self, df: pd.DataFrame, transform: Callable, has_labels: bool = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        with Image.open(image_path) as source:
            img = source.convert("RGB")
        img = self.transform(img)
        if self.has_labels:
            label = torch.tensor(int(row["label"]), dtype=torch.long)
            return img, label
        return img, image_path


@lru_cache(maxsize=None)
def get_data_config_from_arch(arch: str) -> dict:
    """Get timm's resolved data config (input_size, mean, std) without needing an
    already-instantiated model. Uses pretrained=False so no weights are downloaded.

    Cached per architecture: the cross-validation loop rebuilds transforms for
    every fold, and instantiating a throwaway model each time is wasteful. The
    returned dict is treated as read-only by all callers."""
    dummy = timm.create_model(arch, pretrained=False, num_classes=0)
    return timm.data.resolve_data_config({}, model=dummy)


def build_transforms(
    data_cfg: dict, augment_cfg: AugmentConfig, is_training: bool
) -> transforms.Compose:
    """Build the train/eval transform pipeline from a timm resolved data config."""
    mean, std = data_cfg["mean"], data_cfg["std"]
    input_size = data_cfg["input_size"][1:]  # (H, W)

    if is_training:
        ops: list = []
        if augment_cfg.horizontal_flip:
            ops.append(transforms.RandomHorizontalFlip())
        if augment_cfg.random_resized_crop:
            scale = tuple(augment_cfg.rrc_scale)
            ratio = tuple(augment_cfg.rrc_ratio)
            ops.append(transforms.RandomResizedCrop(size=input_size, scale=scale, ratio=ratio))
        else:
            ops.append(transforms.Resize(input_size))
        ops += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        return transforms.Compose(ops)

    return transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def denormalize(tensor: torch.Tensor, mean: list, std: list):
    """Convert a normalized (C, H, W) tensor back to a NumPy image (H, W, C) in [0, 1]."""
    import numpy as np

    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * np.array(std) + np.array(mean)
    return np.clip(arr, 0, 1)


class AISDataModule(pl.LightningDataModule):
    """Data loaders for one precomputed train/validation/test fold."""

    def __init__(
        self,
        data_cfg: DataConfig,
        augment_cfg: AugmentConfig,
        arch: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        super().__init__()
        self.data_cfg = data_cfg
        self.augment_cfg = augment_cfg
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        timm_data_cfg = get_data_config_from_arch(arch)
        self.train_transform = build_transforms(timm_data_cfg, augment_cfg, is_training=True)
        self.eval_transform = build_transforms(timm_data_cfg, augment_cfg, is_training=False)

    def train_dataloader(self) -> DataLoader:
        ds = AISDataset(self.train_df, self.train_transform, has_labels=True)
        return DataLoader(
            ds,
            batch_size=self.data_cfg.batch_size,
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.data_cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        ds = AISDataset(self.val_df, self.eval_transform, has_labels=True)
        return DataLoader(
            ds,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.data_cfg.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        ds = AISDataset(self.test_df, self.eval_transform, has_labels=True)
        return DataLoader(
            ds,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.data_cfg.num_workers > 0,
        )
