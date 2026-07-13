"""Ensemble inference using the ten best fold checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ais_progression.checkpointing import load_trusted_checkpoint
from ais_progression.config import load_config
from ais_progression.data import AISDataset, build_transforms, get_data_config_from_arch, load_and_validate_predict_input
from ais_progression.utils import get_device


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict with the cross-validation ensemble.")
    parser.add_argument("--run-dir", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv")
    source.add_argument("--input-dir")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or not 0 <= args.threshold <= 1:
        parser.error("batch-size must be positive, num-workers non-negative, and threshold in [0, 1]")

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    checkpoints = [
        run_dir / f"fold_{fold:02d}" / "checkpoints" / "best.ckpt"
        for fold in range(cfg.cross_validation.num_folds)
    ]
    missing = [path for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing best fold checkpoint(s), e.g. {missing[0]}")

    frame = load_and_validate_predict_input(args.input_csv, args.input_dir)
    if frame.empty:
        raise ValueError("No input images were found.")
    transform = build_transforms(
        get_data_config_from_arch(cfg.model.arch), cfg.augment, is_training=False
    )
    device = get_device()
    loader = DataLoader(
        AISDataset(frame, transform, has_labels=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    fold_probabilities = []
    for checkpoint in checkpoints:
        model = load_trusted_checkpoint(checkpoint, device).eval().to(device)
        batches = []
        # Full-precision inference (no autocast) so the ensemble probabilities
        # match the fp32 out-of-fold probabilities reported during training.
        with torch.inference_mode():
            for images, _ in loader:
                logits = model(images.to(device, non_blocking=True))
                batches.append(torch.softmax(logits, dim=1).cpu().numpy())
        fold_probabilities.append(np.concatenate(batches))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    probabilities = np.mean(fold_probabilities, axis=0)
    output = pd.DataFrame(
        {
            "image_path": frame["image_path"],
            "prob_class0": probabilities[:, 0],
            "prob_class1": probabilities[:, 1],
            "predicted_label": (probabilities[:, 1] >= args.threshold).astype(int),
            "threshold": args.threshold,
            "ensemble_folds": len(checkpoints),
        }
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Predicted {len(output)} image(s). Saved: {output_path}")
