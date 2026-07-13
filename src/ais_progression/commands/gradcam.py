"""Generate Grad-CAM images from one fold's best checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ais_progression.checkpointing import load_trusted_checkpoint
from ais_progression.config import load_config
from ais_progression.data import build_transforms, get_data_config_from_arch, load_and_validate_predict_input
from ais_progression.gradcam_utils import generate_gradcam_images
from ais_progression.utils import get_device


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM from a best fold checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--target", default="test", help='"train", "val", "test", or a CSV path.')
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-class", choices=("pred", "0", "1"), default="pred")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not 0 <= args.alpha <= 1:
        parser.error("alpha must be in [0, 1]")

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml")
    if args.fold not in range(cfg.cross_validation.num_folds):
        parser.error(f"fold must be between 0 and {cfg.cross_validation.num_folds - 1}")
    fold_dir = run_dir / f"fold_{args.fold:02d}"
    if args.target in {"train", "val", "test"}:
        manifest = pd.read_csv(fold_dir / "split.csv")
        frame = manifest[manifest["split"] == args.target][["image_path"]].reset_index(drop=True)
    else:
        frame = load_and_validate_predict_input(args.target, None)

    device = get_device()
    module = load_trusted_checkpoint(fold_dir / "checkpoints" / "best.ckpt", device)
    data_cfg = get_data_config_from_arch(cfg.model.arch)
    output_dir = Path(args.output_dir) if args.output_dir else fold_dir / "gradcam" / args.target
    generate_gradcam_images(
        model=module.model.eval().to(device),
        arch=cfg.model.arch,
        df=frame,
        transform=build_transforms(data_cfg, cfg.augment, is_training=False),
        output_dir=output_dir,
        target_class=args.target_class if args.target_class == "pred" else int(args.target_class),
        alpha=args.alpha,
        device=device,
        mean=data_cfg["mean"],
        std=data_cfg["std"],
    )
    print(f"Grad-CAM output: {output_dir}")
