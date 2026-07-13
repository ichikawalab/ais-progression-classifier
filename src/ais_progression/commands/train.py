"""Run the patient-level stratified 10-fold experiment."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.utils.class_weight import compute_class_weight

from ais_progression.checkpointing import load_trusted_checkpoint
from ais_progression.config import load_config, parse_set_args, save_config
from ais_progression.data import AISDataModule, load_and_validate_train_csv
from ais_progression.evaluation import (
    aggregate_patient_predictions,
    binary_metrics,
    bootstrap_confidence_intervals,
    predict_dataframe,
    save_json,
)
from ais_progression.lit_module import TransferLightningModule
from ais_progression.splitting import (
    assign_stratified_group_folds,
    build_split_manifest,
    file_sha256,
    split_for_outer_fold,
)
from ais_progression.utils import get_device, make_run_dir, resolve_precision, set_seed

BOOTSTRAP_RESAMPLES = 2000


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run patient-level stratified cross-validation for AIS progression."
    )
    parser.add_argument("--config", default=None, help="Optional YAML configuration file.")
    parser.add_argument("--data-csv", default=None, help="CSV with sample_id, patient_id, image_path, label.")
    parser.add_argument("--arch", default=None, help="timm architecture name.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume-run",
        default=None,
        help="Resume an interrupted run directory; other configuration overrides are not allowed.",
    )
    parser.add_argument(
        "--set", dest="set_args", action="append", default=None,
        help="Explicit research override, e.g. --set train.max_epochs=2 (repeatable).",
    )
    return parser


def _cli_overrides(args: argparse.Namespace) -> dict:
    result: dict[str, dict] = {"data": {}, "model": {}, "train": {}, "output": {}}
    mappings = (
        ("data", "csv_path", args.data_csv),
        ("data", "batch_size", args.batch_size),
        ("data", "num_workers", args.num_workers),
        ("model", "arch", args.arch),
        ("train", "seed", args.seed),
        ("output", "dir", args.output_dir),
        ("output", "run_name", args.run_name),
    )
    for section, key, value in mappings:
        if value is not None:
            result[section][key] = value
    return {key: value for key, value in result.items() if value}


def _environment(csv_path: str | Path, base_seed: int) -> dict:
    packages = {}
    for package in ("torch", "pytorch-lightning", "timm", "scikit-learn"):
        try:
            packages[package] = version(package)
        except Exception:
            packages[package] = "unknown"
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "base_seed": base_seed,
        "input_csv_sha256": file_sha256(csv_path),
        "git_commit": git_commit,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.resume_run:
        conflicting = any(
            value is not None
            for value in (
                args.config,
                args.data_csv,
                args.arch,
                args.batch_size,
                args.seed,
                args.num_workers,
                args.output_dir,
                args.run_name,
                args.set_args,
            )
        )
        if conflicting:
            raise ValueError("--resume-run cannot be combined with configuration overrides.")
        run_dir = Path(args.resume_run)
        cfg = load_config(run_dir / "config.yaml")
    else:
        cfg = load_config(
            args.config,
            cli_overrides=_cli_overrides(args),
            dotted_overrides=parse_set_args(args.set_args),
        )
    if cfg.data.csv_path is None:
        raise ValueError("--data-csv is required unless data.csv_path is set in YAML.")

    source_df = load_and_validate_train_csv(cfg.data.csv_path)
    if args.resume_run:
        folded_df = pd.read_csv(run_dir / "folds.csv")
        if set(folded_df["sample_id"]) != set(source_df["sample_id"]):
            raise ValueError("Current training CSV does not match the run's fold manifest.")
        saved_digests = folded_df.set_index("sample_id")["image_sha256"].to_dict()
        current_digests = source_df.set_index("sample_id")["image_sha256"].to_dict()
        if saved_digests != current_digests:
            raise ValueError("Image content has changed since the run was created.")
    else:
        folded_df = assign_stratified_group_folds(
            source_df, cfg.cross_validation.num_folds, cfg.train.seed
        )
        run_dir = make_run_dir(cfg.output.dir, cfg.output.run_name, cfg.model.arch)
        save_config(cfg, run_dir / "config.yaml")
        folded_df.to_csv(run_dir / "folds.csv", index=False)
        save_json(_environment(cfg.data.csv_path, cfg.train.seed), run_dir / "environment.json")
    print(f"Run directory: {run_dir}")

    all_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []
    device = get_device()

    for fold in range(cfg.cross_validation.num_folds):
        fold_dir = run_dir / f"fold_{fold:02d}"
        predictions_path = fold_dir / "predictions.csv"
        metrics_path = fold_dir / "metrics.json"
        if args.resume_run and predictions_path.exists() and metrics_path.exists():
            all_predictions.append(pd.read_csv(predictions_path))
            fold_metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
            print(f"Skipping completed fold {fold + 1}/{cfg.cross_validation.num_folds}")
            continue
        fold_dir.mkdir(parents=True, exist_ok=bool(args.resume_run))
        train_df, val_df, test_df = split_for_outer_fold(
            folded_df, fold, cfg.cross_validation.num_folds
        )
        manifest = build_split_manifest(train_df, val_df, test_df)
        manifest.to_csv(fold_dir / "split.csv", index=False)

        fold_seed = cfg.train.seed + fold
        set_seed(fold_seed, deterministic=cfg.train.deterministic)
        save_json(
            {
                "fold": fold,
                "test_fold": fold,
                "validation_fold": (fold + 1) % cfg.cross_validation.num_folds,
                "base_seed": cfg.train.seed,
                "fold_seed": fold_seed,
                "n_train": len(train_df),
                "n_validation": len(val_df),
                "n_test": len(test_df),
            },
            fold_dir / "metadata.json",
        )

        data_module = AISDataModule(
            data_cfg=cfg.data,
            augment_cfg=cfg.augment,
            arch=cfg.model.arch,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
        )

        class_weights = None
        if cfg.train.use_class_weights:
            y_train = train_df["label"].astype(int).to_numpy()
            if set(np.unique(y_train)) != {0, 1}:
                raise ValueError(f"Fold {fold} training subset does not contain both classes.")
            class_weights = compute_class_weight(
                "balanced", classes=np.array([0, 1]), y=y_train
            ).tolist()

        module = TransferLightningModule(cfg.model, cfg.train, class_weights)
        checkpoint = ModelCheckpoint(
            dirpath=fold_dir / "checkpoints",
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        trainer = pl.Trainer(
            max_epochs=cfg.train.max_epochs,
            min_epochs=cfg.train.min_epochs,
            accelerator="auto",
            devices=1,
            precision=resolve_precision(cfg.train.precision),
            deterministic=cfg.train.deterministic,
            logger=TensorBoardLogger(str(fold_dir), name="tensorboard", version=""),
            callbacks=[
                checkpoint,
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=cfg.train.early_stopping_patience,
                ),
                LearningRateMonitor(logging_interval="epoch"),
            ],
        )
        last_checkpoint = fold_dir / "checkpoints" / "last.ckpt"
        trainer.fit(
            module,
            datamodule=data_module,
            ckpt_path=str(last_checkpoint)
            if args.resume_run and last_checkpoint.exists()
            else None,
        )
        if not checkpoint.best_model_path:
            raise RuntimeError(f"Fold {fold} did not produce a best checkpoint.")

        best_model = load_trusted_checkpoint(checkpoint.best_model_path, device)
        predictions = predict_dataframe(
            best_model,
            data_module.test_dataloader(),
            test_df,
            device,
            fold,
            threshold=0.5,
        )
        predictions.to_csv(predictions_path, index=False)
        metrics = binary_metrics(predictions["true_label"], predictions["prob_class1"])
        metrics.update(
            {
                "fold": fold,
                "best_checkpoint": str(Path(checkpoint.best_model_path).relative_to(run_dir)),
                "best_val_loss": float(checkpoint.best_model_score),
                "threshold_source": "fixed_default",
            }
        )
        save_json(metrics, metrics_path)
        all_predictions.append(predictions)
        fold_metrics.append(metrics)
        print(f"Completed fold {fold + 1}/{cfg.cross_validation.num_folds}")

    oof = pd.concat(all_predictions, ignore_index=True)
    if len(oof) != len(source_df) or oof["sample_id"].duplicated().any():
        raise RuntimeError("Out-of-fold predictions must contain every sample exactly once.")
    oof.to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(fold_metrics).drop(columns=["confusion_matrix"]).to_csv(
        run_dir / "metrics_by_fold.csv", index=False
    )
    patient_oof = aggregate_patient_predictions(oof)
    patient_oof.to_csv(run_dir / "patient_predictions.csv", index=False)
    summary = {
        "analysis_unit": "patient",
        "metrics": binary_metrics(patient_oof["true_label"], patient_oof["prob_class1"]),
        "confidence_intervals_95": bootstrap_confidence_intervals(
            patient_oof, seed=cfg.train.seed, n_resamples=BOOTSTRAP_RESAMPLES
        ),
        "confidence_interval_method": (
            "class-stratified patient-level percentile bootstrap on fixed OOF predictions"
        ),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": cfg.train.seed,
        "threshold_source": "fixed_default",
        "model_selection_warning": (
            "These out-of-fold metrics are unbiased for a single, pre-specified "
            "configuration only. Selecting the architecture or hyperparameters by "
            "comparing this number across runs turns the test folds into a selection "
            "set and biases the reported performance upward. Use nested "
            "cross-validation or an independent external cohort for model selection."
        ),
    }
    save_json(summary, run_dir / "metrics_summary.json")
    print(f"Completed all folds. Results: {run_dir}")


if __name__ == "__main__":
    main()
