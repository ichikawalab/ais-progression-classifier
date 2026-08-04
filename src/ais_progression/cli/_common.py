"""Argument-parser fragments shared by the command-line entry points."""
from __future__ import annotations

import argparse
from pathlib import Path

from ais_progression.config import Config, load_config, parse_set_args


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="YAML configuration file.")
    parser.add_argument(
        "--set",
        dest="set_args",
        action="append",
        default=None,
        help="Explicit override, e.g. --set train.max_epochs=2 (repeatable).",
    )


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-csv",
        default=None,
        help="Unified dataset CSV. Defaults to data.csv_path from the config.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)


def build_config(args: argparse.Namespace) -> Config:
    """Assemble the configuration from the YAML file plus the parsed flags."""
    overrides: dict[str, dict] = {"data": {}, "cross_validation": {}, "output": {}}
    mappings = (
        ("data", "csv_path", getattr(args, "dataset_csv", None)),
        ("data", "batch_size", getattr(args, "batch_size", None)),
        ("data", "num_workers", getattr(args, "num_workers", None)),
        ("cross_validation", "num_reps", getattr(args, "reps", None)),
        ("cross_validation", "num_folds", getattr(args, "folds", None)),
        ("cross_validation", "seed", getattr(args, "seed", None)),
        ("output", "dir", getattr(args, "output_dir", None)),
    )
    for section, key, value in mappings:
        if value is not None:
            overrides[section][key] = value
    return load_config(
        args.config,
        cli_overrides={k: v for k, v in overrides.items() if v},
        dotted_overrides=parse_set_args(getattr(args, "set_args", None)),
    )


def require_dataset_csv(config: Config) -> Path:
    if config.data.csv_path is None:
        raise SystemExit(
            "No dataset CSV. Pass --dataset-csv, or set data.csv_path in the config. "
            "Build one with 'ais-build-dataset'."
        )
    path = Path(config.data.csv_path)
    if not path.exists():
        raise SystemExit(f"Dataset CSV not found: {path}")
    return path
