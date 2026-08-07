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
    overrides: dict[str, dict] = {
        "data": {},
        "cross_validation": {},
        "final": {},
        "output": {},
    }
    mappings = (
        ("data", "csv_path", getattr(args, "dataset_csv", None)),
        ("data", "batch_size", getattr(args, "batch_size", None)),
        ("data", "num_workers", getattr(args, "num_workers", None)),
        ("cross_validation", "num_reps", getattr(args, "reps", None)),
        ("cross_validation", "num_folds", getattr(args, "folds", None)),
        ("cross_validation", "seed", getattr(args, "cv_seed", None)),
        ("final", "seed", getattr(args, "final_seed", None)),
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


def add_cpu_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Train image models without a GPU. Refused by default because a "
        "full run takes days on a GPU and would not finish on a CPU.",
    )


def require_gpu(allow_cpu: bool) -> None:
    """Refuse to start image training on the CPU unless it was asked for.

    Torch reports a missing GPU and carries on, so a run that should take hours
    quietly becomes one that never finishes -- and nothing about the output says
    which it was. The usual cause on Windows is an environment that reverted to
    the CPU wheels: PyPI has no CUDA build there, and ``uv run`` re-syncs from
    the lock file before every command, so an installed CUDA build is replaced
    unless the command passes ``--no-sync``.
    """
    import torch

    if torch.cuda.is_available() or allow_cpu:
        return
    raise SystemExit(
        f"No GPU available to torch (installed build: {torch.__version__}). Image "
        "training would run on the CPU and take far longer than the protocol "
        "assumes.\n"
        "  If the build ends in '+cpu', reinstall the CUDA one:\n"
        "    uv pip install --reinstall torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu130\n"
        "  Then run commands with 'uv run --no-sync ...', or call "
        ".venv/Scripts/ directly: plain 'uv run' re-syncs and puts the CPU "
        "wheels back.\n"
        "  Pass --allow-cpu if you really mean to train on the CPU."
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
