"""Shared helpers: seeding, device selection, run-directory management."""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

import pytorch_lightning as pl
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    if deterministic:
        # Required for deterministic cuBLAS matmul on CUDA >= 10.2; without it,
        # torch.use_deterministic_algorithms leaves those ops non-deterministic
        # (and would raise instead of warn if warn_only were False). Set before
        # the first CUDA context is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_precision(requested: str) -> str:
    """Resolve the training precision to something the current hardware supports.

    - Any mixed precision on CPU falls back to "32-true" (AMP needs CUDA).
    - "bf16-mixed" on a CUDA GPU without bfloat16 support falls back to
      "16-mixed", since bf16 is emulated (slow) on pre-Ampere hardware.
    Inference/evaluation always runs in fp32 regardless of this setting.
    """
    mixed = {"16-mixed", "bf16-mixed"}
    if requested in mixed and not torch.cuda.is_available():
        return "32-true"
    if (
        requested == "bf16-mixed"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        return "16-mixed"
    return requested


def make_run_dir(output_dir: str | Path, run_name: str | None, arch: str) -> Path:
    """Create (and return) a unique run directory under output_dir.

    If run_name is None, auto-generate "{arch}_{YYYYmmdd-HHMMSS}".
    If the resulting directory already exists, a numeric suffix is appended
    to avoid clobbering a previous run.
    """
    output_dir = Path(output_dir)
    if run_name is None:
        timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{arch}_{timestamp}"

    run_dir = output_dir / run_name
    if run_dir.exists():
        i = 1
        while (output_dir / f"{run_name}_{i}").exists():
            i += 1
        run_dir = output_dir / f"{run_name}_{i}"

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_checkpoint_path(run_dir: str | Path, checkpoint: str) -> Path:
    """Resolve "best" / "last" / an explicit path to a concrete .ckpt file."""
    run_dir = Path(run_dir)
    if checkpoint in ("best", "last"):
        ckpt_path = run_dir / "checkpoints" / f"{checkpoint}.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path
