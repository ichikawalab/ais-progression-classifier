"""Shared helpers: seeding, device selection, hashing, run metadata, JSON output."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and torch, and optionally pin deterministic kernels."""
    import random

    import pytorch_lightning as pl

    if deterministic:
        # Required for deterministic cuBLAS matmul on CUDA >= 10.2. Must be set
        # before the first CUDA context is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_matmul_precision(precision: str) -> None:
    """Set the fp32 matmul mode, process-wide.

    On CUDA Ampere and later, "high" lets fp32 matmuls run as TensorFloat-32:
    much faster, with a 10-bit rather than 24-bit mantissa. "highest" keeps true
    fp32. It has no effect on CPU or on pre-Ampere GPUs.

    The published runs enabled "high" globally, so it applied to training *and*
    to the fp32 inference that produced the reported probabilities. Call this
    from every entry point that touches torch, so a model never scores new
    patients under different numerics than it was validated with.
    """
    torch.set_float32_matmul_precision(precision)


def resolve_precision(requested: str) -> str:
    """Downgrade a mixed-precision request to something the hardware supports.

    Inference always runs in fp32 regardless of this setting.
    """
    if requested in {"16-mixed", "bf16-mixed"} and not torch.cuda.is_available():
        return "32-true"
    if (
        requested == "bf16-mixed"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        return "16-mixed"
    return requested


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_safe(value: Any) -> Any:
    """Convert a value into something strict JSON can represent.

    NaN and infinity become null, and NumPy scalars/arrays become plain Python.
    ``json.dumps`` would otherwise emit bare ``NaN``/``Infinity`` tokens, which
    Python reads back happily but which are not valid JSON -- so an artefact
    written here would fail to parse in any other language.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data: Any, path: str | Path) -> None:
    """Write strict JSON. Raises rather than emitting a non-finite literal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(payload, encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Content digest of a file, so a result can be tied to the exact input."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(csv_path: str | Path, dataset) -> dict:
    """Identify the exact cohort a run consumed."""
    labels = dataset["label"] if "label" in dataset.columns else None
    return {
        "path": str(csv_path),
        "sha256": file_sha256(csv_path),
        "n_patients": int(len(dataset)),
        "n_progression": int((labels == 1).sum()) if labels is not None else None,
        "n_non_progression": int((labels == 0).sum()) if labels is not None else None,
    }


def warn_if_not_reproducible(precision: str, deterministic: bool) -> None:
    """Say plainly what ``deterministic`` does and does not guarantee."""
    if not deterministic:
        return
    if precision != "32-true":
        warnings.warn(
            f"train.deterministic is set but train.precision is '{precision}'. "
            "Mixed precision is not bit-reproducible across GPUs; use '32-true' "
            "for strict reproducibility.",
            RuntimeWarning,
            stacklevel=2,
        )
    # torch.use_deterministic_algorithms runs with warn_only=True, so any op
    # without a deterministic kernel proceeds after printing a warning.
    warnings.warn(
        "Deterministic mode is best-effort: operations without a deterministic "
        "implementation fall back to a non-deterministic one and only warn.",
        RuntimeWarning,
        stacklevel=2,
    )


def environment_report(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Capture the software/hardware context needed to interpret a run."""
    # Local import avoids a module cycle: provenance itself uses file_sha256.
    from ais_progression.provenance import software_identity

    software = software_identity()
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": software["packages"],
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": software["git_commit"],
        "git_dirty": software["git_dirty"],
        "source_tree_sha256": software["source_tree_sha256"],
    }
    report.update(extra or {})
    return report
