"""Radiograph preprocessing: CLAHE, then zero-padding to a square canvas.

The reference preprocessing. Intensity normalisation with the
ImageNet mean/std happens later, inside the training transform.

cv2.imread/imwrite silently fail on non-ASCII (e.g. Japanese) paths on Windows,
so all I/O goes through np.fromfile/tofile.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ais_progression.data.schema import IMAGE_COLUMNS

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


def imread_gray(path: str | Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def imwrite_png(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for: {path}")
    buffer.tofile(str(path))


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = CLAHE_TILE_GRID,
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on a grayscale uint8 image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def pad_to_square(image: np.ndarray, fill_value: int = 0) -> np.ndarray:
    """Centre the image on a square canvas sized to its longer side."""
    height, width = image.shape[:2]
    size = max(height, width)
    canvas = np.full((size, size), fill_value, dtype=image.dtype)
    top = (size - height) // 2
    left = (size - width) // 2
    canvas[top : top + height, left : left + width] = image
    return canvas


def preprocess_image(
    input_path: str | Path,
    output_path: str | Path,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = CLAHE_TILE_GRID,
) -> None:
    """CLAHE + square padding for one radiograph, written out as PNG."""
    image = imread_gray(input_path)
    if image is None:
        raise RuntimeError(f"Could not load image: {input_path}")
    image = apply_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    imwrite_png(output_path, pad_to_square(image))


def preprocess_dataset(
    dataset: pd.DataFrame,
    output_dir: str | Path,
    modalities: tuple[str, ...] = ("front", "lateral"),
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple[int, int] = CLAHE_TILE_GRID,
) -> pd.DataFrame:
    """Preprocess every radiograph referenced by the dataset.

    Outputs land at ``<output_dir>/<modality>/<patient_id>.png``, and the
    returned copy of the dataset points at those files. Applying CLAHE twice
    degrades the image, so run this at most once per cohort.
    """
    output_dir = Path(output_dir)
    result = dataset.copy()
    for modality in modalities:
        column = IMAGE_COLUMNS[modality]
        destination_dir = output_dir / modality
        outputs: list[str] = []
        rows = list(result[["patient_id", column]].itertuples(index=False))
        for patient_id, source in tqdm(rows, desc=f"Preprocessing {modality}"):
            destination = destination_dir / f"{patient_id}.png"
            preprocess_image(
                source, destination, clip_limit=clip_limit, tile_grid_size=tile_grid_size
            )
            outputs.append(str(destination.resolve()))
        result[column] = outputs
    return result
