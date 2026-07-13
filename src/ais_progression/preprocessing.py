"""Image preprocessing: CLAHE contrast enhancement + square padding.

Handles non-ASCII (e.g. Japanese) file paths on Windows, where cv2.imread/imwrite
fail silently because they only accept ASCII paths internally.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def imread_gray(path: str | Path) -> np.ndarray | None:
    """Read an image as grayscale, tolerating non-ASCII paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return img


def imwrite_png(path: str | Path, image: np.ndarray) -> None:
    """Write an image as PNG, tolerating non-ASCII paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image for: {path}")
    buf.tofile(str(path))


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to a grayscale uint8 image (H, W)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def pad_to_square(image: np.ndarray, fill_value: int = 0) -> np.ndarray:
    """Center the image on a square canvas sized to the longer side (aspect-ratio preserving)."""
    h, w = image.shape[:2]
    size = max(h, w)
    canvas = np.full((size, size), fill_value, dtype=image.dtype)
    top = (size - h) // 2
    left = (size - w) // 2
    canvas[top : top + h, left : left + w] = image
    return canvas


def preprocess_image(
    input_path: str | Path,
    output_path: str | Path,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> None:
    """Load one image, apply CLAHE + square padding, and save as PNG."""
    img = imread_gray(input_path)
    if img is None:
        raise RuntimeError(f"Could not load image: {input_path}")
    img = apply_clahe(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
    img = pad_to_square(img)
    imwrite_png(output_path, img)


def _relative_output_path(input_path: Path, base_dir: Path, output_dir: Path) -> Path:
    try:
        rel = input_path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        rel = Path(input_path.name)
    return output_dir / rel.with_suffix(".png")


def preprocess_dataset(
    output_dir: str | Path,
    input_csv: str | Path | None = None,
    input_dir: str | Path | None = None,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    skip_errors: bool = False,
) -> pd.DataFrame:
    """Preprocess a set of images from either a CSV (image_path[, label] columns)
    or a directory (recursively globbed for image files).

    Returns a DataFrame with the processed `image_path` column (and `label` if the
    input CSV had one). The relative directory structure of the input is preserved
    under `output_dir`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if (input_csv is None) == (input_dir is None):
        raise ValueError("Exactly one of input_csv or input_dir must be provided.")

    rows: list[dict] = []
    seen_destinations: set[Path] = set()

    if input_csv is not None:
        input_csv = Path(input_csv)
        df = pd.read_csv(input_csv)
        if "image_path" not in df.columns:
            raise ValueError(f"CSV must contain an 'image_path' column: {input_csv}")
        base_dir = input_csv.parent
        metadata_columns = [column for column in df.columns if column != "image_path"]

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
            src = Path(str(row["image_path"]).strip())
            if not src.is_absolute():
                src = base_dir / src
            dst = _relative_output_path(src, base_dir, output_dir)
            resolved_dst = dst.resolve()
            if resolved_dst in seen_destinations:
                raise FileExistsError(f"Multiple inputs map to the same output path: {dst}")
            seen_destinations.add(resolved_dst)
            try:
                preprocess_image(src, dst, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
            except RuntimeError as e:
                if skip_errors:
                    print(f"Warning: {e}. Skipping.")
                    continue
                raise
            # Written as an absolute path: the output CSV may later be read from a
            # different working directory than the one preprocessing ran in, and
            # downstream loaders resolve relative image_path entries against the
            # CSV's own directory (not the cwd), so a relative path here would be
            # ambiguous.
            out_row = {"image_path": str(dst.resolve())}
            out_row.update({column: row[column] for column in metadata_columns})
            rows.append(out_row)
    else:
        input_dir = Path(input_dir)
        src_paths = sorted(
            p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for src in tqdm(src_paths, desc="Preprocessing"):
            dst = _relative_output_path(src, input_dir, output_dir)
            resolved_dst = dst.resolve()
            if resolved_dst in seen_destinations:
                raise FileExistsError(f"Multiple inputs map to the same output path: {dst}")
            seen_destinations.add(resolved_dst)
            try:
                preprocess_image(src, dst, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
            except RuntimeError as e:
                if skip_errors:
                    print(f"Warning: {e}. Skipping.")
                    continue
                raise
            rows.append({"image_path": str(dst.resolve())})

    return pd.DataFrame(rows)
