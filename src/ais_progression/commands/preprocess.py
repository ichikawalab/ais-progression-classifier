"""Preprocess raster spine images with CLAHE and square padding."""
from __future__ import annotations

import argparse
from pathlib import Path

from ais_progression.preprocessing import preprocess_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess raster spine X-rays.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv")
    source.add_argument("--input-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tile-grid-size", type=int, default=8)
    parser.add_argument("--skip-errors", action="store_true")
    args = parser.parse_args(argv)
    if args.clip_limit <= 0 or args.tile_grid_size < 1:
        parser.error("clip-limit and tile-grid-size must be positive")
    result = preprocess_dataset(
        output_dir=args.output_dir,
        input_csv=args.input_csv,
        input_dir=args.input_dir,
        clip_limit=args.clip_limit,
        tile_grid_size=(args.tile_grid_size, args.tile_grid_size),
        skip_errors=args.skip_errors,
    )
    output_csv = Path(args.output_csv) if args.output_csv else Path(args.output_dir) / "processed_files.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Processed {len(result)} image(s). Saved: {output_csv}")
