"""ais-preprocess: CLAHE + square padding for both radiograph modalities."""
from __future__ import annotations

import argparse
from pathlib import Path

from ais_progression.data.preprocess import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    preprocess_dataset,
)
from ais_progression.data.schema import load_dataset, relativize_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply CLAHE and zero-pad each radiograph to a square, then write a new "
            "dataset CSV pointing at the processed images. Do not run this twice on "
            "the same cohort: CLAHE is not idempotent."
        )
    )
    parser.add_argument("--dataset-csv", required=True, help="Unified dataset CSV.")
    parser.add_argument("--output-dir", required=True, help="Where processed PNGs go.")
    parser.add_argument("--output-csv", required=True, help="Dataset CSV for the processed images.")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["front", "lateral"],
        choices=["front", "lateral"],
    )
    parser.add_argument("--clip-limit", type=float, default=CLAHE_CLIP_LIMIT)
    parser.add_argument("--tile-grid", type=int, default=CLAHE_TILE_GRID[0])
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute image paths instead of paths relative to the output CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.clip_limit <= 0 or args.tile_grid < 1:
        raise SystemExit("--clip-limit must be positive and --tile-grid at least 1")

    dataset = load_dataset(
        args.dataset_csv, require_labels=False, required_clinical_features=()
    )
    processed = preprocess_dataset(
        dataset,
        output_dir=args.output_dir,
        modalities=tuple(args.modalities),
        clip_limit=args.clip_limit,
        tile_grid_size=(args.tile_grid, args.tile_grid),
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not args.absolute_paths:
        processed = relativize_paths(processed, output_csv)
    processed.to_csv(output_csv, index=False)
    print(f"Preprocessed {len(processed)} patient(s). Wrote {output_csv}")


if __name__ == "__main__":
    main()
