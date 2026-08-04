"""ais-gradcam: Grad-CAM overlays from an image model inside a bundle."""
from __future__ import annotations

import argparse
from pathlib import Path

from ais_progression.config import resolve_arch
from ais_progression.data.schema import load_dataset
from ais_progression.final.bundle import ModelBundle, member_name
from ais_progression.gradcam import generate_gradcam
from ais_progression.models.image_model import load_image_classifier


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM heatmaps and overlays for one image model in a bundle. "
            "Exploratory only: saliency does not explain a prediction causally."
        )
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--modality", required=True, choices=["front", "lateral"])
    parser.add_argument("--model", required=True, help="Image model name, e.g. convnextv2.")
    parser.add_argument("--input-csv", required=True, help="Patients to visualise.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-class", choices=("pred", "0", "1"), default="pred")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap opacity.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N patients.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if not 0 <= args.alpha <= 1:
        raise SystemExit("--alpha must be in [0, 1]")

    bundle = ModelBundle.load(args.bundle_dir)
    name = member_name(args.modality, args.model)
    member = next((m for m in bundle.members if m.name == name), None)
    if member is None:
        available = sorted(m.name for m in bundle.members if m.is_image)
        raise SystemExit(f"'{name}' is not in this bundle. Available: {available}")

    # Grad-CAM only needs the radiographs, so missing clinical values are fine.
    frame = load_dataset(
        args.input_csv,
        require_labels=False,
        required_modalities=(args.modality,),
        required_clinical_features=(),
    )
    if args.limit:
        frame = frame.head(args.limit)
    if frame.empty:
        raise SystemExit("The input CSV contains no rows.")

    arch = resolve_arch(bundle.config, args.model)
    classifier = load_image_classifier(bundle.dir / member.artifact, bundle.config, arch)
    output_dir = Path(args.output_dir or bundle.dir / "gradcam" / name)
    written = generate_gradcam(
        classifier,
        bundle.config,
        arch,
        args.modality,
        frame,
        output_dir,
        target_class=args.target_class,
        alpha=args.alpha,
    )
    print(f"Wrote {len(written)} overlay(s) and heatmap(s) to {output_dir}")


if __name__ == "__main__":
    main()
