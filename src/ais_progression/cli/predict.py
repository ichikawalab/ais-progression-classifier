"""ais-predict: score new patients with a trained model bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ais_progression.data.schema import (
    ID_COLUMN,
    LABEL_COLUMN,
    load_dataset,
    missing_feature_report,
)
from ais_progression.evaluation import binary_metrics, format_auc, safe_auc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict progression for new patients using a bundle from "
            "'ais-train-final'. The input CSV uses the same schema as training; "
            "the label column is optional and, when present, is scored."
        )
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--input-csv")
    parser.add_argument("--output-csv")
    parser.add_argument(
        "--profile",
        default=None,
        help="Serving profile to run. Defaults to the bundle's default profile.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the profile's cross-validated decision threshold. Doing so "
        "invalidates the sensitivity and specificity reported for the profile.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Impute missing clinical variables with the training medians instead "
        "of refusing. Imputed fields are listed in the output.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print the bundle's profiles and exit.",
    )
    return parser


def _load_manifest(bundle_dir: str | Path) -> dict:
    path = Path(bundle_dir) / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"Model bundle manifest not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _print_profiles(manifest: dict) -> None:
    default = manifest.get("default_profile")
    profiles = sorted(manifest.get("profiles", []), key=lambda profile: profile["name"])
    for profile in profiles:
        name = profile["name"]
        marker = " (default)" if name == default else ""
        cv = profile.get("cv_metrics", {})
        auc = "n/a" if cv.get("auc_mean") is None else f"{cv['auc_mean']:.3f}"
        threshold = profile["operating_point"]["threshold"]
        members = profile["members"]
        print(
            f"{name}{marker}: {len(members)} model(s), CV AUC {auc}, "
            f"threshold {threshold:.3f}"
        )
        print(f"    {', '.join(members)}")


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be in [0, 1]")

    if args.list_profiles:
        _print_profiles(_load_manifest(args.bundle_dir))
        return
    if not args.input_csv or not args.output_csv:
        parser.error("--input-csv and --output-csv are required for prediction")

    from ais_progression.final.bundle import ModelBundle

    bundle = ModelBundle.load(args.bundle_dir)

    if args.batch_size is not None:
        bundle.config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        bundle.config.data.num_workers = args.num_workers

    profile = bundle.profile(args.profile)
    members = {member.name: member for member in bundle.members}
    required_modalities = tuple(
        sorted(
            {
                members[name].modality
                for name in profile.members
                if members[name].is_image
            }
        )
    )
    needs_clinical = any(not members[name].is_image for name in profile.members)
    required_clinical = bundle.config.clinical.features if needs_clinical else []

    header = pd.read_csv(args.input_csv, nrows=0)
    has_labels = LABEL_COLUMN in header.columns
    frame = load_dataset(
        args.input_csv,
        require_labels=has_labels,
        required_modalities=required_modalities,
        required_clinical_features=required_clinical,
        allow_missing_features=args.allow_missing,
    )
    if frame.empty:
        raise SystemExit("The input CSV contains no rows.")

    print(f"Scoring {len(frame)} patient(s) with profile '{profile.name}'")
    bundle.warmup(profile.name)
    predictions = bundle.predict(frame, profile.name)

    if args.threshold is not None:
        predictions["predicted_label"] = (
            predictions["probability"] >= args.threshold
        ).astype(int)
        predictions["threshold"] = args.threshold
        print(
            f"Using the overridden threshold {args.threshold}; the profile's "
            "reported sensitivity and specificity no longer apply."
        )

    output = pd.concat(
        [frame[[ID_COLUMN]].reset_index(drop=True), predictions.reset_index(drop=True)],
        axis=1,
    )
    output["imputed_fields"] = missing_feature_report(frame, required_clinical).to_numpy()
    if has_labels:
        output.insert(1, LABEL_COLUMN, frame[LABEL_COLUMN].to_numpy())

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")

    imputed = int((output["imputed_fields"] != "").sum())
    if imputed:
        print(f"WARNING: {imputed} patient(s) had clinical variables imputed.")

    if has_labels:
        print(f"Ensemble AUC: {format_auc(safe_auc(frame[LABEL_COLUMN], output['probability']))}")
        for name in profile.members:
            print(f"  {name:<22} {format_auc(safe_auc(frame[LABEL_COLUMN], output[name]))}")
        metrics = binary_metrics(
            frame[LABEL_COLUMN],
            output["probability"],
            threshold=float(output["threshold"].iloc[0]),
        )
        print(
            f"At threshold {metrics['threshold']:.3f}: "
            f"sensitivity {metrics['sensitivity']}, specificity {metrics['specificity']}, "
            f"accuracy {metrics['accuracy']:.3f}"
        )


if __name__ == "__main__":
    main()
