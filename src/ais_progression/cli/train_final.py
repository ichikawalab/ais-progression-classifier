"""ais-train-final: train the deployable model bundle on the whole cohort."""
from __future__ import annotations

import argparse
from pathlib import Path

from ais_progression.cli._common import (
    add_config_arguments,
    add_cpu_argument,
    add_data_arguments,
    build_config,
    require_dataset_csv,
    require_gpu,
)
from ais_progression.config import (
    CLINICAL_MODALITY,
    CLINICAL_MODELS,
    IMAGE_MODALITIES,
    IMAGE_MODELS,
)
from ais_progression.data.schema import load_dataset
from ais_progression.experiments.ensemble_cv import load_base_predictions
from ais_progression.final.operating_point import CALIBRATION_METHODS
from ais_progression.final.profiles import (
    DEFAULT_PROFILE_MEMBERS,
    FULL_PROFILE,
    build_profiles,
    select_members,
)
from ais_progression.final.train import (
    resolve_epoch_plan,
    train_final_model,
    validate_cv_compatibility,
)
from ais_progression.utils import dataset_fingerprint, environment_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train every base model on the entire cohort and bundle them with one "
            "or more serving profiles. Epoch counts, ensemble weights, decision "
            "thresholds and calibrators all come from the cross-validation runs, "
            "so no data has to be held back here."
        )
    )
    add_config_arguments(parser)
    add_data_arguments(parser)
    parser.add_argument("--bundle-dir", default=None, help="Where to write the bundle.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reps", type=int, default=None, help="Repetitions to read.")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--cv-dir",
        default=None,
        help="Modality-CV runs to derive epochs, weights and thresholds from "
        "(default <output.dir>/cv).",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=None,
        metavar="NAME=MODALITIES",
        help="Serving profile, e.g. --profile cheap=clinical or "
        f"--profile mixed=front,clinical. Default: {', '.join(DEFAULT_PROFILE_MEMBERS)}.",
    )
    parser.add_argument(
        "--default-profile",
        default=FULL_PROFILE,
        help="Profile used when a caller does not name one.",
    )
    parser.add_argument(
        "--threshold-policy",
        default="youden",
        choices=("youden", "target_sensitivity"),
        help="How the decision threshold is chosen from cross-validation.",
    )
    parser.add_argument("--target-sensitivity", type=float, default=0.90)
    parser.add_argument("--calibration", default="isotonic", choices=CALIBRATION_METHODS)
    parser.add_argument("--image-models", nargs="+", default=list(IMAGE_MODELS))
    parser.add_argument("--clinical-models", nargs="+", default=list(CLINICAL_MODELS))
    parser.add_argument(
        "--modalities", nargs="+", default=list(IMAGE_MODALITIES), choices=list(IMAGE_MODALITIES)
    )
    parser.add_argument(
        "--save-full-checkpoints",
        action="store_true",
        help="Store full Lightning checkpoints. By default only the model weights "
        "(state_dict) are written, which is smaller and loads without unpickling.",
    )
    add_cpu_argument(parser)
    return parser


def parse_profiles(entries: list[str] | None) -> dict[str, tuple[str, ...] | None]:
    if not entries:
        return dict(DEFAULT_PROFILE_MEMBERS)
    requested: dict[str, tuple[str, ...] | None] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"--profile expects NAME=MODALITIES, got '{entry}'")
        name, modalities = entry.split("=", 1)
        parsed = tuple(part.strip() for part in modalities.split(",") if part.strip())
        valid = {*IMAGE_MODALITIES, CLINICAL_MODALITY}
        unknown = sorted(set(parsed) - valid)
        if unknown:
            raise SystemExit(f"Profile '{name}' names unknown modalities: {unknown}")
        requested[name.strip()] = parsed or None
    return requested


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.threshold_policy == "target_sensitivity" and not 0 < args.target_sensitivity <= 1:
        raise SystemExit("--target-sensitivity must be in (0, 1]")
    # Only a profile that names an image modality causes image training; a
    # clinical-only bundle is scikit-learn throughout and needs no GPU. ``None``
    # means "every available base model", which may well include images.
    requested_profiles = parse_profiles(args.profile)
    if any(
        modalities is None or any(m != CLINICAL_MODALITY for m in modalities)
        for modalities in requested_profiles.values()
    ):
        require_gpu(args.allow_cpu)
    config = build_config(args)
    dataset_csv = require_dataset_csv(config)

    cv_dir = Path(args.cv_dir) if args.cv_dir else Path(config.output.dir) / "cv"
    sources = {
        child.name: child
        for child in sorted(cv_dir.glob("*"))
        if child.is_dir() and (child / "predictions.csv").exists()
    }
    if not sources:
        raise SystemExit(
            f"No completed modality-CV runs under {cv_dir}. Run 'ais-cv-modality' for "
            "the individual models first."
        )
    base_predictions = load_base_predictions(sources)
    print(f"Base models from cross-validation: {', '.join(sorted(base_predictions))}")

    selected_members = {
        member
        for modalities in requested_profiles.values()
        for member in select_members(list(base_predictions), modalities)
    }
    required_modalities = tuple(
        modality
        for modality in IMAGE_MODALITIES
        if any(name.startswith(f"{modality}_") for name in selected_members)
    )
    needs_clinical = any(
        name.startswith(f"{CLINICAL_MODALITY}_") for name in selected_members
    )
    dataset = load_dataset(
        dataset_csv,
        required_modalities=required_modalities,
        required_clinical_features=config.clinical.features if needs_clinical else (),
    )

    bundle_dir = Path(args.bundle_dir or Path(config.output.dir) / "final")
    selected_sources = {name: sources[name] for name in selected_members}
    try:
        cv_identities = validate_cv_compatibility(config, dataset, selected_sources)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    profiles = build_profiles(
        requested=requested_profiles,
        config=config,
        dataset=dataset,
        base_predictions=base_predictions,
        ensemble_root=Path(config.output.dir) / "ensemble",
        threshold_policy=args.threshold_policy,
        target_sensitivity=args.target_sensitivity,
        calibration=args.calibration,
    )
    available = {profile.name for profile in profiles}
    if args.default_profile not in available:
        raise SystemExit(
            f"--default-profile '{args.default_profile}' is not among the built "
            f"profiles: {sorted(available)}"
        )

    image_members = sorted(
        name for name in selected_members if not name.startswith(f"{CLINICAL_MODALITY}_")
    )
    print("Epoch plan (median best epoch from cross-validation):")
    try:
        epoch_plan = resolve_epoch_plan(cv_dir, image_members)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    for name, epochs in epoch_plan.items():
        print(f"  {name}: {epochs}")

    metrics = train_final_model(
        config=config,
        dataset=dataset,
        bundle_dir=bundle_dir,
        profiles=profiles,
        cv_identities=cv_identities,
        environment=environment_report(
            {"dataset": dataset_fingerprint(dataset_csv, dataset)}
        ),
        epoch_plan=epoch_plan,
        image_models=tuple(args.image_models),
        clinical_models=tuple(args.clinical_models),
        image_modalities=tuple(args.modalities),
        weights_only=not args.save_full_checkpoints,
        default_profile=args.default_profile,
    )
    print(f"\nBundle: {bundle_dir}")
    print("Cross-validated performance by profile")
    for name, entry in metrics["profiles"].items():
        cv = entry["cv_metrics"]
        point = entry["operating_point"]
        auc = "n/a" if cv.get("auc_mean") is None else f"{cv['auc_mean']:.3f}"
        sensitivity = point.get("sensitivity_mean")
        specificity = point.get("specificity_mean")
        print(
            f"  {name:<16} AUC {auc}  threshold {point['threshold']:.3f}  "
            f"sens {sensitivity:.3f}  spec {specificity:.3f}"
            if sensitivity is not None and specificity is not None
            else f"  {name:<16} AUC {auc}  threshold {point['threshold']:.3f}"
        )


if __name__ == "__main__":
    main()
