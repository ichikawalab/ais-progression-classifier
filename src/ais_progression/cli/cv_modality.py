"""ais-cv-modality: repeated cross-validation for one modality x one model."""
from __future__ import annotations

import argparse
from pathlib import Path

from ais_progression.cli._common import (
    add_config_arguments,
    add_data_arguments,
    build_config,
    require_dataset_csv,
)
from ais_progression.config import (
    CLINICAL_MODELS,
    IMAGE_MODELS,
    MODALITIES,
    is_image_modality,
    models_for_modality,
    save_config,
)
from ais_progression.data.schema import load_dataset
from ais_progression.evaluation import format_auc
from ais_progression.experiments.modality_cv import run_modality_cv, run_name
from ais_progression.experiments.reporting import assert_run_identity
from ais_progression.provenance import modality_run_identity
from ais_progression.utils import (
    dataset_fingerprint,
    environment_report,
    save_json,
    warn_if_not_reproducible,
)

ALL_MODELS = sorted({*IMAGE_MODELS, *CLINICAL_MODELS})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repeated stratified cross-validation for one of the nine individual "
            "models. Image modalities take vit/swint/convnextv2; the clinical "
            "modality takes logreg/svm/rf."
        )
    )
    parser.add_argument("--modality", required=True, choices=MODALITIES)
    # Not a fixed choice list: image.archs is configurable, so the valid model
    # names depend on the resolved config and are checked in main().
    parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help=f"Model name. Built-in: {', '.join(ALL_MODELS)}.",
    )
    add_config_arguments(parser)
    add_data_arguments(parser)
    parser.add_argument("--reps", type=int, default=None, help="Repetitions (default 10).")
    parser.add_argument("--folds", type=int, default=None, help="Outer folds (default 10).")
    parser.add_argument("--seed", type=int, default=None, help="Base seed (default 42).")
    parser.add_argument("--output-dir", default=None, help="Root output directory.")
    parser.add_argument("--run-dir", default=None, help="Override the run directory.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute every fold instead of skipping completed ones.",
    )
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep each fold's weights. Off by default: only predictions are needed, "
        "and a full run would otherwise write hundreds of gigabytes.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = build_config(args)

    valid = models_for_modality(config, args.modality)
    if args.model not in valid:
        raise SystemExit(
            f"Model '{args.model}' is not available for modality '{args.modality}'. "
            f"Choose from: {sorted(valid)}"
        )

    needs_images = is_image_modality(args.modality)
    dataset_csv = require_dataset_csv(config)
    dataset = load_dataset(
        dataset_csv,
        required_modalities=(args.modality,) if needs_images else (),
        required_clinical_features=() if needs_images else config.clinical.features,
        check_files=needs_images,
    )
    warn_if_not_reproducible(config.train.precision, config.train.deterministic)

    run_dir = Path(
        args.run_dir or Path(config.output.dir) / "cv" / run_name(args.modality, args.model)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    assert_run_identity(
        run_dir, modality_run_identity(config, dataset, args.modality, args.model)
    )
    save_config(config, run_dir / "config.yaml")
    save_json(
        environment_report(
            {
                "modality": args.modality,
                "model": args.model,
                "dataset": dataset_fingerprint(dataset_csv, dataset),
                "precision": config.train.precision,
                "deterministic": config.train.deterministic,
            }
        ),
        run_dir / "environment.json",
    )
    print(f"Run directory: {run_dir}")

    summary = run_modality_cv(
        config=config,
        dataset=dataset,
        modality=args.modality,
        model=args.model,
        run_dir=run_dir,
        resume=not args.no_resume,
        keep_checkpoints=args.keep_checkpoints,
    )
    test = summary["test_auc_pooled_per_rep"]
    for source, stats in summary.get("selection_auc_by_source", {}).items():
        print(f"Done. Selection AUC {format_auc(stats['mean'])} ({source}, per fold)")
    print(
        f"      Test AUC {format_auc(test['mean'])} +/- {format_auc(test['sd'])} "
        f"(pooled per repetition)"
    )


if __name__ == "__main__":
    main()
