"""ais-cv-ensemble: nested cross-validation of a multimodal ensemble."""
from __future__ import annotations

import argparse
from pathlib import Path

from ais_progression.cli._common import (
    add_config_arguments,
    add_data_arguments,
    build_config,
    require_dataset_csv,
)
from ais_progression.config import ENSEMBLE_METHODS, save_config
from ais_progression.data.schema import load_dataset
from ais_progression.evaluation import format_auc
from ais_progression.experiments.ensemble_cv import (
    ensemble_run_identity,
    load_base_predictions,
    run_ensemble_cv,
)
from ais_progression.experiments.reporting import assert_run_identity
from ais_progression.utils import dataset_fingerprint, environment_report, save_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the individual models' out-of-fold probabilities. "
            "'weighted' is the best-performing method; 'average', "
            "'logreg', 'svm' and 'rf' are the comparators."
        )
    )
    parser.add_argument("--method", required=True, choices=ENSEMBLE_METHODS)
    parser.add_argument(
        "--cv-dir",
        default=None,
        help="Directory holding the modality-CV runs (default <output.dir>/cv). "
        "Every subdirectory with a predictions.csv is used as a base model.",
    )
    parser.add_argument(
        "--base",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="Use an explicit base model instead of scanning --cv-dir (repeatable).",
    )
    add_config_arguments(parser)
    add_data_arguments(parser)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def resolve_base_sources(args: argparse.Namespace, default_cv_dir: Path) -> dict[str, Path]:
    """Collect the base models, either from --base entries or by scanning --cv-dir."""
    if args.base:
        sources: dict[str, Path] = {}
        for entry in args.base:
            if "=" not in entry:
                raise SystemExit(f"--base expects NAME=PATH, got '{entry}'")
            name, path = entry.split("=", 1)
            sources[name.strip()] = Path(path.strip())
        return sources

    cv_dir = Path(args.cv_dir) if args.cv_dir else default_cv_dir
    if not cv_dir.exists():
        raise SystemExit(
            f"{cv_dir} does not exist. Run 'ais-cv-modality' for the individual models first, "
            "or pass --base NAME=PATH."
        )
    found = {
        child.name: child
        for child in sorted(cv_dir.iterdir())
        if child.is_dir() and (child / "predictions.csv").exists()
    }
    if not found:
        raise SystemExit(f"No completed modality-CV runs under {cv_dir}.")
    return found


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = build_config(args)
    dataset_csv = require_dataset_csv(config)
    dataset = load_dataset(
        dataset_csv,
        required_modalities=(),
        required_clinical_features=(),
        check_files=False,
    )

    sources = resolve_base_sources(args, Path(config.output.dir) / "cv")
    base_predictions = load_base_predictions(sources)
    print(f"Base models ({len(base_predictions)}): {', '.join(sorted(base_predictions))}")

    run_dir = Path(args.run_dir or Path(config.output.dir) / "ensemble" / args.method)
    run_dir.mkdir(parents=True, exist_ok=True)
    assert_run_identity(
        run_dir, ensemble_run_identity(config, dataset, base_predictions, args.method)
    )
    save_config(config, run_dir / "config.yaml")
    save_json(
        environment_report(
            {
                "method": args.method,
                "base_models": {name: str(path) for name, path in sources.items()},
                "dataset": dataset_fingerprint(dataset_csv, dataset),
            }
        ),
        run_dir / "environment.json",
    )
    print(f"Run directory: {run_dir}")

    summary = run_ensemble_cv(
        config=config,
        dataset=dataset,
        base_predictions=base_predictions,
        method=args.method,
        run_dir=run_dir,
        resume=not args.no_resume,
    )
    test = summary["test_auc_pooled_per_rep"]
    print(
        f"Done. Test AUC {format_auc(test['mean'])} +/- {format_auc(test['sd'])} "
        f"(pooled per repetition)"
    )


if __name__ == "__main__":
    main()
