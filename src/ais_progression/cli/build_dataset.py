"""ais-build-dataset: turn the source workbooks into the unified dataset CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

from ais_progression.data.build import build_dataset
from ais_progression.data.schema import relativize_paths
from ais_progression.utils import save_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join the clinical workbook and the two image-path workbooks into one "
            "patient-level CSV: patient_id, front_path, lateral_path, age, sex, "
            "risser, cobb_baseline, label."
        )
    )
    parser.add_argument("--data-dir", default="data", help="Root holding the source files.")
    parser.add_argument("--clinical-xlsx", default=None, help="Clinical workbook.")
    parser.add_argument("--front-xlsx", default=None, help="Frontal InputPath workbook.")
    parser.add_argument("--lateral-xlsx", default=None, help="Lateral InputPath workbook.")
    parser.add_argument("--front-root", default=None, help="Directory holding frontal images.")
    parser.add_argument("--lateral-root", default=None, help="Directory holding lateral images.")
    parser.add_argument("--output-csv", default=None, help="Where to write the dataset CSV.")
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute image paths. By default they are relative to the "
        "output CSV, which keeps the dataset portable when the cohort moves.",
    )
    return parser


# Where to look for each workbook, relative to --data-dir. Globbed rather than
# hard-coded so the cohort can be reorganised into subdirectories without
# breaking the command; the first match in listed order wins.
WORKBOOK_PATTERNS = {
    "clinical_xlsx": ("Clinical_Info*.xlsx", "**/Clinical_Info*.xlsx"),
    "front_xlsx": ("Front/**/InputPath*.xlsx", "Front/**/*Front*.xlsx"),
    "lateral_xlsx": ("Lateral/**/InputPath*.xlsx", "Lateral/**/*Lateral*.xlsx"),
}
IMAGE_ROOTS = {"front_root": "Front", "lateral_root": "Lateral"}


def find_workbook(data_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        matches = sorted(data_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def resolve_sources(args) -> dict[str, Path]:
    """Resolve every input path, preferring explicit flags over discovery."""
    data_dir = Path(args.data_dir)
    resolved: dict[str, Path] = {}
    not_found: list[str] = []

    for key, patterns in WORKBOOK_PATTERNS.items():
        override = getattr(args, key)
        if override:
            resolved[key] = Path(override)
            continue
        found = find_workbook(data_dir, patterns)
        if found is None:
            not_found.append(
                f"{key}: no match for {' or '.join(patterns)} under {data_dir}"
            )
        else:
            resolved[key] = found

    for key, subdir in IMAGE_ROOTS.items():
        resolved[key] = Path(getattr(args, key) or data_dir / subdir)

    resolved["output_csv"] = Path(args.output_csv or data_dir / "dataset.csv")

    missing = [
        f"{key}: {path}"
        for key, path in resolved.items()
        if key != "output_csv" and not path.exists()
    ]
    if not_found or missing:
        raise SystemExit(
            "Could not locate every input:\n  "
            + "\n  ".join([*not_found, *missing])
            + "\nPass the paths explicitly with --clinical-xlsx / --front-xlsx / "
            "--lateral-xlsx / --front-root / --lateral-root."
        )
    return resolved


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    resolved = resolve_sources(args)
    for key in ("clinical_xlsx", "front_xlsx", "lateral_xlsx"):
        print(f"  {key}: {resolved[key]}")

    dataset, report = build_dataset(
        clinical_xlsx=resolved["clinical_xlsx"],
        front_xlsx=resolved["front_xlsx"],
        lateral_xlsx=resolved["lateral_xlsx"],
        front_root=resolved["front_root"],
        lateral_root=resolved["lateral_root"],
    )
    output_csv = resolved["output_csv"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not args.absolute_paths:
        dataset = relativize_paths(dataset, output_csv)
    dataset.to_csv(output_csv, index=False)
    save_json(report, output_csv.with_name(output_csv.stem + "_report.json"))

    cohort = report["cohort"]
    print(f"Wrote {output_csv} ({report['n_kept']} patients, {report['n_dropped']} dropped)")
    print(
        f"  progression {cohort.get('n_progression')} / "
        f"non-progression {cohort.get('n_non_progression')}, "
        f"female {cohort.get('n_female')}"
    )
    if report["n_dropped"]:
        print(f"  dropped: {report['dropped_patient_ids'][:10]}")


if __name__ == "__main__":
    main()
