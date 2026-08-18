"""The single input CSV: one row per patient, all three modalities.

    patient_id,front_path,lateral_path,age,sex,risser,cobb_baseline,label

* ``patient_id``  non-identifying ID; unique within the file.
* ``front_path``  frontal whole-spine radiograph (absolute, or relative to the CSV).
* ``lateral_path`` lateral whole-spine radiograph.
* ``age``         age at the initial visit, in years.
* ``sex``         1 = male, 2 = female (the source workbook's ``Sex_M1_F2`` coding).
* ``risser``      Risser sign 0-5, treated as ordinal.
* ``cobb_baseline`` baseline Cobb angle in degrees.
* ``label``       0 = non-progression (<=5 deg), 1 = progression (>=10 deg).

Borderline patients (6-9 deg) are excluded upstream and must not appear here.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "patient_id"
LABEL_COLUMN = "label"
IMAGE_COLUMNS = {"front": "front_path", "lateral": "lateral_path"}
CLINICAL_COLUMNS = ["age", "sex", "risser", "cobb_baseline"]

SEX_CODES = (1, 2)  # 1 = male, 2 = female
# Plausible ranges for adolescent idiopathic scoliosis at the initial visit.
# The Risser and Cobb bounds are deliberately generous: they catch data-entry
# errors and unit mix-ups rather than restricting the cohort. The age bound is
# tighter on purpose, and does restrict the cohort to the adolescent range the
# models were trained on.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "age": (10.0, 18.0),
    "risser": (0.0, 5.0),
    "cobb_baseline": (0.0, 150.0),
}
DATASET_COLUMNS = [
    ID_COLUMN,
    IMAGE_COLUMNS["front"],
    IMAGE_COLUMNS["lateral"],
    *CLINICAL_COLUMNS,
    LABEL_COLUMN,
]


def image_column(modality: str) -> str:
    """Column holding the radiograph path for an image modality."""
    try:
        return IMAGE_COLUMNS[modality]
    except KeyError:
        raise ValueError(
            f"Unknown image modality '{modality}'. Available: {sorted(IMAGE_COLUMNS)}"
        ) from None


def resolve_paths(df: pd.DataFrame, csv_path: str | Path, columns: list[str]) -> pd.DataFrame:
    """Resolve relative image paths against the directory containing the CSV."""
    base_dir = Path(csv_path).parent

    def resolve(value: object) -> object:
        if pd.isna(value):
            return value
        path = Path(str(value).strip())
        return str(path if path.is_absolute() else base_dir / path)

    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = df[column].map(resolve)
    return df


def relativize_paths(
    df: pd.DataFrame, csv_path: str | Path, columns: list[str] | None = None
) -> pd.DataFrame:
    """Rewrite image paths relative to the directory the CSV will live in.

    Keeps the dataset portable: the CSV and the images move together, and
    ``load_dataset`` resolves the paths back against the CSV's own directory.
    Separators are normalised to forward slashes so a file written on Windows
    still resolves elsewhere. Falls back to the absolute path when no relative
    one exists (a different Windows drive).
    """
    base_dir = Path(csv_path).parent.resolve()

    def relativize(value: object) -> object:
        if pd.isna(value):
            return value
        absolute = Path(str(value).strip()).resolve()
        try:
            return Path(os.path.relpath(absolute, base_dir)).as_posix()
        except ValueError:
            return absolute.as_posix()

    columns = columns or [c for c in IMAGE_COLUMNS.values() if c in df.columns]
    df = df.copy()
    for column in columns:
        df[column] = df[column].map(relativize)
    return df


def missing_feature_report(
    df: pd.DataFrame, features: list[str] | tuple[str, ...] | None = None
) -> pd.Series:
    """Per row, the clinical columns that have no value. Empty string when none."""
    requested = CLINICAL_COLUMNS if features is None else features
    present = [column for column in requested if column in df.columns]
    if not present:
        return pd.Series("", index=df.index, dtype=str)
    return df[present].isna().apply(
        lambda row: ";".join(sorted(row.index[row])), axis=1
    )


def _validate_clinical_ranges(df: pd.DataFrame, source: str) -> None:
    """Reject physiologically impossible values.

    Without this, a mistyped Cobb angle or Risser sign produces a confident
    prediction rather than an error. The Risser check also keeps the ordinal
    encoder's unknown-value fallback (-1, which a linear model reads as below
    Risser 0) from ever firing.
    """
    if "sex" in df and not df["sex"].dropna().isin(SEX_CODES).all():
        invalid = sorted(set(df["sex"].dropna().unique()) - set(SEX_CODES))
        raise ValueError(f"'sex' must be coded 1 (male) or 2 (female), found: {invalid}")

    risser = df["risser"].dropna() if "risser" in df else pd.Series(dtype=float)
    if not risser.empty and not np.allclose(risser, risser.round()):
        raise ValueError("'risser' must be a whole number between 0 and 5.")

    for column, (low, high) in FEATURE_BOUNDS.items():
        if column not in df:
            continue
        values = df[column].dropna()
        out_of_range = values[(values < low) | (values > high)]
        if not out_of_range.empty:
            examples = sorted(out_of_range.unique().tolist())[:5]
            raise ValueError(
                f"{source}: '{column}' must be between {low} and {high}; "
                f"found {examples}. Fix the input or widen "
                f"ais_progression.data.schema.FEATURE_BOUNDS if the range is wrong."
            )


def load_dataset(
    csv_path: str | Path,
    require_labels: bool = True,
    required_modalities: tuple[str, ...] | None = None,
    required_clinical_features: list[str] | tuple[str, ...] | None = None,
    check_files: bool = True,
    allow_missing_features: bool = False,
    validate_ranges: bool = True,
) -> pd.DataFrame:
    """Load and validate the unified dataset CSV.

    Args:
        require_labels: fail when the ``label`` column is absent (training).
        required_modalities: radiograph modalities that must be present. ``None``
            requires both; an empty tuple requires neither.
        required_clinical_features: clinical columns that must be present.
            ``None`` requires all four; an empty tuple requires none.
        check_files: verify that every referenced image exists on disk.
        allow_missing_features: permit blank clinical variables. Off by default,
            because the clinical pipeline would otherwise impute them with the
            training median and return a confident-looking probability with no
            sign that a value was invented.
        validate_ranges: reject clinical values outside plausible bounds.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [str(column).strip() for column in df.columns]

    required_modalities = (
        tuple(IMAGE_COLUMNS) if required_modalities is None else required_modalities
    )
    required_clinical_features = (
        CLINICAL_COLUMNS
        if required_clinical_features is None
        else list(required_clinical_features)
    )
    required = [ID_COLUMN, *required_clinical_features]
    required += [image_column(modality) for modality in required_modalities]
    if require_labels:
        required.append(LABEL_COLUMN)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s): {missing}")

    df[ID_COLUMN] = df[ID_COLUMN].astype(str).str.strip()
    if (df[ID_COLUMN] == "").any():
        raise ValueError(f"{ID_COLUMN} cannot be empty.")
    if df[ID_COLUMN].duplicated().any():
        duplicates = sorted(df.loc[df[ID_COLUMN].duplicated(), ID_COLUMN].unique())[:10]
        raise ValueError(
            f"{ID_COLUMN} values must be unique (one row per patient); duplicates: {duplicates}"
        )

    for column in required_clinical_features:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if validate_ranges:
        _validate_clinical_ranges(df[required_clinical_features], str(csv_path))

    if not allow_missing_features:
        gaps = missing_feature_report(df, required_clinical_features)
        if (gaps != "").any():
            affected = df.loc[gaps != "", ID_COLUMN].tolist()
            examples = [
                f"{pid}: {gap}"
                for pid, gap in zip(affected[:5], gaps[gaps != ""][:5], strict=True)
            ]
            raise ValueError(
                f"{len(affected)} patient(s) have missing clinical variables, e.g. {examples}. "
                "Supply the values, or pass --allow-missing to impute them (the "
                "imputed fields are then reported alongside the prediction)."
            )

    if require_labels:
        labels = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            invalid = sorted(set(labels.dropna().unique()) - {0, 1})
            raise ValueError(
                f"'{LABEL_COLUMN}' must contain only 0 or 1; found {invalid or 'missing values'}. "
                "Progression (>=10 deg) is 1 and non-progression (<=5 deg) is 0."
            )
        df[LABEL_COLUMN] = labels.astype("int64")

    present_image_columns = [c for c in IMAGE_COLUMNS.values() if c in df.columns]
    df = resolve_paths(df, csv_path, present_image_columns)
    if check_files:
        for column in [image_column(modality) for modality in required_modalities]:
            missing_files = [
                path for path in df[column].dropna() if not Path(path).exists()
            ]
            if missing_files:
                preview = "\n".join(missing_files[:10])
                raise FileNotFoundError(
                    f"{len(missing_files)} path(s) in '{column}' do not exist, e.g.:\n{preview}"
                )
    return df


def describe_dataset(df: pd.DataFrame) -> dict:
    """Cohort summary: the baseline characteristics table."""
    summary: dict = {"n_patients": int(len(df))}
    if LABEL_COLUMN in df.columns:
        counts = df[LABEL_COLUMN].value_counts()
        summary["n_progression"] = int(counts.get(1, 0))
        summary["n_non_progression"] = int(counts.get(0, 0))
    for column in ("age", "cobb_baseline"):
        if column in df.columns:
            summary[column] = {
                "mean": float(df[column].mean()),
                "sd": float(df[column].std(ddof=1)),
                "n_missing": int(df[column].isna().sum()),
            }
    if "sex" in df.columns:
        summary["n_female"] = int((df["sex"] == 2).sum())
        summary["n_male"] = int((df["sex"] == 1).sum())
    if "risser" in df.columns:
        summary["risser"] = {
            str(int(k)): int(v) for k, v in df["risser"].value_counts().sort_index().items()
        }
    return summary
