"""Build the unified dataset CSV from the study's source workbooks.

The cohort ships as three separate artefacts:

* ``Clinical_Info_n471.xlsx``  - Patient ID, Age, Sex_M1_F2, Risser, Cobb_baseline, Label
* ``Front/.../InputPath_n471.xlsx``   - Patient ID -> frontal radiograph path
* ``Lateral/.../InputPath_n471.xlsx`` - Patient ID -> lateral radiograph path

The absolute paths stored in those workbooks point at the original acquisition
machine, so they are re-rooted onto the local image directories by matching the
trailing portion of each path. ``label`` is derived from the source ``Label``
column, where 2 marks progression and 0 marks non-progression; borderline
patients were already dropped from the workbook.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ais_progression.data.schema import DATASET_COLUMNS, describe_dataset

PROGRESSION_LABEL_VALUE = 2
NON_PROGRESSION_LABEL_VALUE = 0

# Source workbook column -> unified CSV column.
CLINICAL_RENAMES = {
    "Patient ID": "patient_id",
    "Age": "age",
    "Sex_M1_F2": "sex",
    "Risser": "risser",
    "Cobb_baseline": "cobb_baseline",
}
ID_ALIASES = ("Patient ID", "ID", "patient_id")


def _id_column(frame: pd.DataFrame, source: Path) -> str:
    for alias in ID_ALIASES:
        if alias in frame.columns:
            return alias
    raise ValueError(f"{source} has no patient identifier column (looked for {ID_ALIASES}).")


def reroot_path(recorded: str, image_root: Path) -> str | None:
    """Map a path recorded on the acquisition machine onto ``image_root``.

    The longest trailing run of path components that exists under ``image_root``
    wins, so both ``.../Input_Original_n471/Toyama/x.png`` and a bare filename
    resolve correctly. Returns None when nothing matches.
    """
    recorded_path = Path(str(recorded).strip().replace("\\", "/"))
    # Drop the anchor ("C:\\", "/"). Path.joinpath discards everything to its
    # left when handed an absolute component, which would silently resolve the
    # match outside image_root.
    parts = [part for part in recorded_path.parts if part != recorded_path.anchor]
    image_root = image_root.resolve()
    for start in range(len(parts)):
        candidate = (image_root.joinpath(*parts[start:])).resolve()
        # A "..' component could still climb out of the root; require containment.
        if candidate.exists() and candidate.is_relative_to(image_root):
            return str(candidate)
    return None


def _path_lookup(workbook: Path, path_column: str, image_root: Path) -> pd.Series:
    """Patient ID -> resolved local image path, from an InputPath workbook."""
    frame = pd.read_excel(workbook)
    frame.columns = [str(column).strip() for column in frame.columns]
    if path_column not in frame.columns:
        raise ValueError(f"{workbook} has no '{path_column}' column.")
    id_column = _id_column(frame, workbook)
    ids = frame[id_column].astype(str).str.strip()
    if ids.duplicated().any():
        # A duplicated index would make the later .map raise an opaque
        # reindexing error instead of naming the problem.
        raise ValueError(
            f"{workbook} has duplicate patient IDs: "
            f"{sorted(ids[ids.duplicated()].unique())[:10]}"
        )
    resolved = frame[path_column].map(lambda value: reroot_path(value, image_root))
    return pd.Series(resolved.to_numpy(), index=ids)


def build_dataset(
    clinical_xlsx: str | Path,
    front_xlsx: str | Path,
    lateral_xlsx: str | Path,
    front_root: str | Path,
    lateral_root: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Join the three workbooks into one patient-level DataFrame.

    Returns the dataset and a report describing what was dropped and why.
    """
    clinical = pd.read_excel(clinical_xlsx)
    clinical.columns = [str(column).strip() for column in clinical.columns]
    missing = [column for column in CLINICAL_RENAMES if column not in clinical.columns]
    if missing:
        raise ValueError(f"{clinical_xlsx} is missing column(s): {missing}")
    if "Label" not in clinical.columns:
        raise ValueError(f"{clinical_xlsx} is missing the 'Label' column.")

    dataset = clinical[list(CLINICAL_RENAMES)].rename(columns=CLINICAL_RENAMES)
    dataset["patient_id"] = dataset["patient_id"].astype(str).str.strip()

    source_label = pd.to_numeric(clinical["Label"], errors="coerce")
    known = source_label.isin([NON_PROGRESSION_LABEL_VALUE, PROGRESSION_LABEL_VALUE])
    dataset["label"] = (source_label == PROGRESSION_LABEL_VALUE).astype("int64")

    front = _path_lookup(Path(front_xlsx), "Front_paths", Path(front_root))
    lateral = _path_lookup(Path(lateral_xlsx), "Lateral_paths", Path(lateral_root))
    dataset["front_path"] = dataset["patient_id"].map(front)
    dataset["lateral_path"] = dataset["patient_id"].map(lateral)

    complete = (
        known
        & dataset["front_path"].notna()
        & dataset["lateral_path"].notna()
        & dataset[["age", "sex", "risser", "cobb_baseline"]].notna().all(axis=1)
    )
    dropped = dataset.loc[~complete, "patient_id"].tolist()
    report = {
        "n_source_rows": int(len(clinical)),
        "n_kept": int(complete.sum()),
        "n_dropped": int((~complete).sum()),
        "dropped_unknown_label": sorted(dataset.loc[~known, "patient_id"]),
        "dropped_missing_front": sorted(dataset.loc[dataset["front_path"].isna(), "patient_id"]),
        "dropped_missing_lateral": sorted(
            dataset.loc[dataset["lateral_path"].isna(), "patient_id"]
        ),
        "dropped_patient_ids": sorted(dropped),
    }

    dataset = dataset.loc[complete, DATASET_COLUMNS].reset_index(drop=True)
    dataset["risser"] = dataset["risser"].astype("int64")
    dataset["sex"] = dataset["sex"].astype("int64")
    report["cohort"] = describe_dataset(dataset)
    return dataset, report
