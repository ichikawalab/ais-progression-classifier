"""Train the deployable model, outside the cross-validation protocol.

Cross-validation estimates how well the approach generalises; it does not leave
behind a model you can run on new patients. This module builds that model, and
it deliberately keeps no validation split of its own:

1. Read each image model's best epoch from its cross-validation runs and take
   the median. That is the epoch count to train for -- discovered without
   holding anything back now.
2. Train every base model on the entire cohort. Image models run for exactly
   that many epochs; clinical models tune with the same inner Optuna search the
   cross-validation used.
3. Build one or more serving profiles, whose weights, decision threshold, and
   calibrator all come from cross-validation out-of-fold predictions.

Every performance number attached to the bundle therefore comes from
cross-validation. A 10% holdout carved out here would cost training data, and
its AUC over ~47 patients would be too noisy to mean anything -- worse, the
ensemble weights are derived from out-of-fold predictions covering those same
patients, so it would also be biased upward.
"""
from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import torch

from ais_progression.config import (
    CLINICAL_MODALITY,
    CLINICAL_MODELS,
    Config,
    resolve_arch,
    save_config,
)
from ais_progression.data.schema import CLINICAL_COLUMNS, LABEL_COLUMN
from ais_progression.final.bundle import (
    BundleMember,
    ModelBundle,
    member_name,
    save_manifest,
)
from ais_progression.final.profiles import FULL_PROFILE, Profile
from ais_progression.models.clinical_model import fit_clinical_model
from ais_progression.models.image_model import fit_image_model
from ais_progression.provenance import (
    final_compatible_algorithm,
    frame_sha256,
    image_content_sha256,
    software_identity,
)
from ais_progression.utils import (
    ensure_dir,
    load_json,
    progress_bar_enabled,
    save_json,
    set_seed,
)


def export_image_weights(
    checkpoint_path: Path, destination: Path, weights_only: bool = True
) -> None:
    """Copy a trained image model into the bundle.

    By default only the ``state_dict`` tensors are written, so the file stays
    small and can be read back with ``torch.load(weights_only=True)``.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not weights_only:
        shutil.copyfile(checkpoint_path, destination)
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # The loss weight buffer belongs to training, not to the model.
    torch.save(
        {k: v for k, v in state_dict.items() if not k.startswith("loss_fn.")}, destination
    )


def epochs_from_cv(cv_dir: Path, name: str) -> int | None:
    """Median number of epochs the cross-validated models actually used.

    ``best_epoch`` is 0-based, so the epoch *count* is one more than the median.
    """
    fold_metrics = cv_dir / name / "fold_metrics.csv"
    if not fold_metrics.exists():
        return None
    frame = pd.read_csv(fold_metrics)
    if "best_epoch" not in frame.columns or frame["best_epoch"].dropna().empty:
        return None
    return int(np.median(frame["best_epoch"].dropna().to_numpy())) + 1


def resolve_epoch_plan(
    cv_dir: Path, image_members: list[str]
) -> dict[str, int]:
    """How long to train each image model on the full cohort.

    A missing cross-validation record is an error. Falling back to some default
    epoch count would train a model nobody chose the stopping point for, and the
    bundle would still carry cross-validated numbers as though the protocol had
    been followed.
    """
    plan: dict[str, int] = {}
    unresolved: list[str] = []
    for name in image_members:
        epochs = epochs_from_cv(cv_dir, name)
        if epochs is None:
            unresolved.append(name)
            continue
        plan[name] = epochs
    if unresolved:
        raise ValueError(
            "No cross-validation record of best_epoch for: "
            f"{sorted(unresolved)} (looked under {cv_dir}). Run 'ais-cv-modality' "
            "for them first."
        )
    return plan


def validate_cv_compatibility(
    config: Config,
    dataset: pd.DataFrame,
    sources: dict[str, Path],
) -> dict[str, dict]:
    """Require final training to use the exact procedure its CV metrics evaluated."""
    current_dataset = frame_sha256(dataset)
    current_software = software_identity()
    identities: dict[str, dict] = {}
    failures: list[str] = []
    for name, source in sorted(sources.items()):
        path = Path(source) / "run_identity.json"
        if not path.exists():
            failures.append(f"{name}: no run_identity.json under {source}")
            continue
        recorded = load_json(path)
        modality, model = name.split("_", 1)
        identities[name] = recorded
        expected = {
            "kind": "modality_cv",
            "modality": modality,
            "model": model,
            "dataset_sha256": current_dataset,
            "cross_validation": asdict(config.cross_validation),
            "algorithm": final_compatible_algorithm(config, modality, model),
            "software": current_software,
        }
        if modality != CLINICAL_MODALITY:
            expected["image_content_sha256"] = image_content_sha256(dataset, (modality,))
        for key, value in expected.items():
            if recorded.get(key) != value:
                failures.append(f"{name}: {key} differs from its cross-validation run")
    if failures:
        raise ValueError(
            "Final training does not match the cross-validation procedure:\n  - "
            + "\n  - ".join(failures)
        )
    return identities


def validate_plan(
    config: Config,
    profiles: list[Profile],
    image_models: tuple[str, ...],
    clinical_models: tuple[str, ...],
    image_modalities: tuple[str, ...],
) -> list[str]:
    """Check up front that every model a profile needs will be trained.

    Training nine models takes hours; discovering a mismatch afterwards wastes
    all of it.
    """
    planned = {
        member_name(modality, model)
        for modality in image_modalities
        for model in image_models
    } | {member_name(CLINICAL_MODALITY, model) for model in clinical_models}

    required: set[str] = set()
    for profile in profiles:
        required.update(profile.members)
    unavailable = sorted(required - planned)
    if unavailable:
        raise ValueError(
            f"The serving profiles need models this run would not train: {unavailable}. "
            f"Planned: {sorted(planned)}. Adjust --image-models / --clinical-models / "
            "--modalities, or drop the profile."
        )
    for model in image_models:
        resolve_arch(config, model)  # raises with a helpful message if unknown
    unknown_clinical = sorted(set(clinical_models) - set(CLINICAL_MODELS))
    if unknown_clinical:
        raise ValueError(f"Unknown clinical model(s): {unknown_clinical}")
    return sorted(required)


def _train_final_model_into(
    config: Config,
    dataset: pd.DataFrame,
    bundle_dir: str | Path,
    profiles: list[Profile],
    cv_identities: dict[str, dict],
    environment: dict,
    epoch_plan: dict[str, int],
    image_models: tuple[str, ...],
    clinical_models: tuple[str, ...] = CLINICAL_MODELS,
    image_modalities: tuple[str, ...] = ("front", "lateral"),
    weights_only: bool = True,
    default_profile: str = FULL_PROFILE,
) -> dict:
    """Train every needed base model on the full cohort and write the bundle."""
    bundle_dir = ensure_dir(bundle_dir)
    ensure_dir(bundle_dir / "models")
    work_dir = bundle_dir / "_work"

    required = validate_plan(config, profiles, image_models, clinical_models, image_modalities)
    print(f"Training {len(required)} model(s) on all {len(dataset)} patients")

    set_seed(config.final.seed, deterministic=config.train.deterministic)
    members: list[BundleMember] = []
    training_report: dict[str, dict] = {}

    for modality in image_modalities:
        for model in image_models:
            name = member_name(modality, model)
            if name not in required:
                continue
            arch = resolve_arch(config, model)
            epochs = epoch_plan[name]
            print(f"Training {name} ({arch}) for {epochs} epoch(s)")
            fit = fit_image_model(
                config,
                arch,
                modality,
                dataset,
                val_df=None,
                work_dir=work_dir / name,
                enable_progress_bar=progress_bar_enabled(),
                fixed_epochs=epochs,
            )
            artifact = f"models/{name}.pt" if weights_only else f"models/{name}.ckpt"
            export_image_weights(fit.checkpoint_path, bundle_dir / artifact, weights_only)
            members.append(
                BundleMember(modality=modality, model=model, artifact=artifact, epochs=epochs)
            )
            training_report[name] = {
                "epochs": epochs,
                "class_weights": fit.class_weights,
            }

    for model in clinical_models:
        name = member_name(CLINICAL_MODALITY, model)
        if name not in required:
            continue
        print(f"Training {name}")
        fit = fit_clinical_model(
            dataset[CLINICAL_COLUMNS],
            dataset[LABEL_COLUMN].astype(int),
            model,
            config.clinical,
            config.final.seed,
        )
        artifact = f"models/{name}.joblib"
        joblib.dump(fit.pipeline, bundle_dir / artifact)
        members.append(BundleMember(modality=CLINICAL_MODALITY, model=model, artifact=artifact))
        # Only the chosen hyperparameters are recorded. The search's own inner-CV
        # AUC is a post-selection score over the whole cohort, so keeping it here
        # would put a number in metrics.json that looks like performance but is
        # not the profile's out-of-fold estimate.
        training_report[name] = {"best_params": fit.best_params}

    profile_entries = []
    for profile in profiles:
        calibrator_artifact = None
        if profile.calibrator is not None:
            calibrator_artifact = f"models/calibrator_{profile.name}.joblib"
            joblib.dump(profile.calibrator, bundle_dir / calibrator_artifact)
        profile_entries.append(profile.as_dict(calibrator_artifact))

    metrics = {
        "n_train": int(len(dataset)),
        "trained_on": "the entire cohort; no holdout was kept",
        "performance_source": (
            "cross-validation. Each profile's cv_metrics carry its out-of-fold AUC; "
            "this model has no independent evaluation set of its own."
        ),
        "training": training_report,
        "cv_provenance": cv_identities,
        "profiles": {entry["name"]: entry for entry in profile_entries},
    }
    save_json(metrics, bundle_dir / "metrics.json")
    save_config(config, bundle_dir / "config.yaml")
    save_json(environment, bundle_dir / "environment.json")
    save_manifest(
        bundle_dir,
        members,
        profile_entries,
        default_profile=default_profile,
        extra={"n_train": metrics["n_train"]},
    )
    shutil.rmtree(work_dir, ignore_errors=True)
    return metrics


def train_final_model(
    config: Config,
    dataset: pd.DataFrame,
    bundle_dir: str | Path,
    profiles: list[Profile],
    cv_identities: dict[str, dict],
    environment: dict,
    epoch_plan: dict[str, int],
    image_models: tuple[str, ...],
    clinical_models: tuple[str, ...] = CLINICAL_MODELS,
    image_modalities: tuple[str, ...] = ("front", "lateral"),
    weights_only: bool = True,
    default_profile: str = FULL_PROFILE,
) -> dict:
    """Build and validate a complete bundle, then replace the destination atomically."""
    destination = Path(bundle_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    staging = destination.parent / f".{destination.name}.staging-{token}"
    backup = destination.parent / f".{destination.name}.backup-{token}"
    try:
        metrics = _train_final_model_into(
            config=config,
            dataset=dataset,
            bundle_dir=staging,
            profiles=profiles,
            cv_identities=cv_identities,
            environment=environment,
            epoch_plan=epoch_plan,
            image_models=image_models,
            clinical_models=clinical_models,
            image_modalities=image_modalities,
            weights_only=weights_only,
            default_profile=default_profile,
        )
        ModelBundle.load(staging)
        if destination.exists():
            destination.rename(backup)
        try:
            staging.rename(destination)
        except Exception:
            if backup.exists() and not destination.exists():
                backup.rename(destination)
            raise
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        return metrics
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not destination.exists():
            backup.rename(destination)


__all__ = [
    "epochs_from_cv",
    "export_image_weights",
    "resolve_epoch_plan",
    "train_final_model",
    "validate_cv_compatibility",
    "validate_plan",
]
