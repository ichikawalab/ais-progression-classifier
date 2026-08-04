"""The deployable artefact: fitted models plus one or more serving profiles.

A bundle directory looks like::

    <bundle>/
      manifest.json          format version, model inventory, profiles
      config.yaml            the configuration the models were trained with
      metrics.json           cross-validated performance of each profile
      models/
        front_vit.pt  ...    one weights file per image model
        clinical_logreg.joblib ...
        calibrator_full.joblib ...

Image models are stored as bare ``state_dict`` tensors, which keeps bundles
small and lets them load with ``torch.load(weights_only=True)``. scikit-learn
pipelines and calibrators have to be pickled, so only load a bundle produced by
a run you trust.

Loaded models are cached, so an application pays the load cost once rather than
per request. Call :meth:`ModelBundle.warmup` at startup to pay it up front.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from ais_progression.config import CLINICAL_MODALITY, Config, load_config, resolve_arch
from ais_progression.data.schema import CLINICAL_COLUMNS
from ais_progression.ensemble.weighted import WeightedEnsemble
from ais_progression.final.operating_point import Calibrator, OperatingPoint
from ais_progression.final.profiles import FULL_PROFILE
from ais_progression.models.clinical_model import predict_clinical_model
from ais_progression.models.image_model import load_image_classifier, predict_image_model
from ais_progression.utils import get_device, load_json, save_json, set_matmul_precision

MANIFEST_NAME = "manifest.json"

# Bumped whenever the manifest layout changes in a way older code cannot read.
BUNDLE_FORMAT_VERSION = 1


def member_name(modality: str, model: str) -> str:
    """Column name identifying one base model inside an ensemble."""
    return f"{modality}_{model}"


@dataclass
class BundleMember:
    """One trained base model inside a bundle."""

    modality: str
    model: str
    artifact: str  # path relative to the bundle directory
    epochs: int | None = None

    @property
    def name(self) -> str:
        return member_name(self.modality, self.model)

    @property
    def is_image(self) -> bool:
        return self.modality != CLINICAL_MODALITY

    def as_dict(self) -> dict:
        return {
            "modality": self.modality,
            "model": self.model,
            "artifact": self.artifact,
            "epochs": self.epochs,
        }


@dataclass
class ServingProfile:
    """A configuration the bundle can be asked to run."""

    name: str
    members: list[str]
    weights: np.ndarray
    operating_point: OperatingPoint
    calibration: dict
    cv_metrics: dict

    @property
    def ensemble(self) -> WeightedEnsemble:
        return WeightedEnsemble(
            columns=list(self.members),
            weights=self.weights,
            inner_cv_auc=None,
        )

    @classmethod
    def from_dict(cls, payload: dict) -> ServingProfile:
        return cls(
            name=payload["name"],
            members=list(payload["members"]),
            weights=np.asarray(payload["weights"], dtype=float),
            operating_point=OperatingPoint.from_dict(payload["operating_point"]),
            calibration=payload.get("calibration", {}),
            cv_metrics=payload.get("cv_metrics", {}),
        )


class ModelBundle:
    """Load a bundle once, then score patients with any of its profiles."""

    def __init__(
        self,
        bundle_dir: str | Path,
        config: Config,
        members: list[BundleMember],
        profiles: dict[str, ServingProfile],
        manifest: dict,
    ):
        self.dir = Path(bundle_dir)
        self.config = config
        self.members = members
        self.profiles = profiles
        self.manifest = manifest
        self._by_name = {member.name: member for member in members}
        self._model_cache: dict[str, object] = {}
        self._calibrator_cache: dict[str, Calibrator | None] = {}

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, bundle_dir: str | Path) -> ModelBundle:
        bundle_dir = Path(bundle_dir)
        manifest_path = bundle_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Not a model bundle (no {MANIFEST_NAME}): {bundle_dir}")
        manifest = load_json(manifest_path)

        found_version = manifest.get("format_version")
        if found_version != BUNDLE_FORMAT_VERSION:
            raise ValueError(
                f"{bundle_dir} is a format-{found_version} bundle but this version of "
                f"ais-progression reads format {BUNDLE_FORMAT_VERSION}. Rebuild it with "
                "'ais-train-final'."
            )

        config = load_config(bundle_dir / "config.yaml")
        members = [BundleMember(**entry) for entry in manifest["members"]]
        missing = [m.name for m in members if not (bundle_dir / m.artifact).exists()]
        if missing:
            raise FileNotFoundError(f"Bundle is missing artefact(s) for: {missing}")

        profiles = {
            entry["name"]: ServingProfile.from_dict(entry) for entry in manifest["profiles"]
        }
        if not profiles:
            raise ValueError(f"{bundle_dir} declares no serving profiles.")
        known = {member.name for member in members}
        for profile in profiles.values():
            unknown = set(profile.members) - known
            if unknown:
                raise ValueError(
                    f"Profile '{profile.name}' references models absent from the bundle: "
                    f"{sorted(unknown)}"
                )
            calibrator_artifact = profile.calibration.get("artifact")
            if calibrator_artifact and not (bundle_dir / calibrator_artifact).exists():
                raise FileNotFoundError(
                    f"Profile '{profile.name}' is missing calibrator artefact: "
                    f"{calibrator_artifact}"
                )

        # Score under the same fp32 matmul mode the models were trained with,
        # rather than whatever the host process happens to default to.
        set_matmul_precision(config.train.matmul_precision)
        return cls(bundle_dir, config, members, profiles, manifest)

    def profile(self, name: str | None = None) -> ServingProfile:
        name = name or self.manifest.get("default_profile") or FULL_PROFILE
        try:
            return self.profiles[name]
        except KeyError:
            raise ValueError(
                f"Unknown profile '{name}'. Available: {sorted(self.profiles)}"
            ) from None

    # ---------------------------------------------------------------- caching

    def _model(self, member: BundleMember):
        """Load a base model once and keep it resident."""
        if member.name not in self._model_cache:
            artifact = self.dir / member.artifact
            if member.is_image:
                arch = resolve_arch(self.config, member.model)
                self._model_cache[member.name] = load_image_classifier(
                    artifact, self.config, arch, get_device()
                )
            else:
                self._model_cache[member.name] = joblib.load(artifact)
        return self._model_cache[member.name]

    def _calibrator(self, profile: ServingProfile) -> Calibrator | None:
        if profile.name not in self._calibrator_cache:
            artifact = profile.calibration.get("artifact")
            self._calibrator_cache[profile.name] = (
                joblib.load(self.dir / artifact) if artifact else None
            )
        return self._calibrator_cache[profile.name]

    def warmup(self, profile: str | None = None) -> None:
        """Load a profile's models into memory ahead of the first request."""
        selected = self.profile(profile)
        for name in selected.members:
            self._model(self._by_name[name])
        self._calibrator(selected)

    def release(self) -> None:
        """Drop cached models and free GPU memory."""
        self._model_cache.clear()
        self._calibrator_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------- inference

    def predict_members(self, df: pd.DataFrame, members: list[str]) -> pd.DataFrame:
        """Per-base-model progression probabilities for ``df``, in row order."""
        device = get_device()
        columns: dict[str, np.ndarray] = {}
        for name in members:
            member = self._by_name[name]
            model = self._model(member)
            if member.is_image:
                arch = resolve_arch(self.config, member.model)
                columns[name] = predict_image_model(
                    model, self.config, arch, member.modality, df, device
                )
            else:
                columns[name] = predict_clinical_model(model, df[CLINICAL_COLUMNS])
        return pd.DataFrame(columns, index=df.index)

    def predict(self, df: pd.DataFrame, profile: str | None = None) -> pd.DataFrame:
        """Base-model probabilities, the ensemble score, and the decision.

        Columns: one per base model, plus ``probability`` (the weighted ensemble
        score), ``calibrated_probability`` when the profile carries a calibrator,
        ``predicted_label`` at the profile's threshold, and ``threshold``.
        """
        selected = self.profile(profile)
        result = self.predict_members(df, selected.members)
        result["probability"] = selected.ensemble.predict(result)

        calibrator = self._calibrator(selected)
        result["calibrated_probability"] = (
            calibrator.transform(result["probability"]) if calibrator else np.nan
        )
        threshold = selected.operating_point.threshold
        result["predicted_label"] = (result["probability"] >= threshold).astype(int)
        result["threshold"] = threshold
        result["profile"] = selected.name
        return result


def save_manifest(
    bundle_dir: str | Path,
    members: list[BundleMember],
    profiles: list[dict],
    default_profile: str,
    extra: dict | None = None,
) -> dict:
    manifest = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "members": [member.as_dict() for member in members],
        "profiles": profiles,
        "default_profile": default_profile,
        **(extra or {}),
    }
    save_json(manifest, Path(bundle_dir) / MANIFEST_NAME)
    return manifest
