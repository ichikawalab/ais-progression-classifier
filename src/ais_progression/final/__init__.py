"""The deployable model: full-cohort training, serving profiles, and inference."""
from ais_progression.final.bundle import (
    BUNDLE_FORMAT_VERSION,
    BundleMember,
    ModelBundle,
    ServingProfile,
    member_name,
)
from ais_progression.final.operating_point import (
    Calibrator,
    OperatingPoint,
    choose_operating_point,
    fit_calibrator,
)
from ais_progression.final.profiles import (
    DEFAULT_PROFILE_MEMBERS,
    FULL_PROFILE,
    Profile,
    build_profiles,
)
from ais_progression.final.train import resolve_epoch_plan, train_final_model

__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "BundleMember",
    "Calibrator",
    "DEFAULT_PROFILE_MEMBERS",
    "FULL_PROFILE",
    "ModelBundle",
    "OperatingPoint",
    "Profile",
    "ServingProfile",
    "build_profiles",
    "choose_operating_point",
    "fit_calibrator",
    "member_name",
    "resolve_epoch_plan",
    "train_final_model",
]
