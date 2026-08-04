"""Late-fusion ensembles over the nine per-modality models."""
from ais_progression.ensemble.meta import (
    META_MODELS,
    MetaEnsemble,
    fit_meta_ensemble,
    simple_average,
)
from ais_progression.ensemble.weighted import WeightedEnsemble, fit_weighted_ensemble

__all__ = [
    "META_MODELS",
    "MetaEnsemble",
    "WeightedEnsemble",
    "fit_meta_ensemble",
    "fit_weighted_ensemble",
    "simple_average",
]
