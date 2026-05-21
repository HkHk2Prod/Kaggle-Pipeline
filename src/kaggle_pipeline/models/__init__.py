"""Models: the abstract base, the registry, and the built-in model zoo.

Importing this package imports :mod:`kaggle_pipeline.models.definitions`, which
registers all built-in models as a side effect.
"""

from kaggle_pipeline.models import definitions  # noqa: F401  (registers models)
from kaggle_pipeline.models.base import Model, sample_parameters
from kaggle_pipeline.models.registry import (
    SINGLE_TARGET_PROB_PRED,
    ModelRegistry,
    register_model,
    registry,
)

__all__ = [
    "Model",
    "sample_parameters",
    "ModelRegistry",
    "register_model",
    "registry",
    "SINGLE_TARGET_PROB_PRED",
]
