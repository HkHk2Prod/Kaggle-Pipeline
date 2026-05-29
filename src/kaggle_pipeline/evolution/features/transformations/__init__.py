"""Feature transformations: operators that turn parent features into new ones.

Transforms are grouped by type into submodules (numeric unary, numeric binary,
categorical). Each operator class is decorated with :func:`register` so
:func:`build_default_registry` returns a registry containing everything that
has been imported. Importing this package imports all submodules and so
populates the default set.
"""

from __future__ import annotations

from kaggle_pipeline.evolution.features.transformations.base import (
    EPS,
    FeatureTransformation,
    TransformationRegistry,
    TransformError,
    build_default_registry,
    register,
)
from kaggle_pipeline.evolution.features.transformations.categorical import (
    CategoryJoin,
    CountEncode,
    FrequencyEncode,
    HashEncode,
    RareGroup,
    TargetEncode,
)
from kaggle_pipeline.evolution.features.transformations.numeric_binary import (
    AbsDiff,
    Add,
    Maximum,
    Minimum,
    Multiply,
    Ratio,
    SafeDivide,
    Subtract,
)
from kaggle_pipeline.evolution.features.transformations.numeric_unary import (
    Bin,
    Clip,
    Log1p,
    MinMax,
    MissingIndicator,
    Rank,
    Sqrt,
    Square,
    ZScore,
)

__all__ = [
    # Base
    "EPS",
    "FeatureTransformation",
    "TransformError",
    "TransformationRegistry",
    "build_default_registry",
    "register",
    # Numeric unary
    "Log1p",
    "Sqrt",
    "Square",
    "Rank",
    "ZScore",
    "MinMax",
    "Clip",
    "Bin",
    "MissingIndicator",
    # Numeric binary
    "Add",
    "Subtract",
    "Multiply",
    "SafeDivide",
    "AbsDiff",
    "Ratio",
    "Minimum",
    "Maximum",
    # Categorical
    "CategoryJoin",
    "FrequencyEncode",
    "CountEncode",
    "RareGroup",
    "HashEncode",
    "TargetEncode",
]
