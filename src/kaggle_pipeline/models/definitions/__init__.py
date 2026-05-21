"""Importing this package registers every built-in model.

Add a new model by dropping a module here that defines a ``Model`` subclass
decorated with ``@register_model`` and importing it below.
"""

from kaggle_pipeline.models.definitions import (  # noqa: F401
    catboost,
    hist_gb,
    lightgbm,
    logistic,
    random_forest,
    xgboost,
)
