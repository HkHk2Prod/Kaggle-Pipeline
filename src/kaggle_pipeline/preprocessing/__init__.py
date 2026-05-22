"""Preprocessing: column inspection, transformers, target transforms, pipeline."""

from kaggle_pipeline.preprocessing.association import (
    association_matrix,
    correlation_ratio,
    cramers_v,
)
from kaggle_pipeline.preprocessing.columns import (
    detect_ordinal_order_cols,
    get_columns,
    get_predictor_names,
    is_num_check,
    make_cat_order,
    split_num_cat,
)
from kaggle_pipeline.preprocessing.encoders import (
    DEFAULT_STRATEGY,
    ENCODING_STRATEGIES,
    FrequencyEncoder,
    categorical_transformer_specs,
    resolve_encoding_plan,
)
from kaggle_pipeline.preprocessing.pipeline import (
    CATEGORICAL_TYPER_STEP,
    build_pretrain_pipeline,
)
from kaggle_pipeline.preprocessing.selection import (
    CorrelationPruner,
    irrelevance_threshold,
    plan_pruning,
    redundancy_lower_bound,
)
from kaggle_pipeline.preprocessing.target import (
    TargetTransforms,
    build_target_transforms,
)
from kaggle_pipeline.preprocessing.transformers import (
    CategoricalTyper,
    FeatureEngineer,
    OrdinalEncoderTransformer,
)

__all__ = [
    "is_num_check",
    "get_columns",
    "get_predictor_names",
    "split_num_cat",
    "detect_ordinal_order_cols",
    "make_cat_order",
    "FeatureEngineer",
    "CategoricalTyper",
    "OrdinalEncoderTransformer",
    "FrequencyEncoder",
    "ENCODING_STRATEGIES",
    "DEFAULT_STRATEGY",
    "resolve_encoding_plan",
    "categorical_transformer_specs",
    "association_matrix",
    "cramers_v",
    "correlation_ratio",
    "CorrelationPruner",
    "plan_pruning",
    "irrelevance_threshold",
    "redundancy_lower_bound",
    "build_pretrain_pipeline",
    "CATEGORICAL_TYPER_STEP",
    "build_target_transforms",
    "TargetTransforms",
]
