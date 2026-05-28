"""Assemble the pre-training preprocessing pipeline from a :class:`Config`."""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from kaggle_pipeline.config import Config
from kaggle_pipeline.preprocessing.transformers import (
    CategoricalTyper,
    FeatureEngineer,
    OrdinalEncoderTransformer,
)

# Step name used to recover learned category orderings after fitting.
CATEGORICAL_TYPER_STEP = "categorical_typer"


def build_pretrain_pipeline(config: Config) -> Pipeline:
    """Feature engineering -> categorical typing -> ordinal encoding.

    Fitted once on the training frame and then applied to test; this is distinct
    from the per-model preprocessing baked into each estimator's own pipeline.
    """
    steps = [
        ("feature_engineer", FeatureEngineer(expressions=config.feature_expressions)),
        (
            CATEGORICAL_TYPER_STEP,
            CategoricalTyper(
                cat_cutoff=config.cat_cutoff,
                cat_order_list=_flatten(config.order_lists),
            ),
        ),
        (
            "ordinal_encoder",
            OrdinalEncoderTransformer(
                target=config.target,
                order_lists=config.order_lists,
            ),
        ),
    ]
    return Pipeline(steps)


def _flatten(order_lists: list[list[str]]) -> list[str]:
    """Concatenate the per-group ordering lists into one flat preference list."""
    flat: list[str] = []
    for group in order_lists:
        flat.extend(group)
    return flat
