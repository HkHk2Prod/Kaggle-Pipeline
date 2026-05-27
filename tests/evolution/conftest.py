"""Fixtures for the evolutionary-pipeline tests: synthetic data and a scored registry."""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.features.materialization import (
    FEATURE_EVAL_SAMPLE,
    MaterializationContext,
)
from kaggle_pipeline.evolution.features.registry import FeatureRegistry
from kaggle_pipeline.scoring.metrics import resolve_scoring


@pytest.fixture
def synthetic() -> tuple[pd.DataFrame, np.ndarray]:
    """A small classification frame whose target depends on the features."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "num3": rng.uniform(0, 10, size=n),
            "cat1": rng.choice(list("abcd"), size=n),
            "cat2": rng.choice(list("xyz"), size=n),
        }
    )
    logit = df["num1"] + 0.5 * df["num2"] - 0.3 * df["num3"] + (df["cat1"] == "a") * 1.2
    y = (logit + rng.normal(scale=0.5, size=n) > 0).astype(int).to_numpy()
    return df, y


@pytest.fixture
def settings() -> EvolutionSettings:
    return EvolutionSettings(default_random_seed=0, max_active_features=40)


@pytest.fixture
def originals() -> list[tuple[str, str]]:
    return [
        ("num1", "numeric"),
        ("num2", "numeric"),
        ("num3", "numeric"),
        ("cat1", "categorical"),
        ("cat2", "categorical"),
    ]


@pytest.fixture
def eval_context(synthetic) -> MaterializationContext:
    df, _ = synthetic
    return MaterializationContext(frame=df, context_id=FEATURE_EVAL_SAMPLE)


@pytest.fixture
def registry(settings, synthetic, originals, eval_context) -> FeatureRegistry:
    """A registry with originals registered and the active pool scored."""
    _, y = synthetic
    reg = FeatureRegistry(settings)
    for column, output_type in originals:
        reg.add_original_feature(column, output_type)
    reg.rescore_active(context=eval_context, y=y, task="classification")
    return reg


@pytest.fixture
def scoring_ctx():
    """A minimal stand-in for the v1 PipelineContext used by CrossValScore."""
    return types.SimpleNamespace(
        scoring_fn=resolve_scoring("roc_auc"), target_width=2, target_is_num=False
    )
