"""Unit tests for ``training.py`` helpers — currently the feature-name sanitiser."""

from __future__ import annotations

import importlib.util
import logging

import numpy as np
import pandas as pd
import pytest

from kaggle_pipeline.evolution.genes.base import BaseModelGene
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.parameter_spaces import build_default_families
from kaggle_pipeline.evolution.models.training import (
    _LGBM_FORBIDDEN_NAME_CHARS,
    ModelTrainer,
    _SanitizeFeatureNames,
)


def test_sanitiser_rewrites_every_forbidden_character():
    df = pd.DataFrame(
        {
            "orig::age": [1, 2],
            'gen::abc__cat_"a":b': [3, 4],
            "weird[col]{x},y\\z": [5, 6],
            "plain": [7, 8],
        }
    )
    out = _SanitizeFeatureNames().fit_transform(df)
    assert list(out.columns) == [
        "orig__age",
        "gen__abc__cat__a__b",
        "weird_col__x__y_z",
        "plain",
    ]
    assert not any(_LGBM_FORBIDDEN_NAME_CHARS.search(c) for c in out.columns)


def test_sanitiser_preserves_data_and_index():
    df = pd.DataFrame({"orig::x": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
    out = _SanitizeFeatureNames().fit_transform(df)
    assert list(out.index) == [10, 20, 30]
    np.testing.assert_array_equal(out["orig__x"].to_numpy(), df["orig::x"].to_numpy())


def test_sanitiser_passes_numpy_through_unchanged():
    arr = np.arange(6).reshape(3, 2)
    out = _SanitizeFeatureNames().fit_transform(arr)
    assert out is arr  # no copy, no rename


@pytest.mark.skipif(
    importlib.util.find_spec("lightgbm") is None,
    reason="lightgbm not installed",
)
def test_lightgbm_accepts_sanitised_names_from_a_pipeline():
    """End-to-end: a Pipeline with the sanitiser lets LightGBM fit on ``::`` names.

    Without the sanitiser LightGBM raises ``Do not support special JSON characters
    in feature name`` -- this regression-guards that fix.
    """
    from lightgbm import LGBMClassifier
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "orig::age": rng.normal(size=40),
            "gen::ratio": rng.normal(size=40),
        }
    )
    y = (X["orig::age"] + X["gen::ratio"] > 0).astype(int).to_numpy()
    pipe = Pipeline(
        [
            ("sanitize", _SanitizeFeatureNames()),
            ("model", LGBMClassifier(n_estimators=5, verbose=-1)),
        ]
    )
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (40, 2)


def _numeric_cols(pipeline) -> list[str]:
    for name, _transformer, cols in pipeline.named_steps["prep"].transformers:
        if name == "num":
            return list(cols)
    return []


@pytest.fixture
def training_logs(caplog):
    """Caplog wrapper that bypasses ``propagate=False`` on the package logger.

    ``kaggle_pipeline.logconfig`` sets ``propagate=False`` so root handlers stay
    quiet; pytest's caplog attaches to root, so it sees nothing without help.
    Attaching caplog's handler directly to the training logger fixes this.
    """
    target = logging.getLogger("kaggle_pipeline.evolution.models.training")
    target.addHandler(caplog.handler)
    prev_level = target.level
    target.setLevel(logging.WARNING)
    try:
        yield caplog
    finally:
        target.removeHandler(caplog.handler)
        target.setLevel(prev_level)


def test_build_pipeline_dedups_duplicate_feature_refs(registry, synthetic, training_logs):
    # A genome with the same feature_id twice would otherwise emit two `num__orig::num1`
    # columns out of the ColumnTransformer and trip LightGBM's duplicate-feature check.
    df, _ = synthetic
    genome = ModelGenome(
        base_model_gene=BaseModelGene("lightgbm"),
        feature_reference_genes=[
            FeatureReferenceGene("orig::num1"),
            FeatureReferenceGene("orig::num2"),
            FeatureReferenceGene("orig::num1"),
        ],
    )
    X = pd.DataFrame({"orig::num1": df["num1"].to_numpy(), "orig::num2": df["num2"].to_numpy()})
    trainer = ModelTrainer(registry, families=build_default_families())

    pipeline = trainer._build_pipeline(genome, X, seed=0)

    assert _numeric_cols(pipeline) == ["orig::num1", "orig::num2"]
    assert any(
        "duplicate feature references" in r.getMessage() and "orig::num1" in r.getMessage()
        for r in training_logs.records
    )


def test_build_pipeline_unique_refs_does_not_warn(registry, synthetic, training_logs):
    df, _ = synthetic
    genome = ModelGenome(
        base_model_gene=BaseModelGene("lightgbm"),
        feature_reference_genes=[
            FeatureReferenceGene("orig::num1"),
            FeatureReferenceGene("orig::num2"),
        ],
    )
    X = pd.DataFrame({"orig::num1": df["num1"].to_numpy(), "orig::num2": df["num2"].to_numpy()})
    trainer = ModelTrainer(registry, families=build_default_families())

    trainer._build_pipeline(genome, X, seed=0)

    assert not any("duplicate feature references" in r.getMessage() for r in training_logs.records)
