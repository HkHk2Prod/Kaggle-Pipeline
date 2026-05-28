"""Unit tests for ``training.py`` helpers — currently the feature-name sanitiser."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from kaggle_pipeline.evolution.models.training import (
    _LGBM_FORBIDDEN_NAME_CHARS,
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
