"""Unit tests for the data-prep helpers extracted from KagglePipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kaggle_pipeline.evolution.features.recipe import CATEGORICAL, NUMERIC
from kaggle_pipeline.evolution.prepare import (
    autodetect_problem,
    build_search_sample,
    engineer_features,
    infer_feature_type,
)


@pytest.fixture
def classification_frame():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "id": range(n),
            "num1": rng.normal(size=n),
            "cat1": rng.choice(list("abc"), n),
            "target": rng.integers(0, 2, size=n),
        }
    )
    return df


def test_autodetect_fills_missing_target_task_and_scoring(classification_frame):
    target, task, scoring, prediction_aim = autodetect_problem(
        classification_frame, None, None, None, None, "id"
    )
    assert target == "target"
    assert task == "classification"
    assert isinstance(scoring, str) and scoring
    assert prediction_aim in {"probability", "category"}


def test_engineer_features_returns_input_when_no_expressions(classification_frame):
    out = engineer_features(classification_frame, None)
    assert out is classification_frame  # no copy when nothing to do
    out2 = engineer_features(classification_frame, [])
    assert out2 is classification_frame


def test_engineer_features_applies_eval_expressions(classification_frame):
    out = engineer_features(classification_frame, ["num1_sq = num1 ** 2"])
    assert "num1_sq" in out.columns
    assert out["num1_sq"].equals(classification_frame["num1"] ** 2)


def test_build_search_sample_returns_full_data_when_fraction_invalid(classification_frame):
    features = classification_frame.drop(columns=["target", "id"])
    y = classification_frame["target"].to_numpy()
    sampled, sy, used = build_search_sample(
        features, y, "classification", fraction=0.0, cv_splits=3, seed=0
    )
    assert sampled is features and sy is y and used is False


def test_build_search_sample_returns_full_data_when_n_sample_too_small(classification_frame):
    features = classification_frame.drop(columns=["target", "id"])
    y = classification_frame["target"].to_numpy()
    # fraction=0.05 of 120 rows = 6 rows, below min_rows = max(2*cv, 30)
    sampled, sy, used = build_search_sample(
        features, y, "classification", fraction=0.05, cv_splits=3, seed=0
    )
    assert sampled is features and sy is y and used is False


def test_build_search_sample_returns_subsample_when_fraction_in_range(classification_frame):
    features = classification_frame.drop(columns=["target", "id"])
    y = classification_frame["target"].to_numpy()
    sampled, sy, used = build_search_sample(
        features, y, "classification", fraction=0.5, cv_splits=3, seed=0
    )
    assert used is True
    assert len(sampled) == 60 and len(sy) == 60
    # Stratification preserves class proportions roughly: both classes present.
    assert set(np.unique(sy).tolist()) == {0, 1}


def test_infer_feature_type_honors_overrides(classification_frame):
    assert infer_feature_type("num1", classification_frame, {"num1": CATEGORICAL}) == CATEGORICAL


def test_infer_feature_type_numeric_vs_categorical(classification_frame):
    assert infer_feature_type("num1", classification_frame, None) == NUMERIC
    assert infer_feature_type("cat1", classification_frame, None) == CATEGORICAL
