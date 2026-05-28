"""Tests for per-column categorical encoding and capability-driven wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_pipeline import Config
from kaggle_pipeline.models import registry
from kaggle_pipeline.models.definitions.hist_gb import HistGBClassifierModel
from kaggle_pipeline.pipeline import build_pipeline
from kaggle_pipeline.preprocessing import (
    ONEHOT_MAX_CARDINALITY,
    FrequencyEncoder,
    categorical_transformer_specs,
    resolve_encoding_plan,
)

# A categorical predictor with far more levels than one-hot encoding could
# reasonably expand into a readable/cheap feature set.
N_DRIVERS = 40


# --------------------------------------------------------------------------- #
# FrequencyEncoder
# --------------------------------------------------------------------------- #
def test_frequency_encoder_maps_to_training_frequencies():
    enc = FrequencyEncoder().fit(pd.DataFrame({"c": ["a", "a", "a", "b"]}))
    out = enc.transform(pd.DataFrame({"c": ["a", "b", "z"]}))  # "z" unseen
    assert list(out["c"]) == [0.75, 0.25, 0.0]  # unseen level -> 0.0
    assert out["c"].dtype == float


def test_frequency_encoder_is_one_column_in_one_column_out():
    df = pd.DataFrame({"driver": [f"d{i % N_DRIVERS}" for i in range(200)]})
    out = FrequencyEncoder().fit_transform(df)
    assert out.shape == (200, 1)  # never widens, unlike one-hot


# --------------------------------------------------------------------------- #
# Plan resolution
# --------------------------------------------------------------------------- #
def test_resolve_plan_defaults_low_cardinality_to_onehot():
    # Both columns have 2 levels (<= ONEHOT_MAX_CARDINALITY), so the unspecified
    # "city" defaults to onehot; the explicit "driver" override still wins.
    df = pd.DataFrame({"driver": ["a", "b"], "city": ["x", "y"]})
    plan = resolve_encoding_plan({"driver": "target"}, df, ["driver", "city"], announce=False)
    assert plan == {"driver": "target", "city": "onehot"}


def test_resolve_plan_defaults_high_cardinality_to_frequency():
    # A column with more than ONEHOT_MAX_CARDINALITY levels would explode under
    # one-hot, so the unspecified default falls back to frequency.
    df = pd.DataFrame({"driver": [f"d{i}" for i in range(ONEHOT_MAX_CARDINALITY + 1)]})
    plan = resolve_encoding_plan({}, df, ["driver"], announce=False)
    assert plan == {"driver": "frequency"}


def test_resolve_plan_ignores_non_categorical_columns():
    df = pd.DataFrame({"driver": ["a", "b"]})
    plan = resolve_encoding_plan({"ghost": "onehot"}, df, ["driver"], announce=False)
    # "ghost" is not a categorical predictor; "driver" (2 levels) defaults to onehot.
    assert plan == {"driver": "onehot"}


def test_resolve_plan_respects_explicit_onehot_max_cardinality():
    # With the cut-off lowered to 1, a 2-level column is now "high" cardinality
    # and falls back to frequency instead of the onehot it would get by default.
    df = pd.DataFrame({"city": ["x", "y"]})
    plan = resolve_encoding_plan({}, df, ["city"], onehot_max_cardinality=1, announce=False)
    assert plan == {"city": "frequency"}


def test_resolve_plan_onehot_max_cardinality_none_uses_module_default():
    # None means "use ONEHOT_MAX_CARDINALITY": a column at exactly the default
    # cut-off still one-hots, one level above it does not.
    at_cap = pd.DataFrame({"c": [f"v{i}" for i in range(ONEHOT_MAX_CARDINALITY)]})
    above_cap = pd.DataFrame({"c": [f"v{i}" for i in range(ONEHOT_MAX_CARDINALITY + 1)]})
    assert resolve_encoding_plan({}, at_cap, ["c"], announce=False) == {"c": "onehot"}
    assert resolve_encoding_plan({}, above_cap, ["c"], announce=False) == {"c": "frequency"}


# --------------------------------------------------------------------------- #
# Transformer specs
# --------------------------------------------------------------------------- #
def test_specs_group_columns_by_strategy_and_native_falls_back_to_frequency():
    specs = categorical_transformer_specs(
        {"a": "frequency", "b": "onehot", "c": "native"}, ["a", "b", "c"]
    )
    grouped = {name: cols for name, _, cols in specs}
    # "native" cannot be honoured by a model that needs encoding -> frequency.
    assert grouped["cat_frequency"] == ["a", "c"]
    assert grouped["cat_onehot"] == ["b"]


def test_specs_reject_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown categorical encoding strategy"):
        categorical_transformer_specs({"a": "bogus"}, ["a"])


def test_specs_support_ordinal_and_drop():
    specs = categorical_transformer_specs({"a": "ordinal", "b": "drop"}, ["a", "b"])
    by_name = {name: transformer for name, transformer, _ in specs}
    assert "cat_ordinal" in by_name
    assert by_name["cat_drop"] == "drop"  # sklearn ColumnTransformer drop sentinel


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def test_config_rejects_unknown_encoding_strategy():
    with pytest.raises(ValueError, match="Unknown categorical_encoding"):
        Config(competition="x", categorical_encoding={"driver": "bogus"})


def test_config_rejects_non_positive_onehot_max_cardinality():
    with pytest.raises(ValueError, match="onehot_max_cardinality must be a positive int"):
        Config(competition="x", onehot_max_cardinality=0)


# --------------------------------------------------------------------------- #
# End-to-end wiring on a high-cardinality column
# --------------------------------------------------------------------------- #
def _write_highcard_competition(data_dir: Path) -> None:
    """Train/test CSVs with a high-cardinality nominal `driver` column.

    The test set deliberately includes driver names unseen in training, so the
    encoders must tolerate unseen levels rather than error (as the old
    ``OneHotEncoder(handle_unknown="error")`` would).
    """
    rng = np.random.default_rng(0)
    n_train, n_test = 180, 60
    train_drivers = rng.choice([f"d{i}" for i in range(N_DRIVERS)], size=n_train)
    num1 = rng.normal(size=n_train)
    y = np.where(num1 + rng.normal(scale=0.5, size=n_train) > 0, "yes", "no")
    pd.DataFrame({"id": range(n_train), "num1": num1, "driver": train_drivers, "y": y}).to_csv(
        data_dir / "train.csv", index=False
    )

    # Half the test drivers are names never seen in training.
    test_drivers = rng.choice([f"d{i}" for i in range(N_DRIVERS, N_DRIVERS + 60)], size=n_test)
    pd.DataFrame(
        {"id": range(n_test), "num1": rng.normal(size=n_test), "driver": test_drivers}
    ).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": range(n_test), "y": ["no"] * n_test}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )


def _build_highcard_ctx(tmp_path: Path, categorical_encoding: dict[str, str] | None = None):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_highcard_competition(data_dir)
    config = Config(
        competition="synthetic",
        target="y",
        id_col="id",
        task="classification",
        scoring="balanced_accuracy",
        prediction_aim="category",
        categorical_encoding=categorical_encoding or {},
        seed=0,
        data_dir=data_dir,
        storage_dir=tmp_path / "models",
    )
    ctx, _ = build_pipeline(config)
    return ctx


@pytest.fixture
def highcard_ctx(tmp_path: Path):
    return _build_highcard_ctx(tmp_path)


def _fit_predict(model, ctx):
    X = ctx.train_df[ctx.predictor_columns]
    y = ctx.target_transforms.forward(ctx.train_df[ctx.target])
    model.fit(X, y)
    return model.predict(ctx.test_df[ctx.predictor_columns])


def test_driver_is_a_categorical_predictor_defaulting_to_frequency(highcard_ctx):
    assert "driver" in highcard_ctx.cat_cols_x
    assert highcard_ctx.categorical_encoding["driver"] == "frequency"


@pytest.mark.parametrize("name", ["RandomForestClassifier", "LogisticRegression"])
def test_non_native_models_encode_without_exploding_or_erroring(name, highcard_ctx):
    ctx = highcard_ctx
    model = registry[name](ctx)
    proba = _fit_predict(model, ctx)

    assert proba.shape[0] == len(ctx.test_df)  # unseen test drivers did not error
    # Frequency encoding keeps one column per categorical: width == #features,
    # not the ~40 columns one-hot would have produced.
    preprocessor = model._model.named_steps["preprocessor"]
    width = preprocessor.transform(ctx.train_df[ctx.predictor_columns]).shape[1]
    assert width == len(ctx.num_cols_x) + len(ctx.cat_cols_x)


def test_histgb_passes_moderate_cardinality_natively(highcard_ctx):
    ctx = highcard_ctx
    model = HistGBClassifierModel(ctx)
    proba = _fit_predict(model, ctx)
    assert proba.shape[0] == len(ctx.test_df)
    # driver (~40 levels) is under the 255 cap, so it is a native categorical.
    assert model._model.named_steps["model"].categorical_features == [0 + len(ctx.num_cols_x)]


def test_explicit_target_encoding_wires_through_to_a_non_native_model(tmp_path: Path):
    # `target` encoding uses sklearn's supervised TargetEncoder, so this also
    # checks that y propagates to it through the model pipeline's fit.
    ctx = _build_highcard_ctx(tmp_path, categorical_encoding={"driver": "target"})
    assert ctx.categorical_encoding["driver"] == "target"
    model = registry["LogisticRegression"](ctx)
    proba = _fit_predict(model, ctx)
    assert proba.shape[0] == len(ctx.test_df)
    preprocessor = model._model.named_steps["preprocessor"]
    width = preprocessor.transform(ctx.train_df[ctx.predictor_columns]).shape[1]
    assert width == len(ctx.num_cols_x) + len(ctx.cat_cols_x)  # target enc: 1 col per column


def test_drop_strategy_removes_the_column_for_a_non_native_model(tmp_path: Path):
    ctx = _build_highcard_ctx(tmp_path, categorical_encoding={"driver": "drop"})
    assert ctx.categorical_encoding["driver"] == "drop"
    model = registry["LogisticRegression"](ctx)
    proba = _fit_predict(model, ctx)
    assert proba.shape[0] == len(ctx.test_df)
    # The dropped categorical contributes no features: only the numerics remain.
    preprocessor = model._model.named_steps["preprocessor"]
    width = preprocessor.transform(ctx.train_df[ctx.predictor_columns]).shape[1]
    assert width == len(ctx.num_cols_x)


def test_histgb_encodes_above_native_cardinality_cap(monkeypatch, highcard_ctx):
    # Force the cap below driver's cardinality so it must be encoded instead.
    monkeypatch.setattr(HistGBClassifierModel, "native_cardinality_cap", 5)
    ctx = highcard_ctx
    model = HistGBClassifierModel(ctx)
    proba = _fit_predict(model, ctx)
    assert proba.shape[0] == len(ctx.test_df)
    # No native categorical features: the over-cap column was frequency-encoded.
    assert model._model.named_steps["model"].categorical_features == []
