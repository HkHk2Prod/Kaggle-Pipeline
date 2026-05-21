"""Tests for column helpers and the preprocessing transformers."""

from __future__ import annotations

import warnings

import pandas as pd

from kaggle_pipeline.preprocessing import (
    CategoricalTyper,
    FeatureEngineer,
    OrdinalEncoderTransformer,
    detect_ordinal_order_cols,
    is_num_check,
    split_num_cat,
)


def test_detect_ordinal_is_case_insensitive():
    # Mixed case must still match the lower-case ordering list.
    df = pd.DataFrame({"size": ["Low", "HIGH", "medium"], "other": ["a", "b", "c"]})
    result = detect_ordinal_order_cols(df, [["low", "medium", "high"]])
    assert "size" in result
    assert result["size"] == ["Low", "medium", "HIGH"]  # natural order, original casing
    assert "other" not in result


def test_is_num_check_for_graph_uses_cutoff():
    df = pd.DataFrame({"x": [1, 1, 2, 2, 3]})  # numeric, only 3 uniques
    assert is_num_check("x", df) is True
    assert is_num_check("x", df, for_graph=True, cat_cutoff=5) is False


def test_split_num_cat():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    num, cat = split_num_cat(["a", "b"], df)
    assert num == ["a"] and cat == ["b"]


def test_feature_engineer_casts_bool_to_int():
    fe = FeatureEngineer(expressions=["flag = a > 1"])
    out = fe.fit_transform(pd.DataFrame({"a": [0, 2, 3]}))
    assert list(out["flag"]) == [0, 1, 1]
    assert out["flag"].dtype.kind in "iu"


def test_ordinal_encoder_maps_known_orderings():
    enc = OrdinalEncoderTransformer(target=["y"], order_lists=[["low", "medium", "high"]])
    df = pd.DataFrame({"size": ["low", "high", "medium"], "y": ["a", "b", "c"]})
    out = enc.fit_transform(df)
    assert list(out["size"]) == [0, 2, 1]
    assert list(out["y"]) == ["a", "b", "c"]  # target untouched


def test_categorical_typer_orders_and_retypes():
    typer = CategoricalTyper(cat_cutoff=5, cat_order_list=["no", "yes"])
    df = pd.DataFrame({"flag": ["yes", "no", "yes"]})
    out = typer.fit_transform(df)
    assert isinstance(out["flag"].dtype, pd.CategoricalDtype)
    assert list(out["flag"].cat.categories) == ["no", "yes"]


def test_categorical_typer_maps_unseen_levels_to_nan_without_deprecation():
    # A test-only category (here "maybe") must become NaN without tripping the
    # pandas deprecation for constructing a Categorical with out-of-dtype values.
    typer = CategoricalTyper(cat_cutoff=5, cat_order_list=["no", "yes"])
    typer.fit(pd.DataFrame({"flag": ["yes", "no", "yes"]}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = typer.transform(pd.DataFrame({"flag": ["yes", "maybe"]}))
    assert out["flag"].isna().tolist() == [False, True]  # unseen -> NaN
    assert list(out["flag"].cat.categories) == ["no", "yes"]  # ordering kept
    assert not any("Constructing a Categorical" in str(w.message) for w in caught)
