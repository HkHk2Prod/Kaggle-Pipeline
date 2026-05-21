"""Tests for the forward / inverse target transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kaggle_pipeline.preprocessing import build_target_transforms


def _binary_train_df():
    return pd.DataFrame({"y": ["no", "yes", "no", "yes"]})


def test_binary_forward_maps_to_ints():
    tt = build_target_transforms(
        _binary_train_df(),
        target=["y"],
        target_is_num=False,
        ordered_cats={"y": ["no", "yes"]},
        prediction_aim="category",
    )
    out = tt.forward(_binary_train_df()[["y"]])
    assert list(out) == [0, 1, 0, 1]
    assert tt.width == 2


def test_category_inverse_takes_argmax_label():
    tt = build_target_transforms(
        _binary_train_df(),
        target=["y"],
        target_is_num=False,
        ordered_cats={"y": ["no", "yes"]},
        prediction_aim="category",
    )
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert list(tt.inverse(probs)) == ["no", "yes"]


def test_probability_inverse_returns_positive_column():
    tt = build_target_transforms(
        _binary_train_df(),
        target=["y"],
        target_is_num=False,
        ordered_cats={"y": ["no", "yes"]},
        prediction_aim="probability",
    )
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    np.testing.assert_allclose(tt.inverse(probs), [0.1, 0.8])


def test_multicolumn_target_raises():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="Multicolumn"):
        build_target_transforms(
            df, target=["a", "b"], target_is_num=False, ordered_cats={}, prediction_aim="category"
        )
