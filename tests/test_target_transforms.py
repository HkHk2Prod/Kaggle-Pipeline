"""Tests for the forward / inverse target transforms."""

from __future__ import annotations

import logging

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


def test_numeric_target_orders_classes_deterministically():
    """A numeric 0/1 target is never typed into ordered_cats, so it hits the
    fallback ordering. That ordering must be sorted, not first-appearance: column
    1 (the positive-class probability) must be P(class 1) regardless of row order.

    Regression test: an unsorted target whose first row is the positive class used
    to invert the submission (P(class 0) = 1 - P(positive)), turning a 0.95 CV into
    a ~0.05 leaderboard score.
    """
    # First row is the positive class (1) -- the case that used to invert.
    train = pd.DataFrame({"y": [1, 0, 1, 0, 1]})
    tt = build_target_transforms(
        train, target=["y"], target_is_num=False, ordered_cats={}, prediction_aim="probability"
    )
    # forward must map class 0 -> 0 and class 1 -> 1 (sorted), not flip them.
    assert list(tt.forward(train[["y"]])) == [1, 0, 1, 0, 1]
    # predict_proba columns follow sklearn classes_ = [0, 1] = [P(0), P(1)].
    probs = np.array([[0.05, 0.95], [0.8, 0.2]])
    np.testing.assert_allclose(tt.inverse(probs), [0.95, 0.2])  # returns P(class 1)

    # The category aim must label by the same sorted order.
    tt_cat = build_target_transforms(
        train, target=["y"], target_is_num=False, ordered_cats={}, prediction_aim="category"
    )
    assert list(tt_cat.inverse(probs)) == [1, 0]


def _capture_target_warnings(build_call) -> list[str]:
    """Run ``build_call`` and return messages logged by the target module.

    Captures via a handler on the module logger directly: robust to the package's
    logging config, which disables propagation to the root (where caplog sits).
    """
    messages: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())  # type: ignore[method-assign]
    tgt_logger = logging.getLogger("kaggle_pipeline.preprocessing.target")
    tgt_logger.addHandler(handler)
    previous_level = tgt_logger.level
    tgt_logger.setLevel(logging.WARNING)
    try:
        build_call()
    finally:
        tgt_logger.removeHandler(handler)
        tgt_logger.setLevel(previous_level)
    return messages


def test_ambiguous_binary_positive_class_warns():
    """A binary probability submission with an unconventional class ordering warns.

    ``cat``/``dog`` is neither a 0/1 target nor an order_lists entry, so which class
    is "positive" (the submitted column) is a guess that may invert the leaderboard.
    """
    train = pd.DataFrame({"y": ["cat", "dog", "cat", "dog"]})
    messages = _capture_target_warnings(
        lambda: build_target_transforms(
            train,
            target=["y"],
            target_is_num=False,
            ordered_cats={"y": ["cat", "dog"]},
            prediction_aim="probability",
        )
    )
    assert any("AMBIGUOUS positive class" in m for m in messages)
    # The message names the guessed positive class plainly, not as a numpy scalar.
    assert any("P('dog')" in m for m in messages)


def test_conventional_binary_positive_class_does_not_warn():
    """No warning for a 0/1 target, an order_lists-matched target, or a category aim."""
    train_num = pd.DataFrame({"y": [1, 0, 1, 0]})
    train_yn = pd.DataFrame({"y": ["no", "yes", "no", "yes"]})
    messages = _capture_target_warnings(
        lambda: (
            # 0/1 numeric target: sorted -> [0, 1], P(1) is the convention.
            build_target_transforms(
                train_num, target=["y"], target_is_num=False, ordered_cats={},
                prediction_aim="probability",
            ),
            # Values matching a configured order_lists entry: ordering is intentional.
            build_target_transforms(
                train_yn, target=["y"], target_is_num=False, ordered_cats={"y": ["no", "yes"]},
                prediction_aim="probability", order_lists=[["no", "yes"]],
            ),
            # category aim can't invert (order[argmax] recovers the right label).
            build_target_transforms(
                train_yn, target=["y"], target_is_num=False, ordered_cats={"y": ["cat", "dog"]},
                prediction_aim="category",
            ),
        )
    )
    assert not [m for m in messages if "AMBIGUOUS positive class" in m]


def test_multicolumn_target_raises():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="Multicolumn"):
        build_target_transforms(
            df, target=["a", "b"], target_is_num=False, ordered_cats={}, prediction_aim="category"
        )
