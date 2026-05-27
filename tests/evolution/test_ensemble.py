"""Ensemble primitives: proba reconstruction, weighted average, greedy selection."""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.ensemble.greedy import greedy_weights
from kaggle_pipeline.evolution.ensemble.weighted import (
    reconstruct_proba,
    weighted_average,
)
from kaggle_pipeline.scoring.metrics import resolve_scoring


def test_reconstruct_proba_binary():
    full = reconstruct_proba(np.array([[0.2], [0.7]]))
    assert full.shape == (2, 2)
    np.testing.assert_allclose(full[:, 1], [0.8, 0.3])


def test_reconstruct_proba_multiclass():
    full = reconstruct_proba(np.array([[0.2, 0.3], [0.1, 0.1]]))
    assert full.shape == (2, 3)
    np.testing.assert_allclose(full.sum(axis=1), [1.0, 1.0])


def test_weighted_average_normalises_weights():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    out = weighted_average([a, b], [3.0, 1.0])
    np.testing.assert_allclose(out, [[0.75, 0.25]])


def test_greedy_not_worse_than_best_single():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    good = np.array([[0.9], [0.8], [0.2], [0.1], [0.3], [0.7], [0.4], [0.6]])  # P(class0)
    useless = np.full((8, 1), 0.5)
    scoring_fn = resolve_scoring("roc_auc")

    weights, score = greedy_weights(
        ["good", "useless"],
        {"good": good, "useless": useless},
        y,
        scoring_fn,
        max_models=5,
        min_models=1,
    )
    best_single = scoring_fn(y, reconstruct_proba(good))
    assert score >= best_single - 1e-9
    assert abs(sum(weights.values()) - 1.0) < 1e-9
