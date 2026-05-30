"""Signed Pearson + Olkin-Pratt small-sample correction used for model penalties."""

from __future__ import annotations

import math

import numpy as np

from kaggle_pipeline.evolution.utils.arrays import (
    pearson_correlation,
    small_sample_adjusted_correlation,
    standardize_for_correlation,
)


def _z(values):
    return standardize_for_correlation(np.asarray(values, dtype=float))


def test_pearson_correlation_preserves_sign():
    # Two perfectly anti-correlated standardized vectors give r = -1.
    a = _z([1.0, 2.0, 3.0, 4.0])
    b = _z([4.0, 3.0, 2.0, 1.0])
    assert math.isclose(pearson_correlation(a, b), -1.0, abs_tol=1e-12)


def test_pearson_correlation_returns_one_for_identical_vectors():
    a = _z([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(pearson_correlation(a, a), 1.0, abs_tol=1e-12)


def test_pearson_correlation_returns_none_on_mismatched_sizes():
    assert pearson_correlation(_z([1.0, 2.0, 3.0]), _z([1.0, 2.0])) is None


def test_olkin_pratt_is_identity_for_huge_samples():
    assert math.isclose(small_sample_adjusted_correlation(0.99, 50_000), 0.99, abs_tol=1e-5)


def test_olkin_pratt_lifts_r_toward_true_rho_at_small_n():
    r, n = 0.9, 10
    adj = small_sample_adjusted_correlation(r, n)
    assert adj > r
    assert math.isclose(adj, r * (1 + (1 - r * r) / (2 * (n - 3))), rel_tol=1e-12)


def test_olkin_pratt_falls_back_to_r_when_n_too_small():
    assert small_sample_adjusted_correlation(0.5, 3) == 0.5
    assert small_sample_adjusted_correlation(0.5, 2) == 0.5


def test_olkin_pratt_handles_zero_and_unit_r():
    assert small_sample_adjusted_correlation(0.0, 100) == 0.0
    assert small_sample_adjusted_correlation(1.0, 100) == 1.0
    assert small_sample_adjusted_correlation(-1.0, 100) == -1.0
