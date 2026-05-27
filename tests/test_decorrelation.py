"""Tests for the residual-correlation redundancy rule and greedy de-correlation.

These cover the pure maths (the Fisher z lower confidence bound) and the
score-ordered greedy selection, with no models or leaderboard involved -- the
leaderboard-level wiring is exercised in ``test_judge_decorrelation.py``.
"""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.search.decorrelation import (
    residual_correlation_lower_bound,
    select_redundant,
    select_redundant_indices,
    standardize,
)


def test_lower_bound_approaches_observed_on_large_data():
    # With many rows the confidence interval is tiny, so the bound ~ the observed r.
    r = 0.99
    bound = residual_correlation_lower_bound(r, n_eff=10_000_000)
    assert abs(bound - r) < 1e-3


def test_lower_bound_is_below_observed_and_widens_as_data_shrinks():
    r = 0.99
    big = residual_correlation_lower_bound(r, n_eff=100_000)
    small = residual_correlation_lower_bound(r, n_eff=50)
    assert small < big < r  # smaller dataset -> looser (lower) bound


def test_small_data_does_not_prune_a_noisy_high_correlation():
    # r = 0.98 measured on only 50 rows: not enough evidence to clear tau = 0.98.
    assert residual_correlation_lower_bound(0.98, n_eff=50) < 0.98


def test_clearly_real_correlation_clears_tau():
    assert residual_correlation_lower_bound(0.999, n_eff=10_000) > 0.98


def test_tiny_n_eff_never_prunes():
    # n_eff <= 3 leaves the standard error undefined; the bound must sit below any tau.
    assert residual_correlation_lower_bound(0.999, n_eff=3) <= 0.0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def test_identical_residuals_drop_the_later_worse_model():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    # Input is score-ordered (best first); index 1 is the worse copy -> dropped.
    assert select_redundant_indices([a, a.copy()], n_eff=500, tau=0.98) == {1}


def test_independent_residuals_are_all_kept():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=500), rng.normal(size=500)
    assert abs(_corr(a, b)) < 0.2  # genuinely uncorrelated
    assert select_redundant_indices([a, b], n_eff=500, tau=0.98) == set()


def test_anticorrelated_residuals_are_kept():
    # Negatively correlated errors cancel in an ensemble -- valuable, never pruned.
    rng = np.random.default_rng(2)
    a = rng.normal(size=500)
    assert select_redundant_indices([a, -a], n_eff=500, tau=0.98) == set()


def test_only_the_redundant_member_is_dropped_in_a_mixed_set():
    rng = np.random.default_rng(3)
    a = rng.normal(size=2000)
    near_copy = a + 1e-4 * rng.normal(size=2000)  # correlation ~ 1
    independent = rng.normal(size=2000)
    drop = select_redundant_indices([a, near_copy, independent], n_eff=2000, tau=0.98)
    assert drop == {1}  # keep the best (0) and the diverse one (2)


def test_select_redundant_reports_the_match_and_correlation():
    # The richer return maps each dropped index to (the better model it duplicates,
    # the observed correlation) so callers can explain why it was evicted.
    rng = np.random.default_rng(5)
    a = rng.normal(size=2000)
    near_copy = a + 1e-4 * rng.normal(size=2000)
    independent = rng.normal(size=2000)
    units = [standardize(a), standardize(near_copy), standardize(independent)]
    dropped = select_redundant(units, n_eff=2000, tau=0.98)
    assert set(dropped) == {1}  # only the near-copy goes
    kept_index, corr = dropped[1]
    assert kept_index == 0  # duplicated the better-scoring model at index 0
    assert corr > 0.99  # near-identical residuals


def test_zero_variance_residual_is_kept_and_ignored():
    # A perfect model (all-zero residual) has no defined correlation: keep it, and
    # don't let it suppress a genuinely independent model either.
    rng = np.random.default_rng(4)
    perfect = np.zeros(500)
    other = rng.normal(size=500)
    assert select_redundant_indices([perfect, other], n_eff=500, tau=0.98) == set()
