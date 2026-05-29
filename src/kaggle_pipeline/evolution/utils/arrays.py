"""Small array helpers shared across the evolutionary layer.

Centralises two idioms so each is defined once rather than re-spelled at every
call site: the "is this categorical value missing?" test (a missing marker in an
object array is either ``None`` or a float ``NaN``), and the standardize-then-
correlate primitive used for feature similarity and OOF behaviour deltas.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_EPS = 1e-12


def standardize_for_correlation(values: Any) -> np.ndarray | None:
    """Zero-mean unit-std vector (NaNs/Infs -> 0), or ``None`` if (near-)constant.

    The length-scaled dot product of two such vectors is their Pearson
    correlation (see :func:`abs_correlation`) -- the primitive behind feature
    similarity and OOF behaviour deltas, defined once here rather than re-spelled
    in each. (The v1 leaderboard de-correlation keeps its own unit-*norm* variant:
    it caches float32 residuals and reads correlation as a bare dot product, a
    different trade-off in the base layer that must not depend on this one.)
    """
    x = np.asarray(values, dtype=float).ravel()
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if not np.isfinite(std) or std < _EPS:
        return None
    z = (x - mean) / std
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def abs_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """``|Pearson r|`` of two standardized vectors, or ``None`` if undefined.

    ``a``/``b`` come from :func:`standardize_for_correlation`; their dot product
    divided by length is the correlation.
    """
    corr = pearson_correlation(a, b)
    return abs(corr) if corr is not None else None


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Signed Pearson ``r`` of two standardized vectors, or ``None`` if undefined.

    Same primitive as :func:`abs_correlation` but preserves sign -- needed when
    direction matters (e.g. residual-error correlation: anti-correlated errors
    are *helpful* in a blend and should not be penalised).
    """
    if a.size == 0 or a.size != b.size:
        return None
    corr = float(np.dot(a, b) / a.size)
    return corr if np.isfinite(corr) else None


def small_sample_adjusted_correlation(r: float, n: int) -> float:
    """Olkin-Pratt unbiased estimate of the population correlation from sample ``r``.

    Sample ``r`` is biased toward 0 (``E[r] ≈ ρ − ρ(1−ρ²)/(2(N−1))``); this lifts
    ``|r|`` toward the true ``|ρ|``. For large ``n`` the correction is microscopic;
    at small ``n`` it appreciably tightens the estimate. Falls back to ``r``
    unchanged when ``n`` is too small for the formula to be defined.
    """
    if n <= 3 or not np.isfinite(r):
        return r
    return r * (1.0 + (1.0 - r * r) / (2.0 * (n - 3)))


def is_missing(value: Any) -> bool:
    """True if ``value`` is a missing marker (``None`` or float ``NaN``)."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def missing_mask(values: Any) -> np.ndarray:
    """Boolean mask of missing entries over the raveled values."""
    raveled = np.asarray(values, dtype=object).ravel()
    return np.array([is_missing(v) for v in raveled], dtype=bool)
