"""Pairwise residual-error correlation cache for OOF predictions.

A model's standardized residual vector (``standardize(oof - y_target)``) does
not change once training is done, so the absolute Pearson correlation between
any two such vectors is fixed for the life of the pair. We exploit that here:
the cache stores one standardized vector per model and one ``|r|`` per
unordered model pair, both updated incrementally by :class:`OOFStore` as it
sees ``store`` / ``remove`` calls. ``compute_correlation_penalties`` then
reads pair values in O(1) instead of redoing the pairwise dot products every
batch.

Standardized vectors are kept as ``float32``: the standardization plus the
dot product is dominated by sample noise long before float32 precision bites,
and the memory footprint halves (200 models × 43k samples × 4 bytes ≈ 34 MB
vs ≈ 68 MB at float64). Pair entries store the *signed* Olkin-Pratt-corrected
Pearson ``r`` so anti-correlated errors (helpful in a blend) survive into the
penalty pass unchanged -- the caller compares against the threshold directly.
"""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.utils.arrays import (
    pearson_correlation,
    small_sample_adjusted_correlation,
    standardize_for_correlation,
)


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _residuals(oof: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute residuals ``oof - y_target`` as a flat float vector.

    Mirrors the shape handling in ``registry._residuals`` so both call sites
    stay consistent. Returns an empty array on shape mismatch -- callers
    treat that as "no correlation available" and skip the model.
    """
    if oof.ndim == 1:
        if oof.shape[0] != y.shape[0]:
            return np.empty(0, dtype=np.float32)
        return (oof.astype(np.float32) - y.astype(np.float32)).ravel()
    if oof.ndim == 2:
        if oof.shape[0] != y.shape[0]:
            return np.empty(0, dtype=np.float32)
        n_classes = oof.shape[1]
        y_int = y.astype(int)
        if y_int.size == 0 or y_int.min() < 0 or y_int.max() >= n_classes:
            return np.empty(0, dtype=np.float32)
        target = np.zeros_like(oof, dtype=np.float32)
        target[np.arange(y_int.size), y_int] = 1.0
        return (oof.astype(np.float32) - target).ravel()
    return np.empty(0, dtype=np.float32)


class OOFCorrelationCache:
    """Incremental store of standardized residuals + pairwise ``|r|``.

    Owned by :class:`OOFStore` and kept in step with it: every store/remove on
    the underlying OOFs is mirrored here. Reads (``correlation`` / ``has``)
    are O(1). Writes (``add``) are O(N) in the number of cached models for
    the pair update -- the cost we used to pay per recompute, now paid once
    per admit.
    """

    def __init__(self) -> None:
        self._z: dict[str, np.ndarray] = {}
        self._r: dict[tuple[str, str], float] = {}
        self._target: np.ndarray | None = None

    # --- target -------------------------------------------------------------
    def set_target(self, y: np.ndarray | None, oofs: dict[str, np.ndarray]) -> None:
        """Bind the residual target ``y`` and rebuild the cache from scratch.

        Called when the pipeline first knows ``y`` (post-``fit``) and again on
        resume. We compare to the bound vector by *identity*: an unchanged
        ``y`` (same object) is a no-op. A new array forces a rebuild because
        every residual is recomputed.
        """
        if self._target is y:
            return
        self._target = y
        self._z.clear()
        self._r.clear()
        if y is None:
            return
        for mid, oof in oofs.items():
            self._add_uncached(mid, oof)

    def has_target(self) -> bool:
        return self._target is not None

    # --- mutation -----------------------------------------------------------
    def add(self, model_id: str, oof: np.ndarray) -> None:
        """Add or replace a model's standardized residual and update its pair entries.

        No-op if the target ``y`` is not bound yet -- the caller (OOFStore)
        will populate the cache lazily when ``set_target`` runs.
        """
        if self._target is None:
            return
        self._add_uncached(model_id, oof)

    def remove(self, model_id: str) -> None:
        """Drop a model's standardized residual and every pair it participates in."""
        if model_id not in self._z:
            return
        del self._z[model_id]
        stale = [pair for pair in self._r if model_id in pair]
        for pair in stale:
            del self._r[pair]

    def clear(self) -> None:
        self._z.clear()
        self._r.clear()
        self._target = None

    # --- read ---------------------------------------------------------------
    def has(self, model_id: str) -> bool:
        return model_id in self._z

    def correlation(self, a: str, b: str) -> float | None:
        """Return the signed Olkin-Pratt-corrected ``r`` for the pair, or ``None`` if absent."""
        if a == b:
            return None
        return self._r.get(_pair_key(a, b))

    def size(self) -> int:
        return len(self._z)

    # --- internals ----------------------------------------------------------
    def _add_uncached(self, model_id: str, oof: np.ndarray) -> None:
        assert self._target is not None
        residuals = _residuals(oof, self._target)
        if residuals.size == 0:
            # Drop any stale entries for this id so a previously-cached value
            # doesn't linger after the OOF is replaced with a mismatched one.
            self.remove(model_id)
            return
        z = standardize_for_correlation(residuals)
        if z is None:
            self.remove(model_id)
            return
        z32 = z.astype(np.float32, copy=False)
        # Drop the model's existing pair entries before re-adding so a replace
        # call cannot leave stale pairs behind.
        if model_id in self._z:
            self.remove(model_id)
        self._z[model_id] = z32
        n = int(z32.size)
        for other_id, other_z in self._z.items():
            if other_id == model_id or other_z.size != z32.size:
                continue
            r = pearson_correlation(z32, other_z)
            if r is None:
                continue
            self._r[_pair_key(model_id, other_id)] = small_sample_adjusted_correlation(r, n)
