"""Out-of-fold prediction store and behaviour-delta computation.

OOF predictions let us measure how *differently* a child model behaves from its
parent: ``behavior_delta = 1 - corr(parent_oof, child_oof)``. A high behaviour
delta means a mutation genuinely changed predictions (not just nudged the score),
which is the multiplier the README uses to weight gene credit. OOF arrays are kept
only for models we still care about (the store is pruned as models are dropped).

The store also owns a sibling :class:`OOFCorrelationCache` keyed on the same
model ids: every ``store`` / ``remove`` cascades into it so pairwise residual
correlations are maintained incrementally and the per-batch correlation-penalty
pass becomes a pure read. The cache is empty until ``set_residual_target`` binds
``y``; before then ``store`` only stashes the OOF.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kaggle_pipeline.evolution.evaluation.oof_correlation_cache import OOFCorrelationCache
from kaggle_pipeline.evolution.utils.arrays import abs_correlation, standardize_for_correlation


class OOFStore:
    """Keeps OOF prediction matrices keyed by model id."""

    def __init__(self) -> None:
        self._oof: dict[str, np.ndarray] = {}
        self.correlation_cache = OOFCorrelationCache()

    def store(self, model_id: str, oof: np.ndarray | None) -> None:
        if oof is None:
            return
        arr = np.asarray(oof, dtype=float)
        self._oof[model_id] = arr
        self.correlation_cache.add(model_id, arr)

    def get(self, model_id: str) -> np.ndarray | None:
        return self._oof.get(model_id)

    def has(self, model_id: str) -> bool:
        return model_id in self._oof

    def remove(self, model_id: str) -> None:
        self._oof.pop(model_id, None)
        self.correlation_cache.remove(model_id)

    def set_residual_target(self, y: np.ndarray | None) -> None:
        """Bind the residual target ``y`` and (re)populate the correlation cache.

        Identity-checked inside the cache, so calling this repeatedly with the
        same array is a no-op. The first non-``None`` call after stores have
        already happened backfills pair entries for every stored OOF.
        """
        self.correlation_cache.set_target(y, self._oof)

    def __getstate__(self) -> dict[str, Any]:
        # Don't pickle the correlation cache -- it's a pure derived view of
        # ``_oof`` + ``y``, cheaper to rebuild on resume than to ship.
        state = self.__dict__.copy()
        state.pop("correlation_cache", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.correlation_cache = OOFCorrelationCache()

    def behavior_delta(self, parent_id: str, child_id: str) -> float | None:
        """``1 - |corr|`` of the two OOF matrices, or ``None`` if unavailable.

        ``None`` covers a missing model, mismatched shapes and a (near-)constant
        OOF -- all surfaced by the shared standardize/correlate helpers.
        """
        a, b = self._oof.get(parent_id), self._oof.get(child_id)
        if a is None or b is None:
            return None
        za = standardize_for_correlation(a)
        zb = standardize_for_correlation(b)
        if za is None or zb is None:
            return None
        corr = abs_correlation(za, zb)
        if corr is None:
            return None
        return float(np.clip(1.0 - corr, 0.0, 1.0))
