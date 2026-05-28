"""Out-of-fold prediction store and behaviour-delta computation.

OOF predictions let us measure how *differently* a child model behaves from its
parent: ``behavior_delta = 1 - corr(parent_oof, child_oof)``. A high behaviour
delta means a mutation genuinely changed predictions (not just nudged the score),
which is the multiplier the README uses to weight gene credit. OOF arrays are kept
only for models we still care about (the store is pruned as models are dropped).
"""

from __future__ import annotations

import numpy as np


class OOFStore:
    """Keeps OOF prediction matrices keyed by model id."""

    def __init__(self) -> None:
        self._oof: dict[str, np.ndarray] = {}

    def store(self, model_id: str, oof: np.ndarray | None) -> None:
        if oof is not None:
            self._oof[model_id] = np.asarray(oof, dtype=float)

    def get(self, model_id: str) -> np.ndarray | None:
        return self._oof.get(model_id)

    def has(self, model_id: str) -> bool:
        return model_id in self._oof

    def remove(self, model_id: str) -> None:
        self._oof.pop(model_id, None)

    def behavior_delta(self, parent_id: str, child_id: str) -> float | None:
        """``1 - |corr|`` of the two OOF matrices, or ``None`` if unavailable."""
        a, b = self._oof.get(parent_id), self._oof.get(child_id)
        if a is None or b is None:
            return None
        fa, fb = a.ravel(), b.ravel()
        if fa.shape != fb.shape or fa.size < 2:
            return None
        if np.std(fa) < 1e-12 or np.std(fb) < 1e-12:
            return None
        corr = float(np.corrcoef(fa, fb)[0, 1])
        if not np.isfinite(corr):
            return None
        return float(np.clip(1.0 - abs(corr), 0.0, 1.0))
