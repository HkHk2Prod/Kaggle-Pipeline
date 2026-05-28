"""Prediction-combination primitives shared by the ensemble strategies.

Training stores cross-validated OOF predictions with the *last* class column
dropped (it is redundant given the others sum to one). :func:`reconstruct_proba`
restores the full probability matrix so scoring and averaging are correct for both
binary and multiclass targets.
"""

from __future__ import annotations

import numpy as np


def reconstruct_proba(oof: np.ndarray) -> np.ndarray:
    """Restore a full probability matrix from a column-dropped OOF array.

    Binary OOF ``(n, 1)`` -> ``(n, 2)`` as ``[p, 1 - p]``; multiclass ``(n, k-1)``
    -> ``(n, k)`` by appending ``1 - rowsum``. A 1-D array is treated as a single
    column. Regression-style single columns pass through as ``(n, 1)``.
    """
    arr = np.asarray(oof, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    last = 1.0 - arr.sum(axis=1, keepdims=True)
    return np.hstack([arr, last])


def weighted_average(matrices: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted average of equally-shaped prediction matrices (weights renormalised)."""
    if not matrices:
        raise ValueError("no matrices to average")
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(matrices), 1.0 / len(matrices))
    stacked = np.stack([np.asarray(m, dtype=float) for m in matrices], axis=0)
    return np.tensordot(w, stacked, axes=(0, 0))


def equal_weights(n: int) -> list[float]:
    return [1.0 / n] * n if n else []
