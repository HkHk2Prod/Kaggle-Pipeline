"""Cross-validation split construction for the evolutionary trainer.

Mirrors the v1 search's stratified K-fold scheme so reported scores are
comparable. Splits are built once per batch and reused across the genomes trained
in that batch, which also keeps fold assignment identical for parent and child --
important for a clean behaviour-delta comparison.
"""

from __future__ import annotations

import numpy as np


def make_cv_splits(
    y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int | None = None,
    task: str = "classification",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``(train_idx, val_idx)`` folds; stratified for classification."""
    from sklearn.model_selection import KFold, StratifiedKFold

    n = len(y)
    n_splits = max(2, min(n_splits, n))
    if task == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(n), y))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(n)))
