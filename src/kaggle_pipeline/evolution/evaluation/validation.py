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
    from kaggle_pipeline.search.cv import make_cv_splitter

    n = len(y)
    # Clamp to the row count so a tiny search subsample still yields valid folds.
    n_splits = max(2, min(n_splits, n))
    splitter = make_cv_splitter(n_splits=n_splits, seed=seed, task=task)
    if task == "classification":
        return list(splitter.split(np.zeros(n), y))
    return list(splitter.split(np.zeros(n)))
