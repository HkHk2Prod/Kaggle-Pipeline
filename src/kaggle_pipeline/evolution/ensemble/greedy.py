"""Greedy forward selection ensemble (Caruana-style) on OOF predictions.

Start from the single best model, then repeatedly add whichever candidate most
improves the OOF score of the running average (selection *with replacement*, which
naturally yields integer weights). Stop when no addition helps, the model cap is
reached, or the time budget runs low.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from kaggle_pipeline.evolution.ensemble.weighted import reconstruct_proba

if TYPE_CHECKING:
    from kaggle_pipeline.scoring.metrics import ScoringFn


def greedy_weights(
    candidate_ids: list[str],
    oof_by_id: dict[str, np.ndarray],
    y: np.ndarray,
    scoring_fn: ScoringFn,
    *,
    max_models: int = 50,
    min_models: int = 2,
    time_left: Callable[[], bool] | None = None,
) -> tuple[dict[str, float], float]:
    """Return ``(weights, oof_score)`` from greedy forward selection.

    ``weights`` maps model id -> normalised weight (selection multiplicity).
    Falls back to the single best candidate if greedy cannot improve.
    """
    probas = {mid: reconstruct_proba(oof_by_id[mid]) for mid in candidate_ids if mid in oof_by_id}
    ids = list(probas)
    if not ids:
        return {}, float("-inf")

    def score(matrix: np.ndarray) -> float:
        return float(scoring_fn(y, matrix))

    # Seed with the single best model.
    best_single = max(ids, key=lambda m: score(probas[m]))
    selected = [best_single]
    running = probas[best_single].copy()
    best_score = score(running)

    while len(selected) < max_models:
        if time_left is not None and not time_left():
            break
        best_gain_id, best_gain_score, best_gain_sum = None, best_score, None
        for mid in ids:
            trial_sum = running + probas[mid]
            trial_score = score(trial_sum / (len(selected) + 1))
            if trial_score > best_gain_score:
                best_gain_id, best_gain_score, best_gain_sum = mid, trial_score, trial_sum
        if best_gain_id is None:
            if len(selected) >= min_models:
                break
            # Force progress toward the minimum size with the least-harmful add.
            mid = max(ids, key=lambda m: score((running + probas[m]) / (len(selected) + 1)))
            running = running + probas[mid]
            selected.append(mid)
            best_score = score(running / len(selected))
            continue
        running = best_gain_sum
        selected.append(best_gain_id)
        best_score = best_gain_score

    counts = Counter(selected)
    total = sum(counts.values())
    weights = {mid: c / total for mid, c in counts.items()}
    return weights, best_score
