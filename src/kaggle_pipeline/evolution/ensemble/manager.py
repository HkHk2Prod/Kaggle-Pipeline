"""The :class:`EnsembleManager` -- finalization into a single prediction.

Selects candidate models that have OOF predictions, builds an ensemble by the
configured strategy (greedy / mean / weighted), scores it on OOF, and -- given
test data -- refits each member on the full train set and weighted-averages their
test probabilities. Falls back to the best single model when there are too few
candidates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from kaggle_pipeline.evolution.config import KagglePipelineSettings
from kaggle_pipeline.evolution.ensemble.greedy import greedy_weights
from kaggle_pipeline.evolution.ensemble.weighted import (
    equal_weights,
    reconstruct_proba,
    weighted_average,
)
from kaggle_pipeline.evolution.evaluation.oof_store import OOFStore
from kaggle_pipeline.evolution.models.registry import ModelPopulation
from kaggle_pipeline.evolution.utils.logging import get_logger

if TYPE_CHECKING:
    from kaggle_pipeline.evolution.models.training import ModelTrainer

logger = get_logger(__name__)


@dataclass
class EnsembleResult:
    """The finalized ensemble: members, weights, and its OOF score."""

    status: str  # greedy | weighted | mean | single | none
    member_ids: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    oof_score: float | None = None
    n_members: int = 0
    note: str = ""

    def to_serializable(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "enabled": True,
            "member_ids": list(self.member_ids),
            "weights": dict(self.weights),
            "oof_score": self.oof_score,
            "n_members": self.n_members,
            "candidate_count": self.n_members,
            "note": self.note,
        }


class EnsembleManager:
    """Builds and applies the final ensemble from the model population."""

    def __init__(self, settings: KagglePipelineSettings):
        self.settings = settings

    def select_candidates(self, population: ModelPopulation, oof_store: OOFStore) -> list[str]:
        ranking = population.ensemble_candidate_ranking()
        min_score = self.settings.ensemble_candidate_min_score
        candidates = [
            g.model_id
            for g in ranking
            if g.score_set is not None
            and oof_store.has(g.model_id)
            and (min_score is None or g.score_set.score >= min_score)
        ]
        return candidates[: self.settings.ensemble_max_models]

    def build(
        self,
        population: ModelPopulation,
        oof_store: OOFStore,
        y: np.ndarray,
        scoring_fn: Callable[[np.ndarray, np.ndarray], float],
        *,
        time_left: Callable[[], bool] | None = None,
    ) -> EnsembleResult:
        candidates = self.select_candidates(population, oof_store)
        if len(candidates) < self.settings.ensemble_min_models:
            return self._single_fallback(population, oof_store, y, scoring_fn, "too_few_candidates")

        oof_by_id: dict[str, np.ndarray] = {}
        for mid in candidates:
            arr = oof_store.get(mid)
            if arr is not None:
                oof_by_id[mid] = arr
        strategy = self.settings.ensemble_strategy
        if strategy == "greedy":
            weights, score = greedy_weights(
                candidates,
                oof_by_id,
                y,
                scoring_fn,
                max_models=self.settings.ensemble_max_models,
                min_models=self.settings.ensemble_min_models,
                time_left=time_left,
            )
        elif strategy == "weighted":
            weights, score = self._weighted_by_score(
                candidates, population, oof_by_id, y, scoring_fn
            )
        else:  # "mean"
            weights, score = self._mean(candidates, oof_by_id, y, scoring_fn)

        members = [mid for mid in weights if weights[mid] > 0]
        if not members:
            return self._single_fallback(population, oof_store, y, scoring_fn, "empty_selection")
        logger.info("ensemble (%s): %d members, oof_score=%.4f", strategy, len(members), score)
        return EnsembleResult(strategy, members, weights, score, len(members))

    def predict(
        self,
        result: EnsembleResult,
        *,
        trainer: ModelTrainer,
        population: ModelPopulation,
        train_frame: pd.DataFrame,
        y: np.ndarray,
        test_frame: pd.DataFrame,
        task: str = "classification",
        seed: int | None = None,
    ) -> np.ndarray:
        """Refit each member on full train data and weighted-average test predictions."""
        matrices: list[np.ndarray] = []
        weights: list[float] = []
        for mid, weight in result.weights.items():
            genome = population.get(mid)
            preds = trainer.fit_predict_test(
                genome, train_frame=train_frame, y=y, test_frame=test_frame, task=task, seed=seed
            )
            matrices.append(np.asarray(preds, dtype=float))
            weights.append(weight)
        return weighted_average(matrices, weights)

    # --- strategies ---------------------------------------------------------
    def _single_fallback(self, population, oof_store, y, scoring_fn, note) -> EnsembleResult:
        ranking = population.absolute_score_ranking()
        if not ranking:
            return EnsembleResult("none", note="no completed models")
        best = ranking[0]
        score = None
        if oof_store.has(best.model_id):
            score = float(scoring_fn(y, reconstruct_proba(oof_store.get(best.model_id))))
        logger.warning("ensemble falling back to best single model (%s)", note)
        return EnsembleResult("single", [best.model_id], {best.model_id: 1.0}, score, 1, note)

    def _mean(self, candidates, oof_by_id, y, scoring_fn):
        weights = dict(zip(candidates, equal_weights(len(candidates)), strict=True))
        score = self._score_weighted(weights, oof_by_id, y, scoring_fn)
        return weights, score

    def _weighted_by_score(self, candidates, population, oof_by_id, y, scoring_fn):
        raw = np.array([max(0.0, population.get(mid).score_set.score) for mid in candidates])
        raw = raw / raw.sum() if raw.sum() > 0 else np.full(len(candidates), 1.0 / len(candidates))
        weights = dict(zip(candidates, raw.tolist(), strict=True))
        score = self._score_weighted(weights, oof_by_id, y, scoring_fn)
        return weights, score

    @staticmethod
    def _score_weighted(weights, oof_by_id, y, scoring_fn) -> float:
        ids = list(weights)
        matrices = [reconstruct_proba(oof_by_id[mid]) for mid in ids]
        blended = weighted_average(matrices, [weights[mid] for mid in ids])
        return float(scoring_fn(y, blended))
