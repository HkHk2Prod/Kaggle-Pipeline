"""The :class:`EnsembleManager` -- finalization into a single prediction.

Selects candidate models that have OOF predictions, builds an ensemble by the
configured strategy (greedy / mean / weighted), scores it on OOF, and -- given
test data -- refits each member on the full train set and weighted-averages their
test probabilities. Falls back to the best single model when there are too few
candidates.
"""

from __future__ import annotations

from collections import Counter
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
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.parameter_spaces import (
    DEFAULT_MIN_MODELS,
    FamilyDefinition,
    resolve_min_count,
)
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

    def __init__(
        self,
        settings: KagglePipelineSettings,
        *,
        families: dict[str, FamilyDefinition] | None = None,
    ):
        self.settings = settings
        # Per-family floor lookup; families absent here fall back to the default.
        self.families = families or {}

    def _eligible_ranking(
        self, population: ModelPopulation, oof_store: OOFStore
    ) -> list[ModelGenome]:
        """Score-filtered ensemble candidates, best-first (uncapped)."""
        min_score = self.settings.ensemble_candidate_min_score
        return [
            g
            for g in population.ensemble_candidate_ranking()
            if g.score_set is not None
            and oof_store.has(g.model_id)
            and (min_score is None or g.score_set.score >= min_score)
        ]

    def family_min_count(self, family: str) -> int:
        """Per-family ensemble floor, resolved against ``ensemble_max_models``."""
        fam = self.families.get(family)
        spec = fam.min_models if fam is not None else DEFAULT_MIN_MODELS
        return resolve_min_count(spec, self.settings.ensemble_max_models)

    def _required_members(self, ranking: list[ModelGenome]) -> list[str]:
        """Each family's top ``min_models`` candidates -- guaranteed a seat."""
        kept: Counter = Counter()
        required: list[str] = []
        for g in ranking:
            if kept[g.family] < self.family_min_count(g.family):
                required.append(g.model_id)
                kept[g.family] += 1
        return required

    def select_candidates(self, population: ModelPopulation, oof_store: OOFStore) -> list[str]:
        """Top candidates up to the cap, with each family's floor guaranteed in.

        The cap (``ensemble_max_models``) is filled best-first, but every
        family's required floor is admitted first -- so a family is never shut
        out of the blend purely because its models score lower. As with elite
        protection in pruning, an oversized floor may push the count past the
        cap rather than evict a required member.
        """
        ranking = self._eligible_ranking(population, oof_store)
        max_models = self.settings.ensemble_max_models
        selected = self._required_members(ranking)
        seen = set(selected)
        for g in ranking:
            if len(selected) >= max_models:
                break
            if g.model_id not in seen:
                selected.append(g.model_id)
                seen.add(g.model_id)
        return selected

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
            # Per-family floor members are forced in so greedy can't drop a
            # family the candidate stage deliberately kept; the candidate set is
            # already family-balanced, so ``mean``/``weighted`` need no seed.
            ranked_genomes = [population.get(mid) for mid in candidates]
            required = [mid for mid in self._required_members(ranked_genomes) if mid in oof_by_id]
            weights, score = greedy_weights(
                candidates,
                oof_by_id,
                y,
                scoring_fn,
                max_models=self.settings.ensemble_max_models,
                min_models=self.settings.ensemble_min_models,
                required_ids=required,
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
        executor: Any = None,
    ) -> np.ndarray:
        """Refit each member on full train data and weighted-average test predictions.

        Members are independent (each rebuilds its own pipeline from its genome
        and reads the registry/materializer in read-only mode), so when an
        ``executor`` is supplied each refit runs in parallel. With 30 members
        on an 8-core box this turns the longest member's wall-time into the
        bottleneck instead of the sum -- typically a 5-8x speed-up on the
        submission window. When ``executor`` is ``None`` we fall back to the
        original sequential loop.
        """
        items = list(result.weights.items())

        def _refit_one(mid: str) -> np.ndarray:
            genome = population.get(mid)
            preds = trainer.fit_predict_test(
                genome, train_frame=train_frame, y=y, test_frame=test_frame, task=task, seed=seed
            )
            return np.asarray(preds, dtype=float)

        if executor is None:
            matrices = [_refit_one(mid) for mid, _ in items]
        else:
            # Preserve member order so the weight alignment is unambiguous --
            # ``submit``/``result`` is order-preserving across a fixed list.
            futures = [executor.submit(_refit_one, mid) for mid, _ in items]
            matrices = [f.result() for f in futures]
        weights = [w for _, w in items]
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
