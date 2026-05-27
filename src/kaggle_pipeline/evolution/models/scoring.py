"""Model scoring: the extensible :class:`ModelScoreSet` and the utility model.

Initial scores are ``score`` / ``score_std`` / ``compute_time``; many more are
reserved (see :class:`ReservedMetrics`). Every metric is converted to an internal
**larger-is-better** convention before it is stored, so utility maths never has to
branch on direction.

Utility is **cost-aware** and only meaningful *within* a comparable set (same
competition, metric, validation scheme and fidelity level):

    adj_score     = score - score_std_penalty * score_std
    t_ref         = median compute time over comparable trials
    model_utility = (adj_score - comparable_adj_avg) / (1 + log(1 + time / t_ref))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from kaggle_pipeline.evolution.config import EvolutionSettings


class ReservedMetrics:
    """Names reserved for future scores so call sites can rely on stable keys."""

    TRAIN_SCORE = "train_score"
    VALIDATION_SCORE = "validation_score"
    SCORE_GAP = "score_gap"
    MEMORY_USAGE = "memory_usage"
    INFERENCE_TIME = "inference_time"
    NUMBER_OF_FEATURES = "number_of_features"
    MATERIALIZED_WIDTH = "materialized_width"
    MODEL_SIZE = "model_size"
    PREDICTION_DIVERSITY = "prediction_diversity"
    FOLD_STABILITY = "fold_stability"
    PUBLIC_LEADERBOARD_SCORE = "public_leaderboard_score"
    ENSEMBLE_CONTRIBUTION = "ensemble_contribution"
    FAILURE_PENALTY = "failure_penalty"


def to_internal(raw_metric: float, higher_is_better: bool) -> float:
    """Convert a metric to the internal larger-is-better convention."""
    return float(raw_metric) if higher_is_better else -float(raw_metric)


@dataclass
class ModelScoreSet:
    """Scores for one trained model. ``score`` is already larger-is-better."""

    score: float = 0.0
    score_std: float = 0.0
    compute_time: float = 0.0
    n_features: int = 0
    fidelity_level: int = 1
    failure_penalty: float = 0.0
    # Reserved/extra metrics keyed by name (see :class:`ReservedMetrics`).
    extra: dict[str, float] = field(default_factory=dict)

    def adj_score(self, std_penalty: float) -> float:
        """Stability-penalised score: ``score - std_penalty * score_std``."""
        return self.score - std_penalty * self.score_std - self.failure_penalty

    def set(self, name: str, value: float) -> None:
        self.extra[name] = float(value)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "score_std": self.score_std,
            "compute_time": self.compute_time,
            "n_features": self.n_features,
            "fidelity_level": self.fidelity_level,
            "failure_penalty": self.failure_penalty,
            "extra": dict(self.extra),
        }


@dataclass
class ComparableStats:
    """Summary of a comparable set used to normalise a model's utility."""

    adj_avg: float = 0.0
    t_ref: float = 1.0
    n: int = 0


def comparable_stats(score_sets: list[ModelScoreSet], *, std_penalty: float) -> ComparableStats:
    """Robust averages over a comparable set: mean adj-score, median compute time."""
    if not score_sets:
        return ComparableStats()
    adj = np.array([s.adj_score(std_penalty) for s in score_sets])
    times = np.array([s.compute_time for s in score_sets if s.compute_time > 0])
    t_ref = float(np.median(times)) if times.size else 1.0
    return ComparableStats(adj_avg=float(adj.mean()), t_ref=max(t_ref, 1e-6), n=len(score_sets))


class ModelUtility:
    """Computes the cost-aware utility of a model against a comparable set."""

    def __init__(self, settings: EvolutionSettings):
        self.settings = settings

    def adj_score(self, score_set: ModelScoreSet) -> float:
        return score_set.adj_score(self.settings.score_std_penalty)

    def utility(self, score_set: ModelScoreSet, comparable: ComparableStats) -> float:
        """Cost-aware utility relative to ``comparable`` (same fidelity/metric)."""
        numerator = self.adj_score(score_set) - comparable.adj_avg
        if self.settings.compute_penalty_enabled and comparable.t_ref > 0:
            denom = 1.0 + math.log1p(score_set.compute_time / comparable.t_ref)
        else:
            denom = 1.0
        return float(numerator / denom)
