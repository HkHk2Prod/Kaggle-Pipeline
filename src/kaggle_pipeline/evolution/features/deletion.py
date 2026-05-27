"""Deletion / deactivation policy for the active feature pool.

The active pool is capped. Original features are protected and are never deletion
candidates; generated features can be deactivated but remain reproducible through
their recipe. A fresh feature is protected by a cooldown so it is not evicted
before it can be used.

Deletion does not rank on intrinsic utility alone -- a feature that elite models
lean on should survive even if its intrinsic score is mediocre:

    deletion_score = feature_utility + active_model_usage_bonus
                   + elite_model_usage_bonus - redundancy_penalty - cost_penalty
"""

from __future__ import annotations

from dataclasses import dataclass

from kaggle_pipeline.evolution.features.genome import FeatureGenome
from kaggle_pipeline.evolution.features.scoring import GENERATION_COST, REDUNDANCY


def _normalized(genome: FeatureGenome, name: str) -> float:
    score = genome.score_set.get(name)
    if score is None:
        return 0.0
    return score.normalized_value if score.normalized_value is not None else score.value


@dataclass
class DeletionPolicy:
    """Scores removable features; the lowest score is evicted first."""

    active_usage_weight: float = 0.10
    elite_usage_weight: float = 0.30
    redundancy_weight: float = 0.50
    cost_weight: float = 0.10

    def score(self, genome: FeatureGenome) -> float:
        usage = genome.usage_stats
        return (
            genome.score_set.utility
            + self.active_usage_weight * usage.times_in_completed_model
            + self.elite_usage_weight * usage.times_in_elite_model
            - self.redundancy_weight * _normalized(genome, REDUNDANCY)
            - self.cost_weight * _normalized(genome, GENERATION_COST)
        )

    def weakest(self, features: list[FeatureGenome]) -> FeatureGenome | None:
        """The lowest-scoring (most evictable) feature, or ``None`` if empty."""
        if not features:
            return None
        return min(features, key=self.score)
