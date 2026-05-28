"""Deletion / deactivation policy for the active feature pool.

The active pool is capped. Original features are protected and are never deletion
candidates; generated features can be deactivated but remain reproducible through
their recipe. A fresh feature is protected by a cooldown so it is not evicted
before it can be used.

Deletion does not rank on intrinsic utility alone -- a feature that elite models
lean on should survive even if its intrinsic score is mediocre. Usage credit is
counted *per batch alive*, not as a raw count, so a long-lived feature does not
accumulate an unbounded head start over a newcomer:

    deletion_score = feature_utility
                   + active_usage_weight  * (completed_uses / max(1, age))
                   + elite_usage_weight   * (elite_uses     / max(1, age))
                   - redundancy_weight    * normalized_redundancy
                   - cost_weight          * normalized_generation_cost
"""

from __future__ import annotations

from dataclasses import dataclass

from kaggle_pipeline.evolution.features.genome import FeatureGenome
from kaggle_pipeline.evolution.features.scoring import GENERATION_COST, REDUNDANCY


@dataclass
class DeletionPolicy:
    """Scores removable features; the lowest score is evicted first."""

    active_usage_weight: float = 0.10
    elite_usage_weight: float = 0.30
    redundancy_weight: float = 0.50
    cost_weight: float = 0.10

    def score(self, genome: FeatureGenome, current_batch: int) -> float:
        usage = genome.usage_stats
        age = max(1, current_batch - genome.created_at_batch)
        return (
            genome.score_set.utility
            + self.active_usage_weight * (usage.times_in_completed_model / age)
            + self.elite_usage_weight * (usage.times_in_elite_model / age)
            - self.redundancy_weight * genome.score_set.normalized(REDUNDANCY)
            - self.cost_weight * genome.score_set.normalized(GENERATION_COST)
        )

    def weakest(self, features: list[FeatureGenome], current_batch: int) -> FeatureGenome | None:
        """The lowest-scoring (most evictable) feature, or ``None`` if empty."""
        if not features:
            return None
        return min(features, key=lambda g: self.score(g, current_batch))
