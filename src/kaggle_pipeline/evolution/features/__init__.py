"""Global feature layer: recipes, genomes, registry, transformations and scoring.

Generated features are *global* logical definitions
(:class:`~kaggle_pipeline.evolution.features.genome.FeatureGenome`) recorded in the
:class:`~kaggle_pipeline.evolution.features.registry.FeatureRegistry`. Models
reference them by ``feature_id``; they never own feature definitions. See the
README "Evolutionary architecture" section for the full contract.
"""

from __future__ import annotations

from kaggle_pipeline.evolution.features.genome import FeatureGenome, FeatureUsageStats
from kaggle_pipeline.evolution.features.recipe import (
    BOOLEAN,
    CATEGORICAL,
    NUMERIC,
    OUTPUT_TYPES,
    FeatureRecipe,
)
from kaggle_pipeline.evolution.features.scoring import FeatureScoreSet, Score

__all__ = [
    "FeatureRecipe",
    "FeatureGenome",
    "FeatureUsageStats",
    "Score",
    "FeatureScoreSet",
    "NUMERIC",
    "CATEGORICAL",
    "BOOLEAN",
    "OUTPUT_TYPES",
]
