"""The :class:`FeatureController` -- per-batch feature generation and scoring.

Wraps the registry and the generator: registers originals at start-up, and each
batch proposes a batch of candidates, inserts the useful ones (evicting weak
generated features past the cap), then rescoring the active pool so selection
probabilities reflect the new state.
"""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.features.generation import BatchReport, FeatureGenerator
from kaggle_pipeline.evolution.features.materialization import MaterializationContext
from kaggle_pipeline.evolution.features.registry import FeatureRegistry


class FeatureController:
    """Drives feature generation, insertion and scoring for the registry."""

    def __init__(
        self,
        registry: FeatureRegistry,
        settings: EvolutionSettings,
        *,
        generator: FeatureGenerator | None = None,
    ):
        self.registry = registry
        self.settings = settings
        self.generator = generator or FeatureGenerator(registry, settings)

    def initialize(
        self,
        originals: list[tuple[str, str]],
        *,
        eval_context: MaterializationContext,
        y: np.ndarray,
        task: str = "classification",
        rng: np.random.Generator | None = None,
    ) -> None:
        """Register original ``(column, output_type)`` features and score the pool."""
        for column, output_type in originals:
            self.registry.add_original_feature(column, output_type)
        self.registry.rescore_active(context=eval_context, y=y, task=task, rng=rng)

    def run_batch(
        self,
        rng: np.random.Generator,
        *,
        eval_context: MaterializationContext,
        y: np.ndarray,
        task: str = "classification",
    ) -> BatchReport:
        """Generate + insert candidates, then rescore the active pool."""
        report = self.generator.generate_batch(rng, context=eval_context, y=y, task=task)
        self.registry.rescore_active(context=eval_context, y=y, task=task, rng=rng)
        return report
