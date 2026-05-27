"""The :class:`EvolutionController` -- the top-level evolutionary loop.

Each batch it: (1) generates and scores feature candidates, inserting good ones and
evicting weak generated ones past the cap; (2) builds one CV split scheme so a
parent and child are compared on identical folds; (3) runs a number of model steps,
each generating a new model or mutating an existing one, training it, scoring it,
and assigning gene/feature credit; (4) optionally promotes a strong elite to a
higher fidelity. Everything is recorded in the population.

It wires the collaborators together but holds little logic itself -- the feature,
model and promotion controllers and the credit assigner each own their concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.controllers.credit_assignment import CreditAssigner
from kaggle_pipeline.evolution.controllers.feature_controller import FeatureController
from kaggle_pipeline.evolution.controllers.model_controller import (
    GENERATE,
    MUTATE,
    ModelController,
)
from kaggle_pipeline.evolution.controllers.promotion_controller import PromotionController
from kaggle_pipeline.evolution.evaluation.oof_store import OOFStore
from kaggle_pipeline.evolution.evaluation.validation import make_cv_splits
from kaggle_pipeline.evolution.features.materialization import (
    FEATURE_EVAL_SAMPLE,
    MaterializationContext,
)
from kaggle_pipeline.evolution.features.registry import FeatureRegistry
from kaggle_pipeline.evolution.models.factory import ModelFactory
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus
from kaggle_pipeline.evolution.models.mutation import ModelMutator
from kaggle_pipeline.evolution.models.parameter_spaces import build_default_families
from kaggle_pipeline.evolution.models.registry import ModelPopulation
from kaggle_pipeline.evolution.models.training import ModelTrainer
from kaggle_pipeline.evolution.utils.logging import get_logger
from kaggle_pipeline.evolution.utils.random import spawn_rng

if TYPE_CHECKING:
    from kaggle_pipeline.context import PipelineContext

logger = get_logger(__name__)


@dataclass
class BatchSummary:
    """A compact summary of one evolutionary batch."""

    batch: int
    n_generated: int = 0
    n_mutated: int = 0
    n_completed: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    n_features_active: int = 0
    n_features_generated: int = 0
    best_score: float | None = None
    promoted: list[str] = field(default_factory=list)


class EvolutionController:
    """Owns the registry, population and collaborators; runs the per-batch loop."""

    def __init__(
        self,
        settings: EvolutionSettings | None = None,
        *,
        registry: FeatureRegistry | None = None,
        population: ModelPopulation | None = None,
        families: dict | None = None,
        n_splits: int = 5,
        seed: int | None = None,
    ):
        self.settings = settings or EvolutionSettings()
        self.families = families or build_default_families()
        self.registry = registry or FeatureRegistry(self.settings)
        self.population = population or ModelPopulation(self.settings)
        self.oof_store = OOFStore()
        self.credit = CreditAssigner(self.registry, self.settings, oof_store=self.oof_store)
        self.factory = ModelFactory(self.registry, self.settings, families=self.families)
        self.mutator = ModelMutator(self.registry, self.settings, families=self.families)
        self.trainer = ModelTrainer(self.registry, families=self.families)
        self.feature_controller = FeatureController(self.registry, self.settings)
        self.model_controller = ModelController(
            self.population,
            self.settings,
            factory=self.factory,
            mutator=self.mutator,
            trainer=self.trainer,
            credit=self.credit,
            oof_store=self.oof_store,
        )
        self.promotion = PromotionController(self.settings, families=self.families)
        self.n_splits = n_splits
        seed = seed if seed is not None else self.settings.default_random_seed
        self.rng = spawn_rng(seed)
        self._eval_context: MaterializationContext | None = None
        self._eval_y: np.ndarray | None = None
        self._task = "classification"

    # --- setup --------------------------------------------------------------
    def initialize_features(
        self,
        originals: list[tuple[str, str]],
        *,
        eval_frame: pd.DataFrame,
        y: np.ndarray,
        task: str = "classification",
    ) -> None:
        """Register original features and score the initial pool on ``eval_frame``."""
        self._eval_context = MaterializationContext(
            frame=eval_frame, context_id=FEATURE_EVAL_SAMPLE
        )
        self._eval_y = np.asarray(y)
        self._task = task
        self.feature_controller.initialize(
            originals, eval_context=self._eval_context, y=self._eval_y, task=task, rng=self.rng
        )

    # --- per-batch loop -----------------------------------------------------
    def run_batch(
        self,
        *,
        train_frame: pd.DataFrame,
        scoring_ctx: PipelineContext,
        y: np.ndarray,
        n_models: int = 8,
        task: str | None = None,
        promote: bool = True,
    ) -> BatchSummary:
        if self._eval_context is None or self._eval_y is None:
            raise RuntimeError("call initialize_features() before run_batch()")
        task = task or self._task
        batch = self.registry.advance_batch()

        feature_report = self.feature_controller.run_batch(
            self.rng, eval_context=self._eval_context, y=self._eval_y, task=task
        )

        split_seed = int(self.rng.integers(0, 2**31 - 1))
        splits = make_cv_splits(y, n_splits=self.n_splits, seed=split_seed, task=task)

        summary = BatchSummary(
            batch=batch,
            n_features_active=len(self.registry.get_active_features()),
            n_features_generated=feature_report.inserted + feature_report.replaced,
        )

        for _ in range(n_models):
            step = self.model_controller.step(
                self.rng,
                batch=batch,
                train_frame=train_frame,
                scoring_ctx=scoring_ctx,
                y=y,
                splits=splits,
                task=task,
            )
            self._tally(summary, step)

        if promote:
            self._promote_step(
                batch=batch,
                train_frame=train_frame,
                scoring_ctx=scoring_ctx,
                y=y,
                splits=splits,
                task=task,
                summary=summary,
            )

        ranking = self.population.absolute_score_ranking()
        if ranking and ranking[0].score_set is not None:
            summary.best_score = ranking[0].score_set.score
        logger.info(
            "batch %d: gen=%d mut=%d done=%d fail=%d skip=%d feats=%d best=%s",
            batch,
            summary.n_generated,
            summary.n_mutated,
            summary.n_completed,
            summary.n_failed,
            summary.n_skipped,
            summary.n_features_active,
            None if summary.best_score is None else round(summary.best_score, 4),
        )
        return summary

    def _tally(self, summary: BatchSummary, step) -> None:
        if step.action == GENERATE:
            summary.n_generated += 1
        elif step.action == MUTATE:
            summary.n_mutated += 1
        if step.skipped:
            summary.n_skipped += 1
        elif step.result is not None:
            if step.result.status == ModelStatus.COMPLETED:
                summary.n_completed += 1
            elif step.result.status == ModelStatus.FAILED:
                summary.n_failed += 1

    def _promote_step(
        self, *, batch, train_frame, scoring_ctx, y, splits, task, summary: BatchSummary
    ) -> None:
        for genome in self.population.elite_genomes():
            if not self.promotion.can_promote(genome):
                continue
            promoted = self.promotion.promote(genome, batch=batch)
            if self.population.has_genome_hash(promoted.genome_hash):
                continue
            self.credit.assign_selection(promoted)
            result = self.trainer.train(
                promoted,
                train_frame=train_frame,
                y=y,
                splits=splits,
                ctx=scoring_ctx,
                task=task,
                seed=int(self.rng.integers(0, 2**31 - 1)),
            )
            self.population.register(promoted)
            self.oof_store.store(promoted.model_id, result.oof)
            self.population.record_result(promoted)
            if result.status == ModelStatus.COMPLETED:
                promoted.status = ModelStatus.PROMOTED
                self.credit.assign_usage_credit(
                    promoted, is_elite=promoted.model_id in self.population.elite
                )
                summary.promoted.append(promoted.model_id)
            return  # one promotion per batch keeps the budget in check

    # --- queries ------------------------------------------------------------
    def best_genome(self) -> ModelGenome | None:
        ranking = self.population.absolute_score_ranking()
        return ranking[0] if ranking else None
