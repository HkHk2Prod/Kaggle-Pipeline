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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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
    GLOBAL_TRAIN,
    MaterializationContext,
)
from kaggle_pipeline.evolution.features.registry import INSERTED, REPLACED, FeatureRegistry
from kaggle_pipeline.evolution.models.factory import ModelFactory
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus
from kaggle_pipeline.evolution.models.mutation import ModelMutator
from kaggle_pipeline.evolution.models.parameter_spaces import build_default_families
from kaggle_pipeline.evolution.models.registry import ModelPopulation
from kaggle_pipeline.evolution.models.training import ModelTrainer
from kaggle_pipeline.evolution.utils.logging import get_logger
from kaggle_pipeline.evolution.utils.random import spawn_rng

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
    generated_feature_names: list[str] = field(default_factory=list)


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
        self.population = population or ModelPopulation(self.settings, families=self.families)
        # A population handed in (resume / merge) may predate the family-floor
        # wiring; attach the specs so pruning honours per-family minimums.
        if not getattr(self.population, "family_min_models", None):
            self.population.set_family_minimums(self.families)
        self.oof_store = OOFStore()
        # Let the population evict a pruned model's OOF (its only "big" data) while
        # keeping the lightweight genome + scores.
        self.population.oof_store = self.oof_store
        self.credit = CreditAssigner(self.registry, self.settings, oof_store=self.oof_store)
        self.factory = ModelFactory(self.registry, self.settings, families=self.families)
        self.mutator = ModelMutator(self.registry, self.settings, families=self.families)
        self.trainer = ModelTrainer(
            self.registry,
            families=self.families,
            onehot_max_cardinality=self.settings.onehot_max_cardinality,
        )
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
        scoring_ctx: Any,
        y: np.ndarray,
        n_models: int = 8,
        task: str | None = None,
        promote: bool = True,
        executor: Any | None = None,
        should_continue: Callable[[], bool] | None = None,
    ) -> BatchSummary:
        """Run one batch: features, then produce/train/apply ``n_models`` models.

        Training is parallelised across ``executor`` when given. Genomes are
        produced and pre-materialised on the calling (main) thread; workers only
        read the shared materializer cache and registry; results are applied back
        on the main thread. ``should_continue`` (if given) is polled before
        producing each model so a batch can stop launching work near a deadline.
        """
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
            generated_feature_names=[
                r.genome.human_name
                for r in feature_report.insertions
                if r.status in (INSERTED, REPLACED)
            ],
        )

        # 1) Produce genomes (main thread), deduplicating by genome hash.
        produced: list = []
        seeds: list[int] = []
        seen: set[str] = set()
        for _ in range(n_models):
            if should_continue is not None and not should_continue():
                break
            candidate = self.model_controller.produce(self.rng, batch=batch)
            self._tally_action(summary, candidate.action)
            ghash = candidate.genome.genome_hash
            if self.population.has_genome_hash(ghash) or ghash in seen:
                summary.n_skipped += 1
                continue
            seen.add(ghash)
            produced.append(candidate)
            seeds.append(int(self.rng.integers(0, 2**31 - 1)))

        # 2) Pre-materialise every referenced feature once (main thread) so workers
        #    only hit cache reads.
        self._prematerialize(produced, train_frame)

        # 3) Train (parallel when an executor is supplied) and 4) apply results.
        results = self._train_all(
            produced, seeds, train_frame, scoring_ctx, y, splits, task, executor
        )
        for candidate, result in zip(produced, results, strict=True):
            step = self.model_controller.apply_result(candidate, result)
            self._tally_result(summary, step)

        if promote and (should_continue is None or should_continue()):
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

    def _prematerialize(self, produced: list, train_frame: pd.DataFrame) -> None:
        context = MaterializationContext(frame=train_frame, context_id=GLOBAL_TRAIN)
        for candidate in produced:
            for fid in candidate.genome.feature_ids():
                self.registry.materialize(fid, context)

    def _train_all(self, produced, seeds, train_frame, scoring_ctx, y, splits, task, executor):
        def train_one(candidate, seed):
            return self.trainer.train(
                candidate.genome,
                train_frame=train_frame,
                y=y,
                splits=splits,
                ctx=scoring_ctx,
                task=task,
                seed=seed,
            )

        if executor is None:
            return [train_one(c, s) for c, s in zip(produced, seeds, strict=True)]
        futures = [executor.submit(train_one, c, s) for c, s in zip(produced, seeds, strict=True)]
        return [f.result() for f in futures]

    def _tally_action(self, summary: BatchSummary, action: str) -> None:
        if action == GENERATE:
            summary.n_generated += 1
        elif action == MUTATE:
            summary.n_mutated += 1

    def _tally_result(self, summary: BatchSummary, step) -> None:
        if step.result is None:
            return
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
            result = self.trainer.train(
                promoted,
                train_frame=train_frame,
                y=y,
                splits=splits,
                ctx=scoring_ctx,
                task=task,
                seed=int(self.rng.integers(0, 2**31 - 1)),
            )
            promoted.status = result.status
            promoted.score_set = result.score_set
            self.credit.assign_selection(promoted)
            self.population.register(promoted)
            self.oof_store.store(promoted.model_id, result.oof)
            self.population.record_result(promoted)
            if result.status == ModelStatus.COMPLETED:
                promoted.metadata["promoted"] = True
                self.credit.assign_usage_credit(
                    promoted, is_elite=promoted.model_id in self.population.elite
                )
                summary.promoted.append(promoted.model_id)
            return  # one promotion per batch keeps the budget in check

    # --- queries ------------------------------------------------------------
    def best_genome(self) -> ModelGenome | None:
        ranking = self.population.absolute_score_ranking()
        return ranking[0] if ranking else None
