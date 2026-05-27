"""The :class:`ModelController` -- one model step: generate-or-mutate, train, credit.

Decides between generating a new model and mutating an existing one (with the
configured exploration floors so neither action ever stops), builds the genome,
skips it if an identical genome hash was already trained, trains it, registers the
result in the population, and assigns gene/feature credit. Behaviour deltas come
from the OOF store when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.controllers.credit_assignment import CreditAssigner
from kaggle_pipeline.evolution.evaluation.oof_store import OOFStore
from kaggle_pipeline.evolution.models.factory import ModelFactory
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus
from kaggle_pipeline.evolution.models.mutation import ModelMutator, MutationRecord
from kaggle_pipeline.evolution.models.registry import ModelPopulation
from kaggle_pipeline.evolution.models.training import ModelResult, ModelTrainer
from kaggle_pipeline.evolution.utils.random import spawn_rng

if TYPE_CHECKING:
    from kaggle_pipeline.context import PipelineContext

GENERATE = "generate"
MUTATE = "mutate"


@dataclass
class ProducedModel:
    """A genome produced (generated or mutated) but not yet trained."""

    action: str
    genome: ModelGenome
    parent: ModelGenome | None = None
    record: MutationRecord | None = None


@dataclass
class StepResult:
    """Outcome of one model step."""

    action: str
    genome: ModelGenome
    result: ModelResult | None = None
    record: MutationRecord | None = None
    skipped: bool = False


class ModelController:
    """Generates or mutates one model per step, trains it, and assigns credit."""

    def __init__(
        self,
        population: ModelPopulation,
        settings: EvolutionSettings,
        *,
        factory: ModelFactory,
        mutator: ModelMutator,
        trainer: ModelTrainer,
        credit: CreditAssigner,
        oof_store: OOFStore | None = None,
    ):
        self.population = population
        self.settings = settings
        self.factory = factory
        self.mutator = mutator
        self.trainer = trainer
        self.credit = credit
        self.oof_store = oof_store

    def choose_action(self, rng: np.random.Generator) -> str:
        """Generate vs. mutate, respecting the configured exploration floors."""
        if not self.population.active:
            return GENERATE
        # Baseline 40% generate / 60% mutate, clamped so neither drops below floor.
        p_generate = min(
            1.0 - self.settings.p_mutate_existing_model_floor,
            max(self.settings.p_generate_new_model_floor, 0.4),
        )
        return GENERATE if rng.random() < p_generate else MUTATE

    def produce(self, rng: np.random.Generator, *, batch: int) -> ProducedModel:
        """Generate or mutate one genome (no training). Runs on the main thread."""
        rng = spawn_rng(rng)
        action = self.choose_action(rng)
        parent: ModelGenome | None = None
        record: MutationRecord | None = None
        if action == MUTATE:
            parent = self.population.select_parent(rng)
            if parent is None:
                action = GENERATE
        if action == MUTATE and parent is not None:
            genome, record = self.mutator.mutate(parent, rng, batch=batch)
        else:
            action = GENERATE
            genome = self.factory.generate(rng, batch=batch)
        return ProducedModel(action, genome, parent, record)

    def apply_result(self, produced: ProducedModel, result: ModelResult) -> StepResult:
        """Apply a (worker-computed) result to the genome and registries (main thread)."""
        genome = produced.genome
        genome.status = result.status
        genome.score_set = result.score_set
        self.credit.assign_selection(genome)
        self.population.register(genome)
        if self.oof_store is not None:
            self.oof_store.store(genome.model_id, result.oof)
        self.population.record_result(genome)
        if result.status == ModelStatus.COMPLETED:
            self._assign_credit(genome, produced.parent, produced.record)
        if produced.record is not None:
            self.population.add_mutation_record(produced.record)
        return StepResult(produced.action, genome, result, produced.record)

    def step(
        self,
        rng: np.random.Generator,
        *,
        batch: int,
        train_frame: pd.DataFrame,
        scoring_ctx: PipelineContext,
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        task: str = "classification",
    ) -> StepResult:
        """Sequential produce -> train -> apply (used when no executor is supplied)."""
        rng = spawn_rng(rng)
        seed = int(rng.integers(0, 2**31 - 1))
        produced = self.produce(rng, batch=batch)
        if self.population.has_genome_hash(produced.genome.genome_hash):
            return StepResult(
                produced.action, produced.genome, skipped=True, record=produced.record
            )
        result = self.trainer.train(
            produced.genome,
            train_frame=train_frame,
            y=y,
            splits=splits,
            ctx=scoring_ctx,
            task=task,
            seed=seed,
        )
        return self.apply_result(produced, result)

    def _assign_credit(
        self,
        genome: ModelGenome,
        parent: ModelGenome | None,
        record: MutationRecord | None,
    ) -> None:
        is_elite = genome.model_id in self.population.elite
        if record is not None and parent is not None:
            behavior = (
                self.oof_store.behavior_delta(parent.model_id, genome.model_id)
                if self.oof_store is not None
                else None
            )
            record.attach_outcome(parent, genome, behavior_delta=behavior)
            self.credit.assign_gene_credit(record, genome)
            self.credit.assign_feature_mutation_credit(record)
        self.credit.assign_usage_credit(genome, is_elite=is_elite)
