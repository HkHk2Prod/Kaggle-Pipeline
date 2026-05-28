"""Feature generation: proposing new global :class:`FeatureGenome` candidates.

A :class:`FeatureGenerator` samples a transform, samples its parent features from
the registry by selection probability, samples parameters, builds a canonical
recipe (rejecting duplicate hashes), materializes and scores the candidate on the
evaluation context, and hands a *scored* genome to the registry for insertion.

Generation is gene-compatible but global: candidates are global features, not
private model genes. Initially only original features may be parents
(``allow_generated_feature_parents = False``, ``max_feature_depth = 1``); the
registry enforces that via :meth:`FeatureRegistry.get_candidate_parents`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.features.genome import FeatureGenome
from kaggle_pipeline.evolution.features.materialization import MaterializationContext
from kaggle_pipeline.evolution.features.recipe import NUMERIC
from kaggle_pipeline.evolution.features.registry import FeatureRegistry, InsertionResult
from kaggle_pipeline.evolution.features.scoring import (
    GENERATION_COST,
    MISSINGNESS,
    REDUNDANCY,
    TARGET_CORRELATION,
)
from kaggle_pipeline.evolution.features.transformations import (
    FeatureTransformation,
    TransformError,
)
from kaggle_pipeline.evolution.utils.logging import get_logger
from kaggle_pipeline.evolution.utils.random import spawn_rng, weighted_choice

logger = get_logger(__name__)

CREATED = "created"
DUPLICATE = "duplicate"
FAILED = "failed"
NO_PARENTS = "no_parents"


@dataclass
class GenerationOutcome:
    """Result of proposing one candidate (before registry insertion)."""

    status: str
    genome: FeatureGenome | None = None
    transform_name: str = ""
    reason: str = ""


@dataclass
class BatchReport:
    """Summary of one generation batch."""

    proposed: int = 0
    inserted: int = 0
    replaced: int = 0
    duplicates: int = 0
    failed: int = 0
    no_parents: int = 0
    stored_inactive: int = 0
    insertions: list[InsertionResult] = field(default_factory=list)


class FeatureGenerator:
    """Proposes and scores new feature candidates against a registry."""

    def __init__(self, registry: FeatureRegistry, settings: EvolutionSettings):
        self.registry = registry
        self.settings = settings
        # How many times each transform produced an invalid candidate; used to
        # down-weight chronically-failing transforms in selection.
        self.transform_failures: dict[str, int] = {}

    # --- single candidate ---------------------------------------------------
    def generate_candidate(
        self,
        rng: np.random.Generator,
        *,
        context: MaterializationContext,
        y: np.ndarray,
        task: str = "classification",
    ) -> GenerationOutcome:
        transform = self._sample_transform(rng)
        if transform is None:
            return GenerationOutcome(NO_PARENTS, reason="no transform has available parents")

        parents = self._sample_parents(transform, rng)
        if parents is None:
            return GenerationOutcome(
                NO_PARENTS, transform_name=transform.name, reason="not enough parents"
            )

        params = transform.sample_parameters(rng)
        recipe = transform.generate_recipe([p.feature_id for p in parents], params)
        if self.registry.has_recipe_hash(recipe.recipe_hash):
            return GenerationOutcome(DUPLICATE, transform_name=transform.name)

        try:
            transform.validate_inputs(parents)
            parent_values = [self.registry.materialize(p.feature_id, context) for p in parents]
            values = transform.apply(parent_values, recipe.parameters)
        except (TransformError, NotImplementedError, ValueError) as exc:
            reason = getattr(exc, "reason", type(exc).__name__)
            self.transform_failures[transform.name] = (
                self.transform_failures.get(transform.name, 0) + 1
            )
            return GenerationOutcome(FAILED, transform_name=transform.name, reason=reason)

        genome = self._build_genome(transform, parents, recipe)
        self._score_candidate(genome, values, y=y, task=task)
        return GenerationOutcome(CREATED, genome=genome, transform_name=transform.name)

    def _build_genome(
        self,
        transform: FeatureTransformation,
        parents: list[FeatureGenome],
        recipe,
    ) -> FeatureGenome:
        depth = 1 + max(p.depth for p in parents)
        complexity = sum(p.complexity for p in parents) + transform.cost_estimate
        name = transform.generate_name([p.human_name for p in parents], recipe)
        return FeatureGenome(
            recipe=recipe,
            human_name=name,
            created_at_batch=self.registry.current_batch,
            parent_genome_id=parents[0].feature_id if len(parents) == 1 else None,
            depth=depth,
            complexity=complexity,
        )

    def _score_candidate(
        self, genome: FeatureGenome, values: np.ndarray, *, y: np.ndarray, task: str
    ) -> None:
        """Score a candidate's values transiently (without storing similarity state)."""
        scorer = self.registry.scorer
        ss = genome.score_set
        ss.set(TARGET_CORRELATION, scorer.target_correlation(values, y, task=task))
        ss.set(MISSINGNESS, scorer.missingness(values), higher_is_better=False)
        if genome.output_type in (NUMERIC, "boolean"):
            ss.set(
                REDUNDANCY,
                self.registry.similarity.redundancy_of_vector(values),
                higher_is_better=False,
            )
        try:
            cost = self.registry.transformations.get(genome.transform_name).cost_estimate
        except KeyError:
            cost = 1.0
        ss.set(GENERATION_COST, cost, higher_is_better=False)
        # Combine on raw values; the registry re-normalises across the pool after
        # insertion. Good enough for the admission decision.
        self.registry.utility.combine(genome)

    # --- sampling helpers ---------------------------------------------------
    def _sample_transform(self, rng: np.random.Generator) -> FeatureTransformation | None:
        available_types = {g.output_type for g in self.registry.get_candidate_parents()}
        if not available_types:
            return None
        transforms = self.registry.transformations.for_inputs(available_types)
        # Drop transforms whose required parent count cannot be met by type.
        viable = [t for t in transforms if self._has_parents_for(t)]
        if not viable:
            return None
        # Down-weight transforms that keep failing (penalise repeated failures).
        weights = np.array([1.0 / (1.0 + self.transform_failures.get(t.name, 0)) for t in viable])
        weights = weights / weights.sum()
        return viable[int(rng.choice(len(viable), p=weights))]

    def _has_parents_for(self, transform: FeatureTransformation) -> bool:
        input_type = transform.input_types[0]
        return len(self.registry.get_candidate_parents(output_type=input_type)) >= transform.arity

    def _sample_parents(
        self, transform: FeatureTransformation, rng: np.random.Generator
    ) -> list[FeatureGenome] | None:
        input_type = transform.input_types[0]
        pool = self.registry.get_candidate_parents(output_type=input_type)
        if len(pool) < transform.arity:
            return None
        probs = self.registry.compute_selection_probabilities(pool)
        chosen = weighted_choice(rng, probs, transform.arity)
        return [self.registry.get_feature(fid) for fid in chosen]

    # --- batch --------------------------------------------------------------
    def generate_batch(
        self,
        rng: np.random.Generator | None = None,
        *,
        context: MaterializationContext,
        y: np.ndarray,
        task: str = "classification",
        n_candidates: int | None = None,
    ) -> BatchReport:
        """Propose and insert a batch of candidates; return a summary report."""
        rng = spawn_rng(rng)
        n = n_candidates or self.settings.num_feature_candidates_per_batch
        report = BatchReport()
        for _ in range(n):
            report.proposed += 1
            outcome = self.generate_candidate(rng, context=context, y=y, task=task)
            if outcome.status == CREATED and outcome.genome is not None:
                result = self.registry.maybe_insert_generated_feature(outcome.genome)
                report.insertions.append(result)
                if result.status == "inserted":
                    report.inserted += 1
                elif result.status == "replaced":
                    report.replaced += 1
                elif result.status == "stored_inactive":
                    report.stored_inactive += 1
                elif result.status == "duplicate":
                    report.duplicates += 1
            elif outcome.status == DUPLICATE:
                report.duplicates += 1
            elif outcome.status == FAILED:
                report.failed += 1
            else:
                report.no_parents += 1
        logger.info(
            "feature generation: proposed=%d inserted=%d replaced=%d dup=%d failed=%d",
            report.proposed,
            report.inserted,
            report.replaced,
            report.duplicates,
            report.failed,
        )
        return report
