"""Model mutation: deriving a child :class:`ModelGenome` from a parent.

**Mutation creates a child and never changes the parent.** The mutator clones the
parent's genes (faithful copies), mutates a *small* number of them (the count is
drawn from the configured distribution), and assembles a new genome with a
:class:`MutationRecord`. After the child trains, the record's
:meth:`attach_outcome` fills the parent/child deltas that drive gene and feature
credit assignment.

Resource genes are never touched here -- raising fidelity is *promotion*, handled
elsewhere. Changing model family is a *new genome* from the factory, not a
mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.features.recipe import CATEGORICAL
from kaggle_pipeline.evolution.features.registry import FeatureRegistry
from kaggle_pipeline.evolution.genes.base import MutationContext
from kaggle_pipeline.evolution.genes.encoding_gene import (
    FREQUENCY,
    NATIVE,
    NATIVE_CAPABLE_ENCODINGS,
    NON_NATIVE_ENCODINGS,
    EncodingGene,
)
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.genes.mutation import (
    sample_num_mutated_genes,
    sample_signed_amount,
)
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus
from kaggle_pipeline.evolution.models.parameter_spaces import (
    FamilyDefinition,
    build_default_families,
)
from kaggle_pipeline.evolution.utils.logging import get_logger
from kaggle_pipeline.evolution.utils.random import spawn_rng

logger = get_logger(__name__)

# Mutation types.
LOCAL_HYPERPARAMETER = "local_hyperparameter"
COORDINATED_HYPERPARAMETER = "coordinated_hyperparameter"
ADD_FEATURE = "add_feature"
REMOVE_FEATURE = "remove_feature"
REPLACE_FEATURE = "replace_feature"
CHANGE_FEATURE_ENCODING = "change_feature_encoding"

# Named coordinated moves: (param_a, param_b, sign_b_relative_to_a). Applied only
# when both parameters are present; the same base signed amount drives both.
COORDINATED_RULES: list[tuple[str, str, float]] = [
    ("num_leaves", "min_child_samples", 1.0),  # bigger trees + more leaf samples
    ("max_depth", "reg_lambda", 1.0),  # deeper + stronger regularization
    ("max_leaf_nodes", "l2_regularization", 1.0),
    ("colsample_bytree", "subsample", 1.0),
]


@dataclass
class MutationRecord:
    """The full record of a parent -> child mutation (pre- and post-training)."""

    child_model_id: str
    parent_model_id: str
    mutation_type: str
    mutated_gene_ids: list[str] = field(default_factory=list)
    signed_amounts: list[float] = field(default_factory=list)
    old_values: list[Any] = field(default_factory=list)
    new_values: list[Any] = field(default_factory=list)
    added_feature_ids: list[str] = field(default_factory=list)
    removed_feature_ids: list[str] = field(default_factory=list)
    created_at_batch: int = 0
    random_seed: int | None = None
    notes: str = ""
    # Filled by attach_outcome after the child has been scored.
    parent_utility: float | None = None
    child_utility: float | None = None
    delta_utility: float | None = None
    parent_score: float | None = None
    child_score: float | None = None
    delta_score: float | None = None
    parent_compute_time: float | None = None
    child_compute_time: float | None = None
    delta_compute_time: float | None = None
    behavior_delta: float | None = None

    def attach_outcome(
        self, parent: ModelGenome, child: ModelGenome, *, behavior_delta: float | None = None
    ) -> None:
        """Fill parent/child deltas once both have score sets and utilities."""
        if parent.score_set and child.score_set:
            self.parent_score = parent.score_set.score
            self.child_score = child.score_set.score
            self.delta_score = self.child_score - self.parent_score
            self.parent_compute_time = parent.score_set.compute_time
            self.child_compute_time = child.score_set.compute_time
            self.delta_compute_time = self.child_compute_time - self.parent_compute_time
        if parent.utility is not None and child.utility is not None:
            self.parent_utility = parent.utility
            self.child_utility = child.utility
            self.delta_utility = child.utility - parent.utility
        if behavior_delta is not None:
            self.behavior_delta = behavior_delta

    def to_serializable(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ModelMutator:
    """Produces child genomes from parents and records what changed."""

    def __init__(
        self,
        registry: FeatureRegistry,
        settings: EvolutionSettings,
        *,
        families: dict[str, FamilyDefinition] | None = None,
    ):
        self.registry = registry
        self.settings = settings
        self.families = families or build_default_families()

    def mutate(
        self, parent: ModelGenome, rng: np.random.Generator | None = None, *, batch: int = 0
    ) -> tuple[ModelGenome, MutationRecord]:
        rng = spawn_rng(rng)
        ctx = MutationContext(rng=rng, settings=self.settings, registry=self.registry)

        # Faithful clones so the parent is never touched.
        feature_genes = [g.copy() for g in parent.feature_reference_genes]
        parameter_genes = [g.copy() for g in parent.parameter_genes]
        resource_genes = [g.copy() for g in parent.resource_genes]
        base = parent.base_model_gene.copy()

        record = MutationRecord(
            child_model_id="",  # filled after the child is built
            parent_model_id=parent.model_id,
            mutation_type="",
            created_at_batch=batch,
            random_seed=int(rng.integers(0, 2**31 - 1)),
        )

        mutation_type = self._choose_type(parent, rng)
        record.mutation_type = mutation_type

        if mutation_type == LOCAL_HYPERPARAMETER:
            self._mutate_local(parameter_genes, feature_genes, ctx, record, rng)
        elif mutation_type == COORDINATED_HYPERPARAMETER:
            self._mutate_coordinated(parameter_genes, ctx, record, rng)
        elif mutation_type == ADD_FEATURE:
            self._add_feature(feature_genes, parent, ctx, record, rng)
        elif mutation_type == REMOVE_FEATURE:
            self._remove_feature(feature_genes, record, rng)
        elif mutation_type == REPLACE_FEATURE:
            self._replace_feature(feature_genes, ctx, record, rng)
        elif mutation_type == CHANGE_FEATURE_ENCODING:
            self._change_encoding(feature_genes, ctx, record, rng)

        child = ModelGenome(
            base_model_gene=base,
            feature_reference_genes=feature_genes,
            parameter_genes=parameter_genes,
            resource_genes=resource_genes,
            parent_model_id=parent.model_id,
            created_at_batch=batch,
            fidelity_level=parent.fidelity_level,
            mutation_history=[*parent.mutation_history, mutation_type],
            status=ModelStatus.CREATED,
            metadata=dict(parent.metadata),
        )
        record.child_model_id = child.model_id
        logger.debug(
            "mutated %s -> %s via %s (%d genes)",
            parent.model_id,
            child.model_id,
            mutation_type,
            len(record.mutated_gene_ids),
        )
        return child, record

    # --- type selection -----------------------------------------------------
    def _choose_type(self, parent: ModelGenome, rng: np.random.Generator) -> str:
        weights: dict[str, float] = {}
        if parent.parameter_genes:
            weights[LOCAL_HYPERPARAMETER] = 0.45
            if self._coordinated_pairs(parent):
                weights[COORDINATED_HYPERPARAMETER] = 0.15
        n_active = len(self.registry.get_active_features())
        if len(parent.feature_reference_genes) < n_active:
            weights[ADD_FEATURE] = 0.15
            weights[REPLACE_FEATURE] = 0.10
        if len(parent.feature_reference_genes) > 1:
            weights[REMOVE_FEATURE] = 0.10
        if any(fr.encoding is not None for fr in parent.feature_reference_genes):
            weights[CHANGE_FEATURE_ENCODING] = 0.05
        if not weights:
            return LOCAL_HYPERPARAMETER
        types = list(weights)
        probs = np.array([weights[t] for t in types])
        probs = probs / probs.sum()
        return types[int(rng.choice(len(types), p=probs))]

    def _coordinated_pairs(self, genome: ModelGenome) -> list[tuple[str, str, float]]:
        present = {g.parameter_name for g in genome.parameter_genes}
        return [rule for rule in COORDINATED_RULES if rule[0] in present and rule[1] in present]

    # --- operators ----------------------------------------------------------
    def _mutate_local(self, parameter_genes, feature_genes, ctx, record, rng) -> None:
        candidates = [g for g in parameter_genes if g.mutable]
        # also allow mutating encodings / feature replacements in the local pool
        if not candidates:
            return
        n = min(sample_num_mutated_genes(self.settings, rng), len(candidates))
        chosen_idx = rng.choice(len(candidates), size=n, replace=False)
        for i in np.atleast_1d(chosen_idx):
            gene = candidates[int(i)]
            amount = sample_signed_amount(self.settings, rng)
            new_gene = gene.mutate(amount, ctx)
            self._replace_in(parameter_genes, gene, new_gene)
            self._record_gene(record, gene, new_gene, amount)

    def _mutate_coordinated(self, parameter_genes, ctx, record, rng) -> None:
        present = {g.parameter_name: g for g in parameter_genes}
        rules = [r for r in COORDINATED_RULES if r[0] in present and r[1] in present]
        if not rules:
            return self._mutate_local(parameter_genes, [], ctx, record, rng)
        a_name, b_name, sign = rules[int(rng.integers(len(rules)))]
        amount = sample_signed_amount(self.settings, rng)
        for name, amt in ((a_name, amount), (b_name, amount * sign)):
            gene = present[name]
            new_gene = gene.mutate(amt, ctx)
            self._replace_in(parameter_genes, gene, new_gene)
            self._record_gene(record, gene, new_gene, amt)
        record.notes = f"coordinated({a_name},{b_name})"

    def _add_feature(self, feature_genes, parent, ctx, record, rng) -> None:
        used = {fr.feature_id for fr in feature_genes}
        new_id = self.registry.sample_feature(rng, exclude=used)
        if new_id is None:
            return
        gene = FeatureReferenceGene(new_id)
        feature = self.registry.get_feature(new_id)
        if feature.output_type == CATEGORICAL:
            fam = self.families.get(parent.family)
            native = fam.handles_categoricals if fam else False
            gene.set_encoding(
                EncodingGene(
                    NATIVE if native else FREQUENCY,
                    alternatives=NATIVE_CAPABLE_ENCODINGS if native else NON_NATIVE_ENCODINGS,
                )
            )
        feature_genes.append(gene)
        record.mutated_gene_ids.append(gene.gene_id)
        record.signed_amounts.append(1.0)
        record.old_values.append(None)
        record.new_values.append(new_id)
        record.added_feature_ids.append(new_id)

    def _remove_feature(self, feature_genes, record, rng) -> None:
        if len(feature_genes) <= 1:
            return
        idx = int(rng.integers(len(feature_genes)))
        removed = feature_genes.pop(idx)
        record.mutated_gene_ids.append(removed.gene_id)
        record.signed_amounts.append(-1.0)
        record.old_values.append(removed.feature_id)
        record.new_values.append(None)
        record.removed_feature_ids.append(removed.feature_id)

    def _replace_feature(self, feature_genes, ctx, record, rng) -> None:
        if not feature_genes:
            return
        idx = int(rng.integers(len(feature_genes)))
        gene = feature_genes[idx]
        new_gene = gene.mutate(0.0, ctx)
        if new_gene.gene_id == gene.gene_id:  # replacement found nothing
            return
        feature_genes[idx] = new_gene
        record.mutated_gene_ids.append(new_gene.gene_id)
        record.signed_amounts.append(0.0)
        record.old_values.append(gene.feature_id)
        record.new_values.append(new_gene.feature_id)
        record.removed_feature_ids.append(gene.feature_id)
        record.added_feature_ids.append(new_gene.feature_id)

    def _change_encoding(self, feature_genes, ctx, record, rng) -> None:
        with_encoding = [fr for fr in feature_genes if fr.encoding is not None]
        if not with_encoding:
            return
        fr = with_encoding[int(rng.integers(len(with_encoding)))]
        old_enc = fr.encoding
        if old_enc is None:
            return
        new_enc = old_enc.mutate(0.0, ctx)
        if new_enc.value == old_enc.value:
            return
        fr.set_encoding(new_enc)
        record.mutated_gene_ids.append(new_enc.gene_id)
        record.signed_amounts.append(0.0)
        record.old_values.append(old_enc.value)
        record.new_values.append(new_enc.value)

    # --- small helpers ------------------------------------------------------
    @staticmethod
    def _replace_in(genes: list, old, new) -> None:
        for i, g in enumerate(genes):
            if g is old:
                genes[i] = new
                return

    @staticmethod
    def _record_gene(record: MutationRecord, old, new, amount: float) -> None:
        record.mutated_gene_ids.append(new.gene_id)
        record.signed_amounts.append(amount)
        record.old_values.append(old.value)
        record.new_values.append(new.value)
