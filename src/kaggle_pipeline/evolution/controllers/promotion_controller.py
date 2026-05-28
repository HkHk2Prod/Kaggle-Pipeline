"""The :class:`PromotionController` -- raising a promising model's fidelity.

Promotion is *not* mutation: it clones a genome and bumps its resource/fidelity
genes (more folds/iterations/seeds) to re-evaluate a strong candidate at higher
fidelity. Behaviour genes are untouched. Utilities are only ever compared within a
fidelity level, so a promoted model competes against other full-fidelity trials,
never against the cheap ones that produced it.
"""

from __future__ import annotations

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus
from kaggle_pipeline.evolution.models.parameter_spaces import (
    FamilyDefinition,
    build_default_families,
)


class PromotionController:
    """Decides which models to promote and builds the promoted child genome."""

    def __init__(
        self,
        settings: EvolutionSettings,
        *,
        families: dict[str, FamilyDefinition] | None = None,
        max_fidelity: int = 4,
    ):
        self.settings = settings
        self.families = families or build_default_families()
        self.max_fidelity = max_fidelity

    def can_promote(self, genome: ModelGenome) -> bool:
        return genome.status == ModelStatus.COMPLETED and genome.fidelity_level < self.max_fidelity

    def promote(self, genome: ModelGenome, *, batch: int = 0) -> ModelGenome:
        """Return a higher-fidelity clone of ``genome`` (a new genome to train)."""
        new_fidelity = genome.fidelity_level + 1
        fam = self.families.get(genome.family)
        n_estimators = fam.n_estimators_for(new_fidelity) if fam else None

        resource_genes = []
        for resource in genome.resource_genes:
            if resource.resource_name == "n_estimators" and n_estimators is not None:
                resource_genes.append(resource.promoted(n_estimators, new_fidelity))
            else:
                clone = resource.copy()
                clone.fidelity_level = new_fidelity
                resource_genes.append(clone)

        return ModelGenome(
            base_model_gene=genome.base_model_gene.copy(),
            feature_reference_genes=[g.copy() for g in genome.feature_reference_genes],
            parameter_genes=[g.copy() for g in genome.parameter_genes],
            resource_genes=resource_genes,
            parent_model_id=genome.model_id,
            created_at_batch=batch,
            fidelity_level=new_fidelity,
            mutation_history=[*genome.mutation_history, "promotion"],
            status=ModelStatus.CREATED,
            metadata=dict(genome.metadata),
        )
