"""The :class:`ModelFactory` -- generates fresh :class:`ModelGenome` objects.

New-model generation: pick a base family, choose a feature-count upper bound
(penalising very large counts), sample features from the registry by selection
probability, wrap them in :class:`FeatureReferenceGene`s with model-appropriate
:class:`EncodingGene` children, sample parameters from the family's spaces, attach
resource/fidelity genes, validate and hash. The base model gene is immutable
within the genome; families are pluggable via :mod:`parameter_spaces`.
"""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.features.recipe import CATEGORICAL
from kaggle_pipeline.evolution.features.registry import FeatureRegistry
from kaggle_pipeline.evolution.genes.base import BaseModelGene
from kaggle_pipeline.evolution.genes.encoding_gene import (
    FREQUENCY,
    NATIVE,
    NATIVE_CAPABLE_ENCODINGS,
    NON_NATIVE_ENCODINGS,
    EncodingGene,
)
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.genes.parameter_gene import ParameterGene
from kaggle_pipeline.evolution.genes.resource_gene import ResourceGene
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.parameter_spaces import (
    FamilyDefinition,
    build_default_families,
)
from kaggle_pipeline.evolution.utils.logging import get_logger
from kaggle_pipeline.evolution.utils.random import spawn_rng

logger = get_logger(__name__)

# Upper bound on how many features a freshly generated model may select.
DEFAULT_MAX_FEATURES = 32


class ModelFactory:
    """Builds new model genomes from the available families and the feature pool."""

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
        if not self.families:
            raise RuntimeError("no model families available (install sklearn / boosters)")

    def family_names(self) -> list[str]:
        return list(self.families)

    def generate(
        self,
        rng: np.random.Generator | None = None,
        *,
        family: str | None = None,
        batch: int = 0,
        fidelity_level: int = 1,
        max_features: int | None = None,
    ) -> ModelGenome:
        rng = spawn_rng(rng)
        family = family or self.family_names()[int(rng.integers(len(self.families)))]
        fam = self.families[family]

        n_active = len(self.registry.get_active_features())
        cap = min(n_active, max_features or DEFAULT_MAX_FEATURES)
        count = self._sample_feature_count(cap, rng)
        feature_ids = self.registry.sample_features(count, rng)
        if not feature_ids:
            raise RuntimeError("feature registry has no active features to build a model from")

        feature_genes = [self._feature_reference(fid, fam, rng) for fid in feature_ids]
        parameter_genes = [ParameterGene(spec, spec.sample(rng)) for spec in fam.parameter_specs]
        resource_genes = [
            ResourceGene(
                "n_estimators",
                fam.n_estimators_for(fidelity_level),
                bounds=(1, max(fam.fidelity_n_estimators.values())),
                fidelity_level=fidelity_level,
            )
        ]
        genome = ModelGenome(
            base_model_gene=BaseModelGene(family),
            feature_reference_genes=feature_genes,
            parameter_genes=parameter_genes,
            resource_genes=resource_genes,
            created_at_batch=batch,
            fidelity_level=fidelity_level,
        )
        genome.validate()
        logger.debug(
            "generated model %s family=%s features=%d", genome.model_id, family, len(feature_genes)
        )
        return genome

    # --- helpers ------------------------------------------------------------
    def _sample_feature_count(self, cap: int, rng: np.random.Generator) -> int:
        """Pick a feature count, biased toward smaller models (large counts penalised)."""
        if cap <= 1:
            return max(1, cap)
        mode = max(1.0, cap * 0.25)
        return int(np.clip(round(rng.triangular(1, mode, cap)), 1, cap))

    def _feature_reference(
        self, feature_id: str, fam: FamilyDefinition, rng: np.random.Generator
    ) -> FeatureReferenceGene:
        gene = FeatureReferenceGene(feature_id)
        feature = self.registry.get_feature(feature_id)
        if feature.output_type == CATEGORICAL:
            if fam.handles_categoricals:
                gene.set_encoding(EncodingGene(NATIVE, alternatives=NATIVE_CAPABLE_ENCODINGS))
            else:
                gene.set_encoding(EncodingGene(FREQUENCY, alternatives=NON_NATIVE_ENCODINGS))
        return gene
