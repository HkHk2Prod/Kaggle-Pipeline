"""The gene system: the building blocks of a model genome.

A :class:`~kaggle_pipeline.evolution.genes.base.Gene` can represent the base
model family, a selected feature reference, an encoding choice, a model parameter
or a resource/fidelity setting. Genes are immutable in the sense that ``mutate``
returns a *new* gene (with a new id and a link to its parent), never editing the
original -- which is what lets a child model be built from a parent without
disturbing it.
"""

from __future__ import annotations

from kaggle_pipeline.evolution.genes.base import (
    BaseModelGene,
    Gene,
    MutationContext,
    MutationStats,
    new_gene_id,
)
from kaggle_pipeline.evolution.genes.encoding_gene import EncodingGene
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.genes.mutation import (
    sample_num_mutated_genes,
    sample_signed_amount,
)
from kaggle_pipeline.evolution.genes.parameter_gene import ParameterGene, ParameterSpec
from kaggle_pipeline.evolution.genes.resource_gene import ResourceGene

__all__ = [
    "Gene",
    "BaseModelGene",
    "MutationStats",
    "MutationContext",
    "new_gene_id",
    "ParameterGene",
    "ParameterSpec",
    "FeatureReferenceGene",
    "EncodingGene",
    "ResourceGene",
    "sample_signed_amount",
    "sample_num_mutated_genes",
]
