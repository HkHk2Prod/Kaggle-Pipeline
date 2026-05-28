"""The :class:`ModelGenome` -- a model defined as a set of dependent genes.

Immutable once created/trained: mutation produces a *child* genome (the mutator
clones the genes, mutates a few, and builds a new genome). A genome references
global features by id via :class:`FeatureReferenceGene`s and owns model-local
encoding/parameter/resource genes; the :class:`BaseModelGene` (model family) is
immutable within the genome -- changing family is a new genome, not a mutation.

The :attr:`genome_hash` covers the base model, the *set* of feature ids + their
encodings, the parameter values, and the resource/fidelity settings (order
independent), plus any validation-scheme / target-transform tags in
``metadata``. Identical genomes hash identically, which is what lets the trainer
skip retraining duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaggle_pipeline.evolution.genes.base import BaseModelGene, Gene
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.genes.parameter_gene import ParameterGene
from kaggle_pipeline.evolution.genes.resource_gene import ResourceGene
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus
from kaggle_pipeline.evolution.models.scoring import ModelScoreSet
from kaggle_pipeline.evolution.storage.hashing import short_hash, stable_hash


@dataclass
class ModelGenome:
    """A model genome: base model + feature references + parameters + resources."""

    base_model_gene: BaseModelGene
    feature_reference_genes: list[FeatureReferenceGene] = field(default_factory=list)
    parameter_genes: list[ParameterGene] = field(default_factory=list)
    resource_genes: list[ResourceGene] = field(default_factory=list)
    parent_model_id: str | None = None
    created_at_batch: int = 0
    fidelity_level: int = 1
    mutation_history: list[str] = field(default_factory=list)
    status: str = ModelStatus.CREATED
    metadata: dict[str, Any] = field(default_factory=dict)
    score_set: ModelScoreSet | None = None
    utility: float | None = None
    # Derived; set in __post_init__.
    genome_hash: str = field(default="", compare=False)
    model_id: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        self.genome_hash = self.compute_hash()
        if not self.model_id:
            self.model_id = f"m_{short_hash(self.genome_hash, 16)}"

    # --- hashing ------------------------------------------------------------
    def compute_hash(self) -> str:
        """Order-independent hash over the genome's identity-defining genes."""
        payload = {
            "base_model": self.base_model_gene.hash_component(),
            "features": sorted(
                (g.hash_component() for g in self.feature_reference_genes),
                key=lambda c: c.get("feature_id", ""),
            ),
            "parameters": sorted(
                (g.hash_component() for g in self.parameter_genes),
                key=lambda c: c.get("name", ""),
            ),
            "resources": sorted(
                (g.hash_component() for g in self.resource_genes),
                key=lambda c: c.get("resource_name", ""),
            ),
            "fidelity_level": self.fidelity_level,
            "validation_scheme": self.metadata.get("validation_scheme"),
            "target_transform": self.metadata.get("target_transform"),
            "config_version": self.metadata.get("config_version"),
        }
        return stable_hash(payload)

    # --- views --------------------------------------------------------------
    @property
    def family(self) -> str:
        return self.base_model_gene.family

    def feature_ids(self) -> list[str]:
        return [g.feature_id for g in self.feature_reference_genes]

    def all_genes(self) -> list[Gene]:
        genes: list[Gene] = [self.base_model_gene]
        genes.extend(self.feature_reference_genes)
        genes.extend(self.parameter_genes)
        genes.extend(self.resource_genes)
        return genes

    def mutable_genes(self) -> list[Gene]:
        """Genes eligible for ordinary mutation (feature refs, encodings, params).

        Resources and the base model are excluded (resources change by promotion,
        the base model defines the genome). Encoding children of feature references
        are included so encodings can be mutated too.
        """
        out: list[Gene] = []
        for fr in self.feature_reference_genes:
            if fr.mutable:
                out.append(fr)
            enc = fr.encoding
            if enc is not None and enc.mutable:
                out.append(enc)
        out.extend(g for g in self.parameter_genes if g.mutable)
        return out

    def get_parameter(self, name: str) -> ParameterGene | None:
        for g in self.parameter_genes:
            if g.parameter_name == name:
                return g
        return None

    def get_resource(self, name: str) -> ResourceGene | None:
        for g in self.resource_genes:
            if g.resource_name == name:
                return g
        return None

    def validate(self) -> None:
        if not self.feature_reference_genes:
            raise ValueError(f"genome {self.model_id} has no feature references")
        for gene in self.parameter_genes:
            gene.validate()

    def to_serializable(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "genome_hash": self.genome_hash,
            "parent_model_id": self.parent_model_id,
            "family": self.family,
            "status": self.status,
            "fidelity_level": self.fidelity_level,
            "created_at_batch": self.created_at_batch,
            "feature_ids": self.feature_ids(),
            "base_model_gene": self.base_model_gene.to_serializable(),
            "feature_reference_genes": [g.to_serializable() for g in self.feature_reference_genes],
            "parameter_genes": [g.to_serializable() for g in self.parameter_genes],
            "resource_genes": [g.to_serializable() for g in self.resource_genes],
            "mutation_history": list(self.mutation_history),
            "metadata": dict(self.metadata),
            "score_set": self.score_set.to_serializable() if self.score_set else None,
            "utility": self.utility,
        }
