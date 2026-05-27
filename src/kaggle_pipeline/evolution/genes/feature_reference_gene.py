"""The :class:`FeatureReferenceGene` -- a model's use of a global feature.

A model genome owns *references* to global features (by ``feature_id``), not
feature definitions. A reference may have an :class:`EncodingGene` child saying how
this model encodes the feature; removing the reference removes its encoding child.
Gene-level mutation here *replaces* the referenced feature with another of the same
output type sampled from the registry; adding/removing references is a
genome-level mutation handled by the :class:`ModelMutator`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaggle_pipeline.evolution.genes.base import FEATURE_REFERENCE, Gene
from kaggle_pipeline.evolution.genes.encoding_gene import EncodingGene

if TYPE_CHECKING:
    from kaggle_pipeline.evolution.genes.base import MutationContext


class FeatureReferenceGene(Gene):
    """A reference to a global feature id, optionally with an encoding child."""

    def __init__(self, feature_id: str, *, selected: bool = True, **kwargs: Any):
        super().__init__(FEATURE_REFERENCE, feature_id, **kwargs)
        self.selected = selected

    @property
    def feature_id(self) -> str:
        return self.value

    @property
    def encoding(self) -> EncodingGene | None:
        for child in self.children:
            if isinstance(child, EncodingGene):
                return child
        return None

    def set_encoding(self, gene: EncodingGene) -> None:
        """Attach (or replace) the single encoding child for this reference."""
        self.children = [c for c in self.children if not isinstance(c, EncodingGene)]
        self.child_gene_ids = [c.gene_id for c in self.children]
        self.add_child(gene)

    def hash_component(self) -> dict[str, Any]:
        comp: dict[str, Any] = {"gene_type": FEATURE_REFERENCE, "feature_id": self.feature_id}
        encoding = self.encoding
        if encoding is not None:
            comp["encoding"] = encoding.hash_component()
        return comp

    def mutate(self, signed_amount: float, context: MutationContext) -> FeatureReferenceGene:
        """Replace the referenced feature with another of the same output type."""
        if not self.mutable or context.registry is None:
            return self.copy()
        try:
            output_type = context.registry.get_feature(self.feature_id).output_type
        except KeyError:
            return self.copy()
        new_id = context.registry.sample_feature(
            context.rng, output_type=output_type, exclude={self.feature_id}
        )
        if new_id is None or new_id == self.feature_id:
            return self.copy()
        # fresh_child deep-copies the encoding child, which stays valid because the
        # replacement has the same output type.
        return self.fresh_child(new_id)
