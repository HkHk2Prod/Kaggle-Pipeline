"""The :class:`EncodingGene` -- how a model encodes a referenced feature.

Encoding is **model-specific**, so it lives as a child of a
:class:`~kaggle_pipeline.evolution.genes.feature_reference_gene.FeatureReferenceGene`
inside a model genome, never in the global feature definition. The same logical
feature can be native-categorical in one model, one-hot in another, count-encoded
in a third.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaggle_pipeline.evolution.genes.base import ENCODING, Gene

if TYPE_CHECKING:
    from kaggle_pipeline.evolution.genes.base import MutationContext

# Encoding strategies (a superset of the v1 pipeline's options; "target" is a
# planned/stubbed OOF encoding).
NATIVE = "native"
ONEHOT = "onehot"
ORDINAL = "ordinal"
FREQUENCY = "frequency"
COUNT = "count"
TARGET = "target"

# Encodings a model that cannot consume raw categoricals may use. Models that
# handle categoricals natively get no encoding gene at all (the factory skips it).
NON_NATIVE_ENCODINGS = (FREQUENCY, COUNT, ONEHOT, ORDINAL)


def allowed_encodings_for(cardinality: int | None, onehot_max_cardinality: int) -> tuple[str, ...]:
    """Encodings permitted for a non-native model's categorical of this cardinality.

    One-hot is dropped from the allowed set when the categorical has more distinct
    levels than ``onehot_max_cardinality`` (or its cardinality is unknown), so a
    single feature cannot explode the materialized width -- the constraint is
    applied here, at gene-construction time, rather than only at training time.
    """
    alternatives = list(NON_NATIVE_ENCODINGS)
    too_many = cardinality is None or cardinality > onehot_max_cardinality
    if too_many and ONEHOT in alternatives:
        alternatives.remove(ONEHOT)
    return tuple(alternatives)


class EncodingGene(Gene):
    """A model-local encoding choice for one referenced feature."""

    def __init__(
        self,
        encoding_type: str,
        *,
        parameters: dict[str, Any] | None = None,
        alternatives: tuple[str, ...] = (),
        **kwargs: Any,
    ):
        super().__init__(ENCODING, encoding_type, **kwargs)
        self.parameters: dict[str, Any] = parameters or {}
        self.alternatives: tuple[str, ...] = tuple(alternatives) or (encoding_type,)

    @property
    def encoding_type(self) -> str:
        return self.value

    def hash_component(self) -> dict[str, Any]:
        return {
            "gene_type": ENCODING,
            "encoding_type": self.encoding_type,
            "parameters": dict(self.parameters),
        }

    def to_serializable(self) -> dict[str, Any]:
        out = super().to_serializable()
        out["parameters"] = dict(self.parameters)
        out["alternatives"] = list(self.alternatives)
        return out

    def mutate(self, signed_amount: float, context: MutationContext) -> EncodingGene:
        if not self.mutable or len(self.alternatives) <= 1:
            return self.copy()
        others = [e for e in self.alternatives if e != self.encoding_type]
        if not others:
            return self.copy()
        return self.fresh_child(others[int(context.rng.integers(len(others)))])
