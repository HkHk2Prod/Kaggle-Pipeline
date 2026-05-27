"""The base :class:`Gene`, mutation bookkeeping, and the :class:`BaseModelGene`.

Design rules (see the README):

* ``copy()`` is a faithful clone -- same ``gene_id``, deep-copied children/stats --
  used when a child genome reuses an unmutated parent gene.
* ``mutate(signed_amount, context)`` returns a **new** gene with a fresh
  ``gene_id`` and ``parent_gene_id`` set to the original, so the parent is never
  changed in place. It may return a *list* (e.g. when a mutation also rewrites
  child genes).
* ``hash_component()`` is the part that defines the genome hash; it deliberately
  **excludes** ``gene_id`` and mutation stats (instance bookkeeping, not identity).

``Gene`` is a plain class (not a dataclass) so subclasses can add required fields
without the dataclass field-ordering constraints.
"""

from __future__ import annotations

import copy as _copy
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    import numpy as np

    from kaggle_pipeline.evolution.config import EvolutionSettings
    from kaggle_pipeline.evolution.features.registry import FeatureRegistry

_G = TypeVar("_G", bound="Gene")

# Gene type tags (also used as hash components and in serialisation).
BASE_MODEL = "base_model"
FEATURE_REFERENCE = "feature_reference"
ENCODING = "encoding"
PARAMETER = "parameter"
RESOURCE = "resource"


def new_gene_id(prefix: str = "g") -> str:
    """A unique gene id. Not part of any hash, so randomness here is harmless."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class MutationStats:
    """Per-gene accumulators of how its mutations have fared.

    Positive vs. negative buckets follow the sign of ``signed_amount`` (more
    complex vs. simpler). Means/variances come from running sums so updates are
    O(1) and one lucky mutation cannot dominate the mean.
    """

    positive_mutation_score: float = 0.0
    negative_mutation_score: float = 0.0
    positive_mutation_count: int = 0
    negative_mutation_count: int = 0
    positive_sum: float = 0.0
    negative_sum: float = 0.0
    positive_sumsq: float = 0.0
    negative_sumsq: float = 0.0
    behavior_change_score: float = 0.0
    compute_change_score: float = 0.0

    def record(
        self,
        signed_amount: float,
        credit: float,
        *,
        behavior_delta: float = 0.0,
        compute_delta: float = 0.0,
    ) -> None:
        if signed_amount >= 0:
            self.positive_mutation_count += 1
            self.positive_mutation_score += credit
            self.positive_sum += credit
            self.positive_sumsq += credit * credit
        else:
            self.negative_mutation_count += 1
            self.negative_mutation_score += credit
            self.negative_sum += credit
            self.negative_sumsq += credit * credit
        self.behavior_change_score += behavior_delta
        self.compute_change_score += compute_delta

    @staticmethod
    def _mean(total: float, count: int) -> float:
        return total / count if count else 0.0

    @staticmethod
    def _variance(total: float, sumsq: float, count: int) -> float:
        if count < 2:
            return 0.0
        mean = total / count
        return max(0.0, sumsq / count - mean * mean)

    @property
    def positive_mutation_mean(self) -> float:
        return self._mean(self.positive_sum, self.positive_mutation_count)

    @property
    def negative_mutation_mean(self) -> float:
        return self._mean(self.negative_sum, self.negative_mutation_count)

    @property
    def positive_mutation_variance(self) -> float:
        return self._variance(self.positive_sum, self.positive_sumsq, self.positive_mutation_count)

    @property
    def negative_mutation_variance(self) -> float:
        return self._variance(self.negative_sum, self.negative_sumsq, self.negative_mutation_count)

    def to_serializable(self) -> dict[str, Any]:
        return {
            "positive_mutation_score": self.positive_mutation_score,
            "negative_mutation_score": self.negative_mutation_score,
            "positive_mutation_count": self.positive_mutation_count,
            "negative_mutation_count": self.negative_mutation_count,
            "positive_mutation_mean": self.positive_mutation_mean,
            "negative_mutation_mean": self.negative_mutation_mean,
            "positive_mutation_variance": self.positive_mutation_variance,
            "negative_mutation_variance": self.negative_mutation_variance,
            "behavior_change_score": self.behavior_change_score,
            "compute_change_score": self.compute_change_score,
        }


@dataclass
class MutationContext:
    """Everything a gene needs to mutate itself: rng, settings, optional registry."""

    rng: np.random.Generator
    settings: EvolutionSettings
    registry: FeatureRegistry | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Gene:
    """Base class for all genes."""

    def __init__(
        self,
        gene_type: str,
        value: Any,
        *,
        mutable: bool = True,
        gene_id: str | None = None,
        parent_gene_id: str | None = None,
        children: list[Gene] | None = None,
        metadata: dict[str, Any] | None = None,
        mutation_stats: MutationStats | None = None,
    ):
        self.gene_type = gene_type
        self.value = value
        self.mutable = mutable
        self.gene_id = gene_id or new_gene_id(gene_type[:2])
        self.parent_gene_id = parent_gene_id
        self.children: list[Gene] = children or []
        self.child_gene_ids: list[str] = [c.gene_id for c in self.children]
        self.metadata: dict[str, Any] = metadata or {}
        self.mutation_stats = mutation_stats or MutationStats()

    # --- lifecycle ----------------------------------------------------------
    def validate(self) -> None:
        """Raise if the gene is in an invalid state. Subclasses extend this."""

    def copy(self: _G) -> _G:
        """A faithful deep clone (same ``gene_id``) for reuse in a child genome."""
        return _copy.deepcopy(self)

    def fresh_child(self: _G, value: Any) -> _G:
        """A new gene of the same kind with a new id, linked to this one as parent.

        Used by ``mutate`` so the parent gene is never edited. Children genes and
        metadata are deep-copied; mutation stats start fresh on the new instance.
        """
        child = _copy.deepcopy(self)
        child.value = value
        child.gene_id = new_gene_id(self.gene_type[:2])
        child.parent_gene_id = self.gene_id
        child.mutation_stats = MutationStats()
        return child

    def mutate(self, signed_amount: float, context: MutationContext) -> Gene | list[Gene]:
        """Return a mutated child gene (default: unchanged clone for immutable genes)."""
        return self.copy()

    # --- hashing / serialisation -------------------------------------------
    def hash_component(self) -> dict[str, Any]:
        """The identity-defining part of this gene (no id, no stats)."""
        comp: dict[str, Any] = {"gene_type": self.gene_type, "value": self.value}
        if self.children:
            comp["children"] = [c.hash_component() for c in self.children]
        return comp

    def to_serializable(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "gene_id": self.gene_id,
            "gene_type": self.gene_type,
            "value": self.value,
            "mutable": self.mutable,
            "parent_gene_id": self.parent_gene_id,
            "metadata": dict(self.metadata),
            "mutation_stats": self.mutation_stats.to_serializable(),
        }
        if self.children:
            out["children"] = [c.to_serializable() for c in self.children]
        return out

    # --- child management ---------------------------------------------------
    def add_child(self, child: Gene) -> None:
        child.parent_gene_id = self.gene_id
        self.children.append(child)
        self.child_gene_ids.append(child.gene_id)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(id={self.gene_id}, value={self.value!r})"


class BaseModelGene(Gene):
    """The model family. Immutable within a genome: changing it is a new genome."""

    def __init__(self, family: str, **kwargs: Any):
        kwargs.setdefault("mutable", False)
        super().__init__(BASE_MODEL, family, **kwargs)

    @property
    def family(self) -> str:
        return self.value

    def hash_component(self) -> dict[str, Any]:
        return {"gene_type": self.gene_type, "family": self.family}
