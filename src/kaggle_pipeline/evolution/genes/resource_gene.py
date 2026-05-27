"""The :class:`ResourceGene` -- compute/fidelity settings.

Resource genes (fold count, seed count, iteration/epoch cap, row sampling) are
**not** mutated by ordinary behaviour mutation: they are ``mutable=False`` and are
changed only by *promotion* (see the promotion controller), which raises a
model's fidelity level. This keeps the "more compute" axis separate from the
"different behaviour" axis so utility is only ever compared within a fidelity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaggle_pipeline.evolution.genes.base import RESOURCE, Gene

if TYPE_CHECKING:
    from kaggle_pipeline.evolution.genes.base import MutationContext


class ResourceGene(Gene):
    """A compute/fidelity setting, changed by promotion rather than mutation."""

    def __init__(
        self,
        resource_name: str,
        value: Any,
        *,
        bounds: tuple[float, float] | None = None,
        fidelity_level: int = 1,
        **kwargs: Any,
    ):
        kwargs.setdefault("mutable", False)
        super().__init__(RESOURCE, value, **kwargs)
        self.resource_name = resource_name
        self.bounds = bounds
        self.fidelity_level = fidelity_level

    def hash_component(self) -> dict[str, Any]:
        return {
            "gene_type": RESOURCE,
            "resource_name": self.resource_name,
            "value": self.value,
            "fidelity_level": self.fidelity_level,
        }

    def to_serializable(self) -> dict[str, Any]:
        out = super().to_serializable()
        out["resource_name"] = self.resource_name
        out["bounds"] = list(self.bounds) if self.bounds else None
        out["fidelity_level"] = self.fidelity_level
        return out

    def mutate(self, signed_amount: float, context: MutationContext) -> Gene:
        # Resource genes never change under ordinary mutation; promotion handles them.
        return self.copy()

    def promoted(self, value: Any, fidelity_level: int) -> ResourceGene:
        """Return a new resource gene at a higher fidelity (used by promotion)."""
        child = self.fresh_child(value)
        assert isinstance(child, ResourceGene)
        child.fidelity_level = fidelity_level
        return child
