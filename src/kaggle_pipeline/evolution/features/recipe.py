"""The canonical :class:`FeatureRecipe` -- the source of truth for a feature.

A feature is *defined* by its recipe, not by its human name. The recipe is the
input to hashing: identical canonical recipes hash identically and therefore
deduplicate to one feature. Callers must hand the recipe already-canonicalised
(e.g. a commutative transform sorts its parent IDs); this module only guarantees a
deterministic digest of whatever it is given.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from kaggle_pipeline.evolution.storage.hashing import stable_hash

# Logical output types a feature can have. Kept as plain strings (not an Enum) so
# they serialise trivially and new types can be added without touching callers.
NUMERIC = "numeric"
CATEGORICAL = "categorical"
BOOLEAN = "boolean"
OUTPUT_TYPES: frozenset[str] = frozenset({NUMERIC, CATEGORICAL, BOOLEAN})


def _to_native(value: Any) -> Any:
    """Coerce numpy scalars/arrays and nested containers to JSON-native types.

    Keeps recipes (and therefore hashes) independent of numpy's scalar ``repr``,
    which differs across versions. Imported lazily so this module stays light.
    """
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    # Duck-type numpy scalars/arrays without a hard import at module load.
    if hasattr(value, "item") and type(value).__module__ == "numpy":
        return _to_native(value.item())
    return value


@dataclass(frozen=True)
class FeatureRecipe:
    """An immutable, canonical recipe for computing one logical feature.

    ``parameters`` and ``metadata`` are dicts (so the dataclass is not hashable via
    ``__hash__``); use :attr:`recipe_hash` as the identity key. ``metadata`` is
    descriptive only and is **excluded** from the hash, so two recipes that differ
    only in annotations still deduplicate.
    """

    transform_name: str
    parent_feature_ids: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    output_type: str = NUMERIC
    # Bump when a transform's behaviour changes so old hashes are invalidated.
    version: int = 1
    uses_target: bool = False
    requires_oof: bool = False
    # Only set for target-dependent recipes whose values depend on a fold scheme.
    fold_context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_type not in OUTPUT_TYPES:
            raise ValueError(
                f"output_type must be one of {sorted(OUTPUT_TYPES)}, got {self.output_type!r}."
            )
        # Normalise parent IDs to a tuple and parameter values to native types so
        # the canonical form (and hash) is stable regardless of the caller.
        object.__setattr__(self, "parent_feature_ids", tuple(self.parent_feature_ids))
        object.__setattr__(self, "parameters", _to_native(dict(self.parameters)))

    def canonical(self) -> dict[str, Any]:
        """The hash-defining structure (everything but descriptive ``metadata``)."""
        return {
            "transform_name": self.transform_name,
            "parent_feature_ids": list(self.parent_feature_ids),
            "parameters": self.parameters,
            "output_type": self.output_type,
            "version": self.version,
            "uses_target": self.uses_target,
            "requires_oof": self.requires_oof,
            "fold_context": self.fold_context,
        }

    @property
    def recipe_hash(self) -> str:
        """Full SHA-256 digest of the canonical recipe (the dedup key)."""
        return stable_hash(self.canonical())

    @property
    def arity(self) -> int:
        return len(self.parent_feature_ids)

    @property
    def is_original(self) -> bool:
        """An original feature has no parent features (identity on a raw column)."""
        return self.transform_name == "identity"

    def with_metadata(self, **extra: Any) -> FeatureRecipe:
        """Return a copy with merged metadata (does not change the hash)."""
        return replace(self, metadata={**self.metadata, **extra})

    def to_serializable(self) -> dict[str, Any]:
        """A JSON-compatible dict including metadata and the recipe hash."""
        out = self.canonical()
        out["metadata"] = _to_native(self.metadata)
        out["recipe_hash"] = self.recipe_hash
        return out
