"""Base classes and registry for feature transformations.

Every transform is a small subclass of :class:`FeatureTransformation` declaring
its arity, input/output types, commutativity, target/OOF requirements, parameter
space and cost, plus ``apply``/``generate_recipe``/``generate_name``. Operators
register themselves with :func:`register`; :func:`build_default_registry`
returns a :class:`TransformationRegistry` populated with everything that was
imported by the package.

Operators are *pure functions of their parent values* (no target leakage).
Target-dependent operators (target encoding) are stubbed and flagged
``uses_target``/``requires_oof`` so materialization can enforce OOF discipline
once they are implemented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from kaggle_pipeline.evolution.features.recipe import (
    BOOLEAN,
    NUMERIC,
    FeatureRecipe,
)
from kaggle_pipeline.evolution.storage.hashing import short_hash
from kaggle_pipeline.evolution.utils.arrays import missing_mask

if TYPE_CHECKING:
    from kaggle_pipeline.evolution.features.genome import FeatureGenome

EPS = 1e-9


class TransformError(Exception):
    """Raised when a transform's inputs or output are invalid.

    Carries a short ``reason`` code so the generator can record *why* a candidate
    failed and penalise the transform if failures repeat.
    """

    def __init__(self, reason: str, message: str = ""):
        super().__init__(message or reason)
        self.reason = reason


class FeatureTransformation(ABC):
    """Base class for all feature operators."""

    name: str = ""
    short: str = ""  # short label used in generated human names
    input_types: tuple[str, ...] = (NUMERIC,)
    output_type: str = NUMERIC
    arity: int = 1
    is_commutative: bool = False
    uses_target: bool = False
    requires_oof: bool = False
    cost_estimate: float = 1.0
    version: int = 1
    # Reject an output whose non-finite (or, for categoricals, missing) fraction
    # exceeds this, or that is (near-)constant.
    max_nan_fraction: float = 0.5

    # --- parameters ---------------------------------------------------------
    def default_parameters(self) -> dict[str, Any]:
        return {}

    def sample_parameters(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample transform parameters; default is the (empty) parameter space."""
        return dict(self.default_parameters())

    # --- validation ---------------------------------------------------------
    def validate_inputs(self, parents: list[FeatureGenome]) -> None:
        if len(parents) != self.arity:
            raise TransformError(
                "arity", f"{self.name} needs {self.arity} parents, got {len(parents)}."
            )
        for p in parents:
            if p.output_type not in self.input_types:
                raise TransformError(
                    "input_type",
                    f"{self.name} accepts {self.input_types}, got {p.output_type} ({p.human_name}).",
                )

    def sanitize(self, values: np.ndarray) -> np.ndarray:
        """Coerce dtype and replace Inf with NaN, without rejecting anything.

        Always safe to apply -- this is what re-materialising an *accepted* feature
        in any context (train/test/fold) uses, where the feature may legitimately
        be (near-)constant on a slice.
        """
        arr = np.asarray(values)
        if self.output_type in (NUMERIC, BOOLEAN):
            arr = arr.astype(float)
            arr[~np.isfinite(arr)] = np.nan
            return arr
        return np.asarray(arr, dtype=object)

    def validate_output(self, values: np.ndarray) -> np.ndarray:
        """Sanitise and validate; raise on degenerate output (used during generation)."""
        arr = self.sanitize(values)
        if arr.size == 0:
            raise TransformError("empty", "empty output")
        if self.output_type in (NUMERIC, BOOLEAN):
            nan_frac = float(np.isnan(arr).mean())
            if nan_frac > self.max_nan_fraction:
                raise TransformError("too_many_nan", f"{nan_frac:.0%} NaN")
            finite = arr[np.isfinite(arr)]
            if finite.size == 0 or np.unique(finite).size < 2:
                raise TransformError("constant", "constant / near-constant output")
        else:  # categorical
            missing = missing_mask(arr)
            if float(missing.mean()) > self.max_nan_fraction:
                raise TransformError("too_many_nan", "too many missing categories")
            if np.unique(arr[~missing]).size < 2:
                raise TransformError("constant", "constant categorical output")
        return arr

    # --- computation --------------------------------------------------------
    @abstractmethod
    def _compute(self, inputs: list[np.ndarray], params: dict[str, Any]) -> np.ndarray:
        """Return raw output values from parent value arrays (no sanitisation)."""

    def apply(
        self, inputs: list[np.ndarray], params: dict[str, Any], *, validate: bool = True
    ) -> np.ndarray:
        """Compute output values; validate (generation) or merely sanitise (materialize)."""
        raw = self._compute(inputs, params)
        return self.validate_output(raw) if validate else self.sanitize(raw)

    # --- recipe & name ------------------------------------------------------
    def generate_recipe(
        self, parent_feature_ids: list[str] | tuple[str, ...], params: dict[str, Any]
    ) -> FeatureRecipe:
        parents = tuple(parent_feature_ids)
        if self.is_commutative:
            # Canonicalise order so a+b and b+a deduplicate to one recipe.
            parents = tuple(sorted(parents))
        return FeatureRecipe(
            transform_name=self.name,
            parent_feature_ids=parents,
            parameters=params,
            output_type=self.output_type,
            version=self.version,
            uses_target=self.uses_target,
            requires_oof=self.requires_oof,
        )

    def label(self, params: dict[str, Any]) -> str:
        """Short human label for the operator (may fold in a key parameter)."""
        return self.short or self.name

    def generate_name(self, parent_names: list[str], recipe: FeatureRecipe) -> str:
        """Readable name: ``label__parent1__parent2__<short hash>``.

        The short hash makes otherwise-identical names unique and ties the name
        back to the recipe; the name is for humans only and is never hashed.
        """
        parents = "__".join(parent_names)
        suffix = short_hash(recipe.recipe_hash, 6)
        return f"{self.label(recipe.parameters)}__{parents}__{suffix}"


class TransformationRegistry:
    """Name -> transformation instance, with type-aware lookup for the generator."""

    def __init__(self) -> None:
        self._by_name: dict[str, FeatureTransformation] = {}

    def register(self, transform: FeatureTransformation) -> None:
        if not transform.name:
            raise ValueError("transformation must define a name")
        self._by_name[transform.name] = transform

    def get(self, name: str) -> FeatureTransformation:
        if name not in self._by_name:
            raise KeyError(f"transformation {name!r} not registered")
        return self._by_name[name]

    def all(self, *, include_target: bool = False) -> list[FeatureTransformation]:
        return [t for t in self._by_name.values() if include_target or not t.uses_target]

    def for_inputs(
        self, available_types: set[str], *, include_target: bool = False
    ) -> list[FeatureTransformation]:
        """Transforms whose input types are all satisfiable from ``available_types``."""
        return [
            t
            for t in self.all(include_target=include_target)
            if set(t.input_types) & available_types
        ]


_REGISTERED: list[type[FeatureTransformation]] = []


def register(cls: type[FeatureTransformation]) -> type[FeatureTransformation]:
    """Decorator: mark a transformation class for inclusion in the default registry."""
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__} must set a non-empty `name` before @register")
    _REGISTERED.append(cls)
    return cls


def build_default_registry() -> TransformationRegistry:
    """A registry preloaded with every ``@register``-decorated transformation."""
    reg = TransformationRegistry()
    for cls in _REGISTERED:
        reg.register(cls())
    return reg
