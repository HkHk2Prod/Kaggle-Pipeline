"""Feature transformations: the operators that turn parent features into new ones.

Every transform is a small class under :class:`FeatureTransformation` declaring
its arity, input/output types, commutativity, target/OOF requirements, parameter
space and cost, plus ``apply``/``generate_recipe``/``generate_name``. The
:class:`FeatureGenerator` samples a transform + parents + parameters, builds a
canonical recipe and a readable name, and validates the output. New operators are
added by subclassing and registering -- the generator picks them up automatically.

Operators here are *pure functions of their parent values* (no target leakage).
Target-dependent operators (target encoding) are stubbed and flagged
``uses_target``/``requires_oof`` so the materialization layer can enforce OOF
discipline once they are implemented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from kaggle_pipeline.evolution.features.recipe import (
    BOOLEAN,
    CATEGORICAL,
    NUMERIC,
    FeatureRecipe,
)
from kaggle_pipeline.evolution.storage.hashing import short_hash

if TYPE_CHECKING:
    from kaggle_pipeline.evolution.features.genome import FeatureGenome

_EPS = 1e-9


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

    def validate_output(self, values: np.ndarray) -> np.ndarray:
        """Sanitise and validate the computed values; raise on degenerate output."""
        arr = np.asarray(values)
        if self.output_type in (NUMERIC, BOOLEAN):
            arr = arr.astype(float)
            arr[~np.isfinite(arr)] = np.nan
            n = arr.size
            if n == 0:
                raise TransformError("empty", "empty output")
            nan_frac = float(np.isnan(arr).mean())
            if nan_frac > self.max_nan_fraction:
                raise TransformError("too_many_nan", f"{nan_frac:.0%} NaN")
            finite = arr[np.isfinite(arr)]
            if finite.size == 0 or np.unique(finite).size < 2:
                raise TransformError("constant", "constant / near-constant output")
        else:  # categorical
            obj = np.asarray(arr, dtype=object)
            n = obj.size
            if n == 0:
                raise TransformError("empty", "empty output")
            missing = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in obj])
            if float(missing.mean()) > self.max_nan_fraction:
                raise TransformError("too_many_nan", "too many missing categories")
            if np.unique(obj[~missing]).size < 2:
                raise TransformError("constant", "constant categorical output")
            arr = obj
        return arr

    # --- computation --------------------------------------------------------
    @abstractmethod
    def _compute(self, inputs: list[np.ndarray], params: dict[str, Any]) -> np.ndarray:
        """Return raw output values from parent value arrays (no sanitisation)."""

    def apply(self, inputs: list[np.ndarray], params: dict[str, Any]) -> np.ndarray:
        """Compute and validate output values for the given parent value arrays."""
        return self.validate_output(self._compute(inputs, params))

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
        base = f"{self.label(recipe.parameters)}__{parents}__{suffix}"
        return base


# --- numeric unary ----------------------------------------------------------


class Log1p(FeatureTransformation):
    name, short = "log1p", "log"

    def _compute(self, inputs, params):
        x = inputs[0]
        # Shift to be non-negative before log1p so negatives are handled safely.
        shift = min(0.0, float(np.nanmin(x)))
        return np.log1p(x - shift)


class Sqrt(FeatureTransformation):
    name, short = "sqrt", "sqrt"

    def _compute(self, inputs, params):
        x = inputs[0]
        shift = min(0.0, float(np.nanmin(x)))
        return np.sqrt(x - shift)


class Square(FeatureTransformation):
    name, short = "square", "sq"

    def _compute(self, inputs, params):
        return np.square(inputs[0])


class Rank(FeatureTransformation):
    name, short = "rank", "rank"

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=float)
        order = np.argsort(np.argsort(x))
        return order.astype(float) / max(1, x.size - 1)


class ZScore(FeatureTransformation):
    name, short = "zscore", "z"

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=float)
        mean, std = np.nanmean(x), np.nanstd(x)
        return (x - mean) / (std + _EPS)


class MinMax(FeatureTransformation):
    name, short = "minmax", "mm"

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo + _EPS)


class Clip(FeatureTransformation):
    name, short = "clip", "clip"

    def default_parameters(self):
        return {"lower_q": 0.01, "upper_q": 0.99}

    def sample_parameters(self, rng):
        q = float(rng.choice([0.005, 0.01, 0.025, 0.05]))
        return {"lower_q": q, "upper_q": round(1.0 - q, 4)}

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=float)
        lo = np.nanquantile(x, params["lower_q"])
        hi = np.nanquantile(x, params["upper_q"])
        return np.clip(x, lo, hi)


class Bin(FeatureTransformation):
    name, short = "bin", "bin"

    def default_parameters(self):
        return {"n_bins": 10}

    def sample_parameters(self, rng):
        return {"n_bins": int(rng.choice([4, 5, 8, 10, 20]))}

    def label(self, params):
        return f"bin{params.get('n_bins', '')}"

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=float)
        n_bins = int(params["n_bins"])
        quantiles = np.nanquantile(x, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(quantiles)
        if edges.size < 3:
            raise TransformError("constant", "not enough distinct bin edges")
        # Ordinal bin index as a numeric feature.
        return np.digitize(x, edges[1:-1]).astype(float)


class MissingIndicator(FeatureTransformation):
    name, short = "missing_indicator", "isna"
    output_type = BOOLEAN

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=float)
        return (~np.isfinite(x)).astype(float)

    def validate_output(self, values):
        # A missing-indicator is allowed to be all-zero only if the parent had no
        # missing values; in that case it is useless, so reject as constant.
        return super().validate_output(values)


# --- numeric binary ---------------------------------------------------------


class _Binary(FeatureTransformation):
    arity = 2
    input_types = (NUMERIC,)
    output_type = NUMERIC


class Add(_Binary):
    name, short, is_commutative = "add", "add", True

    def _compute(self, inputs, params):
        return inputs[0] + inputs[1]


class Subtract(_Binary):
    name, short = "subtract", "sub"

    def _compute(self, inputs, params):
        return inputs[0] - inputs[1]


class Multiply(_Binary):
    name, short, is_commutative = "multiply", "mul", True

    def _compute(self, inputs, params):
        return inputs[0] * inputs[1]


class SafeDivide(_Binary):
    name, short = "safe_divide", "div"

    def _compute(self, inputs, params):
        a, b = inputs[0], inputs[1]
        return a / (b + np.sign(b) * _EPS + (b == 0) * _EPS)


class AbsDiff(_Binary):
    name, short, is_commutative = "abs_diff", "absdiff", True

    def _compute(self, inputs, params):
        return np.abs(inputs[0] - inputs[1])


class Ratio(_Binary):
    name, short = "ratio", "frac"

    def _compute(self, inputs, params):
        a, b = inputs[0], inputs[1]
        return a / (a + b + _EPS)


class Minimum(_Binary):
    name, short, is_commutative = "minimum", "min", True

    def _compute(self, inputs, params):
        return np.minimum(inputs[0], inputs[1])


class Maximum(_Binary):
    name, short, is_commutative = "maximum", "max", True

    def _compute(self, inputs, params):
        return np.maximum(inputs[0], inputs[1])


# --- categorical -------------------------------------------------------------


class CategoryJoin(FeatureTransformation):
    name, short = "catjoin", "catjoin"
    input_types = (CATEGORICAL,)
    output_type = CATEGORICAL
    arity = 2
    is_commutative = True

    def _compute(self, inputs, params):
        a = np.asarray(inputs[0], dtype=object).astype(str)
        b = np.asarray(inputs[1], dtype=object).astype(str)
        return np.char.add(np.char.add(a, "|"), b).astype(object)


class FrequencyEncode(FeatureTransformation):
    name, short = "frequency", "freq"
    input_types = (CATEGORICAL,)
    output_type = NUMERIC

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=object)
        values, counts = np.unique(x.astype(str), return_counts=True)
        freq = dict(zip(values, counts / x.size, strict=True))
        return np.array([freq.get(str(v), 0.0) for v in x], dtype=float)


class CountEncode(FeatureTransformation):
    name, short = "count", "count"
    input_types = (CATEGORICAL,)
    output_type = NUMERIC

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=object)
        values, counts = np.unique(x.astype(str), return_counts=True)
        count = dict(zip(values, counts, strict=True))
        return np.array([count.get(str(v), 0) for v in x], dtype=float)


class RareGroup(FeatureTransformation):
    name, short = "rare_group", "rare"
    input_types = (CATEGORICAL,)
    output_type = CATEGORICAL

    def default_parameters(self):
        return {"min_count": 10}

    def sample_parameters(self, rng):
        return {"min_count": int(rng.choice([5, 10, 20, 50]))}

    def label(self, params):
        return f"rare{params.get('min_count', '')}"

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=object).astype(str)
        values, counts = np.unique(x, return_counts=True)
        rare = {v for v, c in zip(values, counts, strict=True) if c < params["min_count"]}
        return np.array(["__rare__" if v in rare else v for v in x], dtype=object)


class HashEncode(FeatureTransformation):
    name, short = "hash_encode", "hash"
    input_types = (CATEGORICAL,)
    output_type = NUMERIC

    def default_parameters(self):
        return {"n_buckets": 64}

    def sample_parameters(self, rng):
        return {"n_buckets": int(rng.choice([16, 32, 64, 128]))}

    def label(self, params):
        return f"hash{params.get('n_buckets', '')}"

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=object)
        n = int(params["n_buckets"])
        return np.array([int(short_hash(str(v), 12), 16) % n for v in x], dtype=float)


class TargetEncode(FeatureTransformation):
    """Out-of-fold target (mean) encoding. **Planned -- not implemented yet.**

    Declared with ``uses_target``/``requires_oof`` so the materialization layer
    will enforce fold-safe (out-of-fold) evaluation when this is implemented. The
    generator skips it until ``apply`` is provided.
    """

    name, short = "target_encode", "tgt"
    input_types = (CATEGORICAL,)
    output_type = NUMERIC
    uses_target = True
    requires_oof = True

    def _compute(self, inputs, params):
        raise TransformError(
            "not_implemented", "target encoding requires fold-safe OOF materialization (TODO)"
        )


# --- registry ----------------------------------------------------------------


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


def build_default_registry() -> TransformationRegistry:
    """A registry preloaded with the initial transform set (excludes identity)."""
    reg = TransformationRegistry()
    for transform in (
        Log1p(),
        Sqrt(),
        Square(),
        Rank(),
        ZScore(),
        MinMax(),
        Clip(),
        Bin(),
        MissingIndicator(),
        Add(),
        Subtract(),
        Multiply(),
        SafeDivide(),
        AbsDiff(),
        Ratio(),
        Minimum(),
        Maximum(),
        CategoryJoin(),
        FrequencyEncode(),
        CountEncode(),
        RareGroup(),
        HashEncode(),
        TargetEncode(),  # registered but skipped by the generator (uses_target)
    ):
        reg.register(transform)
    return reg
