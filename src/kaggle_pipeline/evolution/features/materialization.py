"""Turning a recipe into data: :class:`FeatureMaterializer` and its records.

A :class:`FeatureGenome` is a recipe; a :class:`FeatureMaterialization` is the
record of its actual computed values in one *context* (global train/test, a fold,
a fixed evaluation sample). The materializer resolves originals from a source
frame and recurses through parent features for generated ones, caching per
``(feature_id, context_id)``.

Target-dependent features (``uses_target``) must be materialized out-of-fold to
avoid leakage; that path is a documented TODO and currently raises rather than
silently leaking.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from kaggle_pipeline.evolution.features.recipe import CATEGORICAL, NUMERIC

if TYPE_CHECKING:
    from kaggle_pipeline.evolution.features.registry import FeatureRegistry
    from kaggle_pipeline.evolution.features.transformations import TransformationRegistry

# Canonical context identifiers (extend as fold/sample schemes are added).
GLOBAL_TRAIN = "global_train"
GLOBAL_TEST = "global_test"
FEATURE_EVAL_SAMPLE = "feature_eval_sample_v1"


def hash_values(values: np.ndarray) -> str:
    """A short, stable digest of materialized values (for dedup/fingerprints)."""
    arr = np.asarray(values)
    if arr.dtype == object:
        payload = "".join("" if v is None else str(v) for v in arr.ravel())
        data = payload.encode("utf-8")
    else:
        # Round to float32 so trivial float noise does not change the hash.
        data = np.ascontiguousarray(arr.astype(np.float32)).tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class MaterializationContext:
    """Where/how to materialize: a context id plus the source frame to read from."""

    frame: Any  # a pandas DataFrame (duck-typed; no hard pandas import here)
    context_id: str = GLOBAL_TRAIN
    fold_id: str | None = None
    sample_id: str | None = None
    data_version_id: str | None = None


@dataclass
class FeatureMaterialization:
    """Metadata record describing one materialization of a feature."""

    feature_id: str
    context_id: str
    values_hash: str
    materialized_width: int
    dtype: str
    n_rows: int
    fold_id: str | None = None
    sample_id: str | None = None
    data_version_id: str | None = None
    memory_estimate: int = 0
    created_at: float = field(default_factory=time.time)
    cache_location: str | None = None

    def to_serializable(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        return out


class FeatureMaterializer:
    """Computes feature values for a context, caching by ``(feature_id, context_id)``.

    Holds a reference to the registry (to resolve recipes by id) and the transform
    registry (to apply operators). It does not own the data -- a
    :class:`MaterializationContext` carries the frame, so the same materializer can
    serve train, test, folds and the evaluation sample.
    """

    def __init__(self, registry: FeatureRegistry, transformations: TransformationRegistry):
        self.registry = registry
        self.transforms = transformations
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def materialize(self, feature_id: str, context: MaterializationContext) -> np.ndarray:
        """Return the 1-D value array for ``feature_id`` in ``context``."""
        key = (feature_id, context.context_id)
        if key in self._cache:
            return self._cache[key]

        genome = self.registry.get_feature(feature_id)
        if genome.uses_target:
            # OOF materialization is required for target-dependent features.
            raise NotImplementedError(
                f"feature {feature_id!r} uses the target; out-of-fold materialization "
                "is not implemented yet (TODO)."
            )

        recipe = genome.recipe
        if recipe.is_original:
            values = self._extract_column(
                context.frame, recipe.parameters["source_column"], genome.output_type
            )
        else:
            transform = self.transforms.get(recipe.transform_name)
            parent_values = [self.materialize(pid, context) for pid in recipe.parent_feature_ids]
            values = transform.apply(parent_values, recipe.parameters)

        self._cache[key] = values
        return values

    def describe(
        self, feature_id: str, values: np.ndarray, context: MaterializationContext
    ) -> FeatureMaterialization:
        """Build the metadata record for an already-computed value array."""
        arr = np.asarray(values)
        return FeatureMaterialization(
            feature_id=feature_id,
            context_id=context.context_id,
            values_hash=hash_values(arr),
            materialized_width=1 if arr.ndim == 1 else int(arr.shape[1]),
            dtype=str(arr.dtype),
            n_rows=int(arr.shape[0]),
            fold_id=context.fold_id,
            sample_id=context.sample_id,
            data_version_id=context.data_version_id,
            memory_estimate=int(arr.nbytes),
        )

    @staticmethod
    def _extract_column(frame: Any, column: str, output_type: str) -> np.ndarray:
        if column not in frame.columns:
            raise KeyError(f"original feature column {column!r} not in frame")
        series = frame[column]
        if output_type == NUMERIC:
            return series.to_numpy(dtype=float, na_value=np.nan)
        if output_type == CATEGORICAL:
            return series.astype(object).to_numpy()
        # BOOLEAN
        return series.to_numpy(dtype=float, na_value=np.nan)
