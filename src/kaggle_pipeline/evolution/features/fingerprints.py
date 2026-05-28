"""Cheap signatures of a feature's values on the reference evaluation sample.

A fingerprint lets the registry detect duplicate / near-duplicate features and
seed similarity without keeping every feature fully materialized. It is computed
once per feature on a fixed sample so signatures are comparable across features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kaggle_pipeline.evolution.features.materialization import hash_values
from kaggle_pipeline.evolution.features.recipe import NUMERIC
from kaggle_pipeline.evolution.utils.arrays import missing_mask


@dataclass
class FeatureFingerprint:
    """A compact summary of one feature's values on the evaluation sample."""

    feature_id: str
    recipe_hash: str
    value_hash_on_sample: str
    output_type: str
    missing_rate: float
    n_rows: int
    # Numeric summary (None for categoricals).
    mean: float | None = None
    std: float | None = None
    quantiles: list[float] | None = None
    # Categorical summary (None for numerics).
    cardinality: int | None = None
    top_categories: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_serializable(self) -> dict[str, Any]:
        return dict(self.__dict__)


def compute_fingerprint(
    feature_id: str, recipe_hash: str, output_type: str, values: np.ndarray
) -> FeatureFingerprint:
    """Summarise ``values`` into a :class:`FeatureFingerprint`."""
    arr = np.asarray(values)
    n = int(arr.shape[0]) if arr.ndim else 0
    if output_type == NUMERIC or arr.dtype != object:
        floats = np.asarray(arr, dtype=float).ravel()
        finite = floats[np.isfinite(floats)]
        missing = float(np.isnan(floats).mean()) if floats.size else 1.0
        quantiles = (
            [float(q) for q in np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0])]
            if finite.size
            else None
        )
        return FeatureFingerprint(
            feature_id=feature_id,
            recipe_hash=recipe_hash,
            value_hash_on_sample=hash_values(arr),
            output_type=output_type,
            missing_rate=missing,
            n_rows=n,
            mean=float(finite.mean()) if finite.size else None,
            std=float(finite.std()) if finite.size else None,
            quantiles=quantiles,
        )

    obj = np.asarray(arr, dtype=object)
    mask = missing_mask(obj)
    present = obj[~mask].astype(str)
    values_u, counts = (
        np.unique(present, return_counts=True) if present.size else (np.array([]), np.array([]))
    )
    order = np.argsort(counts)[::-1][:5]
    return FeatureFingerprint(
        feature_id=feature_id,
        recipe_hash=recipe_hash,
        value_hash_on_sample=hash_values(arr),
        output_type=output_type,
        missing_rate=float(mask.mean()) if obj.size else 1.0,
        n_rows=n,
        cardinality=int(values_u.size),
        top_categories=[str(values_u[i]) for i in order],
    )
