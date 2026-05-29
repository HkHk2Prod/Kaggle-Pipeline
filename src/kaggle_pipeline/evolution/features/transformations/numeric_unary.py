"""Unary numeric -> numeric / boolean transforms."""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.features.recipe import BOOLEAN
from kaggle_pipeline.evolution.features.transformations.base import (
    EPS,
    FeatureTransformation,
    TransformError,
    register,
)


@register
class Log1p(FeatureTransformation):
    name, short = "log1p", "log"

    def _compute(self, inputs, params):
        x = inputs[0]
        # Shift to be non-negative before log1p so negatives are handled safely.
        shift = min(0.0, float(np.nanmin(x)))
        return np.log1p(x - shift)


@register
class Sqrt(FeatureTransformation):
    name, short = "sqrt", "sqrt"

    def _compute(self, inputs, params):
        x = inputs[0]
        shift = min(0.0, float(np.nanmin(x)))
        return np.sqrt(x - shift)


@register
class Square(FeatureTransformation):
    name, short = "square", "sq"

    def _compute(self, inputs, params):
        return np.square(inputs[0])


@register
class Rank(FeatureTransformation):
    name, short = "rank", "rank"

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=float)
        order = np.argsort(np.argsort(x))
        return order.astype(float) / max(1, x.size - 1)


@register
class ZScore(FeatureTransformation):
    name, short = "zscore", "z"

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=float)
        mean, std = np.nanmean(x), np.nanstd(x)
        return (x - mean) / (std + EPS)


@register
class MinMax(FeatureTransformation):
    name, short = "minmax", "mm"

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / (hi - lo + EPS)


@register
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


@register
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


@register
class MissingIndicator(FeatureTransformation):
    name, short = "missing_indicator", "isna"
    output_type = BOOLEAN

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=float)
        return (~np.isfinite(x)).astype(float)
