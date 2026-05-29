"""Categorical transforms (categorical -> categorical or categorical -> numeric)."""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.features.recipe import CATEGORICAL, NUMERIC
from kaggle_pipeline.evolution.features.transformations.base import (
    FeatureTransformation,
    TransformError,
    register,
)
from kaggle_pipeline.evolution.storage.hashing import short_hash


@register
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


@register
class FrequencyEncode(FeatureTransformation):
    name, short = "frequency", "freq"
    input_types = (CATEGORICAL,)
    output_type = NUMERIC

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=object)
        values, counts = np.unique(x.astype(str), return_counts=True)
        freq = dict(zip(values, counts / x.size, strict=True))
        return np.array([freq.get(str(v), 0.0) for v in x], dtype=float)


@register
class CountEncode(FeatureTransformation):
    name, short = "count", "count"
    input_types = (CATEGORICAL,)
    output_type = NUMERIC

    def _compute(self, inputs, params):
        x = np.asarray(inputs[0], dtype=object)
        values, counts = np.unique(x.astype(str), return_counts=True)
        count = dict(zip(values, counts, strict=True))
        return np.array([count.get(str(v), 0) for v in x], dtype=float)


@register
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


@register
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


@register
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
