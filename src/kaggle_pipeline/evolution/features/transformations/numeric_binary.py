"""Binary numeric -> numeric transforms."""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.features.recipe import NUMERIC
from kaggle_pipeline.evolution.features.transformations.base import (
    EPS,
    FeatureTransformation,
    register,
)


class _Binary(FeatureTransformation):
    arity = 2
    input_types = (NUMERIC,)
    output_type = NUMERIC


@register
class Add(_Binary):
    name, short, is_commutative = "add", "add", True

    def _compute(self, inputs, params):
        return inputs[0] + inputs[1]


@register
class Subtract(_Binary):
    name, short = "subtract", "sub"

    def _compute(self, inputs, params):
        return inputs[0] - inputs[1]


@register
class Multiply(_Binary):
    name, short, is_commutative = "multiply", "mul", True

    def _compute(self, inputs, params):
        return inputs[0] * inputs[1]


@register
class SafeDivide(_Binary):
    name, short = "safe_divide", "div"

    def _compute(self, inputs, params):
        a, b = inputs[0], inputs[1]
        return a / (b + np.sign(b) * EPS + (b == 0) * EPS)


@register
class AbsDiff(_Binary):
    name, short, is_commutative = "abs_diff", "absdiff", True

    def _compute(self, inputs, params):
        return np.abs(inputs[0] - inputs[1])


@register
class Ratio(_Binary):
    name, short = "ratio", "frac"

    def _compute(self, inputs, params):
        a, b = inputs[0], inputs[1]
        return a / (a + b + EPS)


@register
class Minimum(_Binary):
    name, short, is_commutative = "minimum", "min", True

    def _compute(self, inputs, params):
        return np.minimum(inputs[0], inputs[1])


@register
class Maximum(_Binary):
    name, short, is_commutative = "maximum", "max", True

    def _compute(self, inputs, params):
        return np.maximum(inputs[0], inputs[1])
