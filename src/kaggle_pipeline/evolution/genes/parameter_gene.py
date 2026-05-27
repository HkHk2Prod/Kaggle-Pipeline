"""The :class:`ParameterGene` and its :class:`ParameterSpec`.

A parameter gene carries a value plus the spec that defines its space (type,
bounds, log-scale, complexity direction). ``mutate`` interprets a *signed amount*:
positive means "more complex / more expressive" where that concept applies,
negative means "simpler / more regularized"; for parameters with no complexity
meaning (``complexity_direction=None``) it is plain numeric up/down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from kaggle_pipeline.evolution.genes.base import PARAMETER, Gene
from kaggle_pipeline.evolution.utils.random import stochastic_round

if TYPE_CHECKING:
    from kaggle_pipeline.evolution.genes.base import MutationContext

FLOAT = "float"
INT = "int"
CATEGORICAL = "categorical"

# Complexity directions.
POSITIVE = "positive"  # larger value == more complex/expressive
NEGATIVE = "negative"  # smaller value == more complex/expressive (regularizers)


@dataclass
class ParameterSpec:
    """The search/mutation space of one model parameter."""

    name: str
    kind: str = FLOAT
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] = ()
    log_scale: bool = False
    complexity_direction: str | None = None
    model_family: str | None = None
    dependencies: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in (FLOAT, INT, CATEGORICAL):
            raise ValueError(f"unknown parameter kind {self.kind!r}")
        if self.kind == CATEGORICAL and not self.choices:
            raise ValueError(f"categorical parameter {self.name!r} needs choices")
        if self.kind in (FLOAT, INT) and (self.low is None or self.high is None):
            raise ValueError(f"numeric parameter {self.name!r} needs low/high bounds")

    def bounds(self) -> tuple[float, float]:
        """The numeric (low, high) bounds; only valid for non-categorical specs."""
        assert self.low is not None and self.high is not None  # ensured in __post_init__
        return float(self.low), float(self.high)

    def sample(self, rng: np.random.Generator) -> Any:
        """Draw one value from the parameter's distribution."""
        if self.kind == CATEGORICAL:
            return self.choices[int(rng.integers(len(self.choices)))]
        low, high = self.bounds()
        if self.log_scale:
            value = math.exp(rng.uniform(math.log(max(low, 1e-12)), math.log(max(high, 1e-12))))
        else:
            value = rng.uniform(low, high)
        if self.kind == INT:
            return int(self.clamp(round(value)))
        return float(self.clamp(value))

    def clamp(self, value: Any) -> Any:
        if self.kind == CATEGORICAL:
            return value
        low, high = self.bounds()
        v = min(max(float(value), low), high)
        return int(round(v)) if self.kind == INT else v


class ParameterGene(Gene):
    """A single model hyperparameter, mutated by signed amount."""

    def __init__(self, spec: ParameterSpec, value: Any, **kwargs: Any):
        super().__init__(PARAMETER, value, **kwargs)
        self.spec = spec

    @property
    def parameter_name(self) -> str:
        return self.spec.name

    def validate(self) -> None:
        spec = self.spec
        if spec.kind == CATEGORICAL:
            if self.value not in spec.choices:
                raise ValueError(f"{spec.name}={self.value!r} not in choices {spec.choices}")
        else:
            low, high = spec.bounds()
            if not (low <= float(self.value) <= high):
                raise ValueError(
                    f"{spec.name}={self.value} out of bounds [{spec.low}, {spec.high}]"
                )

    def hash_component(self) -> dict[str, Any]:
        return {"gene_type": PARAMETER, "name": self.spec.name, "value": self.value}

    def mutate(self, signed_amount: float, context: MutationContext) -> ParameterGene:
        if not self.mutable:
            return self.copy()
        spec = self.spec
        if spec.kind == CATEGORICAL:
            return self.fresh_child(self._mutate_categorical(context.rng))

        # Flip the amount for regularizer-like parameters where *smaller* is more
        # complex, so a positive signed_amount always means "more complex".
        amount = signed_amount if spec.complexity_direction != NEGATIVE else -signed_amount
        base = float(self.value)
        low, high = spec.bounds()
        if spec.log_scale and base > 0:
            new = base * math.exp(amount)
        elif abs(base) < 1e-12:
            new = base + amount * (high - low) * 0.1
        else:
            new = base * (1.0 + amount)

        new = spec.clamp(new)
        if spec.kind == INT:
            new = int(spec.clamp(stochastic_round(float(new), context.rng)))
        child = self.fresh_child(new)
        child.validate()
        return child

    def _mutate_categorical(self, rng: np.random.Generator) -> Any:
        others = [c for c in self.spec.choices if c != self.value]
        if not others:
            return self.value
        return others[int(rng.integers(len(others)))]
