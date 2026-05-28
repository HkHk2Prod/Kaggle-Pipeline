"""Randomness helpers: seed derivation, stochastic rounding, softmax+exploration.

A single master seed should drive the whole run. We derive independent child
generators from a :class:`numpy.random.SeedSequence` so different concerns (feature
generation, parameter sampling, mutation, parent selection) get reproducible but
uncorrelated streams.
"""

from __future__ import annotations

import numpy as np


def spawn_rng(
    seed: int | np.random.SeedSequence | np.random.Generator | None,
) -> np.random.Generator:
    """Coerce a seed/seed-sequence/generator into a fresh ``np.random.Generator``.

    ``None`` yields an unseeded (non-reproducible) generator, matching the v1
    pipeline's default. A ``Generator`` is returned as-is so callers can thread a
    single stream through when they want shared state.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    return np.random.default_rng(seed)


def stochastic_round(value: float, rng: np.random.Generator) -> int:
    """Round ``value`` to an int, rounding up with probability equal to the fraction.

    e.g. ``7.3`` rounds to ``7`` with probability 0.7 and to ``8`` with
    probability 0.3. Keeps integer-parameter mutation unbiased in expectation
    rather than always flooring/rounding-to-nearest.
    """
    floor = np.floor(value)
    frac = float(value - floor)
    if frac <= 0.0:
        return int(floor)
    return int(floor) + int(rng.random() < frac)


def weighted_choice(
    rng: np.random.Generator | None,
    probabilities: dict[str, float],
    k: int = 1,
    *,
    replace: bool = False,
) -> list[str]:
    """Sample up to ``k`` keys from a ``{key: probability}`` mapping.

    Probabilities are renormalised (uniform if they sum to zero). Without
    replacement, at most ``len(probabilities)`` keys are returned. Returns ``[]``
    for an empty mapping.
    """
    keys = list(probabilities)
    if not keys:
        return []
    rng = spawn_rng(rng)
    p = np.asarray([probabilities[key] for key in keys], dtype=float)
    total = p.sum()
    p = p / total if total > 0 else np.full(len(keys), 1.0 / len(keys))
    size = k if replace else min(k, len(keys))
    chosen = rng.choice(len(keys), size=size, replace=replace, p=p)
    return [keys[int(i)] for i in np.atleast_1d(chosen)]


def softmax_with_exploration(
    utilities: np.ndarray,
    *,
    temperature: float = 1.0,
    exploration_rate: float = 0.0,
) -> np.ndarray:
    """Turn a vector of utilities into a probability vector.

    ``p = (1 - exploration_rate) * softmax(utility / T) + exploration_rate * uniform``.
    Numerically stable (subtracts the max before exponentiating). Returns a
    uniform distribution when given an empty or degenerate input is avoided by the
    caller; here an empty input returns an empty array.
    """
    utilities = np.asarray(utilities, dtype=float)
    n = utilities.size
    if n == 0:
        return utilities
    if not 0.0 <= exploration_rate <= 1.0:
        raise ValueError(f"exploration_rate must be in [0, 1], got {exploration_rate}.")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")

    scaled = utilities / temperature
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    total = exp.sum()
    soft = exp / total if total > 0 else np.full(n, 1.0 / n)
    uniform = np.full(n, 1.0 / n)
    probs = (1.0 - exploration_rate) * soft + exploration_rate * uniform
    # Guard against tiny float drift so probabilities sum to exactly 1.
    return probs / probs.sum()
