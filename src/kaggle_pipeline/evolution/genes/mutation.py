"""Mutation policy helpers shared by the model mutator.

Centralises the two random draws the README prescribes:

* the **number** of genes to mutate in a child, from the configured distribution
  (default ``{1: .70, 2: .20, 3: .07, 4: .03}``) -- a small count, *not* an
  independent per-gene probability, which would mutate too many genes in large
  genomes and wreck credit assignment;
* the **signed amount**, ``Uniform(drift - scale, drift + scale)``.
"""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.config import EvolutionSettings


def sample_signed_amount(settings: EvolutionSettings, rng: np.random.Generator) -> float:
    """Draw a signed mutation amount from ``Uniform(drift - scale, drift + scale)``."""
    lo = settings.mutation_drift - settings.mutation_scale
    hi = settings.mutation_drift + settings.mutation_scale
    return float(rng.uniform(lo, hi))


def sample_num_mutated_genes(settings: EvolutionSettings, rng: np.random.Generator) -> int:
    """Draw how many genes to mutate from the configured count distribution."""
    dist = settings.preferred_num_mutated_genes_distribution
    counts = sorted(dist)
    probs = np.array([dist[c] for c in counts], dtype=float)
    probs = probs / probs.sum()
    return int(counts[int(rng.choice(len(counts), p=probs))])
