"""The checkpointable :class:`EcosystemState`.

A snapshot of everything needed to resume a run: the feature registry, the model
population (genomes, results, elite archive, mutation history), the OOF store, the
RNG state, the config snapshot and the score/runtime histories. It deliberately
does **not** hold live thread pools, futures, or the model-family callables --
those are rebuilt from settings on load (the families dict contains lambdas and is
not picklable, and executors must never be serialized).

Checkpointing happens at batch boundaries, so the state always reflects results
already applied to the registries; there are no half-applied partial results.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kaggle_pipeline.evolution.evaluation.oof_store import OOFStore
    from kaggle_pipeline.evolution.features.registry import FeatureRegistry
    from kaggle_pipeline.evolution.models.registry import ModelPopulation

PIPELINE_VERSION = "0.1.0"


@dataclass
class EcosystemState:
    """Picklable snapshot of the evolutionary ecosystem at a batch boundary."""

    config_snapshot: dict[str, Any]
    batch_index: int
    registry: FeatureRegistry
    population: ModelPopulation
    oof_store: OOFStore
    rng_state: dict[str, Any] | None = None
    score_history: list[dict[str, Any]] = field(default_factory=list)
    runtime_history: list[dict[str, Any]] = field(default_factory=list)
    ensemble_state: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    pipeline_version: str = PIPELINE_VERSION
    python_version: str = field(default_factory=lambda: sys.version.split()[0])

    def touch(self) -> None:
        self.last_updated_at = time.time()
