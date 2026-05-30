"""Merge several ecosystem checkpoints into one (the parallel-train -> blend flow).

Parallel training notebooks each evolve their own ecosystem and emit a checkpoint.
The single-thread / blend notebook attaches all of them, loads each into an
:class:`EcosystemState`, and calls :func:`merge_ecosystem_states` to fold them
into one ecosystem that keeps the best genes:

* **Features** are unioned by recipe hash (identical recipes share a feature id,
  so model references stay valid). A feature active in *any* input is active in
  the merge.
* **Models** are unioned by genome hash; a genome seen in more than one input is
  kept once, preferring the copy with the higher CV score (duplicates dropped).
* **OOF predictions** travel with their model. They line up across inputs when
  the notebooks shared ``search_sample_seed``; if a later subsample differs the
  blender recomputes them (see ``KagglePipeline._ensure_oof_compatible``).

After folding, the population's own utility / elite / prune logic caps the active
set, so ``n`` ecosystems of ~``max_active`` models collapse to a single
``max_active`` leaderboard. Progress is logged line-by-line so the merge is
auditable from the run log.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING

from kaggle_pipeline.evolution.ecosystem.state import EcosystemState
from kaggle_pipeline.evolution.evaluation.oof_store import OOFStore
from kaggle_pipeline.evolution.features.registry import FeatureRegistry
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.parameter_spaces import build_default_families
from kaggle_pipeline.evolution.models.registry import ModelPopulation

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kaggle_pipeline.evolution.config import KagglePipelineSettings

Logger = Callable[[str], None]


def _genome_score(genome: ModelGenome) -> float:
    """Raw CV score used to break genome-hash ties; absent score sorts last."""
    return genome.score_set.score if genome.score_set is not None else float("-inf")


def merge_ecosystem_states(
    states: list[EcosystemState],
    *,
    settings: KagglePipelineSettings,
    log: Logger | None = None,
) -> EcosystemState:
    """Fold ``states`` into one ecosystem, keeping the best genes by performance.

    A single input is returned untouched (nothing to merge). Raises if ``states``
    is empty -- the caller decides what "no ecosystems found" means.
    """
    emit: Logger = log or (lambda _msg: None)
    if not states:
        raise ValueError("merge_ecosystem_states requires at least one state")
    if len(states) == 1:
        emit("[merge] single input ecosystem; using it as-is (nothing to merge)")
        return states[0]

    emit(f"[merge] merging {len(states)} input ecosystems")
    for i, state in enumerate(states):
        best = state.population.absolute_score_ranking()
        best_score = best[0].score_set.score if best and best[0].score_set else None
        emit(
            f"[merge]   ecosystem {i + 1}: batch={state.batch_index}, "
            f"{len(state.population.all_genomes())} models, "
            f"{len(state.registry.all_features())} features, "
            f"best_score={best_score if best_score is None else round(best_score, 4)}"
        )

    evo = settings.evolution_settings()
    registry = _merge_registries(states, evo, emit)
    population, oof_store, duplicates = _merge_populations(states, evo, emit)

    # Cap the union the same way a live run would: rank by effective score,
    # keep elites, prune the rest (which also frees their OOF).
    population.update_utilities()
    population.update_elite()
    population.prune_active()

    emit(
        f"[merge] merged ecosystem: {len(population.all_genomes())} models "
        f"({duplicates} duplicate genome(s) dropped), {len(population.active)} active, "
        f"{len(population.elite)} elite, {len(registry.all_features())} features"
    )

    return EcosystemState(
        config_snapshot=asdict(settings),
        batch_index=max(s.batch_index for s in states),
        registry=registry,
        population=population,
        oof_store=oof_store,
        rng_state=None,  # fresh RNG, seeded from the merge run's settings
        score_history=_longest(s.score_history for s in states),
        runtime_history=_longest(s.runtime_history for s in states),
        ensemble_state=None,  # rebuilt by the blender from the merged population
    )


def _merge_registries(states: list[EcosystemState], evo, emit: Logger) -> FeatureRegistry:
    """Union features across inputs, deduplicating by recipe hash."""
    merged = FeatureRegistry(evo)
    duplicates = 0
    for state in states:
        merged.current_batch = max(merged.current_batch, state.registry.current_batch)
        for genome in state.registry.all_features():
            rh = genome.recipe_hash
            if merged.has_recipe_hash(rh):
                duplicates += 1
                # A feature active anywhere stays active in the merge.
                if genome.active:
                    merged.get_feature(merged._by_recipe_hash[rh]).active = True
                continue
            merged._features[genome.feature_id] = genome
            merged._by_recipe_hash[rh] = genome.feature_id
    emit(
        f"[merge] features: {len(merged.all_features())} unique "
        f"({duplicates} duplicate(s) merged), "
        f"{len(merged.get_active_features())} active"
    )
    return merged


def _merge_populations(
    states: list[EcosystemState], evo, emit: Logger
) -> tuple[ModelPopulation, OOFStore, int]:
    """Union genomes by hash (best score wins ties) and carry their OOF over."""
    max_active = max(s.population.max_active for s in states)
    elite_size = max(s.population.elite_size for s in states)
    population = ModelPopulation(
        evo, max_active=max_active, elite_size=elite_size, families=build_default_families()
    )
    oof_store = OOFStore()
    population.oof_store = oof_store

    active_union: set[str] = set()
    duplicates = 0
    for state in states:
        active_union.update(state.population.active)
        for genome in state.population.all_genomes():
            existing = population.get_by_hash(genome.genome_hash)
            if existing is not None:
                duplicates += 1
                if _genome_score(genome) <= _genome_score(existing):
                    continue
                # Better-scoring copy of the same genome -- swap it (and its OOF) in.
                population._by_id[existing.model_id] = genome
                population._wire_score_recomputers(genome)
            else:
                population.register(genome)
            arr = state.oof_store.get(genome.model_id)
            if arr is not None:
                oof_store.store(genome.model_id, arr)

    population.active = [mid for mid in active_union if mid in population._by_id]
    return population, oof_store, duplicates


def _longest(histories) -> list:
    """Pick the longest per-batch history among inputs (display metadata only)."""
    return list(max(histories, key=len, default=[]))
