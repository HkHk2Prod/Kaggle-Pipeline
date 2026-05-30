"""Per-family survival floor: pruning never culls a family below its ``min_models``.

A family's minimum is a property of its :class:`FamilyDefinition` -- an ``int``
(absolute count) or a ``float`` (fraction of the population cap, rounded up).
The default of ``1`` keeps the single best model of every family alive.
"""

from __future__ import annotations

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.genes.base import BaseModelGene
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus
from kaggle_pipeline.evolution.models.parameter_spaces import (
    DEFAULT_MIN_MODELS,
    build_default_families,
    resolve_min_count,
)
from kaggle_pipeline.evolution.models.registry import ModelPopulation
from kaggle_pipeline.evolution.models.scoring import ModelScoreSet


def _completed(
    population: ModelPopulation, model_id: str, family: str, score: float
) -> ModelGenome:
    g = ModelGenome(
        base_model_gene=BaseModelGene(family),
        feature_reference_genes=[FeatureReferenceGene("orig::a")],
        model_id=model_id,
    )
    g.status = ModelStatus.COMPLETED
    g.score_set = ModelScoreSet(score=score, score_std=0.0, compute_time=1.0)
    # Bypass register so each test genome keeps its given id regardless of hash.
    population._by_id[model_id] = g
    population.active.append(model_id)
    return g


# --- resolve_min_count --------------------------------------------------------
def test_resolve_min_count_int_is_literal():
    assert resolve_min_count(3, total=200) == 3


def test_resolve_min_count_float_is_ceil_fraction():
    # 5% of 200 -> 10; any positive fraction rounds up to at least one.
    assert resolve_min_count(0.05, total=200) == 10
    assert resolve_min_count(0.001, total=200) == 1


def test_resolve_min_count_clamps_to_total():
    assert resolve_min_count(50, total=10) == 10
    assert resolve_min_count(2.0, total=10) == 10  # 200% caps at the total


def test_resolve_min_count_bool_is_normalised_to_int():
    # ``bool`` is an ``int`` subclass; True/False must mean 1/0, not a fraction.
    assert resolve_min_count(True, total=200) == 1
    assert resolve_min_count(False, total=200) == 0


# --- FamilyDefinition default -------------------------------------------------
def test_default_families_keep_at_least_one():
    families = build_default_families()
    assert families  # sanity: catalog is non-empty
    for fam in families.values():
        assert fam.min_models == DEFAULT_MIN_MODELS == 1
        assert fam.min_model_count(total=200) == 1


# --- pruning honours the floor ------------------------------------------------
def _pop(max_active: int, min_models: dict[str, int | float]) -> ModelPopulation:
    pop = ModelPopulation(EvolutionSettings(), max_active=max_active, elite_size=0)
    pop.family_min_models = min_models
    return pop


def test_default_floor_keeps_a_weak_family_alive():
    # Three strong "logistic" models and one weak "knn"; cap of 2 would cull the
    # knn on utility, but its default floor of 1 shields the family's best.
    pop = _pop(max_active=2, min_models={})
    for i, score in enumerate((0.90, 0.88, 0.86)):
        _completed(pop, f"m_log_{i}", "logistic", score)
    weak = _completed(pop, "m_knn", "knn", score=0.40)
    pop.update_utilities()
    pop.update_elite()
    pop.prune_active()
    assert weak.model_id in pop.active
    assert weak.status == ModelStatus.COMPLETED  # protected, not pruned


def test_int_min_models_keeps_that_many_per_family():
    # Floor of 2 for "knn" keeps its two best even though both rank below the cap.
    pop = _pop(max_active=2, min_models={"knn": 2})
    for i, score in enumerate((0.90, 0.88, 0.86)):
        _completed(pop, f"m_log_{i}", "logistic", score)
    knn_keep = [_completed(pop, f"m_knn_{i}", "knn", s) for i, s in enumerate((0.50, 0.45))]
    knn_drop = _completed(pop, "m_knn_2", "knn", score=0.30)
    pop.update_utilities()
    pop.update_elite()
    pop.prune_active()
    assert all(g.model_id in pop.active for g in knn_keep)
    # The third (worst) knn is beyond the floor and gets pruned.
    assert knn_drop.model_id not in pop.active
    assert knn_drop.status == ModelStatus.PRUNED


def test_float_min_models_protects_a_fraction_of_the_cap():
    # 30% of a 10-cap -> ceil(3) protected per family.
    pop = _pop(max_active=10, min_models={"knn": 0.3})
    for i in range(12):  # 12 strong logistics overflow the cap on their own
        _completed(pop, f"m_log_{i}", "logistic", score=0.9 - i * 0.001)
    knn = [_completed(pop, f"m_knn_{i}", "knn", score=0.1 + i * 0.001) for i in range(5)]
    pop.update_utilities()
    pop.update_elite()
    pop.prune_active()
    survivors = [g for g in knn if g.model_id in pop.active]
    assert len(survivors) == 3


def test_set_family_minimums_extracts_specs_from_catalog():
    families = build_default_families()
    pop = ModelPopulation(EvolutionSettings(), families=families)
    assert pop.family_min_models["logistic"] == 1
    # Only picklable specs are cached, never the FamilyDefinition objects.
    assert all(isinstance(v, (int, float)) for v in pop.family_min_models.values())
