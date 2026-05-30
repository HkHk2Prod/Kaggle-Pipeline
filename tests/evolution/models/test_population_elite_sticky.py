"""``was_elite`` is set the first time a genome appears in the elite list and never reset."""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.genes.base import BaseModelGene
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus
from kaggle_pipeline.evolution.models.registry import ModelPopulation
from kaggle_pipeline.evolution.models.scoring import ModelScoreSet


def _completed(population: ModelPopulation, model_id: str, score: float) -> ModelGenome:
    g = ModelGenome(
        base_model_gene=BaseModelGene("logistic"),
        feature_reference_genes=[FeatureReferenceGene("orig::a")],
        model_id=model_id,
    )
    g.status = ModelStatus.COMPLETED
    g.score_set = ModelScoreSet(score=score, score_std=0.0, compute_time=1.0)
    # Bypass register so we can give each test genome a distinct id without
    # relying on a distinct genome hash.
    population._by_id[model_id] = g
    population.active.append(model_id)
    return g


def test_elite_membership_is_sticky_across_subsequent_updates():
    # m_top makes elite_size=1, gets the sticky flag, then a better model
    # arrives and bumps it. The bumped genome must keep ``was_elite=True``.
    population = ModelPopulation(EvolutionSettings(), max_active=10, elite_size=1)
    g_top = _completed(population, "m_top", score=0.90)
    population.update_elite()
    assert g_top.was_elite is True

    g_better = _completed(population, "m_better", score=0.95)
    population.update_elite()
    assert g_better.was_elite is True
    # Sticky: m_top stays True even though it is no longer in ``elite``.
    assert "m_top" not in population.elite
    assert g_top.was_elite is True


def test_never_elite_genome_keeps_flag_false():
    # elite_size=1 with two genomes -- only the better one is ever elite.
    population = ModelPopulation(EvolutionSettings(), max_active=10, elite_size=1)
    g_winner = _completed(population, "m_winner", score=0.95)
    g_loser = _completed(population, "m_loser", score=0.40)
    population.update_elite()
    assert g_winner.was_elite is True
    assert g_loser.was_elite is False


def test_old_pickle_without_attribute_is_treated_as_was_elite_false():
    # Simulate a genome unpickled from a version that predates the field.
    population = ModelPopulation(EvolutionSettings(), max_active=10, elite_size=1)
    g = _completed(population, "m_legacy", score=0.5)
    del g.__dict__["was_elite"]
    # ``getattr`` fallback used by the compute-waste summary must not raise.
    from kaggle_pipeline.evolution.ecosystem.compute_waste import classify

    out = classify(g, frozenset())
    assert out == ("waste", "new")  # treated as not-elite, not in ensemble
    # And update_elite must be able to set it back without exploding.
    population.update_elite()
    assert g.was_elite is True


def test_zero_mean_is_zero_for_genome_with_no_compute_time():
    # ``classify`` skips genomes that never finished training.
    from kaggle_pipeline.evolution.ecosystem.compute_waste import classify

    g = ModelGenome(
        base_model_gene=BaseModelGene("logistic"),
        feature_reference_genes=[FeatureReferenceGene("orig::a")],
        model_id="m_ghost",
    )
    # Untrained: no score_set.
    assert classify(g, frozenset()) is None


def test_setting_flag_is_idempotent_across_repeated_updates():
    # Repeated update_elite calls must not toggle the flag back to False.
    rng = np.random.default_rng(0)  # noqa: F841 -- deterministic; not used directly
    population = ModelPopulation(EvolutionSettings(), max_active=10, elite_size=1)
    g = _completed(population, "m_persistent", score=0.9)
    for _ in range(5):
        population.update_elite()
    assert g.was_elite is True
