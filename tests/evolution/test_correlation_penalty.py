"""Per-batch residual-correlation penalty applied to active models.

A model gets a penalty proportional to the most-positive residual-error
correlation it has with any *strictly higher-scoring* active model. The top
scorer in any cluster is therefore never penalised. Once applied, every
ranking lambda (elite, prune, tournament, ensemble candidates) subtracts the
penalty from that model's effective score.
"""

from __future__ import annotations

import numpy as np
import pytest

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.evaluation.oof_store import OOFStore
from kaggle_pipeline.evolution.genes.base import BaseModelGene
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus
from kaggle_pipeline.evolution.models.registry import ModelPopulation
from kaggle_pipeline.evolution.models.scoring import ModelScoreSet


@pytest.fixture
def y():
    rng = np.random.default_rng(0)
    return (rng.uniform(size=200) > 0.5).astype(int)


def _make_active(population: ModelPopulation, model_id: str, raw_score: float) -> ModelGenome:
    """Insert a minimal completed genome at ``raw_score`` directly into the active set.

    Bypasses ``register`` because we want each test genome distinct by id, not by
    genome hash -- the dedup-by-hash path would collapse the lookalikes we want
    to use as correlated pairs. We still wire the score recomputers (which
    ``register`` would normally do) so ``get_score`` works.
    """
    g = ModelGenome(
        base_model_gene=BaseModelGene("logistic"),
        feature_reference_genes=[FeatureReferenceGene("orig::a")],
        model_id=model_id,
    )
    g.status = ModelStatus.COMPLETED
    g.score_set = ModelScoreSet(score=raw_score, score_std=0.0, compute_time=1.0)
    population._by_id[model_id] = g
    population.active.append(model_id)
    population._wire_score_recomputers(g)
    return g


def test_top_model_is_never_penalized(y):
    # Two near-identical OOFs; the better-scoring one has no higher peer, so its
    # penalty stays 0 while the weaker one absorbs the full hit.
    population = ModelPopulation(EvolutionSettings(), max_active=10, elite_size=1)
    store = OOFStore()
    population.oof_store = store

    rng = np.random.default_rng(1)
    base = rng.uniform(size=200)
    store.store("m_top", base)
    store.store("m_dup", base + 1e-6 * rng.normal(size=200))  # essentially identical residuals
    _make_active(population, "m_top", raw_score=0.95)
    _make_active(population, "m_dup", raw_score=0.90)

    n = population.compute_correlation_penalties(y, threshold=0.975, scale=10.0)
    assert n == 1
    assert population.get("m_top").correlation_penalty == 0.0
    assert population.get("m_dup").correlation_penalty > 0.1  # ~10 * (1 - 0.975)


def test_uncorrelated_model_gets_no_penalty(y):
    population = ModelPopulation(EvolutionSettings(), max_active=10, elite_size=1)
    store = OOFStore()
    population.oof_store = store

    rng = np.random.default_rng(2)
    store.store("m_top", rng.uniform(size=200))
    store.store("m_other", rng.uniform(size=200))  # independent draw
    _make_active(population, "m_top", raw_score=0.95)
    _make_active(population, "m_other", raw_score=0.90)

    population.compute_correlation_penalties(y, threshold=0.975, scale=10.0)
    assert population.get("m_top").correlation_penalty == 0.0
    assert population.get("m_other").correlation_penalty == 0.0


def test_anti_correlated_residuals_are_not_penalized(y):
    # Errors going the *opposite* direction are valuable in a blend, not redundant.
    population = ModelPopulation(EvolutionSettings(), max_active=10, elite_size=1)
    store = OOFStore()
    population.oof_store = store

    rng = np.random.default_rng(3)
    noise = rng.normal(size=200)
    store.store("m_top", y + 0.1 * noise)  # residuals ~ 0.1 * noise
    store.store("m_other", y - 0.1 * noise)  # residuals ~ -0.1 * noise (anti-correlated)
    _make_active(population, "m_top", raw_score=0.95)
    _make_active(population, "m_other", raw_score=0.90)

    population.compute_correlation_penalties(y, threshold=0.975, scale=10.0)
    assert population.get("m_other").correlation_penalty == 0.0


def test_penalty_scale_severity_at_099(y):
    # The user's calibration target: scale=10 at r≈0.99 -> penalty ≈ 0.15.
    population = ModelPopulation(EvolutionSettings(), max_active=10, elite_size=1)
    store = OOFStore()
    population.oof_store = store

    rng = np.random.default_rng(4)
    base_resid = rng.normal(size=200)
    # Construct two OOFs whose residuals correlate ~0.99 with each other.
    store.store("m_top", y + base_resid)
    store.store("m_dup", y + base_resid + 0.05 * rng.normal(size=200))
    _make_active(population, "m_top", raw_score=0.95)
    _make_active(population, "m_dup", raw_score=0.90)

    population.compute_correlation_penalties(y, threshold=0.975, scale=10.0)
    pen = population.get("m_dup").correlation_penalty
    # Around 10 * (0.99 - 0.975) ≈ 0.15; loose bounds because the synthetic noise
    # leaves room for sampling variance.
    assert 0.05 < pen < 0.30


def test_penalty_propagates_into_absolute_score_ranking(y):
    # Without a penalty, m_dup (raw 0.94) would rank above m_clean (raw 0.93).
    # With penalty large enough to drop m_dup below 0.93, the order flips.
    population = ModelPopulation(EvolutionSettings(), max_active=10, elite_size=2)
    store = OOFStore()
    population.oof_store = store

    rng = np.random.default_rng(5)
    base = rng.uniform(size=200)
    store.store("m_top", base)
    store.store("m_dup", base + 1e-6 * rng.normal(size=200))
    store.store("m_clean", rng.uniform(size=200))
    _make_active(population, "m_top", raw_score=0.95)
    _make_active(population, "m_dup", raw_score=0.94)
    _make_active(population, "m_clean", raw_score=0.93)

    population.compute_correlation_penalties(y, threshold=0.975, scale=10.0)
    order = [g.model_id for g in population.absolute_score_ranking()]
    # m_top first (untouched); m_clean now beats m_dup despite the lower raw score.
    assert order[0] == "m_top"
    assert order.index("m_clean") < order.index("m_dup")


def test_penalties_reset_each_call(y):
    population = ModelPopulation(EvolutionSettings(), max_active=10, elite_size=1)
    store = OOFStore()
    population.oof_store = store

    rng = np.random.default_rng(6)
    base = rng.uniform(size=200)
    store.store("m_top", base)
    store.store("m_dup", base + 1e-6 * rng.normal(size=200))
    _make_active(population, "m_top", raw_score=0.95)
    _make_active(population, "m_dup", raw_score=0.90)

    population.compute_correlation_penalties(y, threshold=0.975, scale=10.0)
    assert population.get("m_dup").correlation_penalty > 0.0

    # Replace m_dup's OOF with something uncorrelated; the next pass must
    # zero its stale penalty out, not leave it hanging.
    store.store("m_dup", rng.uniform(size=200))
    population.compute_correlation_penalties(y, threshold=0.975, scale=10.0)
    assert population.get("m_dup").correlation_penalty == 0.0


def test_disabled_when_scale_is_zero(y):
    population = ModelPopulation(EvolutionSettings(), max_active=10, elite_size=1)
    store = OOFStore()
    population.oof_store = store

    rng = np.random.default_rng(7)
    base = rng.uniform(size=200)
    store.store("m_top", base)
    store.store("m_dup", base)
    _make_active(population, "m_top", raw_score=0.95)
    _make_active(population, "m_dup", raw_score=0.90)

    n = population.compute_correlation_penalties(y, threshold=0.975, scale=0.0)
    assert n == 0
    assert population.get("m_dup").correlation_penalty == 0.0


def test_missing_individual_score_triggers_lazy_recompute_with_warning(y, caplog):
    # Simulates an "old ecosystem" genome: the dataclass field doesn't exist on
    # this instance at all (as if pickled before the field was added). The
    # first ``get_score`` call should warn and recompute through the registered
    # callback. This is the generic mechanism on ``ModelGenome`` -- it works
    # the same way for any future individual score.
    import logging

    population = ModelPopulation(EvolutionSettings(), max_active=10, elite_size=1)
    store = OOFStore()
    population.oof_store = store
    population.set_search_target(y)

    rng = np.random.default_rng(8)
    base = rng.uniform(size=200)
    store.store("m_top", base)
    store.store("m_dup", base + 1e-6 * rng.normal(size=200))
    g_top = _make_active(population, "m_top", raw_score=0.95)
    g_dup = _make_active(population, "m_dup", raw_score=0.90)

    # Strip the attribute to simulate the pre-field pickle case.
    del g_top.__dict__["correlation_penalty"]
    del g_dup.__dict__["correlation_penalty"]

    with caplog.at_level(logging.WARNING, logger="kaggle_pipeline.evolution.models.genome"):
        order = [g.model_id for g in population.absolute_score_ranking()]

    # Lazy recompute populated the field on both genomes.
    assert g_top.correlation_penalty == 0.0
    assert g_dup.correlation_penalty > 0.0
    # m_top still ranks first; m_dup keeps a smaller effective score.
    assert order == ["m_top", "m_dup"]
    # Each genome warns at most once for that score name. m_dup's value got
    # set by m_top's recompute before m_dup's ranking key was evaluated, so
    # exactly one warning fires across the two genomes.
    warnings = [r for r in caplog.records if "missing for model" in r.getMessage()]
    assert len(warnings) == 1


def test_missing_individual_score_without_recomputer_returns_none(y, caplog):
    # If no recomputer is registered for the missing score, ``get_score``
    # still warns once and then returns ``None`` -- it never crashes.
    import logging

    population = ModelPopulation(EvolutionSettings(), max_active=10, elite_size=1)
    g = _make_active(population, "m_only", raw_score=0.5)
    # Drop the recomputer for "correlation_penalty" to simulate an unknown score.
    g._score_recomputers.pop("correlation_penalty", None)
    del g.__dict__["correlation_penalty"]

    with caplog.at_level(logging.WARNING, logger="kaggle_pipeline.evolution.models.genome"):
        value = g.get_score("correlation_penalty")

    assert value is None
    assert any("missing for model" in r.getMessage() for r in caplog.records)


def test_resume_path_rewires_recomputers_after_pickle():
    # After a pickle/unpickle round-trip the recomputers are gone (their
    # closures captured the live population, which is not picklable). The
    # population must rebind them via ``wire_all_score_recomputers``.
    import pickle

    settings = EvolutionSettings()
    population = ModelPopulation(settings, max_active=10, elite_size=1)
    population.oof_store = OOFStore()
    g = _make_active(population, "m_only", raw_score=0.5)
    # Confirm the live genome has a recomputer wired pre-pickle.
    assert "correlation_penalty" in g._score_recomputers

    restored = pickle.loads(pickle.dumps(population))
    restored_genome = restored.get("m_only")
    # The recomputer dict survives unpickling but is empty (closures dropped).
    assert restored_genome._score_recomputers == {}

    restored.wire_all_score_recomputers()
    assert "correlation_penalty" in restored_genome._score_recomputers
    assert "utility" in restored_genome._score_recomputers
