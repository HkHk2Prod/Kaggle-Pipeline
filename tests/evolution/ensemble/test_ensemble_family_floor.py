"""The ensemble honours each family's ``min_models`` floor.

``select_candidates`` guarantees every family's top ``min_models`` a seat even
when they score below the candidate cap, and greedy selection is seeded with
those required members so the floor survives into the final blend.
"""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.config import EvolutionSettings, KagglePipelineSettings
from kaggle_pipeline.evolution.ensemble.greedy import greedy_weights
from kaggle_pipeline.evolution.ensemble.manager import EnsembleManager
from kaggle_pipeline.evolution.evaluation.oof_store import OOFStore
from kaggle_pipeline.evolution.genes.base import BaseModelGene
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus
from kaggle_pipeline.evolution.models.registry import ModelPopulation
from kaggle_pipeline.evolution.models.scoring import ModelScoreSet
from kaggle_pipeline.scoring.metrics import resolve_scoring


def _add(pop: ModelPopulation, oof: OOFStore, mid: str, family: str, score: float) -> ModelGenome:
    g = ModelGenome(
        base_model_gene=BaseModelGene(family),
        feature_reference_genes=[FeatureReferenceGene("orig::a")],
        model_id=mid,
    )
    g.status = ModelStatus.COMPLETED
    g.score_set = ModelScoreSet(score=score, score_std=0.0, compute_time=1.0)
    pop._by_id[mid] = g
    pop.active.append(mid)
    # A single-column OOF (binary P(class0)); content is irrelevant to selection.
    oof.store(mid, np.full((4, 1), 1.0 - score))
    return g


def _ensemble_setup(max_models: int):
    pop = ModelPopulation(EvolutionSettings(), max_active=50, elite_size=0)
    oof = OOFStore()
    for i, s in enumerate((0.90, 0.88, 0.86)):
        _add(pop, oof, f"m_log_{i}", "logistic", s)
    weak = _add(pop, oof, "m_knn", "knn", 0.40)
    pop.update_utilities()
    settings = KagglePipelineSettings(ensemble_max_models=max_models)
    return EnsembleManager(settings), pop, oof, weak


def test_select_candidates_reserves_a_seat_for_a_weak_family():
    # Cap of 2 would keep only the two best logistics; the knn floor of 1 forces
    # the weak family's best into the candidate set anyway.
    manager, pop, oof, weak = _ensemble_setup(max_models=2)
    candidates = manager.select_candidates(pop, oof)
    assert weak.model_id in candidates


def test_select_candidates_without_floor_drops_the_weak_family():
    # Sanity contrast: a floor of 0 lets the cap evict the weak family entirely.
    manager, pop, oof, weak = _ensemble_setup(max_models=2)
    manager.families = {}
    # Override the default floor for the test by zeroing knn's spec.
    manager.family_min_count = lambda fam: 0  # type: ignore[method-assign]
    candidates = manager.select_candidates(pop, oof)
    assert weak.model_id not in candidates
    assert len(candidates) == 2


def test_family_min_count_resolves_against_ensemble_cap():
    settings = KagglePipelineSettings(ensemble_max_models=20)
    manager = EnsembleManager(settings)
    # Unknown family -> default floor of 1.
    assert manager.family_min_count("logistic") == 1


def test_greedy_required_ids_force_a_member_into_the_blend():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    good = np.array([[0.9], [0.8], [0.2], [0.1], [0.3], [0.7], [0.4], [0.6]])
    useless = np.full((8, 1), 0.5)
    scoring_fn = resolve_scoring("roc_auc")

    # Without forcing, greedy would never pick the useless model.
    weights, _ = greedy_weights(
        ["good", "useless"],
        {"good": good, "useless": useless},
        y,
        scoring_fn,
        max_models=5,
        min_models=1,
    )
    assert "useless" not in weights

    # required_ids seeds it, so it appears in the final weights with mass > 0.
    forced, _ = greedy_weights(
        ["good", "useless"],
        {"good": good, "useless": useless},
        y,
        scoring_fn,
        max_models=5,
        min_models=1,
        required_ids=["useless"],
    )
    assert forced.get("useless", 0.0) > 0.0
    assert abs(sum(forced.values()) - 1.0) < 1e-9
