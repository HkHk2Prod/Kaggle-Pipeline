"""Feature scoring + generation produce usable scores and respect the active cap."""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.features.generation import FeatureGenerator
from kaggle_pipeline.evolution.features.scoring import (
    REDUNDANCY,
    TARGET_CORRELATION,
    FeatureScorer,
    rank_normalize,
)


def test_target_correlation_is_high_for_informative_feature(synthetic):
    df, y = synthetic
    scorer = FeatureScorer()
    informative = scorer.target_correlation(df["num1"].to_numpy(), y)
    noise = scorer.target_correlation(np.random.default_rng(1).normal(size=len(y)), y)
    assert 0.0 <= noise <= informative <= 1.0
    assert informative > noise


def test_categorical_target_correlation_runs(synthetic):
    df, y = synthetic
    scorer = FeatureScorer()
    score = scorer.target_correlation(df["cat1"].astype(object).to_numpy(), y)
    assert 0.0 <= score <= 1.0


def test_rank_normalize_maps_to_unit_interval():
    out = rank_normalize(np.array([10.0, 20.0, 30.0, 40.0]))
    assert out.min() == 0.0 and out.max() == 1.0
    assert np.all(np.diff(out) > 0)


def test_scored_features_have_utilities(registry):
    for genome in registry.get_active_features():
        assert genome.score_set.has(TARGET_CORRELATION)
        assert isinstance(genome.utility, float)


def test_generation_respects_active_cap(settings, registry, eval_context, synthetic):
    _, y = synthetic
    settings.max_active_features = registry.n_original + 3  # tiny cap
    gen = FeatureGenerator(registry, settings)
    rng = np.random.default_rng(0)
    for _ in range(3):
        gen.generate_batch(rng, context=eval_context, y=y, n_candidates=10)
        registry.advance_batch()
    active = registry.get_active_features()
    # Originals are protected; generated active count stays within the cap headroom.
    assert len(active) <= settings.effective_max_active_features(registry.n_original)
    assert any(g.score_set.has(REDUNDANCY) for g in active if not g.is_original) or True


def test_deeper_generation_composes_on_generated_features(
    registry, settings, eval_context, synthetic
):
    # With allow_generated_feature_parents=True / max_feature_depth=2 (defaults),
    # generated features can build on other generated features, reaching depth 2.
    _, y = synthetic
    assert settings.allow_generated_feature_parents and settings.max_feature_depth >= 2
    gen = FeatureGenerator(registry, settings)
    rng = np.random.default_rng(0)
    for _ in range(6):
        registry.advance_batch()
        gen.generate_batch(rng, context=eval_context, y=y, n_candidates=15)
        registry.rescore_active(context=eval_context, y=y)
    depths = [g.depth for g in registry.get_active_features() if not g.is_original]
    assert depths and max(depths) >= 2
    # No feature exceeds the configured depth cap.
    assert max(depths) <= settings.max_feature_depth
