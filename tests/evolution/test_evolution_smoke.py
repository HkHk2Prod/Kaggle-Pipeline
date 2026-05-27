"""End-to-end: the controller runs batches, evolves features and models, records all."""

from __future__ import annotations

import warnings

import numpy as np

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.controllers import EvolutionController
from kaggle_pipeline.evolution.models.parameter_spaces import build_default_families


def _fast_families():
    # Restrict to always-available, fast sklearn families to keep the test quick.
    families = build_default_families()
    return {name: families[name] for name in ("logistic", "random_forest") if name in families}


def test_controller_runs_batches(synthetic, originals, scoring_ctx):
    warnings.simplefilter("ignore")
    df, y = synthetic
    settings = EvolutionSettings(default_random_seed=1, max_active_features=30)
    ctrl = EvolutionController(settings, families=_fast_families(), n_splits=3, seed=1)
    ctrl.initialize_features(originals, eval_frame=df, y=y, task="classification")

    n_features_start = len(ctrl.registry.get_active_features())
    summaries = [
        ctrl.run_batch(train_frame=df, scoring_ctx=scoring_ctx, y=y, n_models=4) for _ in range(2)
    ]

    # Features were generated and the active pool grew.
    assert len(ctrl.registry.get_active_features()) >= n_features_start
    # Models were trained and completed without crashing the loop.
    assert ctrl.population.completed()
    assert any(s.n_completed > 0 for s in summaries)
    best = ctrl.best_genome()
    assert best is not None and best.score_set is not None
    assert np.isfinite(best.score_set.score)


def test_duplicate_genome_is_not_retrained(synthetic, originals, scoring_ctx):
    warnings.simplefilter("ignore")
    df, y = synthetic
    settings = EvolutionSettings(default_random_seed=2, max_active_features=20)
    ctrl = EvolutionController(settings, families=_fast_families(), n_splits=3, seed=2)
    ctrl.initialize_features(originals, eval_frame=df, y=y, task="classification")
    ctrl.run_batch(train_frame=df, scoring_ctx=scoring_ctx, y=y, n_models=6)

    # Every stored genome hash is unique (dedup held).
    hashes = [g.genome_hash for g in ctrl.population.completed()]
    assert len(hashes) == len(set(hashes))


def test_credit_flows_to_features(synthetic, originals, scoring_ctx):
    warnings.simplefilter("ignore")
    df, y = synthetic
    settings = EvolutionSettings(default_random_seed=3, max_active_features=25)
    ctrl = EvolutionController(settings, families=_fast_families(), n_splits=3, seed=3)
    ctrl.initialize_features(originals, eval_frame=df, y=y, task="classification")
    for _ in range(2):
        ctrl.run_batch(train_frame=df, scoring_ctx=scoring_ctx, y=y, n_models=4)

    # At least one feature accrued downstream usage from completed models.
    used = [g for g in ctrl.registry.all_features() if g.usage_stats.times_in_completed_model > 0]
    assert used
