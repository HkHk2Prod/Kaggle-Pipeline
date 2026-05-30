"""Merging several ecosystem checkpoints into one (parallel-train -> blend)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from kaggle_pipeline.evolution import KagglePipeline, KagglePipelineSettings
from kaggle_pipeline.evolution.ecosystem.merge import merge_ecosystem_states
from kaggle_pipeline.evolution.ecosystem.serialization import EcosystemSerializer
from kaggle_pipeline.evolution.models.parameter_spaces import build_default_families


def _data(n: int = 240):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "id": range(n),
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "cat1": rng.choice(list("abc"), n),
        }
    )
    logit = df["num1"] + 0.5 * df["num2"] + (df["cat1"] == "a") * 1.0
    df["target"] = (logit + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return df


def _run_worker(tmp_path, *, seed: int) -> EcosystemSerializer:
    """Run a tiny training-only pipeline and return its checkpoint serializer."""
    settings = KagglePipelineSettings(
        max_runtime_seconds=20,
        safety_margin_seconds=1,
        checkpoint_time_reserve_seconds=1,
        ensemble_time_reserve_seconds=2,
        finalization_time_reserve_seconds=1,
        verbosity=0,
        models_per_batch=4,
        cv_splits=3,
        max_active_features=20,
        num_workers=2,
        seed=seed,
        state_dir=str(tmp_path / f"state_{seed}"),
        make_submission_on_run=False,
    )
    pipeline = KagglePipeline(settings)
    families = build_default_families()
    pipeline.families = {
        name: families[name] for name in ("logistic", "random_forest") if name in families
    }
    pipeline.fit(_data(), target="target", scoring="roc_auc", id_col="id")
    return EcosystemSerializer(settings.state_dir)


@pytest.fixture
def two_states(tmp_path):
    warnings.simplefilter("ignore")
    a = _run_worker(tmp_path, seed=1).load()
    b = _run_worker(tmp_path, seed=2).load()
    return a, b


def test_merge_unions_models_and_drops_duplicates(two_states):
    a, b = two_states
    settings = KagglePipelineSettings(max_active_features=20)
    merged = merge_ecosystem_states([a, b], settings=settings)

    hashes_a = {g.genome_hash for g in a.population.all_genomes()}
    hashes_b = {g.genome_hash for g in b.population.all_genomes()}
    merged_hashes = {g.genome_hash for g in merged.population.all_genomes()}

    assert merged_hashes == hashes_a | hashes_b
    # No duplicate genomes survive the union.
    ids = [g.genome_hash for g in merged.population.all_genomes()]
    assert len(ids) == len(set(ids))


def test_merge_keeps_features_by_recipe(two_states):
    a, b = two_states
    settings = KagglePipelineSettings(max_active_features=20)
    merged = merge_ecosystem_states([a, b], settings=settings)
    recipes_a = {g.recipe_hash for g in a.registry.all_features()}
    recipes_b = {g.recipe_hash for g in b.registry.all_features()}
    merged_recipes = {g.recipe_hash for g in merged.registry.all_features()}
    assert merged_recipes == recipes_a | recipes_b


def test_merge_carries_oof_for_kept_models(two_states):
    a, b = two_states
    settings = KagglePipelineSettings(max_active_features=20)
    merged = merge_ecosystem_states([a, b], settings=settings)
    # Every active model in the merge that had OOF in an input still has it.
    for mid in merged.population.active:
        had_oof = a.oof_store.has(mid) or b.oof_store.has(mid)
        if had_oof:
            assert merged.oof_store.has(mid)


def test_merge_with_self_drops_all_duplicates(two_states):
    a, _ = two_states
    settings = KagglePipelineSettings(max_active_features=20)
    merged = merge_ecosystem_states([a, a], settings=settings)
    # Merging a state with itself yields exactly the original genome set.
    assert {g.genome_hash for g in merged.population.all_genomes()} == {
        g.genome_hash for g in a.population.all_genomes()
    }


def test_single_state_returned_as_is(two_states):
    a, _ = two_states
    settings = KagglePipelineSettings()
    assert merge_ecosystem_states([a], settings=settings) is a


def test_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        merge_ecosystem_states([], settings=KagglePipelineSettings())


def test_merge_logs_progress(two_states):
    a, b = two_states
    settings = KagglePipelineSettings(max_active_features=20)
    lines: list[str] = []
    merge_ecosystem_states([a, b], settings=settings, log=lines.append)
    text = "\n".join(lines)
    assert "merging 2 input ecosystems" in text
    assert "merged ecosystem:" in text
