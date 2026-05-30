"""Registry behaviour: protection, deactivation, dedup, selection probabilities."""

from __future__ import annotations

import numpy as np
import pytest

from kaggle_pipeline.evolution.features.genome import FeatureGenome
from kaggle_pipeline.evolution.features.recipe import NUMERIC, FeatureRecipe


def test_original_features_are_protected_from_deletion(registry):
    original = registry.get_feature("orig::num1")
    assert original.protected
    assert original not in registry.get_removable_features()
    with pytest.raises(ValueError):
        registry.deactivate_feature("orig::num1")


def test_generated_feature_can_be_deactivated(registry):
    recipe = FeatureRecipe("log1p", ("orig::num3",), {}, NUMERIC)
    genome = FeatureGenome(recipe=recipe, human_name="log__num3", created_at_batch=0)
    registry.add_generated_feature(genome)
    registry.activate_feature(genome.feature_id)
    assert genome.active
    registry.deactivate_feature(genome.feature_id)
    assert not genome.active
    # Still reproducible: the recipe survives deactivation.
    assert registry.get_feature(genome.feature_id).recipe_hash == recipe.recipe_hash


def test_registry_reuses_duplicate_recipe(registry):
    recipe = FeatureRecipe("square", ("orig::num1",), {}, NUMERIC)
    first = FeatureGenome(recipe=recipe, human_name="sq__num1__a")
    assert registry.add_generated_feature(first) is first
    duplicate = FeatureGenome(recipe=recipe, human_name="sq__num1__b")
    # Same recipe hash -> rejected as duplicate.
    assert registry.add_generated_feature(duplicate) is None
    assert registry.has_recipe_hash(recipe.recipe_hash)


def test_selection_probabilities_sum_to_one(registry):
    probs = registry.compute_selection_probabilities()
    assert probs
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(p >= 0 for p in probs.values())


def test_effective_limit_respects_original_count(settings):
    from kaggle_pipeline.evolution.features.registry import FeatureRegistry

    small = FeatureRegistry(settings)
    # More originals than the cap: effective limit must rise to the original count.
    settings.max_active_features = 2
    for i in range(5):
        small.add_original_feature(f"c{i}", NUMERIC)
    assert small.effective_max_active_features == 5


def test_sample_features_respects_output_type(registry):
    rng = np.random.default_rng(0)
    ids = registry.sample_features(2, rng, output_type="categorical")
    for fid in ids:
        assert registry.get_feature(fid).output_type == "categorical"
