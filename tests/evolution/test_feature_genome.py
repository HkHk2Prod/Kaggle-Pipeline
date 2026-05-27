"""Feature recipe hashing, commutativity canonicalisation, and genome identity."""

from __future__ import annotations

from kaggle_pipeline.evolution.features.genome import FeatureGenome
from kaggle_pipeline.evolution.features.recipe import NUMERIC, FeatureRecipe
from kaggle_pipeline.evolution.features.transformations import Add, SafeDivide


def test_same_recipe_same_hash():
    a = FeatureRecipe("log1p", ("orig::price",), {}, NUMERIC)
    b = FeatureRecipe("log1p", ("orig::price",), {}, NUMERIC)
    assert a.recipe_hash == b.recipe_hash


def test_metadata_excluded_from_hash():
    a = FeatureRecipe("log1p", ("orig::price",), {}, NUMERIC)
    b = a.with_metadata(note="anything")
    assert a.recipe_hash == b.recipe_hash


def test_noncommutative_recipes_differ_by_parent_order():
    # subtract / divide are non-commutative: order is preserved and matters.
    ab = SafeDivide().generate_recipe(["orig::a", "orig::b"], {})
    ba = SafeDivide().generate_recipe(["orig::b", "orig::a"], {})
    assert ab.recipe_hash != ba.recipe_hash


def test_commutative_recipes_canonicalize_parent_order():
    # add is commutative: a+b and b+a canonicalise to the same recipe.
    ab = Add().generate_recipe(["orig::a", "orig::b"], {})
    ba = Add().generate_recipe(["orig::b", "orig::a"], {})
    assert ab.recipe_hash == ba.recipe_hash
    assert ab.parent_feature_ids == ba.parent_feature_ids


def test_parameters_affect_hash():
    a = FeatureRecipe("bin", ("orig::age",), {"n_bins": 10}, NUMERIC)
    b = FeatureRecipe("bin", ("orig::age",), {"n_bins": 5}, NUMERIC)
    assert a.recipe_hash != b.recipe_hash


def test_feature_id_derives_from_recipe():
    recipe = FeatureRecipe("log1p", ("orig::price",), {}, NUMERIC)
    g1 = FeatureGenome(recipe=recipe, human_name="log__price__x")
    g2 = FeatureGenome(recipe=recipe, human_name="different_name")
    # Identical recipes share a feature id regardless of human name.
    assert g1.feature_id == g2.feature_id
    assert g1.feature_id.startswith("gen::")


def test_original_feature_id_is_readable():
    g = FeatureGenome.original("price", NUMERIC)
    assert g.feature_id == "orig::price"
    assert g.is_original and g.protected
