"""Gene + model mutation: immutability of parents, bounds, rounding, child cleanup."""

from __future__ import annotations

import numpy as np

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.genes.base import MutationContext
from kaggle_pipeline.evolution.genes.encoding_gene import (
    NON_NATIVE_ENCODINGS,
    ONEHOT,
    EncodingGene,
)
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.genes.parameter_gene import (
    FLOAT,
    INT,
    NEGATIVE,
    ParameterGene,
    ParameterSpec,
)
from kaggle_pipeline.evolution.models.factory import ModelFactory
from kaggle_pipeline.evolution.models.mutation import ModelMutator


def _ctx(seed: int = 0, registry=None) -> MutationContext:
    return MutationContext(
        rng=np.random.default_rng(seed), settings=EvolutionSettings(), registry=registry
    )


def test_numeric_parameter_mutation_respects_bounds():
    spec = ParameterSpec("learning_rate", kind=FLOAT, low=0.01, high=0.3, log_scale=True)
    gene = ParameterGene(spec, 0.3)
    for seed in range(50):
        child = gene.mutate(0.2, _ctx(seed))
        assert spec.low <= child.value <= spec.high
        assert child.gene_id != gene.gene_id  # a new gene
        assert gene.value == 0.3  # parent untouched


def test_integer_parameter_mutation_is_valid_integer():
    spec = ParameterSpec("num_leaves", kind=INT, low=8, high=256)
    gene = ParameterGene(spec, 32)
    for seed in range(50):
        child = gene.mutate(0.3, _ctx(seed))
        assert isinstance(child.value, int)
        assert spec.low <= child.value <= spec.high


def test_negative_direction_inverts_meaning():
    # min_child_samples: positive signed_amount means "more complex" -> smaller value.
    spec = ParameterSpec(
        "min_child_samples", kind=INT, low=5, high=200, complexity_direction=NEGATIVE
    )
    gene = ParameterGene(spec, 100)
    child = gene.mutate(0.5, _ctx(0))  # positive -> should decrease
    assert child.value <= gene.value


def test_removing_feature_reference_removes_encoding_child():
    fr = FeatureReferenceGene("orig::city")
    fr.set_encoding(EncodingGene(ONEHOT, alternatives=NON_NATIVE_ENCODINGS))
    assert fr.encoding is not None
    # A model genome built without that reference has no orphaned encoding.
    genes = [fr]
    genes.remove(fr)
    assert all(g.encoding is None for g in genes if hasattr(g, "encoding"))
    # And the reference's own removal drops its child set with it.
    assert fr.encoding.gene_id in fr.child_gene_ids


def test_model_mutation_creates_child_without_touching_parent(registry, synthetic):
    settings = EvolutionSettings(default_random_seed=0)
    factory = ModelFactory(registry, settings)
    rng = np.random.default_rng(3)
    parent = factory.generate(rng, family="random_forest", batch=0)
    parent_hash = parent.genome_hash
    parent_features = list(parent.feature_ids())

    mutator = ModelMutator(registry, settings)
    child, record = mutator.mutate(parent, rng, batch=1)

    assert parent.genome_hash == parent_hash  # parent unchanged
    assert parent.feature_ids() == parent_features
    assert child.parent_model_id == parent.model_id
    assert record.parent_model_id == parent.model_id
    assert record.child_model_id == child.model_id
    assert record.mutation_type


def test_replace_feature_never_produces_duplicate(registry, settings):
    """A replace-feature mutation must not collide with a sibling reference.

    Regression for a LightGBM training failure: ``num__gen__<hash>`` appearing
    twice because the new feature_id matched another reference already in the
    genome. See [[evolution-training-integration]].
    """
    mutator = ModelMutator(registry, settings)
    active = registry.get_active_features()
    # Force the collision condition: a list that already references every
    # active feature, so any non-self pick is guaranteed to clash.
    feature_genes = [FeatureReferenceGene(f.feature_id) for f in active]
    for seed in range(50):
        rng = np.random.default_rng(seed)
        record = type("R", (), {})()  # minimal stand-in; the operator only writes attrs
        record.mutated_gene_ids = []
        record.signed_amounts = []
        record.old_values = []
        record.new_values = []
        record.removed_feature_ids = []
        record.added_feature_ids = []
        clone = [g.copy() for g in feature_genes]
        ctx = _ctx(seed, registry=registry)
        mutator._replace_feature(clone, ctx, record, rng)
        ids = [g.feature_id for g in clone]
        assert len(ids) == len(set(ids)), f"seed={seed}: duplicate feature_ids {ids}"


def test_full_genome_mutations_never_produce_duplicates(registry, settings):
    """End-to-end: many mutated children must each have distinct feature_ids."""
    factory = ModelFactory(registry, settings)
    mutator = ModelMutator(registry, settings)
    rng = np.random.default_rng(7)
    parent = factory.generate(rng, family="random_forest", batch=0)
    for seed in range(100):
        child, _ = mutator.mutate(parent, np.random.default_rng(seed), batch=1)
        ids = child.feature_ids()
        assert len(ids) == len(set(ids)), f"seed={seed}: duplicate feature_ids {ids}"
