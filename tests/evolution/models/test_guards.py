"""Guards: high-cardinality one-hot fallback and pruned-model OOF eviction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kaggle_pipeline.evolution.evaluation.oof_store import OOFStore
from kaggle_pipeline.evolution.genes.base import BaseModelGene
from kaggle_pipeline.evolution.genes.encoding_gene import (
    NON_NATIVE_ENCODINGS,
    ONEHOT,
    EncodingGene,
    allowed_encodings_for,
)
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.genes.resource_gene import ResourceGene
from kaggle_pipeline.evolution.models.factory import ModelFactory, make_encoding_gene
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.parameter_spaces import build_default_families
from kaggle_pipeline.evolution.models.registry import ModelPopulation
from kaggle_pipeline.evolution.models.scoring import ModelScoreSet
from kaggle_pipeline.evolution.models.training import ModelTrainer
from kaggle_pipeline.preprocessing.encoders import FrequencyEncoder


def _encoder_for_column(pipeline, column):
    prep = pipeline.named_steps["prep"]
    for _name, transformer, cols in prep.transformers:
        if cols == [column]:
            return transformer
    return None


def test_onehot_falls_back_for_high_cardinality(registry, synthetic):
    df, _ = synthetic  # cat1 has 4 distinct levels
    genome = ModelGenome(
        base_model_gene=BaseModelGene("logistic"),
        feature_reference_genes=[
            FeatureReferenceGene(
                "orig::cat1",
                children=[EncodingGene(ONEHOT, alternatives=NON_NATIVE_ENCODINGS)],
            )
        ],
        resource_genes=[ResourceGene("n_estimators", 50)],
    )
    X = pd.DataFrame({"orig::cat1": df["cat1"].astype(object).to_numpy()})

    capped = ModelTrainer(registry, families=build_default_families(), onehot_max_cardinality=2)
    # 4 distinct > cap of 2 -> falls back to frequency (one column), no explosion.
    # The fallback reuses the shared v1 FrequencyEncoder.
    assert isinstance(
        _encoder_for_column(capped._build_pipeline(genome, X, seed=0), "orig::cat1"),
        FrequencyEncoder,
    )

    allowed = ModelTrainer(registry, families=build_default_families(), onehot_max_cardinality=10)
    enc = _encoder_for_column(allowed._build_pipeline(genome, X, seed=0), "orig::cat1")
    assert type(enc).__name__ == "OneHotEncoder"


def test_prune_evicts_oof_but_keeps_structure(registry, settings):
    population = ModelPopulation(settings, max_active=1, elite_size=1)
    store = OOFStore()
    population.oof_store = store
    factory = ModelFactory(registry, settings, families=build_default_families())
    rng = np.random.default_rng(0)

    genomes = []
    for i in range(3):
        g = factory.generate(rng, family="logistic")
        g.status = "completed"
        g.score_set = ModelScoreSet(score=0.9 - 0.1 * i, score_std=0.0, compute_time=1.0)
        population.register(g)
        store.store(g.model_id, np.zeros((10, 1)))
        population.record_result(g)
        genomes.append(g)

    pruned = [g for g in genomes if g.status == "pruned"]
    assert pruned  # max_active=1 forced eviction
    for g in pruned:
        assert g.score_set is not None  # structure + scores kept
        assert not store.has(g.model_id)  # big OOF data freed


def test_allowed_encodings_excludes_onehot_for_high_cardinality():
    assert ONEHOT in allowed_encodings_for(5, 20)  # low cardinality -> one-hot ok
    assert ONEHOT not in allowed_encodings_for(50, 20)  # too many levels -> excluded
    assert ONEHOT not in allowed_encodings_for(None, 20)  # unknown -> conservative


def test_registry_records_categorical_cardinality(registry):
    assert registry.get_feature("orig::cat1").cardinality == 4  # a, b, c, d
    assert registry.get_feature("orig::num1").cardinality is None  # numeric -> unset


def test_native_family_gets_no_encoding_gene(registry):
    rng = np.random.default_rng(0)
    cat = registry.get_feature("orig::cat1")
    # Native-categorical family -> encoding decided by the model, no gene generated.
    assert (
        make_encoding_gene(cat, handles_categoricals=True, onehot_max_cardinality=20, rng=rng)
        is None
    )
    # Numeric feature -> never an encoding gene.
    num = registry.get_feature("orig::num1")
    assert (
        make_encoding_gene(num, handles_categoricals=False, onehot_max_cardinality=20, rng=rng)
        is None
    )


def test_nonnative_encoding_gene_constrained_by_cardinality(registry):
    rng = np.random.default_rng(0)
    cat = registry.get_feature("orig::cat1")  # cardinality 4
    enc = make_encoding_gene(cat, handles_categoricals=False, onehot_max_cardinality=20, rng=rng)
    assert enc is not None and ONEHOT in enc.alternatives  # 4 <= 20
    capped = make_encoding_gene(cat, handles_categoricals=False, onehot_max_cardinality=2, rng=rng)
    assert ONEHOT not in capped.alternatives and capped.value != ONEHOT  # 4 > 2
