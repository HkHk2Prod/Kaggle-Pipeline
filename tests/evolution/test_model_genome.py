"""Model genome hashing and identity."""

from __future__ import annotations

from kaggle_pipeline.evolution.genes.base import BaseModelGene
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.genes.parameter_gene import FLOAT, ParameterGene, ParameterSpec
from kaggle_pipeline.evolution.models.genome import ModelGenome


def _genome(lr: float = 0.05) -> ModelGenome:
    spec = ParameterSpec("learning_rate", kind=FLOAT, low=0.01, high=0.3, log_scale=True)
    return ModelGenome(
        base_model_gene=BaseModelGene("lightgbm"),
        feature_reference_genes=[FeatureReferenceGene("orig::a"), FeatureReferenceGene("orig::b")],
        parameter_genes=[ParameterGene(spec, lr)],
    )


def test_genome_hash_is_deterministic():
    assert _genome().genome_hash == _genome().genome_hash


def test_genome_hash_changes_when_parameter_changes():
    assert _genome(0.05).genome_hash != _genome(0.10).genome_hash


def test_genome_hash_is_feature_order_independent():
    a = _genome()
    b = ModelGenome(
        base_model_gene=BaseModelGene("lightgbm"),
        feature_reference_genes=[FeatureReferenceGene("orig::b"), FeatureReferenceGene("orig::a")],
        parameter_genes=a.parameter_genes,
    )
    assert a.genome_hash == b.genome_hash


def test_changing_family_changes_hash():
    a = _genome()
    b = ModelGenome(
        base_model_gene=BaseModelGene("xgboost"),
        feature_reference_genes=a.feature_reference_genes,
        parameter_genes=a.parameter_genes,
    )
    assert a.genome_hash != b.genome_hash


def test_model_id_derives_from_hash():
    g = _genome()
    assert g.model_id.startswith("m_")
    assert g.feature_ids() == ["orig::a", "orig::b"]


def test_gene_summary_lists_structure_not_hyperparameters():
    summary = _genome().gene_summary()
    assert any(s.startswith("base=") for s in summary)  # base model gene
    assert any(s.startswith("feat:") for s in summary)  # feature reference genes
    assert not any("learning_rate" in s for s in summary)  # hyperparameters excluded
