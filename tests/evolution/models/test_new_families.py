"""End-to-end fit/predict for the diversity-boosting families.

These are all sklearn-bundled, so we can fit them on the synthetic fixture
without any optional-dependency gating. Each test exercises the full
``ModelTrainer._build_pipeline`` plus ``fit`` path so a wiring regression in
``parameter_spaces`` (bad param name, missing predict_proba, wrong
``handles_categoricals`` flag) shows up immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kaggle_pipeline.evolution.genes.base import BaseModelGene
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.genes.parameter_gene import ParameterGene
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.parameter_spaces import build_default_families
from kaggle_pipeline.evolution.models.training import ModelTrainer

NEW_NUMERIC_FAMILIES = ("mlp", "gaussian_nb", "lda", "qda", "knn", "sgd")


def _numeric_genome(family: str, families) -> ModelGenome:
    fam = families[family]
    # Sample a deterministic mid-range value from each parameter spec so the
    # builder gets concrete keyword args (no defaults masking a typo).
    rng = np.random.default_rng(0)
    parameter_genes = [
        ParameterGene(spec=spec, value=spec.sample(rng)) for spec in fam.parameter_specs
    ]
    return ModelGenome(
        base_model_gene=BaseModelGene(family),
        feature_reference_genes=[
            FeatureReferenceGene("orig::num1"),
            FeatureReferenceGene("orig::num2"),
            FeatureReferenceGene("orig::num3"),
        ],
        parameter_genes=parameter_genes,
    )


@pytest.mark.parametrize("family", NEW_NUMERIC_FAMILIES)
def test_family_builds_and_fits(family, registry, synthetic):
    df, y = synthetic
    families = build_default_families()
    if family not in families:
        pytest.skip(f"{family} not available")

    genome = _numeric_genome(family, families)
    trainer = ModelTrainer(registry, families=families)
    X = pd.DataFrame(
        {
            "orig::num1": df["num1"].to_numpy(),
            "orig::num2": df["num2"].to_numpy(),
            "orig::num3": df["num3"].to_numpy(),
        }
    )

    pipeline = trainer._build_pipeline(genome, X, seed=0)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    # Probabilities are normalised (row sums ~1) -- catches a stray
    # decision_function leak from a non-proba estimator.
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_mlp_and_knn_carry_the_train_size_cap():
    families = build_default_families()
    assert families["mlp"].max_train_rows is not None
    assert families["knn"].max_train_rows is not None
    # Sanity: caps are looser than typical-search sizes so they never bite
    # during the 10%-subsample search but DO bite at submission refit.
    assert families["mlp"].max_train_rows >= 50_000
    assert families["knn"].max_train_rows >= families["mlp"].max_train_rows
