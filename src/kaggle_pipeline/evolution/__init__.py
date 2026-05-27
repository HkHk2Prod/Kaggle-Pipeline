"""Evolutionary AutoML rework for the Kaggle pipeline.

This subpackage treats feature engineering and model fitting as an evolutionary
search problem. It lives *alongside* the v1 ``run``/``analyze`` flow and reuses
that pipeline's model registry, cross-validation, scoring and
:class:`~kaggle_pipeline.context.PipelineContext` rather than replacing them.

The design contract is documented in the project README under "Evolutionary
architecture". The single most important rules:

* **Features are global** -- a generated feature is a global
  :class:`~kaggle_pipeline.evolution.features.genome.FeatureGenome` recorded once
  in the :class:`~kaggle_pipeline.evolution.features.registry.FeatureRegistry`.
* **Feature usage is model-specific** -- a model genome references features by
  ``feature_id`` via
  :class:`~kaggle_pipeline.evolution.genes.feature_reference_gene.FeatureReferenceGene`,
  and encodings are model-local child genes.
* **Model mutation produces a child model** and never mutates the parent.
* **Feature mutation produces a child feature** and never mutates the parent.
* **Evaluation combines intrinsic feature scores with downstream model impact.**

Submodules are imported lazily by callers; importing this package does not pull
in numpy/pandas/sklearn at import time.
"""

from __future__ import annotations

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.controllers import EvolutionController
from kaggle_pipeline.evolution.features.genome import FeatureGenome
from kaggle_pipeline.evolution.features.registry import FeatureRegistry
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.registry import ModelPopulation

__all__ = [
    "EvolutionSettings",
    "EvolutionController",
    "FeatureRegistry",
    "FeatureGenome",
    "ModelGenome",
    "ModelPopulation",
]
