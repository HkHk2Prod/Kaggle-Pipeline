"""Model layer: genomes, factory, mutation, scoring, training and population.

A :class:`~kaggle_pipeline.evolution.models.genome.ModelGenome` is immutable once
created; mutation produces a *child* genome. Genomes reference global features by
id and carry model-local encoding/parameter/resource genes. The factory generates
new genomes, the mutator derives children, the trainer evaluates them (reusing the
v1 cross-validation + scoring), and the population/elite archive track them.
"""

from __future__ import annotations

from kaggle_pipeline.evolution.models.factory import ModelFactory
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import FailureReason, ModelStatus
from kaggle_pipeline.evolution.models.mutation import ModelMutator, MutationRecord
from kaggle_pipeline.evolution.models.registry import ModelPopulation
from kaggle_pipeline.evolution.models.scoring import ModelScoreSet, ModelUtility
from kaggle_pipeline.evolution.models.training import ModelResult, ModelTrainer

__all__ = [
    "ModelGenome",
    "ModelStatus",
    "FailureReason",
    "ModelScoreSet",
    "ModelUtility",
    "ModelFactory",
    "ModelMutator",
    "MutationRecord",
    "ModelTrainer",
    "ModelResult",
    "ModelPopulation",
]
