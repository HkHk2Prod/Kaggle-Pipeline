"""Controllers orchestrating the evolutionary loop.

The :class:`EvolutionController` runs each batch: generate & score features,
decide between generating a new model and mutating an existing one, train, score,
assign gene and feature credit, promote, and record everything. The smaller
controllers (feature, model, promotion) and the :class:`CreditAssigner` are its
collaborators, each with one reason to change.
"""

from __future__ import annotations

from kaggle_pipeline.evolution.controllers.credit_assignment import CreditAssigner
from kaggle_pipeline.evolution.controllers.evolution_controller import (
    BatchSummary,
    EvolutionController,
)
from kaggle_pipeline.evolution.controllers.feature_controller import FeatureController
from kaggle_pipeline.evolution.controllers.model_controller import ModelController
from kaggle_pipeline.evolution.controllers.promotion_controller import PromotionController

__all__ = [
    "EvolutionController",
    "BatchSummary",
    "FeatureController",
    "ModelController",
    "PromotionController",
    "CreditAssigner",
]
