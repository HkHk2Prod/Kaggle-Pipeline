"""Ensembling: combine the best models into a final prediction.

The :class:`EnsembleManager` selects candidate models (needing OOF predictions),
builds an ensemble (greedy forward selection by default, with mean/weighted
fallbacks), scores it on OOF, and -- given test data -- refits members on the full
train set to produce the submission. If too few candidates or too little time
exist, it falls back to the best single model.
"""

from __future__ import annotations

from kaggle_pipeline.evolution.ensemble.manager import EnsembleManager, EnsembleResult
from kaggle_pipeline.evolution.ensemble.submission import write_submission

__all__ = ["EnsembleManager", "EnsembleResult", "write_submission"]
