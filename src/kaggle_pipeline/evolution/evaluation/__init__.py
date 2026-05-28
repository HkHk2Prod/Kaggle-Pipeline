"""Evaluation helpers: CV splits, metric direction, and the OOF store.

Thin layer over the v1 scoring and the model utility maths -- it does not
re-implement them, it wires them for the evolutionary controller.
"""

from __future__ import annotations

from kaggle_pipeline.evolution.evaluation.metrics import metric_higher_is_better
from kaggle_pipeline.evolution.evaluation.oof_store import OOFStore
from kaggle_pipeline.evolution.evaluation.validation import make_cv_splits

__all__ = ["OOFStore", "make_cv_splits", "metric_higher_is_better"]
