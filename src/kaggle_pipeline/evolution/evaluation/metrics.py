"""Metric direction helpers, reusing the v1 scoring resolver.

The evolutionary layer converts every metric to a larger-is-better internal score
(see :mod:`kaggle_pipeline.evolution.models.scoring`). The v1 scoring functions are
already higher-is-better (``roc_auc``, ``balanced_accuracy`` and the *negated*
RMSE), so the conversion is the identity today; this helper centralises the
direction knowledge so a future lower-is-better metric is handled in one place.
"""

from __future__ import annotations

from kaggle_pipeline.scoring.metrics import ScoringFn, resolve_scoring

# Metrics whose raw value is already larger-is-better. Anything resolvable by the
# v1 resolver is, by construction (it negates RMSE), so this is currently total.
_HIGHER_IS_BETTER: dict[str, bool] = {
    "roc_auc": True,
    "balanced_accuracy": True,
    "neg_root_mean_squared_error": True,
}


def metric_higher_is_better(name: str) -> bool:
    """Whether a scoring name's raw value is already larger-is-better."""
    return _HIGHER_IS_BETTER.get(name, True)


__all__ = ["metric_higher_is_better", "resolve_scoring", "ScoringFn"]
