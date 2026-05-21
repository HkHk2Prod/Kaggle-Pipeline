"""Resolve a scoring name into a callable used for cross-validation.

The returned function takes ``(y_true, y_pred)`` where ``y_pred`` is whatever a
model emits -- a probability matrix for classifiers. Metrics that need hard
labels (e.g. balanced accuracy) take the arg-max internally.

The same ``scoring`` string is also a valid scikit-learn scorer name, so it is
reused directly by the ensembling search in :mod:`kaggle_pipeline.search.judge`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ScoringFn = Callable[[np.ndarray, np.ndarray], float]


def resolve_scoring(name: str) -> ScoringFn:
    """Return the cross-validation scoring function for ``name``.

    Implemented: ``'roc_auc'``, ``'balanced_accuracy'``,
    ``'neg_root_mean_squared_error'``. Each is also a valid scikit-learn scorer
    name, so the same string is reused by the ensembling search.
    """
    if name == "roc_auc":
        from sklearn.metrics import roc_auc_score

        def roc_auc_from_proba(y_val: np.ndarray, y_prob: np.ndarray) -> float:
            # Models emit a probability matrix; roc_auc_score wants the
            # positive-class scores for binary, and the full matrix (OvR) for
            # multiclass.
            y_prob = np.asarray(y_prob)
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                return roc_auc_score(y_val, y_prob[:, 1])
            if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                return roc_auc_score(y_val, y_prob, multi_class="ovr")
            return roc_auc_score(y_val, y_prob)

        return roc_auc_from_proba

    if name == "balanced_accuracy":
        from sklearn.metrics import balanced_accuracy_score

        def balanced_accuracy_from_proba(y_val: np.ndarray, y_prob: np.ndarray) -> float:
            y_pred = np.argmax(y_prob, axis=1)
            return balanced_accuracy_score(y_val, y_pred)

        return balanced_accuracy_from_proba

    if name == "neg_root_mean_squared_error":
        from sklearn.metrics import mean_squared_error

        def neg_rmse(y_val: np.ndarray, y_pred: np.ndarray) -> float:
            # Negated so that, like the classification metrics, higher is better.
            return -float(mean_squared_error(y_val, y_pred) ** 0.5)

        return neg_rmse

    raise ValueError(
        f"Unknown scoring {name!r}. Implemented: 'roc_auc', 'balanced_accuracy', "
        "'neg_root_mean_squared_error'."
    )
