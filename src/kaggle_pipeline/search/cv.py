"""Cross-validate a model and store its out-of-fold predictions.

Running CV here (rather than via ``cross_val_score``) lets us keep the OOF
prediction matrix on the model, which is what the stacking meta-model is later
trained on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kaggle_pipeline.context import PipelineContext
    from kaggle_pipeline.models.base import Model


class CrossValScore:
    """Fit/score ``model`` across ``splits`` and attach OOF preds to the model."""

    def __init__(self, model: Model, X, y, *, splits, ctx: PipelineContext):
        scores: list[float] = []
        y_oof = np.zeros((len(y), ctx.target_width))
        for train_idx, val_idx in splits:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            scores.append(ctx.scoring_fn(y_val, y_pred))
            y_oof[val_idx] = y_pred
        self._scores: np.ndarray = np.array(scores)
        # Drop the redundant last probability column for multiclass OOF features.
        if ctx.target_width > 1:
            y_oof = y_oof[:, :-1]
        model.set_oof(y_oof)

    @property
    def score(self) -> tuple[float, float]:
        """Mean and standard deviation of the per-fold scores."""
        return self._scores.mean(), self._scores.std()
