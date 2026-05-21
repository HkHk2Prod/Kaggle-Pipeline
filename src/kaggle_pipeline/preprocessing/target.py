"""Forward / inverse transforms for the target column.

The forward transform is applied to ``y`` before training (categories -> ints).
The inverse transform turns a model's raw output back into the submission format
implied by ``prediction_aim`` -- a probability column, probability matrix, or a
hard category label. Applying these to the whole target column is enough to get
a correctly-formatted submission without any further special-casing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TargetTransforms:
    """Bundled target encode/decode functions plus the model output width."""

    forward: Callable[[pd.DataFrame], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]
    # Number of columns a model's prediction has (1 for regression, n_classes
    # for classification probabilities).
    width: int


def build_target_transforms(
    train_df: pd.DataFrame,
    *,
    target: Sequence[str],
    target_is_num: bool,
    ordered_cats: dict[str, list],
    prediction_aim: str,
) -> TargetTransforms:
    """Construct the forward/inverse target transforms for this problem.

    Only single-column targets are supported. For regression both transforms are
    identities. For classification the forward map sends each category to its
    index; the inverse map depends on ``prediction_aim``:

    * ``'probability'`` -- positive-class column (binary) or all-but-first
      columns (multiclass).
    * ``'category'`` -- the arg-max category label.
    """
    target = list(target)
    if train_df[target].shape[1] != 1:
        raise ValueError("Multicolumn target is not implemented yet.")

    def forward_identity(y: pd.DataFrame) -> np.ndarray:
        return y.values

    def inverse_identity(y: np.ndarray) -> np.ndarray:
        return y

    if target_is_num:
        width = 1
        return TargetTransforms(forward_identity, inverse_identity, width)

    # --- Classification ---
    if target[0] in ordered_cats:
        order = ordered_cats[target[0]]
    else:
        order = train_df[target[0]].dropna().unique()
    order = np.array(order)
    mapping = {cat: i for i, cat in enumerate(order)}

    def forward(y: pd.DataFrame) -> np.ndarray:
        y = y.squeeze()
        return y.map(mapping).astype(int).values

    if prediction_aim == "probability":
        if len(order) == 2:

            def inverse(p: np.ndarray) -> np.ndarray:
                return p[:, 1]
        else:

            def inverse(p: np.ndarray) -> np.ndarray:
                return p[:, 1:]
    elif prediction_aim == "category":

        def inverse(p: np.ndarray) -> np.ndarray:
            return order[np.argmax(p, axis=1)]
    else:
        raise ValueError(f"Unknown prediction_aim: {prediction_aim!r}")

    width = len(train_df[target[0]].dropna().unique())
    return TargetTransforms(forward, inverse, width)
