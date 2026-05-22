"""Forward / inverse transforms for the target column.

The forward transform is applied to ``y`` before training (categories -> ints).
The inverse transform turns a model's raw output back into the submission format
implied by ``prediction_aim`` -- a probability column, probability matrix, or a
hard category label. Applying these to the whole target column is enough to get
a correctly-formatted submission without any further special-casing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TargetTransforms:
    """Bundled target encode/decode functions plus the model output width."""

    forward: Callable[[pd.DataFrame], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]
    # Number of columns a model's prediction has (1 for regression, n_classes
    # for classification probabilities).
    width: int


def _positive_class_is_conventional(
    order: np.ndarray, order_lists: Sequence[Sequence[str]]
) -> bool:
    """Whether a binary ``order`` ties its positive class (``order[1]``) to a convention.

    Two cases are confident, so no submission flip can be silently introduced:

    * a numeric ``{0, 1}`` (or boolean) target -- sorted ascending, ``P(class 1)``
      is the near-universal Kaggle binary convention; and
    * a target whose values are a subset of a configured ``order_lists`` entry --
      the ordering is intentional and the positive class is the later entry.

    Anything else (e.g. an arbitrary string pair like ``cat``/``dog``) leaves the
    positive class a guess, which is what the caller warns about.
    """
    try:
        if {int(v) for v in order} == {0, 1}:
            return True
    except (TypeError, ValueError):
        pass  # non-numeric labels -- fall through to the order_lists check
    values = {str(v).lower() for v in order}
    return any(values.issubset({str(o).lower() for o in ol}) for ol in order_lists)


def build_target_transforms(
    train_df: pd.DataFrame,
    *,
    target: Sequence[str],
    target_is_num: bool,
    ordered_cats: dict[str, list],
    prediction_aim: str,
    order_lists: Sequence[Sequence[str]] = (),
) -> TargetTransforms:
    """Construct the forward/inverse target transforms for this problem.

    Only single-column targets are supported. For regression both transforms are
    identities. For classification the forward map sends each category to its
    index; the inverse map depends on ``prediction_aim``:

    * ``'probability'`` -- positive-class column (binary) or all-but-first
      columns (multiclass).
    * ``'category'`` -- the arg-max category label.

    For a binary ``probability`` submission the positive class is ``order[1]``;
    when that ordering can't be tied to a ``0/1`` target or a configured
    ``order_lists`` entry, a loud warning is logged because the submission may be
    the *complement* of what the competition expects (see
    :func:`_positive_class_is_conventional`).
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
    # ``order`` fixes which class is index 1 -- i.e. the positive column emitted
    # for a binary ``probability`` submission. It MUST be deterministic: ``unique()``
    # returns values in first-appearance order, so a numeric target (never typed
    # into ``ordered_cats``) whose first row was the positive class got order
    # ``[1, 0]`` and a submission of ``1 - P(positive)`` -- a 0.95 CV scoring ~0.05
    # on the leaderboard. Sorting matches sklearn's ``classes_`` (the predict_proba
    # column order the inverse relies on) and Kaggle's binary convention (P(class 1)).
    if target[0] in ordered_cats:
        order = np.array(ordered_cats[target[0]])
    else:
        order = np.array(sorted(train_df[target[0]].dropna().unique()))
    mapping = {cat: i for i, cat in enumerate(order)}

    # A binary probability submission emits P(order[1]). If that ordering isn't
    # anchored to a convention, we may be submitting 1 - P(positive) -- the silent
    # flip that turns a strong CV into a near-zero leaderboard score. Warn loudly
    # (WARNING is shown even in quiet mode) with the exact fix.
    if (
        prediction_aim == "probability"
        and len(order) == 2
        and not _positive_class_is_conventional(order, order_lists)
    ):
        # .tolist() unwraps numpy scalars so the message reads `'dog'`, not `np.str_('dog')`.
        negative, positive = order.tolist()
        logger.warning(
            "[target] AMBIGUOUS positive class: submitting P(%r) as the positive "
            "probability and treating %r as negative, but this ordering came from "
            "neither a 0/1 target nor an order_lists entry -- it may be the opposite "
            "of the competition's positive class. If your leaderboard score is "
            "~(1 - your CV score), the classes are flipped: set order_lists=[[%r, %r]] "
            "(positive class last) to fix it.",
            positive,
            negative,
            negative,
            positive,
        )

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
