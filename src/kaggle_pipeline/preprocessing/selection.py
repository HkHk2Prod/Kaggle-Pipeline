"""Automatic correlation-based pruning of predictor columns.

Before training, drop predictors that carry no usable signal:

* **irrelevant** -- association with the target is indistinguishable from noise
  (below a size-inferred significance threshold ``tau(n)``);
* **redundant** -- essentially a duplicate of another predictor (we are
  ``1 - alpha`` confident their true association clears a high floor); the more
  target-relevant one of the pair is kept.

Both thresholds adapt to the dataset size: with more rows, smaller correlations
become real, so ``tau`` shrinks and the redundancy confidence interval tightens.

A predictor that looks irrelevant *yet* is strongly correlated with a
target-relevant predictor is a correlation-transitivity anomaly (suppression /
non-linearity / leakage). Such a predictor is **kept** and a loud warning is
logged rather than silently dropped.

The association math is reused from
:mod:`kaggle_pipeline.preprocessing.association`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from sklearn.base import BaseEstimator, TransformerMixin

from kaggle_pipeline.preprocessing.association import association_matrix
from kaggle_pipeline.preprocessing.columns import get_predictor_names, split_num_cat

logger = logging.getLogger(__name__)


def irrelevance_threshold(n: int, alpha: float) -> float:
    """Smallest ``|correlation|`` distinguishable from zero at level ``alpha``.

    The two-sided critical Pearson r for ``n`` samples; it shrinks as ``n`` grows,
    so larger datasets keep weaker-but-real associations. Used as a generic floor
    across all association measures -- exact for Pearson, a reasonable proxy for
    Cramer's V / correlation ratio, which also live on ``[0, 1]``.
    """
    df = n - 2
    if df <= 0:
        return 1.0
    t_crit = float(t.ppf(1 - alpha / 2, df))
    return float(np.sqrt(t_crit**2 / (t_crit**2 + df)))


def redundancy_lower_bound(assoc: float, n: int, alpha: float) -> float:
    """One-sided ``1 - alpha`` lower confidence bound on a ``[0, 1]`` association.

    Fisher-z transform with standard error ``1 / sqrt(n - 3)``: exact for a
    numeric Pearson correlation, an accepted approximation for the bias-corrected
    Cramer's V / correlation ratio. Returns ``0.0`` when ``n`` is too small to
    form an interval.
    """
    if n <= 3:
        return 0.0
    a = min(max(assoc, 0.0), 1 - 1e-9)
    z = np.arctanh(a)
    se = 1.0 / np.sqrt(n - 3)
    lower = np.tanh(z - float(norm.ppf(1 - alpha)) * se)
    return float(max(0.0, lower))


@dataclass
class PruneResult:
    """Outcome of :func:`plan_pruning`."""

    threshold: float = 0.0
    dropped: list[str] = field(default_factory=list)
    # (irrelevant predictor, its relevant correlated partner) kept + warned about.
    anomalies: list[tuple[str, str]] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)


def plan_pruning(
    matrix: pd.DataFrame,
    target: str,
    n: int,
    *,
    alpha: float,
    redundancy_floor: float,
) -> PruneResult:
    """Decide which predictors to drop from a precomputed association ``matrix``.

    ``matrix`` is the symmetric association matrix over the predictors plus
    ``target`` (see :func:`association_matrix`). Returns the columns to drop, the
    anomalies to keep + warn about, and a reason per dropped column.
    """
    predictors = [c for c in matrix.columns if c != target]
    result = PruneResult(threshold=irrelevance_threshold(n, alpha))
    if len(predictors) < 2 or n <= 3:
        return result
    tau = result.threshold
    target_assoc = {p: float(matrix.loc[p, target]) for p in predictors}

    def strongly_correlated(a: str, b: str) -> bool:
        return redundancy_lower_bound(float(matrix.loc[a, b]), n, alpha) >= redundancy_floor

    # 1) Anomalies take precedence: irrelevant to the target yet strongly tied to
    #    a relevant predictor. Keep them, warn loudly, and protect from dropping.
    protected: set[str] = set()
    for p in predictors:
        if target_assoc[p] > tau:
            continue
        for q in predictors:
            if q != p and target_assoc[q] > tau and strongly_correlated(p, q):
                protected.add(p)
                result.anomalies.append((p, q))
                logger.warning(
                    "[prune] SUSPICIOUS: %r looks unrelated to the target "
                    "(assoc=%.3f <= tau=%.3f) yet is strongly correlated with %r "
                    "(assoc=%.3f) which IS related to the target (assoc=%.3f). "
                    "Keeping %r -- check for suppression / non-linearity / leakage.",
                    p,
                    target_assoc[p],
                    tau,
                    q,
                    float(matrix.loc[p, q]),
                    target_assoc[q],
                    p,
                )
                break

    kept = [p for p in predictors if p not in protected]
    dropped: set[str] = set()

    # 2) Redundancy: over non-protected pairs in descending association, drop the
    #    lower-target-association member of each strongly-correlated pair.
    pairs = [(float(matrix.loc[a, b]), a, b) for i, a in enumerate(kept) for b in kept[i + 1 :]]
    for _, a, b in sorted(pairs, key=lambda item: item[0], reverse=True):
        if a in dropped or b in dropped or not strongly_correlated(a, b):
            continue
        loser, winner = (a, b) if target_assoc[a] <= target_assoc[b] else (b, a)
        dropped.add(loser)
        result.reasons[loser] = f"redundant with {winner!r} (kept the more target-correlated)"

    # 3) Irrelevance: drop remaining non-protected predictors at/below tau.
    for p in kept:
        if p not in dropped and target_assoc[p] <= tau:
            dropped.add(p)
            result.reasons[p] = (
                f"uncorrelated with target (assoc={target_assoc[p]:.3f} <= {tau:.3f})"
            )

    # 4) Safeguard: never drop every predictor; keep the most target-correlated.
    if dropped and len(dropped) == len(predictors):
        best = max(predictors, key=lambda p: target_assoc[p])
        dropped.discard(best)
        result.reasons.pop(best, None)
        logger.warning(
            "[prune] every predictor was flagged for removal; keeping the most "
            "target-correlated one (%r, assoc=%.3f). Inspect the data / thresholds.",
            best,
            target_assoc[best],
        )

    result.dropped = [p for p in predictors if p in dropped]  # stable column order
    return result


class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Drop irrelevant / redundant predictors; the set is learned on ``fit``.

    Fitted as the last step of the pre-training pipeline, where the target column
    is still present, so it can measure each predictor's association with the
    target. ``transform`` drops the learned columns from train and test alike;
    the target and id columns are never dropped.
    """

    def __init__(
        self,
        target: Sequence[str] | None = None,
        id_col: Sequence[str] | None = None,
        alpha: float = 0.05,
        redundancy_floor: float = 0.90,
    ):
        self.target = target
        self.id_col = id_col
        self.alpha = alpha
        self.redundancy_floor = redundancy_floor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.dropped_ = []
        target = list(self.target or [])
        if not target or target[0] not in X.columns:
            return self
        predictors = list(get_predictor_names(X, target, self.id_col or []))
        if len(predictors) < 2:
            return self

        tcol = target[0]
        # Classification is by dtype, consistent with the rest of the modelling
        # pipeline (CategoricalTyper / OrdinalEncoder settle dtypes upstream).
        # The cat_cutoff "low-cardinality numeric counts as categorical" rule is
        # deliberately graph-only -- see is_num_check(for_graph=...).
        num_pred, cat_pred = split_num_cat(predictors, X)
        t_num, t_cat = split_num_cat([tcol], X)
        matrix, _, _ = association_matrix(
            X[predictors + [tcol]], num_pred + t_num, cat_pred + t_cat
        )
        result = plan_pruning(
            matrix, tcol, len(X), alpha=self.alpha, redundancy_floor=self.redundancy_floor
        )
        self.dropped_ = result.dropped

        if self.dropped_:
            logger.info(
                "[prune] removing %d/%d predictor(s) (tau=%.3f): %s",
                len(self.dropped_),
                len(predictors),
                result.threshold,
                {c: result.reasons.get(c, "") for c in self.dropped_},
            )
        else:
            logger.info(
                "[prune] no predictors removed (tau=%.3f, %d predictors).",
                result.threshold,
                len(predictors),
            )
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        cols = [c for c in getattr(self, "dropped_", []) if c in X.columns]
        return X.drop(columns=cols) if cols else X
