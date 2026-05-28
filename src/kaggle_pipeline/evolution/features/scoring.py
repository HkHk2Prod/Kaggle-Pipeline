"""Feature scoring: the extensible :class:`FeatureScoreSet` and the scorers.

Each feature carries a :class:`FeatureScoreSet` -- a mapping of named
:class:`Score` objects -- so new scores are added without reshaping a rigid
struct. Intrinsic scores (target correlation, redundancy, tree importance, ...)
are computed at the *registry* level on global materializations; downstream credit
is attached by the credit assigner. The :class:`FeatureUtility` combiner folds
both into a single utility with confidence weighting, and converts utilities to
selection probabilities.

This module deliberately keeps the statistics simple (the README's "do not
overimplement initially"): the structure is built to grow, the formulae are not
meant to be final.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from kaggle_pipeline.evolution.utils.arrays import missing_mask

if TYPE_CHECKING:
    from kaggle_pipeline.evolution.config import EvolutionSettings
    from kaggle_pipeline.evolution.features.genome import FeatureGenome

# --- Reserved score names ---------------------------------------------------
# Implemented initially.
TARGET_CORRELATION = "target_correlation"
REDUNDANCY = "redundancy"  # negative score: higher means more redundant
TREE_IMPORTANCE = "tree_importance"
# Reserved for future use; declared so callers can rely on stable names.
MISSINGNESS = "missingness"
DRIFT = "drift"
STABILITY = "stability"
GENERATION_COST = "generation_cost"
MATERIALIZED_WIDTH = "materialized_width"
COMPLEXITY = "complexity"
DOWNSTREAM_MODEL = "downstream_model"
DOWNSTREAM_MUTATION = "downstream_mutation"
ELITE_USAGE = "elite_usage"
FINAL_UTILITY = "final_feature_utility"


@dataclass
class Score:
    """One named score with the metadata needed to combine it with others."""

    name: str
    value: float
    higher_is_better: bool = True
    weight: float = 1.0
    normalized_value: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def directed_value(self) -> float:
        """Value flipped to a larger-is-better convention."""
        return self.value if self.higher_is_better else -self.value

    def to_serializable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "higher_is_better": self.higher_is_better,
            "weight": self.weight,
            "normalized_value": self.normalized_value,
            "metadata": dict(self.metadata),
        }


@dataclass
class FeatureScoreSet:
    """A flexible bag of named scores plus the combined utility components.

    ``intrinsic_score``/``downstream_score``/``utility`` are filled by
    :class:`FeatureUtility`; the raw per-name scores are filled by the scorers.
    """

    scores: dict[str, Score] = field(default_factory=dict)
    intrinsic_score: float = 0.0
    downstream_score: float = 0.0
    utility: float = 0.0

    def set(
        self,
        name: str,
        value: float,
        *,
        higher_is_better: bool = True,
        weight: float = 1.0,
        normalized_value: float | None = None,
        **metadata: Any,
    ) -> None:
        self.scores[name] = Score(
            name=name,
            value=float(value),
            higher_is_better=higher_is_better,
            weight=weight,
            normalized_value=normalized_value,
            metadata=dict(metadata),
        )

    def get(self, name: str) -> Score | None:
        return self.scores.get(name)

    def value(self, name: str, default: float = 0.0) -> float:
        score = self.scores.get(name)
        return score.value if score is not None else default

    def normalized(self, name: str, default: float = 0.0) -> float:
        """A score's normalized value when set, else its raw value, else ``default``."""
        score = self.scores.get(name)
        if score is None:
            return default
        return score.normalized_value if score.normalized_value is not None else score.value

    def has(self, name: str) -> bool:
        return name in self.scores

    def to_serializable(self) -> dict[str, Any]:
        return {
            "scores": {name: s.to_serializable() for name, s in self.scores.items()},
            "intrinsic_score": self.intrinsic_score,
            "downstream_score": self.downstream_score,
            "utility": self.utility,
        }


# --- Normalization helpers ---------------------------------------------------


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map values to ``[0, 1]`` by rank (ties share the average rank).

    Robust to scale and outliers, which matters when combining heterogeneous
    scores (a correlation, a tree importance, a cost). A single value maps to 0.5.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return values
    if n == 1:
        return np.array([0.5])
    order = values.argsort()
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    # Average ranks for ties so equal scores get equal normalized values.
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    starts = cum - counts
    avg_rank = (starts + cum - 1) / 2.0
    ranks = avg_rank[inverse]
    return ranks / (n - 1)


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Scale values to ``[0, 1]`` by min/max; constant input maps to 0.5."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.full(values.shape, 0.5)
    return (values - lo) / (hi - lo)


# --- Intrinsic scorers -------------------------------------------------------


class FeatureScorer:
    """Computes intrinsic per-feature scores from materialized values + target.

    Stateless; methods take arrays so they can be reused across contexts/samples.
    ``task`` is the v1 task string (``'classification'`` / ``'regression'``).
    Categorical features should be passed already encoded to numeric (the registry
    materializes a numeric view for scoring).
    """

    def target_correlation(
        self, values: np.ndarray, y: np.ndarray, *, task: str = "classification"
    ) -> float:
        """Absolute association of a feature with the target, in ``[0, 1]``.

        Numeric feature, regression / binary classification: ``|Pearson r|`` (which
        equals the point-biserial correlation for a 0/1 target). Numeric,
        multiclass: the correlation ratio (eta) treating classes as groups.
        Categorical feature: the correlation ratio of the (numeric-encoded) target
        across the feature's categories -- "target mean separation". NaNs are
        dropped pairwise.
        """
        raw = np.asarray(values)
        y = np.asarray(y).ravel()
        if raw.dtype == object:
            return self._categorical_target_assoc(raw, y, task=task)
        values = raw.astype(float).ravel()
        mask = np.isfinite(values)
        if mask.sum() < 3:
            return 0.0
        values = values[mask]
        y_masked = y[mask]
        if np.nanstd(values) < 1e-12:
            return 0.0

        if task == "classification":
            classes = np.unique(y_masked)
            if classes.size > 2:
                return self._correlation_ratio(values, y_masked)
            # Binary (or degenerate) -> point-biserial == |Pearson|.
            y_num = (
                (y_masked == classes[-1]).astype(float)
                if classes.size == 2
                else y_masked.astype(float)
            )
            return self._abs_pearson(values, y_num)
        return self._abs_pearson(values, y_masked.astype(float))

    def _categorical_target_assoc(self, feature: np.ndarray, y: np.ndarray, *, task: str) -> float:
        """Correlation ratio of the numeric-encoded target across feature categories."""
        groups = np.asarray(feature, dtype=object).astype(str)
        if groups.size < 3 or np.unique(groups).size < 2:
            return 0.0
        if task == "classification":
            classes = np.unique(y)
            mapping = {c: i for i, c in enumerate(classes)}
            y_num = np.array([mapping[v] for v in y], dtype=float)
        else:
            y_num = np.asarray(y, dtype=float)
        return self._correlation_ratio(y_num, groups)

    @staticmethod
    def _abs_pearson(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        r = float(np.corrcoef(a, b)[0, 1])
        return 0.0 if not np.isfinite(r) else abs(r)

    @staticmethod
    def _correlation_ratio(values: np.ndarray, groups: np.ndarray) -> float:
        """Correlation ratio eta in ``[0, 1]``: between-group variance fraction."""
        overall_mean = values.mean()
        ss_total = float(((values - overall_mean) ** 2).sum())
        if ss_total < 1e-12:
            return 0.0
        ss_between = 0.0
        for g in np.unique(groups):
            grp = values[groups == g]
            ss_between += grp.size * (grp.mean() - overall_mean) ** 2
        return float(np.sqrt(max(0.0, ss_between) / ss_total))

    def missingness(self, values: np.ndarray) -> float:
        """Fraction of missing/non-finite entries (a negative signal). In ``[0, 1]``."""
        raw = np.asarray(values)
        if raw.size == 0:
            return 1.0
        if raw.dtype == object:
            return float(missing_mask(raw).mean())
        return float((~np.isfinite(raw.astype(float).ravel())).mean())

    def tree_importances(
        self,
        named_values: dict[str, np.ndarray],
        y: np.ndarray,
        *,
        task: str = "classification",
        rng: np.random.Generator | None = None,
        max_rows: int = 5000,
    ) -> dict[str, float]:
        """Fit a small tree ensemble on a row sample and return per-feature importance.

        One signal among several -- known to be biased toward continuous /
        high-cardinality features, so callers should not treat it as ground truth.
        Returns ``{}`` (gracefully) if sklearn is unavailable or the inputs are
        degenerate. Importances are normalised to sum to 1 over the given features.
        """
        names = list(named_values)
        if not names:
            return {}
        try:
            if task == "classification":
                from sklearn.ensemble import ExtraTreesClassifier as _Forest
            else:
                from sklearn.ensemble import ExtraTreesRegressor as _Forest
        except Exception:  # pragma: no cover - sklearn always present in this project
            return {}

        rng = rng or np.random.default_rng()
        cols = [np.asarray(named_values[n], dtype=float).ravel() for n in names]
        matrix = np.column_stack(cols)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        y_arr = np.asarray(y).ravel()
        n = matrix.shape[0]
        if n != y_arr.size or n < 5:
            return {}
        if n > max_rows:
            idx = rng.choice(n, size=max_rows, replace=False)
            matrix, y_arr = matrix[idx], y_arr[idx]

        seed = int(rng.integers(0, 2**31 - 1))
        forest = _Forest(n_estimators=64, max_depth=8, n_jobs=1, random_state=seed)
        try:
            forest.fit(matrix, y_arr)
        except Exception:
            return {}
        importances = np.asarray(forest.feature_importances_, dtype=float)
        total = importances.sum()
        if total <= 0:
            return dict.fromkeys(names, 0.0)
        importances = importances / total
        return {name: float(imp) for name, imp in zip(names, importances, strict=True)}


# --- Utility combiner --------------------------------------------------------


class FeatureUtility:
    """Combines intrinsic scores and downstream credit into a feature's utility.

    Implements the README formula with confidence weighting:
    ``beta = n_obs / (n_obs + k)`` shifts weight from intrinsic to downstream as
    evidence accrues. Scores are read by name from the genome's score set; absent
    scores contribute 0, so partially-scored features still get a utility.
    """

    def __init__(self, settings: EvolutionSettings):
        self.settings = settings

    def intrinsic(self, genome: FeatureGenome) -> float:
        w = self.settings.feature_scoring_weights
        s = genome.score_set
        # normalized() prefers a score's normalized_value (set by the registry's
        # normalization pass), falling back to the raw value, else 0.
        return (
            w.target_correlation * s.normalized(TARGET_CORRELATION)
            + w.tree_importance * s.normalized(TREE_IMPORTANCE)
            - w.redundancy * s.normalized(REDUNDANCY)
            - w.complexity * self._normalized_complexity(genome)
            - w.cost * s.normalized(GENERATION_COST)
        )

    def downstream(self, genome: FeatureGenome) -> float:
        w = self.settings.downstream_weights
        u = genome.usage_stats
        return (
            w.add_mutation_delta * u.avg_add_delta
            - w.remove_mutation_delta * u.avg_remove_delta
            + w.model_usage_credit * u.avg_model_utility_when_used
            + w.elite_usage_rate * u.elite_usage_rate
        )

    def combine(self, genome: FeatureGenome) -> float:
        """Compute, store and return the genome's final utility."""
        intrinsic = self.intrinsic(genome)
        downstream = self.downstream(genome)
        n_obs = genome.usage_stats.downstream_observation_count
        beta = n_obs / (n_obs + self.settings.downstream_confidence_k)
        alpha = 1.0 - beta
        bonus = self.settings.original_feature_bonus if genome.is_original else 0.0
        utility = alpha * intrinsic + beta * downstream + bonus

        genome.score_set.intrinsic_score = float(intrinsic)
        genome.score_set.downstream_score = float(downstream)
        genome.score_set.utility = float(utility)
        genome.score_set.set(FINAL_UTILITY, utility)
        return float(utility)

    @staticmethod
    def _normalized_complexity(genome: FeatureGenome) -> float:
        # Cheap monotone squashing of depth+complexity into [0, 1); originals are 0.
        raw = genome.depth + genome.complexity
        return float(raw / (1.0 + raw)) if raw > 0 else 0.0
