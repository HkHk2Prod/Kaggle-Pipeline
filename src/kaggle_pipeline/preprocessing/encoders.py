"""Per-column categorical encoding for models that need numeric input.

Some estimators handle categorical columns natively (CatBoost, XGBoost,
LightGBM, HistGB) and are simply handed the raw column -- see each model's
``handles_categoricals`` flag. The rest (RandomForest, LogisticRegression) need
the categoricals turned into numbers first.

This module decides *how* each categorical predictor is encoded for those
models. The choice is per column and user-controlled via
``Config.categorical_encoding``; any column left unspecified defaults to
:data:`DEFAULT_STRATEGY` (frequency encoding). :func:`resolve_encoding_plan`
fills in the defaults and logs an ``[encoding]`` summary so the rule is
explicit in the run log, and :func:`categorical_transformer_specs` turns a
resolved plan into ``ColumnTransformer`` entries.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# Valid values for a ``Config.categorical_encoding`` entry.
#   native    -- pass the raw column to the model (only honoured by models that
#                handle categoricals natively; otherwise falls back to frequency).
#   frequency -- replace each level with its training-set frequency.
#   target    -- cross-fitted mean-target encoding (sklearn ``TargetEncoder``).
#   onehot    -- one indicator column per level (unseen levels ignored).
#   ordinal   -- arbitrary integer per level (unseen -> -1).
#   drop      -- discard the column.
ENCODING_STRATEGIES: frozenset[str] = frozenset(
    {"native", "frequency", "target", "onehot", "ordinal", "drop"}
)
DEFAULT_STRATEGY = "frequency"


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace each category with its training-set relative frequency.

    Frequencies are learned on ``fit``; categories unseen at ``transform`` time
    (e.g. a driver that only appears in the test set) map to ``0.0``. Output is
    numeric, one column in -> one column out, so it never inflates the feature
    count the way one-hot encoding does on high-cardinality columns.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.frequencies_ = {
            col: X[col].value_counts(normalize=True, dropna=True) for col in X.columns
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            freq = self.frequencies_.get(col)
            mapped = X[col].map(freq) if freq is not None else X[col]
            X[col] = pd.to_numeric(mapped, errors="coerce").fillna(0.0).astype(float)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


def resolve_encoding_plan(
    categorical_encoding: dict[str, str],
    train_df: pd.DataFrame,
    cat_cols_x: Sequence[str],
    *,
    announce: bool = True,
) -> dict[str, str]:
    """Return ``{column -> strategy}`` for every categorical predictor.

    Columns absent from ``categorical_encoding`` default to
    :data:`DEFAULT_STRATEGY`. Configured columns that are not categorical
    predictors are reported but ignored. When ``announce`` is set, logs one
    ``[encoding]`` line per column (with its cardinality) plus a header making
    the capability-wins rule explicit.
    """
    cat_set = list(cat_cols_x)
    plan = {col: categorical_encoding.get(col, DEFAULT_STRATEGY) for col in cat_set}

    if announce:
        unknown_cols = sorted(set(categorical_encoding) - set(cat_set))
        logger.info(
            "[encoding] categorical encoding plan (used by models without native "
            "categorical support, e.g. RandomForest / LogisticRegression; "
            "native-capable models receive the raw column):"
        )
        for col in cat_set:
            n_unique = train_df[col].nunique() if col in train_df.columns else "?"
            default_note = "" if col in categorical_encoding else " (default)"
            logger.info(
                "[encoding]   %s = %r%s  (%s unique)", col, plan[col], default_note, n_unique
            )
        for col in unknown_cols:
            logger.info(
                "[encoding]   %r in categorical_encoding is not a categorical predictor; ignored.",
                col,
            )
    return plan


def categorical_transformer_specs(
    plan: dict[str, str], columns: Sequence[str]
) -> list[tuple[str, object, list[str]]]:
    """Group ``columns`` by their encoding strategy into ColumnTransformer entries.

    Returns a list of ``(name, transformer, columns)`` triples suitable for a
    :class:`sklearn.compose.ColumnTransformer`. Only ``columns`` are considered
    (so a caller can encode a subset, e.g. only the over-cap columns for HistGB).
    A ``native`` strategy is treated as frequency here, because this path is only
    taken by models that cannot consume a raw categorical column.
    """
    by_strategy: dict[str, list[str]] = {}
    for col in columns:
        strategy = plan.get(col, DEFAULT_STRATEGY)
        # ``native`` is meaningless for a model that needs encoding; fall back.
        if strategy == "native":
            strategy = DEFAULT_STRATEGY
        by_strategy.setdefault(strategy, []).append(col)

    specs: list[tuple[str, object, list[str]]] = []
    for strategy, cols in by_strategy.items():
        specs.append((f"cat_{strategy}", _make_encoder(strategy), cols))
    return specs


def _make_encoder(strategy: str):
    """Build the transformer implementing a single encoding strategy."""
    if strategy == "frequency":
        return FrequencyEncoder()
    if strategy == "drop":
        return "drop"
    if strategy == "onehot":
        from sklearn.preprocessing import OneHotEncoder

        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if strategy == "ordinal":
        from sklearn.preprocessing import OrdinalEncoder

        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if strategy == "target":
        from sklearn.preprocessing import TargetEncoder

        return TargetEncoder()
    raise ValueError(
        f"Unknown categorical encoding strategy {strategy!r}; "
        f"expected one of {sorted(ENCODING_STRATEGIES)}."
    )
