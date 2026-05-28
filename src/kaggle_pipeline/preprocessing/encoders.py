"""Per-column categorical encoding for models that need numeric input.

Some estimators handle categorical columns natively (CatBoost, XGBoost,
LightGBM, HistGB) and are simply handed the raw column. The rest
(RandomForest, LogisticRegression) need the categoricals turned into numbers
first. :func:`_make_encoder` builds the transformer implementing a single
strategy; :class:`FrequencyEncoder` is the default fallback for high-cardinality
columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# An unconfigured categorical with at most this many distinct levels defaults to
# one-hot encoding (cheap and lossless at low cardinality); wider columns fall
# back to frequency encoding so one-hot can't explode the feature count.
ONEHOT_MAX_CARDINALITY = 20


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
    raise ValueError(f"Unknown categorical encoding strategy {strategy!r}.")
