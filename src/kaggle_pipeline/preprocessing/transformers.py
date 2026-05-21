"""Stateless-config scikit-learn transformers used before training.

Three transformers, applied in order by :func:`build_pretrain_pipeline`:

* :class:`FeatureEngineer`     -- derive new columns from ``df.eval`` expressions.
* :class:`CategoricalTyper`    -- order categories and cast objects to ``category``.
* :class:`OrdinalEncoderTransformer` -- integer-encode detected ordinal columns.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from kaggle_pipeline.preprocessing.columns import (
    detect_ordinal_order_cols,
    is_num_check,
    make_cat_order,
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Apply a list of ``pandas.eval`` expressions, casting bool results to int.

    Each expression looks like ``"new_col = expr(existing_cols)"``. Boolean
    outputs are converted to 0/1 so downstream models treat them as numeric.
    """

    def __init__(self, expressions: Sequence[str] | None = None):
        self.expressions = expressions

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for expr in self.expressions or []:
            X = X.eval(expr, engine="python")
        bool_cols = X.select_dtypes(include="bool").columns
        X[bool_cols] = X[bool_cols].astype(int)
        return X


class CategoricalTyper(BaseEstimator, TransformerMixin):
    """Order categorical values (No < Yes, Low < Medium < High, ...) and retype.

    Learns, on ``fit``, which columns are categorical and an ordering for each;
    on ``transform`` it applies that ordering as an ordered ``Categorical`` and
    casts any remaining object/string columns to the ``category`` dtype. The
    ordering is primarily for readable EDA graphs.
    """

    def __init__(self, cat_cutoff: int = 5, cat_order_list: Sequence[str] | None = None):
        self.cat_cutoff = cat_cutoff
        self.cat_order_list = cat_order_list

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cat_order = make_cat_order(self.cat_order_list or [])
        self.cat_cols_ = [
            col for col in X.columns if not is_num_check(col, X, cat_cutoff=self.cat_cutoff)
        ]
        self.ordered_cats_ = {
            col: (
                sorted(X[col].unique(), key=cat_order)
                if not pd.api.types.is_numeric_dtype(X[col])
                else sorted(X[col].unique())
            )
            for col in self.cat_cols_
        }
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col, cats in self.ordered_cats_.items():
            if col in X.columns:
                X[col] = pd.Categorical(X[col], categories=cats, ordered=True)
        obj_cols = X.select_dtypes(include=["object", "string"]).columns
        X[obj_cols] = X[obj_cols].astype("category")
        return X


class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    """Integer-encode columns whose values match a known ordering.

    Ordinal columns (excluding the target) are detected via
    :func:`detect_ordinal_order_cols` and mapped to ``0, 1, 2, ...`` in natural
    order. The target is left untouched here -- it is handled separately by the
    target transforms so submissions can be decoded back to labels.
    """

    def __init__(
        self,
        target: Sequence[str] | None = None,
        order_lists: Sequence[Sequence[str]] | None = None,
    ):
        self.target = target
        self.order_lists = order_lists

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        target = list(self.target or [])
        non_target_cols = [c for c in X.columns if c not in target]
        ordered = detect_ordinal_order_cols(X[non_target_cols], self.order_lists or [])
        self.mappings_ = {
            col: {cat: i for i, cat in enumerate(order)} for col, order in ordered.items()
        }
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col, mapping in self.mappings_.items():
            if col in X.columns:
                # ``pd.to_numeric`` forces an integer (or float, if there are
                # unmapped values) result. Without it, pandas >= 3.0 keeps the
                # source ``category`` dtype through ``.map``, so the encoded
                # column would still be treated as categorical downstream.
                X[col] = pd.to_numeric(X[col].map(mapping))
        return X
