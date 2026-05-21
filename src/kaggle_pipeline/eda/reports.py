"""Textual / heatmap EDA: dataset metadata and correlation matrices."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_RULE = "\n\n" + "-" * 40 + "\n"


def print_meta_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print shapes, describe(), null counts and unique counts for the train set."""
    print("Train shape: ", train_df.shape, "Test Shape: ", test_df.shape)
    print(_RULE)
    print(train_df.describe().T)
    print(_RULE)
    print("Nulls:\n\n", train_df.isnull().sum().sort_values())
    print(_RULE)
    print("Uniques:\n\n", train_df.nunique().sort_values())
    print(_RULE)
    print(train_df.head(10))


def correlation_matrices(
    train_df: pd.DataFrame,
    target: Sequence[str],
    columns_x: Sequence[str],
    cat_cols: Sequence[str],
) -> None:
    """Plot correlation heatmaps, both raw-numeric and with one-hot categoricals."""
    df = train_df[list(target) + list(columns_x)]  # target first
    df_encoded = pd.get_dummies(df, columns=list(cat_cols), drop_first=True)

    sns.heatmap(df.corr(numeric_only=True), cmap="BrBG", annot=True)
    plt.show()

    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(df_encoded.corr(), cmap="BrBG", annot=True, ax=ax)
    plt.show()
