"""Textual / heatmap EDA: dataset metadata and correlation matrices."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from kaggle_pipeline.preprocessing.association import association_matrix

logger = logging.getLogger(__name__)

_RULE = "\n\n" + "-" * 40 + "\n"


def print_meta_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Log shapes, describe(), null counts and unique counts for the train set.

    Goes through the package logger (not ``print``) so it honours
    ``Config.verbosity``: shown at 'normal'/'verbose', suppressed at 'quiet'.
    """
    logger.info("Train shape: %s  Test Shape: %s", train_df.shape, test_df.shape)
    logger.info("%s", _RULE)
    logger.info("%s", train_df.describe().T)
    logger.info("%s", _RULE)
    logger.info("Nulls:\n\n%s", train_df.isnull().sum().sort_values())
    logger.info("%s", _RULE)
    logger.info("Uniques:\n\n%s", train_df.nunique().sort_values())
    logger.info("%s", _RULE)
    logger.info("%s", train_df.head(10))


def correlation_matrices(
    train_df: pd.DataFrame,
    target: Sequence[str],
    columns_x: Sequence[str],
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
) -> None:
    """Plot a numeric Pearson heatmap and a mixed-type association heatmap.

    The first heatmap is the usual *signed* Pearson correlation over numeric
    columns only. The second is an *association* matrix (see
    :mod:`kaggle_pipeline.preprocessing.association`) with one row/column per original
    column, so a high-cardinality categorical stays a single cell instead of
    exploding into one dummy per level.
    """
    cols = list(target) + list(columns_x)
    df = train_df[cols]  # target first

    # Signed Pearson over numerics: diverging palette centred at 0, range [-1, 1].
    sns.heatmap(df.corr(numeric_only=True), cmap="BrBG", vmin=-1, vmax=1, annot=True)
    plt.title("Pearson correlation (numeric columns only)")
    plt.show()

    plot_association_heatmap(df, list(num_cols), list(cat_cols))


def plot_association_heatmap(
    df: pd.DataFrame, num_cols: Sequence[str], cat_cols: Sequence[str]
) -> None:
    """Heatmap of the mixed-type association matrix, with non-Pearson cells flagged.

    Columns are ordered numeric-first / categorical-last and the categorical
    block is set off with divider lines; categorical tick labels are suffixed
    ``(V)`` so it is clear which cells are association strengths (Cramér's V /
    correlation ratio) rather than signed Pearson correlations.
    """
    # Restrict the requested splits to columns actually present in df.
    num_present = [c for c in num_cols if c in df.columns]
    cat_present = [c for c in cat_cols if c in df.columns]
    matrix, ordered_num, ordered_cat = association_matrix(df, num_present, cat_present)

    labels = list(ordered_num) + [f"{c} (V)" for c in ordered_cat]
    fig, ax = plt.subplots(figsize=(max(8, len(matrix)), max(8, len(matrix))))
    # Sequential palette on a fixed [0, 1] scale: unsigned strength, not a
    # signed correlation -- a different palette from the Pearson heatmap above.
    sns.heatmap(
        matrix.to_numpy(),
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    # Divider lines separating the numeric block from the categorical block.
    boundary = len(ordered_num)
    if 0 < boundary < len(matrix):
        ax.axhline(boundary, color="white", linewidth=3)
        ax.axvline(boundary, color="white", linewidth=3)
    ax.set_title(
        "Association strength (unsigned, [0, 1])\n"
        "num–num: |Pearson r|   cat–num: correlation ratio η   "
        "cat–cat: Cramér's V  (‘(V)’ marks categorical columns)"
    )
    plt.show()
