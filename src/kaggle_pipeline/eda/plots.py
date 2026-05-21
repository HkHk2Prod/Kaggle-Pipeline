"""Exploratory plots: pairwise relationships across columns.

These mirror the notebook's plotting helpers. They are invoked only through the
standalone ``analyze`` entry point, since rendering a full grid of plots is
expensive and unrelated to training.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class EdaContext:
    """Typed frame plus the column groupings needed to choose a plot kind."""

    df: pd.DataFrame
    ordered_cats: dict[str, list]
    columns: list[str]
    columns_x: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    target: list[str]


def plot_num_vs_cat(ctx: EdaContext, ax, x: str, y: str) -> None:
    """Boxplot of numeric ``y`` grouped by categorical ``x``."""
    ctx.df.boxplot(column=y, by=x, ax=ax)


def plot_num_vs_num(ctx: EdaContext, ax, x_col: str, y_col: str, n_bins: int = 20) -> None:
    """Scatter of ``y_col`` vs ``x_col`` with a binned-mean overlay."""
    x, y = ctx.df[x_col], ctx.df[y_col]
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = [y[(x >= bins[i]) & (x < bins[i + 1])].mean() for i in range(n_bins)]

    ax.scatter(x, y, alpha=0.3, s=15, label="data")
    ax.plot(
        bin_centers,
        bin_means,
        color="red",
        linewidth=2,
        marker="o",
        markersize=4,
        label="binned mean",
    )
    ax.set(xlabel=x_col, ylabel=y_col)


def plot_cat_vs_any(ctx: EdaContext, ax, x: str, y: str, mode: str = "fill") -> None:
    """Stacked/filled histogram of ``x`` coloured by categorical ``y``."""
    # Fall back to sorted unique values for columns the typer did not order
    # (e.g. low-cardinality numerics treated as categorical only for plotting).
    cats = ctx.ordered_cats.get(y)
    if cats is None:
        cats = sorted(ctx.df[y].dropna().unique())
    hue_order = list(reversed(cats))
    if mode == "fill":
        multiple, stat = "fill", "proportion"
    else:  # 'stack'
        multiple, stat = "stack", "count"
    sns.histplot(data=ctx.df, x=x, hue=y, hue_order=hue_order, stat=stat, multiple=multiple, ax=ax)


def plot(ctx: EdaContext, ax, x: str, y: str) -> None:
    """Dispatch to the right plot kind based on the column types."""
    if y in ctx.cat_cols:
        plot_cat_vs_any(ctx, ax, x, y)
    elif x in ctx.num_cols:
        plot_num_vs_num(ctx, ax, x, y)
    else:
        plot_num_vs_cat(ctx, ax, x, y)


def graphs(ctx: EdaContext, target_only: bool = False) -> None:
    """Render a grid of pairwise plots (optionally only against the target)."""
    if target_only:
        cols_x, cols_y = ctx.columns_x, ctx.target
    else:
        cols_x, cols_y = ctx.columns, ctx.columns

    size = (8, 8)
    fig, ax = plt.subplots(
        len(cols_x),
        len(cols_y),
        squeeze=False,
        figsize=(size[0] * len(cols_y), size[1] * len(cols_x)),
    )
    for i, x in enumerate(cols_x):
        for j, y in enumerate(cols_y):
            if i != j:
                plot(ctx, ax[i, j], x, y)
    plt.show()
