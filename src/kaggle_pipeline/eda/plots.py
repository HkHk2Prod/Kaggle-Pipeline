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

#: Default cap on distinct levels a categorical may have before EDA plots fold it
#: to its top-N levels plus an "Other" bucket. A high-cardinality column (e.g. a
#: ``driver`` with hundreds of levels) otherwise makes seaborn build one stacked
#: layer / boxplot group per level -- slow, illegible, and the source of pandas'
#: "highly fragmented DataFrame" PerformanceWarning. Overridable per run via
#: ``Config.max_plot_cats``; kept in sync with that field's default.
MAX_PLOT_CATEGORIES = 20
_OTHER_LABEL = "Other"


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
    max_plot_cats: int = MAX_PLOT_CATEGORIES


def _cap_categories(
    values: pd.Series, ordered: list | None, max_levels: int
) -> tuple[pd.Series, list]:
    """Fold all but the ``max_levels`` most frequent levels into an ``"Other"`` bucket.

    Returns the (possibly rewritten) series plus the level order to plot it in.
    Columns at or below the cap are returned unchanged, in ``ordered`` order when
    given else sorted-unique. NaNs are preserved, never bucketed into ``"Other"``.
    """
    non_null = values.dropna()
    if non_null.nunique() <= max_levels:
        cats = list(ordered) if ordered is not None else sorted(non_null.unique())
        return values, cats
    keep = list(non_null.value_counts().nlargest(max_levels).index)
    to_other = values.notna() & ~values.isin(keep)
    capped = values.astype(object).mask(to_other, _OTHER_LABEL)
    # Keep the ordering the typer gave us where available; otherwise frequency
    # order (value_counts is already most-frequent-first). "Other" sorts last.
    kept_order = [c for c in ordered if c in set(keep)] if ordered is not None else keep
    return capped, [*kept_order, _OTHER_LABEL]


def plot_num_vs_cat(ctx: EdaContext, ax, x: str, y: str) -> None:
    """Boxplot of numeric ``y`` grouped by categorical ``x`` (high-cardinality capped)."""
    groups, _ = _cap_categories(ctx.df[x], ctx.ordered_cats.get(x), ctx.max_plot_cats)
    pd.DataFrame({y: ctx.df[y], x: groups}).boxplot(column=y, by=x, ax=ax)


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
    # _cap_categories falls back to sorted-unique for columns the typer did not
    # order (e.g. low-cardinality numerics treated as categorical only for
    # plotting) and folds high-cardinality columns to top-N + "Other".
    hue, cats = _cap_categories(ctx.df[y], ctx.ordered_cats.get(y), ctx.max_plot_cats)
    hue_order = list(reversed(cats))
    if mode == "fill":
        multiple, stat = "fill", "proportion"
    else:  # 'stack'
        multiple, stat = "stack", "count"
    plot_df = pd.DataFrame({x: ctx.df[x], y: hue})
    sns.histplot(data=plot_df, x=x, hue=y, hue_order=hue_order, stat=stat, multiple=multiple, ax=ax)


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
