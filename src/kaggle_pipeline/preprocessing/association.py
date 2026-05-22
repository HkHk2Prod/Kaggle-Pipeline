"""Pairwise *association* measures for a mixed numeric/categorical frame.

A Pearson correlation matrix only makes sense between numeric columns, and
one-hot-expanding a high-cardinality categorical (e.g. ``driver`` with dozens of
levels) explodes the matrix into one row/column per level. These helpers instead
summarise each *original* column as a single row/column, picking a measure that
fits the pair of column types:

* numeric  vs numeric      -> ``|Pearson r|``      (linear strength)
* categorical vs numeric   -> bias-corrected correlation ratio η (ε²) (how much
                                                      the category explains the
                                                      numeric mean)
* categorical vs categorical -> bias-corrected Cramér's V

The two cross-type measures (η and Cramér's V) are *both* bias-corrected so a
high-cardinality column does not show a spurious strength in one block while
correctly reading ~0 in the other.

All three live on ``[0, 1]`` and are *unsigned* strengths, so the resulting
matrix is deliberately not a correlation matrix -- :func:`association_matrix`
returns it together with the per-column type so the plotting code can label and
segregate the non-Pearson cells (see :func:`kaggle_pipeline.eda.reports`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V between two categorical series, in ``[0, 1]``.

    Uses the Bergsma (2013) correction so the value does not inflate with the
    number of categories -- important precisely for high-cardinality columns.
    Returns ``0.0`` for degenerate inputs (a single category on either side).
    """
    confusion = pd.crosstab(x, y)
    if confusion.size == 0 or min(confusion.shape) < 2:
        return 0.0
    chi2 = _chi2_statistic(confusion.to_numpy(dtype=float))
    n = confusion.to_numpy().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion.shape
    # Bias correction.
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Bias-corrected correlation ratio between a categorical and numeric series, ``[0, 1]``.

    The raw η² = SS_between / SS_total inflates with the number of groups: an
    *independent* category still yields η ≈ √((k-1)/(n-1)) purely from the group
    count, so a high-cardinality column (e.g. ``driver``) shows a spurious ~0.5
    against every numeric. That mirrors the uncorrected-Cramér's-V problem, so we
    apply the analogous correction here -- the ε² (epsilon-squared) estimator

        ε² = (SS_between - (k-1)·MS_within) / SS_total,   MS_within = SS_within/(n-k)

    and return ``√max(0, ε²)``. Under independence E[SS_between] ≈ (k-1)·MS_within,
    so ε² collapses to ~0, keeping the cat–num block consistent with the
    bias-corrected :func:`cramers_v`. η = 0 means the category tells you nothing
    about the numeric column and η = 1 means it fully determines it. A
    single-category or one-row-per-category (unique) column returns ``0.0``: it
    has no within-group variance to correct against and carries no generalisable
    association, matching how :func:`cramers_v` treats a degenerate column.
    """
    frame = pd.DataFrame(
        {"cat": categories.to_numpy(), "val": pd.to_numeric(values, errors="coerce")}
    )
    frame = frame.dropna()
    if frame.empty:
        return 0.0
    values_arr = frame["val"].to_numpy(dtype=float)
    n = len(values_arr)
    total_mean = values_arr.mean()
    ss_total = float(((values_arr - total_mean) ** 2).sum())
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    k = 0
    for _, group in frame.groupby("cat", observed=True)["val"]:
        g = group.to_numpy(dtype=float)
        ss_between += len(g) * (g.mean() - total_mean) ** 2
        k += 1
    if k < 2 or n <= k:  # need >= 2 groups and within-group d.o.f. to correct.
        return 0.0
    ms_within = (ss_total - ss_between) / (n - k)
    eps_sq = (ss_between - (k - 1) * ms_within) / ss_total
    return float(np.sqrt(eps_sq)) if eps_sq > 0 else 0.0


def association_matrix(
    df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the symmetric association matrix for ``num_cols + cat_cols``.

    Columns are ordered numeric-first, categorical-last so the result has a clean
    block structure: the top-left block is ``|Pearson r|``, the bottom-right
    block is Cramér's V, and the off-diagonal blocks are the correlation ratio.

    Returns ``(matrix, ordered_num, ordered_cat)`` -- the second and third items
    let the caller know where the categorical block begins for labelling.
    """
    ordered_num = [c for c in num_cols if c in df.columns]
    ordered_cat = [c for c in cat_cols if c in df.columns]
    order = ordered_num + ordered_cat
    is_cat = {c: c in ordered_cat for c in order}

    matrix = pd.DataFrame(np.eye(len(order)), index=order, columns=order, dtype=float)
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            value = _pair_association(df, a, b, is_cat[a], is_cat[b])
            matrix.loc[a, b] = value
            matrix.loc[b, a] = value
    return matrix, ordered_num, ordered_cat


def _pair_association(df: pd.DataFrame, a: str, b: str, a_is_cat: bool, b_is_cat: bool) -> float:
    """Dispatch one pair to the measure that fits its column types."""
    if a_is_cat and b_is_cat:
        return cramers_v(df[a], df[b])
    if a_is_cat:  # a categorical, b numeric
        return correlation_ratio(df[a], df[b])
    if b_is_cat:  # a numeric, b categorical
        return correlation_ratio(df[b], df[a])
    corr = df[[a, b]].corr(numeric_only=True).iloc[0, 1]
    return 0.0 if pd.isna(corr) else float(abs(corr))


def _chi2_statistic(observed: np.ndarray) -> float:
    """Pearson χ² statistic of a contingency table (no SciPy dependency)."""
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    grand = observed.sum()
    expected = row_totals @ col_totals / grand
    mask = expected > 0
    return float((((observed - expected) ** 2)[mask] / expected[mask]).sum())
