"""Column inspection helpers: numeric/categorical splits and ordinal detection.

These were module-level functions in the notebook that read globals
(``train_df``, ``CAT_CUTOFF``, ``ORDER_LISTS``). Here every dependency is an
explicit argument, so the helpers are pure and testable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import pandas as pd


def is_num_check(
    col: str,
    df: pd.DataFrame,
    *,
    for_graph: bool = False,
    cat_cutoff: int = 5,
) -> bool:
    """Is ``col`` numeric (and, for graphs, high-cardinality enough to plot)?

    ``cat_cutoff`` only matters when ``for_graph`` is True: a numeric column with
    few unique values is drawn as categorical rather than as a continuous axis.
    """
    is_numeric = pd.api.types.is_numeric_dtype(df[col])
    return is_numeric and (not for_graph or df[col].nunique() > cat_cutoff)


def get_columns(df: pd.DataFrame, id_col: Sequence[str]) -> pd.Index:
    """All columns except the id column(s)."""
    return df.drop(columns=list(id_col)).columns


def get_predictor_names(df: pd.DataFrame, target: Sequence[str], id_col: Sequence[str]) -> pd.Index:
    """Predictor columns: everything except the target and id column(s)."""
    return df.drop(columns=list(target) + list(id_col)).columns


def split_num_cat(
    columns: Iterable[str],
    df: pd.DataFrame,
    *,
    for_graph: bool = False,
    cat_cutoff: int = 5,
) -> tuple[list[str], list[str]]:
    """Split ``columns`` into ``(numeric, categorical)`` lists."""
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for col in columns:
        if is_num_check(col, df, for_graph=for_graph, cat_cutoff=cat_cutoff):
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return num_cols, cat_cols


def make_cat_order(cat_order_list: Sequence[str]) -> Callable[[object], int]:
    """Build a sort-key that orders categories by ``cat_order_list`` then length.

    Values present in ``cat_order_list`` (compared lower-case) sort by their
    position there; anything else sorts after, by string length. This ordering
    only affects how categories are displayed on graphs.
    """
    lowered = [c.lower() for c in cat_order_list]

    def cat_order(value: object) -> int:
        key = str(value).lower()
        if key in lowered:
            return lowered.index(key)
        return len(lowered) + len(str(value))

    return cat_order
