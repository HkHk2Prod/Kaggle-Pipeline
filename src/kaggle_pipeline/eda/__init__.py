"""Exploratory data analysis.

Decoupled from training: invoke via :func:`kaggle_pipeline.analysis.analyze`
(or the ``kaggle-pipeline analyze`` CLI command), never as part of ``run``.
"""

from __future__ import annotations

import pandas as pd

from kaggle_pipeline.config import Config
from kaggle_pipeline.eda.plots import EdaContext, graphs, plot
from kaggle_pipeline.eda.reports import correlation_matrices, print_meta_data
from kaggle_pipeline.preprocessing import (
    CategoricalTyper,
    get_columns,
    get_predictor_names,
    split_num_cat,
)

__all__ = [
    "EdaContext",
    "plot",
    "graphs",
    "print_meta_data",
    "correlation_matrices",
    "run_eda",
]


def run_eda(config: Config, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Fit a display-only categorical typing and render the full EDA suite.

    Operates on the *raw* training frame (before feature engineering), matching
    the notebook: a standalone :class:`CategoricalTyper` orders categories for
    readable graphs without affecting the model-facing preprocessing.
    """
    cat_order_list = [c for group in config.order_lists for c in group]
    typer = CategoricalTyper(cat_cutoff=config.cat_cutoff, cat_order_list=cat_order_list)
    plot_df = typer.fit_transform(train_df)

    columns = list(get_columns(plot_df, config.id_col))
    columns_x = list(get_predictor_names(plot_df, config.target, config.id_col))
    num_cols, cat_cols = split_num_cat(
        columns, plot_df, for_graph=True, cat_cutoff=config.cat_cutoff
    )

    ctx = EdaContext(
        df=plot_df,
        ordered_cats=typer.ordered_cats_,
        columns=columns,
        columns_x=columns_x,
        num_cols=num_cols,
        cat_cols=cat_cols,
        target=list(config.target),
    )

    print_meta_data(train_df, test_df)
    correlation_matrices(train_df, config.target, columns_x, cat_cols)
    graphs(ctx)
