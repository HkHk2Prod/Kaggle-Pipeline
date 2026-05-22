"""End-to-end training orchestration: ``run(config)`` ties every stage together.

This is the training entry point a Kaggle notebook (or the CLI) calls. Stages:

    detect env -> load data -> preprocess + build context
    -> search + ensemble -> write submission

Exploratory data analysis is deliberately *not* part of this flow -- it lives in
:func:`kaggle_pipeline.analysis.analyze`, so training never imports plotting
dependencies or renders anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kaggle_pipeline.config import Config, resolve_paths
from kaggle_pipeline.context import PipelineContext, build_context
from kaggle_pipeline.data import load_datasets
from kaggle_pipeline.logconfig import configure_logging, level_for_verbosity
from kaggle_pipeline.submission import write_submission
from kaggle_pipeline.training import run_training


def run(config: Config) -> Path:
    """Run the whole pipeline and return the path to the written submission."""
    ctx, _ = build_pipeline(config)
    y_pred = run_training(ctx)
    return write_submission(ctx, y_pred)


def build_pipeline(config: Config) -> tuple[PipelineContext, object]:
    """Resolve paths, load data and build the training context.

    Returns the fitted :class:`PipelineContext` and the resolved paths. Split out
    from :func:`run` so tests and notebooks can inspect the context (column
    splits, target transforms, ...) without launching the search.
    """
    configure_logging(level=level_for_verbosity(config.verbosity))
    paths = resolve_paths(config)
    datasets = load_datasets(config, paths.data_dir)
    ctx = build_context(config, datasets, paths)
    return ctx, paths


def predict(config: Config) -> np.ndarray:
    """Run training and return raw decoded predictions without writing a file."""
    ctx, _ = build_pipeline(config)
    return run_training(ctx)
