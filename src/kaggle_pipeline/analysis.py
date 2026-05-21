"""Standalone exploratory data analysis -- fully decoupled from training.

``analyze(config)`` loads the raw data and renders the EDA suite. It does not
build a :class:`~kaggle_pipeline.context.PipelineContext`, fit the pre-training
pipeline, or touch the leaderboard, and the training flow never imports this
module. Run it interactively to explore a competition before training.
"""

from __future__ import annotations

from kaggle_pipeline.config import Config, resolve_paths
from kaggle_pipeline.data import load_datasets


def analyze(config: Config) -> None:
    """Load the raw train/test data and render plots + reports."""
    paths = resolve_paths(config)
    datasets = load_datasets(config, paths.data_dir)
    # Lazy import keeps matplotlib/seaborn out of the training import path.
    from kaggle_pipeline.eda import run_eda

    run_eda(config, datasets.train, datasets.test)
