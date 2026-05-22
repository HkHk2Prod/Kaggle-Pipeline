"""Standalone exploratory data analysis -- fully decoupled from training.

``analyze(config)`` loads the raw data and renders the EDA suite. It does not
build a :class:`~kaggle_pipeline.context.PipelineContext`, fit the pre-training
pipeline, or touch the leaderboard, and the training flow never imports this
module. Run it interactively to explore a competition before training.
"""

from __future__ import annotations

import logging

from kaggle_pipeline.config import Config, resolve_paths
from kaggle_pipeline.data import load_datasets
from kaggle_pipeline.logconfig import configure_logging, level_for_verbosity

logger = logging.getLogger(__name__)


def analyze(config: Config) -> None:
    """Load the raw train/test data and render plots + reports.

    Gated by ``config.run_eda`` (off by default): when False this logs that EDA
    is disabled and returns immediately, without loading data or importing any
    plotting dependency. Set ``run_eda: true`` (or ``cfg.run_eda = True``) to run.
    """
    configure_logging(level=level_for_verbosity(config.verbosity))
    if not config.run_eda:
        logger.info("EDA disabled (run_eda=False); skipping. Set run_eda=True to render plots.")
        return
    paths = resolve_paths(config)
    datasets = load_datasets(config, paths.data_dir)
    # Lazy import keeps matplotlib/seaborn out of the training import path.
    from kaggle_pipeline.eda import run_eda

    run_eda(config, datasets.train, datasets.test)
