"""kaggle_pipeline -- a config-driven AutoML pipeline for tabular Kaggle data.

Typical usage from a Kaggle notebook::

    from kaggle_pipeline import Config, run

    cfg = Config(competition="playground-series-s6e4", target="Irrigation_Need", ...)
    run(cfg)  # loads data, searches models, ensembles, writes submission.csv
"""

from __future__ import annotations

from kaggle_pipeline.analysis import analyze
from kaggle_pipeline.config import Config, load_config
from kaggle_pipeline.context import PipelineContext, build_context
from kaggle_pipeline.pipeline import build_pipeline, predict, run

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_config",
    "run",
    "analyze",
    "predict",
    "build_pipeline",
    "build_context",
    "PipelineContext",
    "__version__",
]
