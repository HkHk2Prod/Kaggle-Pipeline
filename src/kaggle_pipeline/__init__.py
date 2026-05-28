"""kaggle_pipeline -- a config-driven AutoML pipeline for tabular Kaggle data.

Typical usage from a Kaggle notebook::

    from kaggle_pipeline import Config
    from kaggle_pipeline.evolution import KagglePipeline, KagglePipelineSettings

    cfg = Config()  # autodetects competition/target/task/scoring from the data
    KagglePipeline(KagglePipelineSettings()).fit(train_df, test_df=test_df)
"""

from __future__ import annotations

from kaggle_pipeline.analysis import analyze
from kaggle_pipeline.config import Config, load_config

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_config",
    "analyze",
    "__version__",
]
