"""Configuration: the single place users tune the pipeline per competition."""

from kaggle_pipeline.config.config import (
    DEFAULT_ORDER_LISTS,
    REGRESSION_NOT_IMPLEMENTED,
    Config,
)
from kaggle_pipeline.config.environment import (
    ResolvedPaths,
    autodetect_data_dir,
    detect_environment,
    resolve_paths,
)
from kaggle_pipeline.config.loader import load_config

__all__ = [
    "Config",
    "DEFAULT_ORDER_LISTS",
    "REGRESSION_NOT_IMPLEMENTED",
    "load_config",
    "detect_environment",
    "resolve_paths",
    "autodetect_data_dir",
    "ResolvedPaths",
]
