"""Configuration: the single place users tune the pipeline per competition."""

from kaggle_pipeline.config.config import DEFAULT_ORDER_LISTS, Config
from kaggle_pipeline.config.environment import (
    ResolvedPaths,
    detect_environment,
    resolve_paths,
)
from kaggle_pipeline.config.loader import load_config

__all__ = [
    "Config",
    "DEFAULT_ORDER_LISTS",
    "load_config",
    "detect_environment",
    "resolve_paths",
    "ResolvedPaths",
]
