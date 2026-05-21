"""Package logging: a single configurable channel for the pipeline's output.

The pipeline reports its progress (autodetected fields, the leaderboard after
each step, the chosen ensemble, ...) as it runs. That output goes through the
``kaggle_pipeline`` logger rather than bare ``print`` so a host application can
silence or redirect it, while the notebook/CLI workflow still sees it by default.

Every module logs via ``logging.getLogger(__name__)``; because those names are
children of ``kaggle_pipeline`` they inherit the handler and level configured
here. The entry points (:func:`~kaggle_pipeline.pipeline.run`,
:func:`~kaggle_pipeline.analysis.analyze` and the CLI) call
:func:`configure_logging` so output appears out of the box; embedders can call
it themselves (or configure the ``kaggle_pipeline`` logger directly) instead.
"""

from __future__ import annotations

import logging
import sys

PACKAGE_LOGGER = "kaggle_pipeline"


def configure_logging(level: int = logging.INFO, *, force: bool = False) -> logging.Logger:
    """Attach a plain stdout handler to the package logger and return it.

    Idempotent: if the logger already has handlers this is a no-op unless
    ``force`` is set. The formatter prints the bare message (no level/timestamp
    noise) to match the pipeline's original ``print`` output.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    if logger.handlers and not force:
        logger.setLevel(level)
        return logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    # Don't double-emit through the root logger if the host also configured one.
    logger.propagate = False
    return logger
