"""Logger access for the evolutionary layer.

We reuse the existing ``kaggle_pipeline`` logger hierarchy (configured by
:mod:`kaggle_pipeline.logconfig`) so verbosity flags carry over. Modules ask for
a child logger named after themselves; nothing here configures handlers.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a child of the package logger, e.g. ``get_logger(__name__)``.

    ``name`` is typically a dotted module path under ``kaggle_pipeline.evolution``;
    we hand it straight to :func:`logging.getLogger` so it nests under the
    package logger and inherits its configured level/handlers.
    """
    return logging.getLogger(name)
