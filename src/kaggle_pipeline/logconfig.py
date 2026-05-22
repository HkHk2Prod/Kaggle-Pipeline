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

How *much* is shown is a single knob, ``Config.verbosity``, mapped here to a
logging level by :func:`level_for_verbosity`:

* ``"quiet"`` -> ``WARNING``: only warnings and errors (autodetect anomalies,
  a corrupt leaderboard, an exhausted time budget).
* ``"normal"`` -> ``INFO`` (the default): stage progress, the autodetected
  fields, the prune summary, the chosen ensemble's score and the submission path.
* ``"verbose"`` -> ``DEBUG``: everything above plus the per-model scores, the
  full leaderboard after each step, the encoding plan and the submission preview.
"""

from __future__ import annotations

import logging
import sys

PACKAGE_LOGGER = "kaggle_pipeline"

# The user-facing verbosity names (set on ``Config.verbosity`` / in YAML) mapped
# to the package logger's level. Single source of truth: ``Config`` validates
# against the keys and the entry points configure the level from them.
VERBOSITY_LEVELS: dict[str, int] = {
    "quiet": logging.WARNING,
    "normal": logging.INFO,
    "verbose": logging.DEBUG,
}
DEFAULT_VERBOSITY = "normal"


def level_for_verbosity(verbosity: str) -> int:
    """Map a :data:`VERBOSITY_LEVELS` name to a logging level.

    Raises :class:`ValueError` on an unknown name. ``Config`` validates the
    value too, but this stays defensive for embedders calling it directly.
    """
    try:
        return VERBOSITY_LEVELS[verbosity]
    except KeyError:
        raise ValueError(
            f"Unknown verbosity {verbosity!r}; expected one of {sorted(VERBOSITY_LEVELS)}."
        ) from None


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
