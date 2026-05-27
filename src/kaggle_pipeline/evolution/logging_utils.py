"""Verbosity levels and thread-safe logging helpers for the orchestrator.

Verbosity (0..4) controls three things: the Python logging level, how much
``KagglePipeline.print_state`` emits, and whether debug detail is logged. All
output goes through the standard ``logging`` module (thread-safe), never raw
``print`` -- the pipeline routes everything through ``self.log`` / ``print_state``.
"""

from __future__ import annotations

import logging

PACKAGE_LOGGER = "kaggle_pipeline.evolution"


class Verbosity:
    """Verbosity levels controlling logging and state printing."""

    SILENT = 0  # nothing routine; only critical errors
    SUMMARY = 1  # one-line batch summary
    NORMAL = 2  # important events: batch start/end, counts, checkpoints, best changes
    DETAILED = 3  # feature/model/mutation summaries, runtime reserve, families
    DEBUG = 4  # gene credit, mutation records, similarity, tracebacks

    ALL = (SILENT, SUMMARY, NORMAL, DETAILED, DEBUG)


# Map verbosity -> logging level for the package logger. SUMMARY keeps INFO so the
# one-line state still shows; routine chatter is gated by the numeric verbosity in
# ``KagglePipeline.log`` rather than only by the logging level.
_LEVELS = {
    Verbosity.SILENT: logging.CRITICAL,
    Verbosity.SUMMARY: logging.INFO,
    Verbosity.NORMAL: logging.INFO,
    Verbosity.DETAILED: logging.INFO,
    Verbosity.DEBUG: logging.DEBUG,
}


def verbosity_to_logging_level(verbosity: int) -> int:
    return _LEVELS.get(verbosity, logging.INFO)


def configure_logging(verbosity: int) -> logging.Logger:
    """Set the package logger's level from ``verbosity`` and return it.

    Attaches a basic stream handler once if none is configured, so a bare script
    still sees output; embedders that pre-configure the logger keep their setup.
    """
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(verbosity_to_logging_level(verbosity))
    if not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    return logger


def format_duration(seconds: float) -> str:
    """Compact ``HhMMmSSs`` style duration for state printouts."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
