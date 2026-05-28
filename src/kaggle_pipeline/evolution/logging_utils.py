"""Verbosity levels and thread-safe logging helpers for the orchestrator.

Verbosity (0..4) controls three things: the Python logging level, how much
``KagglePipeline.print_state`` emits, and whether debug detail is logged. All
output goes through the standard ``logging`` module (thread-safe), never raw
``print`` -- the pipeline routes everything through ``self.log`` / ``print_state``.
"""

from __future__ import annotations

import logging


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
    """Configure the shared ``kaggle_pipeline`` logger from an integer verbosity.

    Delegates to :func:`kaggle_pipeline.logconfig.configure_logging` so the
    evolutionary layer and the v1 pipeline share one handler on one logger
    hierarchy: the evolution modules log through children of ``kaggle_pipeline``
    and inherit it. Maintaining a second handler on the ``kaggle_pipeline.evolution``
    child (as before) double-emitted records -- once in each layer's format --
    whenever both configured logging in the same process. The integer 0..4
    :class:`Verbosity` scale is mapped to a logging level here.
    """
    from kaggle_pipeline.logconfig import configure_logging as configure_package_logging

    return configure_package_logging(verbosity_to_logging_level(verbosity))


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
