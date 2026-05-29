"""Verbosity-tiered logging helpers used by :class:`KagglePipeline`.

The pipeline keeps a ``self.log(message, level=...)`` callable that gates on
the configured verbosity. The functions in this module compose multi-tier log
output (e.g. "summary line, then detail line, then debug detail") and emit
through that callback rather than owning their own logger. Splitting them out
keeps ``pipeline.py`` focused on the run loop and makes the formatting trivial
to unit-test by passing in a list-appending callback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from kaggle_pipeline.evolution.logging_utils import Verbosity

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kaggle_pipeline.evolution.config import KagglePipelineSettings
    from kaggle_pipeline.evolution.controllers.evolution_controller import (
        BatchSummary,
        EvolutionController,
    )
    from kaggle_pipeline.evolution.runtime import RuntimeManager

LogFn = Callable[..., None]


def log_runtime_budget(
    log_fn: LogFn,
    *,
    runtime: RuntimeManager,
    settings: KagglePipelineSettings,
) -> None:
    """Print the carved-up runtime budget at DETAILED+ so reserves are visible."""
    if settings.verbosity < Verbosity.DETAILED:
        return
    sub_reserve = runtime.submission_time_reserve_seconds
    log_fn(
        f"runtime budget: total={settings.max_runtime_seconds:.0f}s, "
        f"safety={settings.safety_margin_seconds:.0f}s, "
        f"checkpoint={settings.checkpoint_time_reserve_seconds:.0f}s, "
        f"finalization={settings.finalization_time_reserve_seconds:.0f}s, "
        f"ensemble={settings.ensemble_time_reserve_seconds:.0f}s"
        f"{' (off)' if not settings.enable_ensembling else ''}, "
        f"submission={sub_reserve:.0f}s"
        f"{' (off)' if not settings.make_submission_on_run else ' (bootstrap)'}, "
        f"training_window={runtime.remaining_training_seconds():.0f}s",
        level=Verbosity.DETAILED,
    )


def log_feature_generation(
    log_fn: LogFn,
    summary: BatchSummary,
    *,
    controller: EvolutionController | None,
    verbosity: int,
) -> None:
    """Report newly generated feature columns, scaled by verbosity.

    SUMMARY+: a one-line count; DETAILED+: the new column names; DEBUG: the
    names with their depth so deeper (costlier) compositions are visible.
    """
    names = summary.generated_feature_names
    if not names:
        return
    log_fn(
        f"features: +{len(names)} new ({summary.n_features_active} active)",
        level=Verbosity.SUMMARY,
    )
    log_fn("  new feature columns: " + ", ".join(names), level=Verbosity.DETAILED)
    if verbosity < Verbosity.DEBUG or controller is None:
        return
    by_name = {f.human_name: f for f in controller.registry.all_features()}
    detail = [
        f"{name}(depth={feature.depth}, util={feature.utility:.3f})"
        for name in names
        for feature in (by_name.get(name),)
        if feature is not None
    ]
    if detail:
        log_fn("  new feature detail: " + "; ".join(detail), level=Verbosity.DEBUG)
