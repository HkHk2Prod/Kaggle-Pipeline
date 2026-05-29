"""Snapshot/restore helpers for the :class:`KagglePipeline` ecosystem state.

Module functions, not a class: each one takes only what it needs (controller,
state, settings) so they stay easy to test without the full pipeline. The
pipeline's public ``save_state``/``load_state``/``checkpoint`` methods are
thin wrappers around these.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kaggle_pipeline.evolution.ecosystem.resume import find_previous_state_dir
from kaggle_pipeline.evolution.ecosystem.serialization import EcosystemSerializer
from kaggle_pipeline.evolution.ecosystem.state import PIPELINE_VERSION, EcosystemState
from kaggle_pipeline.evolution.ecosystem.summary import build_ecosystem_summary, format_summary

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kaggle_pipeline.evolution.config import KagglePipelineSettings
    from kaggle_pipeline.evolution.controllers.evolution_controller import EvolutionController
    from kaggle_pipeline.evolution.ensemble.manager import EnsembleResult
    from kaggle_pipeline.evolution.runtime import RuntimeManager


def build_ecosystem_state(
    controller: EvolutionController,
    settings: KagglePipelineSettings,
    *,
    ensemble_result: EnsembleResult | None,
    score_history: list[dict[str, Any]],
    runtime_history: list[dict[str, Any]],
) -> EcosystemState:
    """Snapshot a controller + its surrounding bookkeeping into an EcosystemState."""
    return EcosystemState(
        config_snapshot=asdict(settings),
        batch_index=controller.registry.current_batch,
        registry=controller.registry,
        population=controller.population,
        oof_store=controller.oof_store,
        rng_state=dict(controller.rng.bit_generator.state),
        score_history=list(score_history),
        runtime_history=list(runtime_history),
        ensemble_state=ensemble_result.to_serializable() if ensemble_result else None,
    )


def check_pipeline_version(state: EcosystemState, *, strict: bool) -> str | None:
    """Return a mismatch message (or None) and raise when ``strict``."""
    if state.pipeline_version == PIPELINE_VERSION:
        return None
    message = f"checkpoint pipeline_version {state.pipeline_version} != {PIPELINE_VERSION}"
    if strict:
        raise ValueError(message)
    return message


def rebuild_controller_from_state(
    state: EcosystemState,
    *,
    settings: KagglePipelineSettings,
    families: Any,
    n_splits: int,
    seed: int | None,
) -> EvolutionController:
    """Reconstruct an EvolutionController around a restored ecosystem state."""
    from kaggle_pipeline.evolution.controllers.evolution_controller import EvolutionController

    controller = EvolutionController(
        settings.evolution_settings(),
        registry=state.registry,
        population=state.population,
        families=families,
        n_splits=n_splits,
        seed=seed,
    )
    controller.oof_store = state.oof_store
    if state.rng_state is not None:
        controller.rng.bit_generator.state = state.rng_state
    return controller


def pick_resume_serializer(
    serializer: EcosystemSerializer, settings: KagglePipelineSettings
) -> EcosystemSerializer | None:
    """Find the dir to load from; prefer the local state_dir if it has data.

    Falls back to a previous run's directory when configured. Returns ``None``
    when no checkpoint can be located. The fallback serializer is constructed
    read-only -- it should not inherit atomic-write semantics from the live
    write target.
    """
    if serializer.latest_path() is not None:
        return serializer
    prev = find_previous_state_dir(
        previous_state_dir=settings.previous_state_dir,
        state_dir_name=Path(settings.state_dir).name,
    )
    if prev is None:
        return None
    return EcosystemSerializer(prev, keep_last_n=settings.keep_last_n_checkpoints)


def format_loaded_ecosystem(
    controller: EvolutionController,
    runtime: RuntimeManager | None,
    state: EcosystemState,
    verbosity: int,
) -> str | None:
    """Format the post-restore ecosystem summary at the caller's verbosity."""
    summary = build_ecosystem_summary(
        controller.registry,
        controller.population,
        runtime,
        batch_index=state.batch_index,
        last_batch=None,
        ensemble=state.ensemble_state,
    )
    return format_summary(summary, verbosity) or None
