"""``KagglePipelineSettings`` -- the orchestration-level configuration.

Wraps the ecosystem's :class:`EvolutionSettings` and adds the runtime budget,
verbosity, checkpointing, parallelism, batch and ensembling knobs the
:class:`~kaggle_pipeline.evolution.pipeline.KagglePipeline` orchestrator needs.
Users override fields directly or via ``KagglePipeline``'s constructor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from kaggle_pipeline.evolution.config.settings import EvolutionSettings
from kaggle_pipeline.preprocessing.encoders import ONEHOT_MAX_CARDINALITY

_HOUR = 60 * 60


def _default_mutation_distribution() -> dict[int, float]:
    return {1: 0.70, 2: 0.20, 3: 0.07, 4: 0.03}


@dataclass
class KagglePipelineSettings:
    """All orchestration knobs for a :class:`KagglePipeline` run."""

    # --- runtime ------------------------------------------------------------
    max_runtime_seconds: float = 12 * _HOUR
    safety_margin_seconds: float = 10 * 60
    checkpoint_time_reserve_seconds: float = 2 * 60
    ensemble_time_reserve_seconds: float = 30 * 60
    finalization_time_reserve_seconds: float = 5 * 60
    # Bootstrap reserve held back for the submission step (refit each ensemble
    # member on the FULL train data, predict test, write CSV). Only carved out
    # of the training window when ``make_submission_on_run`` is True. Once the
    # pipeline has measured per-model timings it overwrites the live reserve
    # with a data-aware estimate (see ``_estimated_submission_seconds``), so
    # this value only matters at startup and as a floor.
    submission_time_reserve_seconds: float = 30 * 60

    # --- submission ---------------------------------------------------------
    # When True, ``run()`` writes the submission CSV before returning (using the
    # ``sample_df`` / ``test_df`` passed to ``fit``) and the submission reserve
    # is subtracted from the training budget. When False, the user calls
    # ``make_submission`` themselves and run() reserves nothing for it.
    make_submission_on_run: bool = True
    submission_path: str = "submission.csv"

    # --- run mode -----------------------------------------------------------
    # When False, ``run()`` skips the evolutionary batch loop entirely and goes
    # straight from data prep (including merging any input ecosystems) to
    # finalization/submission. This is what the "blend-only" notebook uses: it
    # combines the parallel training outputs and ships a submission without
    # training any new models. Leave True for the single-thread and
    # parallel-training notebooks. Together with ``make_submission_on_run`` this
    # gives the three notebook modes purely via flags:
    #   single-thread : train_models=True,  make_submission_on_run=True
    #   parallel-train: train_models=True,  make_submission_on_run=False
    #   blend-only    : train_models=False, make_submission_on_run=True
    train_models: bool = True

    # --- verbosity ----------------------------------------------------------
    # Default 3 (DETAILED) mirrors the v1 Config's default ``verbosity="verbose"``.
    verbosity: int = 3

    # --- state saving -------------------------------------------------------
    state_dir: str = "kagglepipeline_state"
    # When ``resume=True`` is passed to ``fit``, the pipeline loads from
    # ``state_dir`` if it already holds a checkpoint, else falls back to
    # ``previous_state_dir`` if set, else (on Kaggle) auto-scans
    # ``/kaggle/input/*/<state_dir basename>/checkpoints/`` for the newest
    # checkpoint. New checkpoints are always written to ``state_dir`` -- the
    # previous-state dir is read-only on Kaggle.
    previous_state_dir: str | None = None
    checkpoint_every_batch: bool = True
    checkpoint_interval_seconds: float = 10 * 60
    save_after_each_completed_model: bool = False
    keep_last_n_checkpoints: int = 5
    atomic_checkpoints: bool = True

    # --- parallelism --------------------------------------------------------
    # ``None`` auto-detects all cores -- the same intent as the v1 Config's
    # ``n_workers = -1``.
    num_workers: int | None = None
    feature_workers: int | None = None
    model_workers: int | None = None
    thread_backend: str = "threadpool"
    avoid_nested_parallelism: bool = True

    # --- batch training -----------------------------------------------------
    # Defaults that have a v1 ``Config`` counterpart match it so a run behaves like
    # the previous pipeline: models_per_batch <- step_batch_size (32),
    # ensemble_max_models <- ensemble_length (30), cv_splits <- cv_splits (5).
    models_per_batch: int = 32
    feature_candidates_per_batch: int | None = None
    feature_generation_ratio: float = 0.10
    max_active_features: int = 300
    # Generated features may build on other generated features up to this depth;
    # the extra depth/complexity is penalised in the feature-utility formula.
    max_feature_depth: int = 2
    allow_generated_feature_parents: bool = True
    cv_splits: int = 5
    # Cardinality cap above which a categorical is frequency-encoded rather than
    # one-hot (keeps materialized width bounded). Shares the v1 default.
    onehot_max_cardinality: int = ONEHOT_MAX_CARDINALITY
    # Fraction of the training rows (randomly, stratified, sampled) that models are
    # trained/cross-validated on during the search, to evaluate many more candidates
    # per unit time. Ensemble winners are refit on the FULL data at finalization.
    # 1.0 disables subsampling. Default 0.10 (10%).
    search_sample_fraction: float = 0.10

    # --- ensembling ---------------------------------------------------------
    enable_ensembling: bool = True
    ensemble_max_models: int = 30
    ensemble_min_models: int = 2
    ensemble_strategy: str = "greedy"
    reserve_time_for_ensemble: bool = True
    ensemble_use_diversity: bool = True
    ensemble_candidate_min_score: float | None = None

    # --- evolution ----------------------------------------------------------
    mutation_scale: float = 0.20
    mutation_drift: float = 0.00
    preferred_num_mutated_genes_distribution: dict[int, float] = field(
        default_factory=_default_mutation_distribution
    )

    # --- reproducibility ----------------------------------------------------
    seed: int | None = None
    # Seed for the *search subsample* only (which rows the search trains/CVs on),
    # kept separate from ``seed`` (which drives the evolutionary search itself).
    # It defaults to a fixed value -- not ``None`` -- on purpose: parallel
    # notebooks that share data and ``search_sample_seed`` draw the *same*
    # subsample, so their out-of-fold predictions line up row-for-row and can be
    # blended directly when their ecosystems are merged. Vary ``seed`` (or leave
    # it ``None``) per parallel notebook so each explores different models while
    # the subsample stays aligned. If a merged ecosystem's OOF was built on a
    # different subsample, the blender recomputes it on its own subsample.
    search_sample_seed: int = 0

    def __post_init__(self) -> None:
        if self.max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive")
        if self.verbosity not in (0, 1, 2, 3, 4):
            raise ValueError(f"verbosity must be 0..4, got {self.verbosity}")
        if not 0.0 < self.search_sample_fraction <= 1.0:
            raise ValueError(
                f"search_sample_fraction must be in (0, 1], got {self.search_sample_fraction}"
            )
        if self.submission_time_reserve_seconds < 0:
            raise ValueError("submission_time_reserve_seconds must be >= 0")

    # --- derived ------------------------------------------------------------
    def evolution_settings(self) -> EvolutionSettings:
        """Build the matching ecosystem :class:`EvolutionSettings`."""
        return EvolutionSettings(
            max_active_features=self.max_active_features,
            feature_generation_ratio=self.feature_generation_ratio,
            max_feature_depth=self.max_feature_depth,
            allow_generated_feature_parents=self.allow_generated_feature_parents,
            mutation_scale=self.mutation_scale,
            mutation_drift=self.mutation_drift,
            preferred_num_mutated_genes_distribution=dict(
                self.preferred_num_mutated_genes_distribution
            ),
            onehot_max_cardinality=self.onehot_max_cardinality,
            default_random_seed=self.seed,
        )

    def resolved_num_workers(self) -> int:
        return self.num_workers or max(1, os.cpu_count() or 1)

    def resolved_model_workers(self) -> int:
        return self.model_workers or self.resolved_num_workers()

    def resolved_feature_workers(self) -> int:
        return self.feature_workers or self.resolved_num_workers()

    def per_model_n_jobs(self) -> int:
        """Threads each model may use, avoiding CPU oversubscription across workers."""
        if self.avoid_nested_parallelism:
            return 1
        return max(1, (os.cpu_count() or 1) // max(1, self.resolved_model_workers()))
