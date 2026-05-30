"""The :class:`KagglePipeline` -- the main orchestration class.

KagglePipeline owns the ecosystem state, feature registry, model population,
runtime manager, thread pools, the batch loop, checkpointing and optional ensemble
finalization. Each batch may generate and score features, create new model genomes,
mutate existing models into *child* models, train models in parallel, update
scores and credit, print the ecosystem state, and save a checkpoint.

It is an orchestrator: the algorithms live in the small collaborator classes
(FeatureRegistry, FeatureGenerator, ModelFactory, ModelMutator, ModelTrainer,
CreditAssigner, EvolutionController, EnsembleManager, EcosystemSerializer). The
pipeline respects a 12-hour runtime limit by default, stops launching work before
the deadline, reserves time for ensembling when enabled, and saves final state
before shutting down.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from kaggle_pipeline.evolution.config import KagglePipelineSettings
from kaggle_pipeline.evolution.controllers.evolution_controller import (
    BatchSummary,
    EvolutionController,
)
from kaggle_pipeline.evolution.ecosystem.serialization import EcosystemSerializer
from kaggle_pipeline.evolution.ecosystem.state import EcosystemState
from kaggle_pipeline.evolution.ecosystem.summary import build_ecosystem_summary, format_summary
from kaggle_pipeline.evolution.ensemble.manager import EnsembleManager, EnsembleResult
from kaggle_pipeline.evolution.logging_utils import (
    Verbosity,
    configure_logging,
    format_batch_banner,
    format_phase_banner,
)
from kaggle_pipeline.evolution.models.parameter_spaces import build_default_families
from kaggle_pipeline.evolution.pipeline_log import (
    log_feature_generation,
    log_runtime_budget,
)
from kaggle_pipeline.evolution.prepare import (
    autodetect_problem,
    build_search_sample,
    engineer_features,
    infer_feature_type,
)
from kaggle_pipeline.evolution.runtime import RuntimeManager
from kaggle_pipeline.evolution.runtime_estimator import RuntimeEstimator
from kaggle_pipeline.evolution.scheduler import TaskScheduler
from kaggle_pipeline.evolution.state_io import (
    build_ecosystem_state,
    check_pipeline_version,
    format_loaded_ecosystem,
    pick_resume_serializer,
    rebuild_controller_from_state,
)
from kaggle_pipeline.evolution.submission import (
    SubmissionWriter,
    submission_skip_reason,
    submission_summary_lines,
)
from kaggle_pipeline.evolution.utils.logging import get_logger
from kaggle_pipeline.scoring.metrics import resolve_scoring

logger = get_logger(__name__)


@dataclass
class _ScoringContext:
    """Minimal stand-in for the v1 PipelineContext that CrossValScore reads."""

    scoring_fn: Any
    target_width: int
    target_is_num: bool


class KagglePipeline:
    """Main orchestrator for the evolutionary feature/model ecosystem."""

    def __init__(
        self,
        settings: KagglePipelineSettings | None = None,
        *,
        max_runtime_hours: float | None = None,
        verbosity: int | None = None,
        enable_ensembling: bool | None = None,
        num_workers: int | None = None,
        state_dir: str | None = None,
        seed: int | None = None,
        cv_splits: int | None = None,
        models_per_batch: int | None = None,
    ):
        self.settings = settings or KagglePipelineSettings()
        if max_runtime_hours is not None:
            self.settings.max_runtime_seconds = max_runtime_hours * 3600
        for name, value in (
            ("verbosity", verbosity),
            ("enable_ensembling", enable_ensembling),
            ("num_workers", num_workers),
            ("state_dir", state_dir),
            ("seed", seed),
            ("cv_splits", cv_splits),
            ("models_per_batch", models_per_batch),
        ):
            if value is not None:
                setattr(self.settings, name, value)

        configure_logging(self.settings.verbosity)
        self.families = build_default_families()
        self.serializer = EcosystemSerializer(
            self.settings.state_dir,
            keep_last_n=self.settings.keep_last_n_checkpoints,
            atomic=self.settings.atomic_checkpoints,
        )
        self.scheduler = TaskScheduler(
            model_workers=self.settings.resolved_model_workers(),
            feature_workers=self.settings.resolved_feature_workers(),
        )
        self.runtime: RuntimeManager | None = None
        self.controller: EvolutionController | None = None
        self.ensemble_result: EnsembleResult | None = None

        # Data / problem state, set in fit().
        self._train_features: pd.DataFrame | None = None  # full training features
        self._search_features: pd.DataFrame | None = None  # subsample used during search
        self._search_y: np.ndarray | None = None
        self._test_features: pd.DataFrame | None = None
        self._test_ids: Any = None
        self._test_has_ids: bool = False  # did the test set carry a real id column?
        self._sample: pd.DataFrame | None = None  # sample_submission template
        self._y: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._task = "classification"
        self._prediction_aim = "probability"
        self._scoring_ctx: _ScoringContext | None = None
        self._id_col = "id"

        # Bookkeeping.
        self._last_batch: BatchSummary | None = None
        self._score_history: list[dict[str, Any]] = []
        self._runtime_history: list[dict[str, Any]] = []
        self._last_checkpoint_time = time.monotonic()

    # --- logging ------------------------------------------------------------
    def log(self, message: str, *, level: int = Verbosity.NORMAL) -> None:
        if self.settings.verbosity <= Verbosity.SILENT or self.settings.verbosity < level:
            return
        if level >= Verbosity.DEBUG:
            logger.debug(message)
        else:
            logger.info(message)

    def _log_phase(self, name: str) -> None:
        """Emit a banner marking a top-level phase boundary.

        Banners go through SUMMARY so they're visible on any non-silent run --
        they're the only structural cue that tells "preparation done, training
        starting" apart from the per-batch lines.
        """
        self.log(format_phase_banner(name), level=Verbosity.SUMMARY)

    def _log_batch_separator(self, batch: int, *, end: bool = False) -> None:
        """Emit a separator line bracketing a batch's logs."""
        self.log(format_batch_banner(batch, end=end), level=Verbosity.SUMMARY)

    # --- fit / run ----------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        target: str | None = None,
        test_df: pd.DataFrame | None = None,
        *,
        task: str | None = None,
        scoring: str | None = None,
        prediction_aim: str | None = None,
        id_col: str = "id",
        sample_df: pd.DataFrame | None = None,
        feature_expressions: list[str] | None = None,
        feature_types: dict[str, str] | None = None,
        resume: bool = False,
    ) -> KagglePipeline:
        """Prepare data + ecosystem, then run the full pipeline (returns self).

        ``target``/``task``/``scoring``/``prediction_aim`` left as ``None`` are
        autodetected from the data using the v1 resolver (same rules as
        ``kaggle_pipeline.run``). ``sample_df`` is the competition's
        ``sample_submission`` -- when given, the written submission matches its
        column names and structure. ``feature_expressions`` are ``df.eval`` strings
        that add engineered columns (no v1 categorical *encodings* are applied --
        encoding is model-specific).
        """
        self._sample = sample_df
        self._prepare(
            train_df,
            target,
            test_df,
            task=task,
            scoring=scoring,
            prediction_aim=prediction_aim,
            id_col=id_col,
            feature_expressions=feature_expressions,
            feature_types=feature_types,
            resume=resume,
        )
        self.run()
        return self

    def _prepare(
        self,
        train_df,
        target,
        test_df,
        *,
        task,
        scoring,
        prediction_aim,
        id_col,
        feature_expressions,
        feature_types,
        resume,
    ) -> None:
        self._log_phase("preparation")
        self._id_col = id_col
        # Autodetect on the RAW frame (before engineering, so the "last non-id
        # column" target heuristic is not fooled by new columns).
        target, task, scoring, self._prediction_aim = autodetect_problem(
            train_df, target, task, scoring, prediction_aim, id_col
        )
        self._task = task

        train_eng = engineer_features(train_df, feature_expressions)
        drop = [c for c in (target, id_col) if c in train_eng.columns]
        self._train_features = train_eng.drop(columns=drop)
        feature_cols = list(self._train_features.columns)

        y_raw = train_eng[target].to_numpy()
        classification = task == "classification"
        if classification:
            self._classes, y = np.unique(y_raw, return_inverse=True)
            target_width = int(self._classes.size)
        else:
            self._classes, y = None, y_raw.astype(float)
            target_width = 1
        self._y = y
        self._scoring_ctx = _ScoringContext(
            scoring_fn=resolve_scoring(scoring),
            target_width=target_width,
            target_is_num=not classification,
        )

        if test_df is not None:
            test_eng = engineer_features(test_df, feature_expressions)
            self._test_has_ids = id_col in test_eng.columns
            self._test_ids = (
                test_eng[id_col].to_numpy() if self._test_has_ids else np.arange(len(test_eng))
            )
            self._test_features = test_eng.drop(columns=[id_col], errors="ignore")

        self._search_features, self._search_y, used_subsample = build_search_sample(
            self._train_features,
            self._y,
            task,
            fraction=self.settings.search_sample_fraction,
            cv_splits=self.settings.cv_splits,
            seed=self.settings.seed,
        )
        if not used_subsample and 0.0 < self.settings.search_sample_fraction < 1.0:
            self.log(
                f"search subsample skipped (n={len(self._train_features)}, "
                f"frac={self.settings.search_sample_fraction}); using full data",
                level=Verbosity.NORMAL,
            )

        originals = [
            (c, infer_feature_type(c, self._train_features, feature_types)) for c in feature_cols
        ]
        evo_settings = self.settings.evolution_settings()
        self.controller = EvolutionController(
            evo_settings,
            families=self.families,
            n_splits=self.settings.cv_splits,
            seed=self.settings.seed,
        )
        self.controller.initialize_features(
            originals, eval_frame=self._search_features, y=self._search_y, task=task
        )
        # Bind y on the population so individual-score recomputers (residual
        # correlation, etc.) can run lazily when a leaderboard touches a
        # missing score on a freshly registered genome.
        self.controller.population.set_search_target(self._search_y)
        if resume:
            self._resume_latest()
            # Resumed population's genomes are unpickled without their score
            # recomputers (closures are dropped from the pickle); re-bind them
            # against this run's population instance.
            self.controller.population.set_search_target(self._search_y)
            self.controller.population.wire_all_score_recomputers()
        self.log(
            f"prepared: target={target!r} task={task} scoring={scoring} | "
            f"{len(feature_cols)} features, {target_width}-class, "
            f"{len(train_df)} rows (search on {len(self._search_y)})",
            level=Verbosity.NORMAL,
        )

    def _estimator(self) -> RuntimeEstimator:
        return RuntimeEstimator(
            self.settings, self.controller, model_workers=self.scheduler.model_workers
        )

    def run(self) -> dict[str, Any]:
        """Run the batch loop under the runtime budget, then finalize."""
        if self.controller is None or self._scoring_ctx is None:
            raise RuntimeError("call fit() before run()")
        # The submission reserve is gated by ``make_submission_on_run`` (0 = no
        # reserve, no submission). The orchestrator updates it after every batch
        # with a measured-time estimate -- this initial value is just the
        # bootstrap default used before any per-model timings exist.
        bootstrap_submission_reserve = (
            self.settings.submission_time_reserve_seconds
            if self.settings.make_submission_on_run
            else 0.0
        )
        self.runtime = RuntimeManager(
            max_runtime_seconds=self.settings.max_runtime_seconds,
            safety_margin_seconds=self.settings.safety_margin_seconds,
            checkpoint_time_reserve_seconds=self.settings.checkpoint_time_reserve_seconds,
            ensemble_time_reserve_seconds=self.settings.ensemble_time_reserve_seconds,
            finalization_time_reserve_seconds=self.settings.finalization_time_reserve_seconds,
            submission_time_reserve_seconds=bootstrap_submission_reserve,
            enable_ensembling=self.settings.enable_ensembling,
        )
        self._last_checkpoint_time = time.monotonic()
        self._log_phase("training")
        log_runtime_budget(self.log, runtime=self.runtime, settings=self.settings)
        try:
            while not self.runtime.should_stop_training():
                # Always run at least one batch so a fresh run makes progress before
                # any timing history exists; afterwards gate on the estimate.
                first_batch = self.controller.registry.current_batch == 0
                batch_estimate = self._estimator().batch_seconds()
                if not first_batch and not self.runtime.can_start_batch(batch_estimate):
                    self.log(
                        f"stopping: not enough time for another batch "
                        f"(estimate={batch_estimate:.0f}s, "
                        f"remaining_training={self.runtime.remaining_training_seconds():.0f}s)",
                        level=Verbosity.NORMAL,
                    )
                    break
                next_batch = self.controller.registry.current_batch + 1
                self._log_batch_separator(next_batch)
                self.log(
                    f"batch start: estimate={batch_estimate:.0f}s, "
                    f"remaining_training={self.runtime.remaining_training_seconds():.0f}s",
                    level=Verbosity.DETAILED,
                )
                summary = self.run_batch()
                self._apply_correlation_penalty()
                self._last_batch = summary
                self._record_history(summary)
                self._refresh_submission_reserve()
                if self.settings.checkpoint_every_batch:
                    self.checkpoint(reason="batch_complete")
                self.print_state()
                self._log_batch_separator(summary.batch, end=True)
            self.checkpoint(reason="training_finished")
            self._log_phase("finalization")
            self._finalize()
            self._maybe_make_submission()
            self._log_compute_waste_summary()
            self.checkpoint(reason="final")
        except KeyboardInterrupt:  # graceful: save what we have
            logger.warning("interrupted; checkpointing before exit")
            self.checkpoint(reason="interrupted")
            raise
        finally:
            self.shutdown()
        return self.summarize_state()

    def run_batch(self) -> BatchSummary:
        """Run a single batch (parallel model training when workers > 1)."""
        controller, scoring_ctx, y, train = self._require_ready()
        assert self.runtime is not None
        runtime = self.runtime
        estimator = self._estimator()
        summary = controller.run_batch(
            train_frame=train,
            scoring_ctx=cast(Any, scoring_ctx),
            y=y,
            n_models=self.settings.models_per_batch,
            task=self._task,
            promote=True,
            executor=self.scheduler.model_pool(),
            should_continue=lambda rt=runtime, est=estimator: rt.can_start_model_training(
                est.model_seconds()
            ),
        )
        log_feature_generation(
            self.log,
            summary,
            controller=self.controller,
            verbosity=self.settings.verbosity,
        )
        return summary

    def _require_ready(self):
        """Return ``(controller, scoring_ctx, search_y, search_features)``.

        The search subsample is what batches train on and what OOF is scored
        against; finalization (``predict``) uses the full data instead.
        """
        if (
            self.controller is None
            or self._scoring_ctx is None
            or self._search_y is None
            or self._search_features is None
        ):
            raise RuntimeError("call fit() before running the pipeline")
        return self.controller, self._scoring_ctx, self._search_y, self._search_features

    # --- finalization -------------------------------------------------------
    def _finalize(self) -> None:
        if not self.settings.enable_ensembling:
            self.log("ensembling disabled; best single model is final", level=Verbosity.NORMAL)
            return
        if not self.runtime or not self.runtime.has_time_for_ensemble():
            remaining = self.runtime.remaining_finalization_seconds() if self.runtime else 0.0
            reserve = self.settings.finalization_time_reserve_seconds
            self.log(
                f"not enough time for ensembling; using best single model "
                f"(remaining={remaining:.0f}s, finalization_reserve={reserve:.0f}s)",
                level=Verbosity.SUMMARY,
            )
        self.ensemble()

    def _maybe_make_submission(self) -> Path | None:
        """Write ``submission.csv`` from inside ``run()`` when the flag is set.

        Skipped (with a log line) when the flag is off, when no test data was
        passed to ``fit``, or when the run is already past the reserved
        submission window. Errors are caught: a failed auto-submission must
        not bring down a run whose checkpoint already exists.
        """
        skip = submission_skip_reason(
            make_submission_on_run=self.settings.make_submission_on_run,
            has_test_features=self._test_features is not None,
            runtime=self.runtime,
        )
        if skip is not None:
            if skip:
                self.log(skip, level=Verbosity.SUMMARY)
            return None
        try:
            self._log_phase("submission")
            self.log(
                f"writing submission (reserve={self.runtime.submission_time_reserve_seconds:.0f}s, "
                f"remaining={self.runtime.remaining_submission_seconds():.0f}s)"
                if self.runtime is not None
                else "writing submission",
                level=Verbosity.DETAILED,
            )
            return self.make_submission(self.settings.submission_path)
        except Exception as exc:  # noqa: BLE001 - the checkpoint above is what matters
            logger.exception("auto-submission failed: %s", exc)
            self.log(f"auto-submission failed: {exc}", level=Verbosity.SUMMARY)
            return None

    def ensemble(self) -> EnsembleResult:
        """Build the ensemble from the current population (OOF-based)."""
        controller, scoring_ctx, y, _ = self._require_ready()
        manager = EnsembleManager(self.settings)
        runtime = self.runtime
        time_left = None
        if runtime is not None:
            time_left = lambda rt=runtime: rt.remaining_finalization_seconds() > 5.0  # noqa: E731
        self.ensemble_result = manager.build(
            controller.population,
            controller.oof_store,
            y,
            scoring_ctx.scoring_fn,
            time_left=time_left,
        )
        self.log(
            f"ensemble: {self.ensemble_result.status} "
            f"({self.ensemble_result.n_members} members, score={self.ensemble_result.oof_score})",
            level=Verbosity.NORMAL,
        )
        return self.ensemble_result

    def predict(self, test_df: pd.DataFrame | None = None) -> np.ndarray:
        """Predict test data with the ensemble (refits members on FULL train data)."""
        from concurrent.futures import ThreadPoolExecutor

        controller, _, _, _ = self._require_ready()
        assert self._train_features is not None and self._y is not None
        if self.ensemble_result is None:
            self.ensemble()
        assert self.ensemble_result is not None
        test = (
            self._test_features
            if test_df is None
            else test_df.drop(columns=[self._id_col], errors="ignore")
        )
        if test is None:
            raise ValueError("no test data provided to predict()")
        manager = EnsembleManager(self.settings)
        # Use a fresh, scoped pool for the submission refit instead of the
        # long-running batch pool: ``make_submission`` (which calls this) is
        # often invoked after ``run()`` 's finally block has already shut the
        # scheduler down, and an already-shutdown pool rejects ``submit``.
        n_workers = self.settings.resolved_model_workers()
        if n_workers <= 1:
            executor: Any = None
        else:
            executor = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="evo-submit")
        try:
            return manager.predict(
                self.ensemble_result,
                trainer=controller.trainer,
                population=controller.population,
                train_frame=self._train_features,  # winners refit on the full data
                y=self._y,
                test_frame=test,
                task=self._task,
                seed=self.settings.seed,
                executor=executor,
            )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    def _submission_writer(self) -> SubmissionWriter:
        return SubmissionWriter(
            task=self._task,
            classes=self._classes,
            prediction_aim=self._prediction_aim,
            id_col=self._id_col,
            test_ids=self._test_ids,
            test_has_ids=self._test_has_ids,
        )

    def make_submission(
        self,
        path: str | Path = "submission.csv",
        *,
        sample_df: pd.DataFrame | None = None,
        target_col: str = "target",
    ) -> Path:
        """Predict and write a submission CSV.

        When a ``sample_submission`` is available (``sample_df`` or the one passed
        to ``fit``), the output matches its column names and structure and the
        target is decoded per ``prediction_aim`` (positive-class probability,
        per-class probability columns, or arg-max label). Otherwise it falls back to
        an ``id,target`` file.
        """
        predictions = self.predict()
        sample = sample_df if sample_df is not None else self._sample
        frame = self._submission_writer().build_frame(predictions, sample, target_col=target_col)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        self.log(
            f"submission written to {out} (columns {list(frame.columns)})", level=Verbosity.NORMAL
        )
        self._log_submission_summary(out, frame, predictions)
        return out

    def _log_submission_summary(
        self, path: Path, frame: pd.DataFrame, predictions: np.ndarray
    ) -> None:
        """Print the submission's OOF score, error and ensemble composition.

        Goes at SUMMARY so the headline numbers stay visible at low verbosity;
        a per-member breakdown is added at NORMAL+ so the user always sees what
        actually shipped without forcing DEBUG-level chatter for the rest of
        the run.
        """
        lookup = None
        if self.controller is not None:
            population = self.controller.population

            def lookup(mid: str, _pop=population):
                try:
                    return _pop.get(mid)
                except KeyError:
                    return None

        lines, composition = submission_summary_lines(
            path,
            frame,
            predictions,
            ensemble_result=self.ensemble_result,
            population_lookup=lookup,
        )
        for message, level in lines:
            self.log(message, level=level)
        if composition is not None:
            self.log(composition, level=Verbosity.NORMAL)

    # --- state --------------------------------------------------------------
    def summarize_state(self) -> dict[str, Any]:
        assert self.controller is not None
        return build_ecosystem_summary(
            self.controller.registry,
            self.controller.population,
            self.runtime,
            batch_index=self.controller.registry.current_batch,
            last_batch=self._last_batch,
            ensemble=self.ensemble_result.to_serializable() if self.ensemble_result else None,
        )

    def print_state(self, detail_level: int | None = None) -> None:
        if self.controller is None:
            return
        level = self.settings.verbosity if detail_level is None else detail_level
        if level <= Verbosity.SILENT:
            return
        text = format_summary(self.summarize_state(), level)
        if text:
            logger.info("ecosystem state\n%s", text)

    def _ecosystem_state(self) -> EcosystemState:
        assert self.controller is not None
        return build_ecosystem_state(
            self.controller,
            self.settings,
            ensemble_result=self.ensemble_result,
            score_history=self._score_history,
            runtime_history=self._runtime_history,
        )

    def save_state(self, path: str | Path | None = None, *, reason: str | None = None) -> Path:
        """Save the full ecosystem state. ``path`` overrides the default state dir."""
        state = self._ecosystem_state()
        serializer = (
            self.serializer
            if path is None
            else EcosystemSerializer(path, keep_last_n=self.settings.keep_last_n_checkpoints)
        )
        return serializer.save(state, reason=reason, summary=self.summarize_state())

    def checkpoint(self, reason: str | None = None) -> Path | None:
        if self.controller is None:
            return None
        out = self.save_state(reason=reason)
        self._last_checkpoint_time = time.monotonic()
        self.log(f"checkpoint saved ({reason}) -> {out}", level=Verbosity.NORMAL)
        return out

    def load_state(
        self,
        path: str | Path | None = None,
        *,
        strict: bool = False,
        serializer: EcosystemSerializer | None = None,
    ) -> EcosystemState:
        """Load a saved ecosystem state and rebuild the controller around it.

        ``serializer`` overrides the default (local ``state_dir``) loader -- used
        for warm-starting from a previous run's directory while continuing to
        write new checkpoints into ``self.serializer``'s local dir.
        """
        state = (serializer or self.serializer).load(path)
        mismatch = check_pipeline_version(state, strict=strict)
        if mismatch:
            logger.warning(mismatch)

        self.controller = rebuild_controller_from_state(
            state,
            settings=self.settings,
            families=self.families,
            n_splits=self.settings.cv_splits,
            seed=self.settings.seed,
        )
        self._score_history = list(state.score_history)
        self._runtime_history = list(state.runtime_history)
        self.log(
            f"restored state: batch={state.batch_index}, "
            f"{len(state.population.all_genomes())} models",
            level=Verbosity.SUMMARY,
        )
        self._log_loaded_ecosystem(state)
        return state

    def _log_loaded_ecosystem(self, state: EcosystemState) -> None:
        """Print the full ecosystem summary right after restoring from a checkpoint."""
        if self.settings.verbosity <= Verbosity.SILENT or self.controller is None:
            return
        text = format_loaded_ecosystem(
            self.controller, self.runtime, state, self.settings.verbosity
        )
        if text:
            logger.info("loaded ecosystem state\n%s", text)

    def restore_from_checkpoint(self, path: str | Path) -> EcosystemState:
        return self.load_state(path)

    def _resume_latest(self) -> None:
        """Merge the latest saved population/registry into the prepared controller."""
        if self.controller is None:
            return
        load_serializer = pick_resume_serializer(self.serializer, self.settings)
        if load_serializer is None:
            self.log(
                "resume requested but no checkpoint found; starting fresh", level=Verbosity.SUMMARY
            )
            return
        if load_serializer is not self.serializer:
            self.log(
                f"resuming from previous run at {load_serializer.state_dir}",
                level=Verbosity.NORMAL,
            )
        assert self.controller._eval_context is not None and self.controller._eval_y is not None
        eval_frame = self.controller._eval_context.frame
        eval_y = self.controller._eval_y
        task = self.controller._task
        self.load_state(serializer=load_serializer)
        assert self.controller is not None
        # Re-attach the feature evaluation context to the restored registry.
        self.controller.initialize_features([], eval_frame=eval_frame, y=eval_y, task=task)

    # --- helpers ------------------------------------------------------------
    def _config_snapshot(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self.settings)

    def _record_history(self, summary: BatchSummary) -> None:
        self._score_history.append({"batch": summary.batch, "best_score": summary.best_score})
        if self.runtime is not None:
            self._runtime_history.append({"batch": summary.batch, **self.runtime.time_summary()})

    def _log_compute_waste_summary(self) -> None:
        """Emit the per-family compute-spend breakdown at end-of-cycle.

        Read-only over the population + ensemble; safe to call after any of
        finalize / submission / interruption. Logged at ``NORMAL`` so it
        appears on a default-verbosity run without flooding ``--quiet`` ones.
        """
        if not self.controller:
            return
        from kaggle_pipeline.evolution.ecosystem.compute_waste import (
            build_compute_waste_summary,
            format_compute_waste_summary,
        )

        summary = build_compute_waste_summary(self.controller.population, self.ensemble_result)
        rendered = format_compute_waste_summary(summary)
        self.log(rendered, level=Verbosity.NORMAL)

    def _apply_correlation_penalty(self) -> None:
        """Subtract a soft penalty from active models too similar to better-scoring peers.

        Threshold + scale live in ``EvolutionSettings``; ``scale <= 0`` disables.
        Operates on residual errors (``oof - y``) so anti-correlated mistakes
        (helpful in a blend) get no penalty. Reapplied every batch so a model's
        penalty drops to 0 when a better peer is removed or its score climbs.
        """
        if not self.controller or self._search_y is None:
            return
        ev = self.controller.population.settings
        scale = ev.correlation_penalty_scale
        if scale <= 0.0:
            return
        affected = self.controller.population.compute_correlation_penalties(
            self._search_y,
            threshold=ev.correlation_penalty_threshold,
            scale=scale,
        )
        if affected:
            self.log(
                f"correlation penalty: {affected} active model(s) "
                f"penalized for residual |r| > {ev.correlation_penalty_threshold:.3f}",
                level=Verbosity.NORMAL,
            )

    def _refresh_submission_reserve(self) -> None:
        """Update the runtime's submission reserve from measured timings.

        No-op when ``make_submission_on_run`` is False (the reserve stays at 0)
        or when nothing has finished yet (the bootstrap default still applies).
        Logs the new reserve at DETAILED so the user can see the estimate
        tracking measured per-model times.
        """
        if not (self.runtime and self.settings.make_submission_on_run):
            return
        estimator = self._estimator()
        previous = self.runtime.submission_time_reserve_seconds
        updated = estimator.submission_seconds()
        self.runtime.submission_time_reserve_seconds = updated
        # Only emit when the value actually changed -- repeating the same number
        # every batch is noise.
        if abs(updated - previous) > 1.0:
            times = estimator.completed_compute_times()
            median = float(np.median(times)) if times else 0.0
            self.log(
                f"submission reserve refined: {previous:.0f}s -> {updated:.0f}s "
                f"(median model={median:.2f}s, "
                f"n_members={self.settings.ensemble_max_models}, "
                f"scale_full={1.0 / max(0.01, self.settings.search_sample_fraction):.1f}x)",
                level=Verbosity.DETAILED,
            )

    def shutdown(self, *, wait: bool = True) -> None:
        """Shut the thread pools down gracefully."""
        self.scheduler.shutdown(wait=wait)

    def best_genome(self):
        return self.controller.best_genome() if self.controller else None
