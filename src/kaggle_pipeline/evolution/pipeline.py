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
from kaggle_pipeline.evolution.features.recipe import CATEGORICAL, NUMERIC
from kaggle_pipeline.evolution.logging_utils import Verbosity, configure_logging
from kaggle_pipeline.evolution.models.parameter_spaces import build_default_families
from kaggle_pipeline.evolution.runtime import RuntimeManager
from kaggle_pipeline.evolution.scheduler import TaskScheduler
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
        self._id_col = id_col
        # 1) Autodetect target/task/scoring/prediction_aim on the RAW frame (before
        #    engineering, so the "last non-id column" target heuristic is not fooled
        #    by new columns).
        target, task, scoring, self._prediction_aim = self._autodetect(
            train_df, target, task, scoring, prediction_aim, id_col
        )
        self._task = task

        # 2) Feature engineering: df.eval expressions add columns; no encodings.
        train_eng = self._engineer(train_df, feature_expressions)
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
            test_eng = self._engineer(test_df, feature_expressions)
            self._test_ids = (
                test_eng[id_col].to_numpy()
                if id_col in test_eng.columns
                else np.arange(len(test_eng))
            )
            self._test_features = test_eng.drop(columns=[id_col], errors="ignore")

        # 3) Search subsample: train/score on a random fraction during the search;
        #    ensemble winners are refit on the full data at finalization.
        self._search_features, self._search_y = self._build_search_sample(
            self._train_features, self._y, task
        )

        originals = [(c, self._infer_type(c, feature_types)) for c in feature_cols]
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
        if resume:
            self._resume_latest()
        self.log(
            f"prepared: target={target!r} task={task} scoring={scoring} | "
            f"{len(feature_cols)} features, {target_width}-class, "
            f"{len(train_df)} rows (search on {len(self._search_y)})",
            level=Verbosity.NORMAL,
        )

    def _autodetect(self, train_df, target, task, scoring, prediction_aim, id_col):
        """Fill target/task/scoring/prediction_aim left as None (v1 autodetect rules)."""
        from kaggle_pipeline.config import Config
        from kaggle_pipeline.data.autodetect import resolve_problem_definition

        cfg = Config(
            target=[target] if target else None,
            id_col=[id_col],
            task=task,
            scoring=scoring,
            prediction_aim=prediction_aim,
        )
        resolve_problem_definition(cfg, train_df)
        return cfg.target[0], cfg.task, cfg.scoring, cfg.prediction_aim

    def _engineer(self, df: pd.DataFrame, feature_expressions: list[str] | None) -> pd.DataFrame:
        """Apply ``df.eval`` feature expressions (no encodings)."""
        if not feature_expressions:
            return df
        from kaggle_pipeline.preprocessing.transformers import FeatureEngineer

        return FeatureEngineer(expressions=feature_expressions).fit_transform(df.copy())

    def _build_search_sample(self, features: pd.DataFrame, y: np.ndarray, task: str):
        """Return a (stratified) random subsample for the search, or the full data."""
        frac = self.settings.search_sample_fraction
        n = len(features)
        if not (0.0 < frac < 1.0):
            return features, y
        n_sample = int(round(n * frac))
        min_rows = max(2 * self.settings.cv_splits, 30)
        if n_sample < min_rows or n_sample >= n:
            self.log(
                f"search subsample skipped (n={n}, frac={frac}); using full data",
                level=Verbosity.NORMAL,
            )
            return features, y
        from sklearn.model_selection import train_test_split

        stratify = y if task == "classification" else None
        seed = self.settings.seed
        try:
            idx, _ = train_test_split(
                np.arange(n), train_size=n_sample, random_state=seed, stratify=stratify
            )
        except ValueError:  # a class too rare to stratify -- sample without it
            idx, _ = train_test_split(np.arange(n), train_size=n_sample, random_state=seed)
        idx = np.sort(idx)
        sampled = features.iloc[idx].reset_index(drop=True)
        return sampled, np.asarray(y)[idx]

    def _infer_type(self, column: str, feature_types: dict[str, str] | None) -> str:
        if feature_types and column in feature_types:
            return feature_types[column]
        assert self._train_features is not None
        series = self._train_features[column]
        return NUMERIC if pd.api.types.is_numeric_dtype(series) else CATEGORICAL

    def run(self) -> dict[str, Any]:
        """Run the batch loop under the runtime budget, then finalize."""
        if self.controller is None or self._scoring_ctx is None:
            raise RuntimeError("call fit() before run()")
        self.runtime = RuntimeManager(
            max_runtime_seconds=self.settings.max_runtime_seconds,
            safety_margin_seconds=self.settings.safety_margin_seconds,
            checkpoint_time_reserve_seconds=self.settings.checkpoint_time_reserve_seconds,
            ensemble_time_reserve_seconds=self.settings.ensemble_time_reserve_seconds,
            finalization_time_reserve_seconds=self.settings.finalization_time_reserve_seconds,
            enable_ensembling=self.settings.enable_ensembling,
        )
        self._last_checkpoint_time = time.monotonic()
        self.log("training started", level=Verbosity.NORMAL)
        try:
            while not self.runtime.should_stop_training():
                # Always run at least one batch so a fresh run makes progress before
                # any timing history exists; afterwards gate on the estimate.
                first_batch = self.controller.registry.current_batch == 0
                if not first_batch and not self.runtime.can_start_batch(
                    self._estimated_batch_seconds()
                ):
                    self.log("stopping: not enough time for another batch", level=Verbosity.NORMAL)
                    break
                summary = self.run_batch()
                self._last_batch = summary
                self._record_history(summary)
                if self.settings.checkpoint_every_batch:
                    self.checkpoint(reason="batch_complete")
                self.print_state()
            self.checkpoint(reason="training_finished")
            self._finalize()
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
        summary = controller.run_batch(
            train_frame=train,
            scoring_ctx=cast(Any, scoring_ctx),
            y=y,
            n_models=self.settings.models_per_batch,
            task=self._task,
            promote=True,
            executor=self.scheduler.model_pool(),
            should_continue=lambda rt=runtime: rt.can_start_model_training(
                self._estimated_model_seconds()
            ),
        )
        self._log_feature_generation(summary)
        return summary

    def _log_feature_generation(self, summary: BatchSummary) -> None:
        """Report newly generated feature columns, scaled by verbosity.

        SUMMARY+: a one-line count; DETAILED+: the new column names; DEBUG: the
        names with their depth so deeper (costlier) compositions are visible.
        """
        names = summary.generated_feature_names
        if not names:
            return
        self.log(f"features: +{len(names)} new ({summary.n_features_active} active)", level=Verbosity.SUMMARY)
        self.log("  new feature columns: " + ", ".join(names), level=Verbosity.DETAILED)
        if self.settings.verbosity >= Verbosity.DEBUG and self.controller is not None:
            detail = []
            for name in names:
                feature = self._feature_by_name(name)
                if feature is not None:
                    detail.append(f"{name}(depth={feature.depth}, util={feature.utility:.3f})")
            if detail:
                self.log("  new feature detail: " + "; ".join(detail), level=Verbosity.DEBUG)

    def _feature_by_name(self, human_name: str):
        if self.controller is None:
            return None
        for feature in self.controller.registry.all_features():
            if feature.human_name == human_name:
                return feature
        return None

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
            self.log(
                "not enough time for ensembling; using best single model", level=Verbosity.SUMMARY
            )
        self.ensemble()

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
        return manager.predict(
            self.ensemble_result,
            trainer=controller.trainer,
            population=controller.population,
            train_frame=self._train_features,  # winners refit on the full data
            y=self._y,
            test_frame=test,
            task=self._task,
            seed=self.settings.seed,
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
        if sample is not None:
            frame = self._submission_from_sample(sample, predictions)
        else:
            frame = self._submission_fallback(predictions, target_col)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        self.log(
            f"submission written to {out} (columns {list(frame.columns)})", level=Verbosity.NORMAL
        )
        return out

    def _submission_from_sample(
        self, sample: pd.DataFrame, predictions: np.ndarray
    ) -> pd.DataFrame:
        """Build a submission matching the sample_submission's columns/structure."""
        columns = list(sample.columns)
        id_col = self._id_col if self._id_col in columns else columns[0]
        target_cols = [c for c in columns if c != id_col]
        proba = np.asarray(predictions, dtype=float)

        frame = pd.DataFrame({id_col: self._test_ids})
        if len(target_cols) == 1:
            frame[target_cols[0]] = self._decode_single(proba)
        else:
            # One probability column per class (sample order assumed = class order).
            matrix = proba if proba.ndim == 2 else proba.reshape(-1, 1)
            for i, col in enumerate(target_cols):
                frame[col] = matrix[:, i] if i < matrix.shape[1] else 0.0
        return frame[columns]  # preserve the sample's column order

    def _decode_single(self, proba: np.ndarray) -> np.ndarray:
        """Decode predictions into one submission column per ``prediction_aim``."""
        if self._task != "classification" or self._classes is None:
            return proba.ravel()
        if self._prediction_aim == "category":
            return self._classes[proba.argmax(axis=1)] if proba.ndim == 2 else proba.ravel()
        # probability
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]  # positive class
        if proba.ndim == 2:
            return self._classes[proba.argmax(axis=1)]  # multiclass single-col: best-effort label
        return proba.ravel()

    def _submission_fallback(self, predictions: np.ndarray, target_col: str) -> pd.DataFrame:
        decoded = self._decode_single(np.asarray(predictions, dtype=float))
        return pd.DataFrame({self._id_col: self._test_ids, target_col: np.asarray(decoded).ravel()})

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
        rng_state = dict(self.controller.rng.bit_generator.state)
        return EcosystemState(
            config_snapshot=self._config_snapshot(),
            batch_index=self.controller.registry.current_batch,
            registry=self.controller.registry,
            population=self.controller.population,
            oof_store=self.controller.oof_store,
            rng_state=rng_state,
            score_history=list(self._score_history),
            runtime_history=list(self._runtime_history),
            ensemble_state=self.ensemble_result.to_serializable() if self.ensemble_result else None,
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

    def load_state(self, path: str | Path | None = None, *, strict: bool = False) -> EcosystemState:
        """Load a saved ecosystem state and rebuild the controller around it."""
        state = self.serializer.load(path)
        from kaggle_pipeline.evolution.ecosystem.state import PIPELINE_VERSION

        if state.pipeline_version != PIPELINE_VERSION:
            message = f"checkpoint pipeline_version {state.pipeline_version} != {PIPELINE_VERSION}"
            if strict:
                raise ValueError(message)
            logger.warning(message)

        evo_settings = self.settings.evolution_settings()
        self.controller = EvolutionController(
            evo_settings,
            registry=state.registry,
            population=state.population,
            families=self.families,
            n_splits=self.settings.cv_splits,
            seed=self.settings.seed,
        )
        self.controller.oof_store = state.oof_store
        if state.rng_state is not None:
            self.controller.rng.bit_generator.state = state.rng_state
        self._score_history = list(state.score_history)
        self._runtime_history = list(state.runtime_history)
        self.log(
            f"restored state: batch={state.batch_index}, "
            f"{len(state.population.all_genomes())} models",
            level=Verbosity.NORMAL,
        )
        return state

    def restore_from_checkpoint(self, path: str | Path) -> EcosystemState:
        return self.load_state(path)

    def _resume_latest(self) -> None:
        """Merge the latest saved population/registry into the prepared controller."""
        if self.serializer.latest_path() is None or self.controller is None:
            self.log(
                "resume requested but no checkpoint found; starting fresh", level=Verbosity.SUMMARY
            )
            return
        assert self.controller._eval_context is not None and self.controller._eval_y is not None
        eval_frame = self.controller._eval_context.frame
        eval_y = self.controller._eval_y
        task = self.controller._task
        self.load_state()
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

    def _completed_compute_times(self) -> list[float]:
        if self.controller is None:
            return []
        return [
            g.score_set.compute_time
            for g in self.controller.population.completed()
            if g.score_set is not None and g.score_set.compute_time > 0
        ]

    def _estimated_model_seconds(self) -> float:
        times = self._completed_compute_times()
        if times:
            # Slight safety factor over the observed median.
            return float(np.median(times)) * 1.2
        return 5.0  # optimistic bootstrap before any timing history exists

    def _estimated_batch_seconds(self) -> float:
        per_model = self._estimated_model_seconds()
        workers = max(1, self.scheduler.model_workers)
        return (self.settings.models_per_batch * per_model) / workers + 5.0

    def shutdown(self, *, wait: bool = True) -> None:
        """Shut the thread pools down gracefully."""
        self.scheduler.shutdown(wait=wait)

    def best_genome(self):
        return self.controller.best_genome() if self.controller else None
