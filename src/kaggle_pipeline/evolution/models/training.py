"""Training a model genome -- reusing the v1 cross-validation and scoring.

The trainer materializes the genome's selected features into a frame, wraps them
in a per-fold preprocessing pipeline (encoding categoricals by each feature's
:class:`EncodingGene`, imputing numerics, optionally scaling for linear models),
builds the family's estimator, and evaluates it through the v1
:class:`~kaggle_pipeline.search.cv.CrossValScore` (so out-of-fold predictions and
the same scoring function are reused). Failures are caught and recorded rather
than crashing the search.

The trainer consumes a genome and produces a result; it knows nothing about
feature *generation*. Encoders fit inside each CV fold, so categorical encoding is
leakage-safe; target encoding remains a TODO (it needs the OOF path).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from kaggle_pipeline.evolution.features.materialization import GLOBAL_TRAIN, MaterializationContext
from kaggle_pipeline.evolution.features.recipe import CATEGORICAL
from kaggle_pipeline.evolution.features.registry import FeatureRegistry
from kaggle_pipeline.evolution.genes.encoding_gene import COUNT, FREQUENCY, ONEHOT, TARGET
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import FailureReason, ModelStatus
from kaggle_pipeline.evolution.models.parameter_spaces import (
    FamilyDefinition,
    build_default_families,
)
from kaggle_pipeline.evolution.models.scoring import ModelScoreSet
from kaggle_pipeline.evolution.utils.logging import get_logger
from kaggle_pipeline.search.cv import CrossValScore

if TYPE_CHECKING:
    from kaggle_pipeline.context import PipelineContext

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """Outcome of training one genome."""

    model_id: str
    status: str
    score_set: ModelScoreSet | None = None
    oof: np.ndarray | None = None
    failure_reason: str | None = None
    error_message: str = ""


class _FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Per-column frequency encoder fit on the training fold (unseen -> 0)."""

    def fit(self, X: Any, y: Any = None) -> _FrequencyEncoder:
        col = np.asarray(X).ravel().astype(str)
        values, counts = np.unique(col, return_counts=True)
        self.freq_ = dict(zip(values, counts / col.size, strict=True))
        return self

    def transform(self, X: Any) -> np.ndarray:
        col = np.asarray(X).ravel().astype(str)
        return np.array([self.freq_.get(v, 0.0) for v in col]).reshape(-1, 1)


class _CountEncoder(BaseEstimator, TransformerMixin):
    """Per-column count encoder fit on the training fold (unseen -> 0)."""

    def fit(self, X: Any, y: Any = None) -> _CountEncoder:
        col = np.asarray(X).ravel().astype(str)
        values, counts = np.unique(col, return_counts=True)
        self.count_ = dict(zip(values, counts.astype(float), strict=True))
        return self

    def transform(self, X: Any) -> np.ndarray:
        col = np.asarray(X).ravel().astype(str)
        return np.array([self.count_.get(v, 0.0) for v in col]).reshape(-1, 1)


class _GenomeModel:
    """Adapter exposing the v1 ``Model`` interface CrossValScore expects."""

    def __init__(self, pipeline: Any, task: str):
        self._pipeline = pipeline
        self._task = task
        self._oof: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._task == "classification":
            return self._pipeline.predict_proba(X)
        return self._pipeline.predict(X)

    def set_oof(self, oof: np.ndarray) -> None:
        self._oof = oof

    @property
    def oof(self) -> np.ndarray | None:
        return self._oof


class ModelTrainer:
    """Materializes features, builds a pipeline, and cross-validates a genome."""

    def __init__(
        self,
        registry: FeatureRegistry,
        *,
        families: dict[str, FamilyDefinition] | None = None,
        context_id: str = GLOBAL_TRAIN,
    ):
        self.registry = registry
        self.families = families or build_default_families()
        self.context_id = context_id

    def train(
        self,
        genome: ModelGenome,
        *,
        train_frame: pd.DataFrame,
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        ctx: PipelineContext,
        task: str = "classification",
        seed: int | None = None,
    ) -> ModelResult:
        """Train + cross-validate ``genome``; return a :class:`ModelResult`."""
        genome.status = ModelStatus.TRAINING
        t0 = perf_counter()
        try:
            X = self._build_feature_frame(genome, train_frame)
            pipeline = self._build_pipeline(genome, X, seed=seed)
            adapter = _GenomeModel(pipeline, task)
            # The adapter duck-types the v1 Model interface CrossValScore needs.
            cv = CrossValScore(cast(Any, adapter), X, y, splits=splits, ctx=ctx)
            mean, std = cv.score
            oof = adapter.oof
            if oof is None or not np.all(np.isfinite(oof)):
                return self._fail(genome, FailureReason.NAN_PREDICTIONS, "non-finite OOF")
            score_set = ModelScoreSet(
                score=float(mean),
                score_std=float(std),
                compute_time=perf_counter() - t0,
                n_features=len(genome.feature_reference_genes),
                fidelity_level=genome.fidelity_level,
            )
            genome.score_set = score_set
            genome.status = ModelStatus.COMPLETED
            return ModelResult(genome.model_id, ModelStatus.COMPLETED, score_set, oof)
        except MemoryError:
            return self._fail(genome, FailureReason.MEMORY_ERROR, "MemoryError")
        except Exception as exc:  # noqa: BLE001 - one bad model must not kill the search
            logger.warning("model %s failed: %s", genome.model_id, exc)
            return self._fail(genome, FailureReason.TRAINING_EXCEPTION, str(exc))

    def _fail(self, genome: ModelGenome, reason: str, message: str) -> ModelResult:
        genome.status = ModelStatus.FAILED
        return ModelResult(
            genome.model_id, ModelStatus.FAILED, failure_reason=reason, error_message=message
        )

    # --- feature frame ------------------------------------------------------
    def _build_feature_frame(self, genome: ModelGenome, train_frame: pd.DataFrame) -> pd.DataFrame:
        context = MaterializationContext(frame=train_frame, context_id=self.context_id)
        data: dict[str, np.ndarray] = {}
        for fr in genome.feature_reference_genes:
            data[fr.feature_id] = self.registry.materialize(fr.feature_id, context)
        return pd.DataFrame(data, index=train_frame.index)

    # --- pipeline -----------------------------------------------------------
    def _build_pipeline(self, genome: ModelGenome, X: pd.DataFrame, *, seed: int | None) -> Any:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        fam = self.families[genome.family]
        params = {g.parameter_name: g.value for g in genome.parameter_genes}
        resource = genome.get_resource("n_estimators")
        n_estimators = (
            int(resource.value) if resource else fam.n_estimators_for(genome.fidelity_level)
        )
        estimator = fam.build_estimator(params, n_estimators=n_estimators, random_state=seed)

        numeric_cols: list[str] = []
        transformers: list[tuple[str, Any, list[str]]] = []
        for fr in genome.feature_reference_genes:
            feature = self.registry.get_feature(fr.feature_id)
            if feature.output_type == CATEGORICAL:
                encoding = fr.encoding.value if fr.encoding is not None else FREQUENCY
                transformers.append(
                    (f"enc_{fr.feature_id}", self._encoder_for(encoding), [fr.feature_id])
                )
            else:
                numeric_cols.append(fr.feature_id)
        if numeric_cols:
            transformers.insert(0, ("num", SimpleImputer(strategy="median"), numeric_cols))

        preprocessor = ColumnTransformer(transformers, remainder="drop")
        steps: list[tuple[str, Any]] = [("prep", preprocessor)]
        if fam.needs_scaling:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        return Pipeline(steps)

    @staticmethod
    def _encoder_for(encoding: str) -> Any:
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

        if encoding == ONEHOT:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if encoding == FREQUENCY:
            return _FrequencyEncoder()
        if encoding == COUNT:
            return _CountEncoder()
        if encoding == TARGET:
            # TODO: real out-of-fold target encoding; fall back to frequency for now.
            return _FrequencyEncoder()
        # native / ordinal: ordinal codes, unseen levels -> -1.
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
