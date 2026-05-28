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

import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from kaggle_pipeline.evolution.features.materialization import GLOBAL_TRAIN, MaterializationContext
from kaggle_pipeline.evolution.features.recipe import CATEGORICAL
from kaggle_pipeline.evolution.features.registry import FeatureRegistry
from kaggle_pipeline.evolution.genes.encoding_gene import (
    COUNT,
    FREQUENCY,
    NATIVE,
    ONEHOT,
    ORDINAL,
    TARGET,
)
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import FailureReason, ModelStatus
from kaggle_pipeline.evolution.models.parameter_spaces import (
    FamilyDefinition,
    build_default_families,
)
from kaggle_pipeline.evolution.models.scoring import ModelScoreSet
from kaggle_pipeline.evolution.utils.arrays import is_missing
from kaggle_pipeline.evolution.utils.logging import get_logger
from kaggle_pipeline.preprocessing.encoders import ONEHOT_MAX_CARDINALITY, _make_encoder
from kaggle_pipeline.search.cv import CrossValScore

logger = get_logger(__name__)

# Sentinel level for a missing categorical value, so encoders stay null-safe.
_NA_CATEGORY = "__nan__"

# LightGBM rejects feature names containing these JSON-special characters (it
# serialises the booster to JSON). Our feature ids use "::" and one-hot
# encoders emit "<col>_<category>" -- category values may contain ":" too --
# so the names arriving at the model can trip its check. The sanitiser below
# rewrites them right before the model step.
_LGBM_FORBIDDEN_NAME_CHARS = re.compile(r"[,\[\]:{}\"\\]")


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
        self.feature_names_in_ = np.asarray(getattr(X, "columns", ["x0"]), dtype=object)
        return self

    def transform(self, X: Any) -> np.ndarray:
        col = np.asarray(X).ravel().astype(str)
        return np.array([self.freq_.get(v, 0.0) for v in col]).reshape(-1, 1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


class _CountEncoder(BaseEstimator, TransformerMixin):
    """Per-column count encoder fit on the training fold (unseen -> 0)."""

    def fit(self, X: Any, y: Any = None) -> _CountEncoder:
        col = np.asarray(X).ravel().astype(str)
        values, counts = np.unique(col, return_counts=True)
        self.count_ = dict(zip(values, counts.astype(float), strict=True))
        self.feature_names_in_ = np.asarray(getattr(X, "columns", ["x0"]), dtype=object)
        return self

    def transform(self, X: Any) -> np.ndarray:
        col = np.asarray(X).ravel().astype(str)
        return np.array([self.count_.get(v, 0.0) for v in col]).reshape(-1, 1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


class _SanitizeFeatureNames(BaseEstimator, TransformerMixin):
    """Replace LightGBM-forbidden JSON characters in DataFrame column names.

    No-op on non-DataFrame input (estimators that get a numpy array don't see
    feature names anyway).
    """

    def fit(self, X: Any, y: Any = None) -> _SanitizeFeatureNames:
        return self

    def transform(self, X: Any) -> Any:
        if hasattr(X, "rename"):
            return X.rename(
                columns=lambda c: _LGBM_FORBIDDEN_NAME_CHARS.sub("_", str(c))
            )
        return X


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
        onehot_max_cardinality: int = ONEHOT_MAX_CARDINALITY,
    ):
        self.registry = registry
        self.families = families or build_default_families()
        self.context_id = context_id
        self.onehot_max_cardinality = onehot_max_cardinality

    def train(
        self,
        genome: ModelGenome,
        *,
        train_frame: pd.DataFrame,
        y: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]],
        ctx: Any,
        task: str = "classification",
        seed: int | None = None,
    ) -> ModelResult:
        """Train + cross-validate ``genome``; return a :class:`ModelResult`.

        Pure with respect to ``genome``: it reads the genome but never mutates it,
        so it is safe to run in a worker thread. The caller (main thread) applies
        the result to the genome and registries.
        """
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
            return ModelResult(genome.model_id, ModelStatus.COMPLETED, score_set, oof)
        except MemoryError:
            return self._fail(genome, FailureReason.MEMORY_ERROR, "MemoryError")
        except Exception as exc:  # noqa: BLE001 - one bad model must not kill the search
            logger.warning("model %s failed: %s", genome.model_id, exc)
            return self._fail(genome, FailureReason.TRAINING_EXCEPTION, str(exc))

    def _fail(self, genome: ModelGenome, reason: str, message: str) -> ModelResult:
        return ModelResult(
            genome.model_id, ModelStatus.FAILED, failure_reason=reason, error_message=message
        )

    def fit_predict_test(
        self,
        genome: ModelGenome,
        *,
        train_frame: pd.DataFrame,
        y: np.ndarray,
        test_frame: pd.DataFrame,
        task: str = "classification",
        seed: int | None = None,
    ) -> np.ndarray:
        """Refit ``genome`` on the full train set and predict ``test_frame``.

        Used at finalization to turn ensemble members into test predictions
        (training only stores cross-validated OOF, not test predictions).
        """
        # Distinct context ids from the search so the full-data refit never reuses
        # cached subsample arrays of a different length.
        X_train = self._build_feature_frame(genome, train_frame, context_id="final_train")
        X_test = self._build_feature_frame(genome, test_frame, context_id="final_test")
        pipeline = self._build_pipeline(genome, X_train, seed=seed)
        pipeline.fit(X_train, y)
        if task == "classification":
            return pipeline.predict_proba(X_test)
        return pipeline.predict(X_test)

    # --- feature frame ------------------------------------------------------
    def _build_feature_frame(
        self, genome: ModelGenome, frame: pd.DataFrame, *, context_id: str | None = None
    ) -> pd.DataFrame:
        context = MaterializationContext(frame=frame, context_id=context_id or self.context_id)
        data: dict[str, np.ndarray] = {}
        for fr in genome.feature_reference_genes:
            values = self.registry.materialize(fr.feature_id, context)
            feature = self.registry.get_feature(fr.feature_id)
            if feature.output_type == CATEGORICAL:
                # Make any encoder null-safe: a missing category becomes its own
                # level. Numeric NaNs are handled downstream by the imputer.
                values = np.array(
                    [_NA_CATEGORY if is_missing(v) else v for v in values], dtype=object
                )
            data[fr.feature_id] = values
        return pd.DataFrame(data, index=frame.index)

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
                if fr.encoding is None:
                    # No encoding gene -> a native-categorical family; feed it
                    # integer codes (the encoding choice is not part of its genome).
                    encoding = ORDINAL
                else:
                    encoding = fr.encoding.value
                    # Defensive backstop: the factory already excludes one-hot for
                    # high-cardinality features, but the train fold could differ.
                    if (
                        encoding == ONEHOT
                        and X[fr.feature_id].nunique(dropna=True) > self.onehot_max_cardinality
                    ):
                        encoding = FREQUENCY
                transformers.append(
                    (f"enc_{fr.feature_id}", self._encoder_for(encoding), [fr.feature_id])
                )
            else:
                numeric_cols.append(fr.feature_id)
        if numeric_cols:
            transformers.insert(0, ("num", SimpleImputer(strategy="median"), numeric_cols))

        # set_output("pandas") so the estimator receives a DataFrame with stable
        # column names: otherwise LightGBM autogenerates "Column_0"... feature
        # names at fit and warns at predict ("X does not have valid feature
        # names, but LGBMClassifier was fitted with feature names").
        preprocessor = ColumnTransformer(transformers, remainder="drop").set_output(
            transform="pandas"
        )
        steps: list[tuple[str, Any]] = [("prep", preprocessor)]
        if fam.needs_scaling:
            steps.append(("scaler", StandardScaler()))
        steps.append(("sanitize", _SanitizeFeatureNames()))
        steps.append(("model", estimator))
        return Pipeline(steps)

    @staticmethod
    def _encoder_for(encoding: str) -> Any:
        """Map an encoding-gene value to a transformer, reusing the v1 encoders.

        ``onehot``/``ordinal``/``frequency`` delegate to the shared v1
        :func:`~kaggle_pipeline.preprocessing.encoders._make_encoder` (identical
        ``OneHotEncoder``/``OrdinalEncoder`` specs and ``FrequencyEncoder``); only
        ``count`` has no v1 counterpart and keeps its local encoder.
        """
        if encoding == COUNT:
            return _CountEncoder()
        if encoding == TARGET:
            # TODO: real out-of-fold target encoding; fall back to frequency for now.
            encoding = FREQUENCY
        # onehot / frequency, plus native -> ordinal (codes, unseen levels -> -1).
        return _make_encoder(ORDINAL if encoding == NATIVE else encoding)
