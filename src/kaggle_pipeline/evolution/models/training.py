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
from collections import Counter
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


_CAP_WARNED_FAMILIES: set[str] = set()


def reset_train_size_cap_warnings() -> None:
    """Forget which families have already logged the train-size cap warning.

    Tests use this to make per-family warnings observable without relying on
    test ordering. Production code never calls it -- one warning per family per
    process is the intended cadence.
    """
    _CAP_WARNED_FAMILIES.clear()


class _TrainSizeCappedEstimator(BaseEstimator):
    """Wraps a final estimator to cap the number of training rows it ever sees.

    Used for families whose fit time scales poorly with ``N`` (MLP single-thread,
    KNN predict). On ``fit`` we draw a stratified subsample (random for
    regression) of size ``max_rows`` and forward to the underlying estimator;
    on the first cap firing per family we log a warning showing the measured
    fit time and a linear extrapolation to the full-data cost so the user can
    decide whether the cut was warranted. Predict-time calls pass through
    unchanged.

    Only ``__init__`` params survive a sklearn ``clone()`` (which cross-val
    invokes per fold), so the cap-warning dedup state lives in a module-level
    set rather than on the instance.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        max_rows: int,
        family_name: str,
        seed: int | None = None,
    ):
        self.estimator = estimator
        self.max_rows = max_rows
        self.family_name = family_name
        self.seed = seed

    def fit(self, X: Any, y: np.ndarray) -> _TrainSizeCappedEstimator:
        n = len(X)
        if n <= self.max_rows:
            self.estimator.fit(X, y)
        else:
            rng = np.random.default_rng(self.seed)
            idx = _stratified_subsample_indices(y, self.max_rows, rng)
            X_capped = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            y_capped = y[idx]
            t0 = perf_counter()
            self.estimator.fit(X_capped, y_capped)
            elapsed = perf_counter() - t0
            # Linear extrapolation: works for the families we cap (MLP / KNN),
            # both roughly linear in N. Users who pick a non-linear-time
            # family should raise the cap.
            extrapolated = elapsed * n / self.max_rows
            if self.family_name not in _CAP_WARNED_FAMILIES:
                logger.warning(
                    "%s capped training data: trained on %d / %d rows; "
                    "measured fit = %.1fs, estimated full-data fit = %.1fs. "
                    "Raise max_train_rows for this family if the cap is too tight.",
                    self.family_name,
                    self.max_rows,
                    n,
                    elapsed,
                    extrapolated,
                )
                _CAP_WARNED_FAMILIES.add(self.family_name)
        # Expose ``classes_`` (and friends) as real attributes after fit so
        # sklearn's ``check_is_fitted`` -- which inspects trailing-underscore
        # attributes on the final step of a Pipeline -- treats the wrapper as
        # fitted. Forwarding via ``__getattr__`` was not enough on sklearn 1.5+.
        if hasattr(self.estimator, "classes_"):
            self.classes_ = self.estimator.classes_
        if hasattr(self.estimator, "n_features_in_"):
            self.n_features_in_ = self.estimator.n_features_in_
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator.predict_proba(X)


def _stratified_subsample_indices(
    y: np.ndarray, max_rows: int, rng: np.random.Generator
) -> np.ndarray:
    """Indices into ``y`` selecting ~``max_rows`` rows that preserve class balance.

    Plain random sampling for regression-like targets (many distinct values)
    where stratification has no meaning; per-class allocation proportional to
    class frequency for classification. Last class absorbs any rounding so the
    total lands on ``max_rows`` exactly.
    """
    y = np.asarray(y).ravel()
    uniq = np.unique(y)
    if uniq.size > 100:  # treat as regression target -- no stratification
        return rng.choice(len(y), size=max_rows, replace=False)
    picks: list[np.ndarray] = []
    remaining = max_rows
    for i, cls in enumerate(uniq):
        idx = np.where(y == cls)[0]
        if i == uniq.size - 1:
            take = remaining
        else:
            take = int(round(max_rows * len(idx) / len(y)))
        take = min(max(take, 1), len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        picks.append(chosen)
        remaining -= take
    return np.concatenate(picks)


class _SanitizeFeatureNames(BaseEstimator, TransformerMixin):
    """Replace LightGBM-forbidden JSON characters in DataFrame column names.

    No-op on non-DataFrame input (estimators that get a numpy array don't see
    feature names anyway).
    """

    def fit(self, X: Any, y: Any = None) -> _SanitizeFeatureNames:
        return self

    def transform(self, X: Any) -> Any:
        if hasattr(X, "rename"):
            return X.rename(columns=lambda c: _LGBM_FORBIDDEN_NAME_CHARS.sub("_", str(c)))
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
        if fam.max_train_rows is not None:
            estimator = _TrainSizeCappedEstimator(
                estimator,
                max_rows=fam.max_train_rows,
                family_name=fam.name,
                seed=seed,
            )

        # Dedup feature references by id: duplicates would cause LightGBM to
        # reject the matrix ("Feature ... appears more than one time") because
        # the ColumnTransformer emits one output column per occurrence. The
        # feature frame already collapses duplicates via dict semantics, so
        # dropping the extra refs here is lossless. Keep first occurrence for
        # deterministic ordering.
        unique_refs = list({fr.feature_id: fr for fr in genome.feature_reference_genes}.values())
        if len(unique_refs) != len(genome.feature_reference_genes):
            counts = Counter(genome.feature_ids())
            dupes = sorted(fid for fid, n in counts.items() if n > 1)
            logger.warning(
                "genome %s had duplicate feature references %s; deduped before fit",
                genome.model_id,
                dupes,
            )

        numeric_cols: list[str] = []
        transformers: list[tuple[str, Any, list[str]]] = []
        for fr in unique_refs:
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
