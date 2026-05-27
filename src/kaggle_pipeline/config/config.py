"""The :class:`Config` object — every metaparameter the pipeline needs.

This is the *only* thing a user is expected to edit per competition. In the
original notebook these values were scattered as module-level globals; here they
live in one explicit, serialisable place so the pipeline can be driven from a
YAML file or a few lines of Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kaggle_pipeline.logconfig import DEFAULT_VERBOSITY, VERBOSITY_LEVELS

# Default ordering lists for categorical variables. Whenever a categorical
# column's values are a subset of one of these lists it is treated as ordinal
# and encoded in this order. Matching is case-insensitive: entries are
# lower-cased in Config.__post_init__, so they may be written in any case here
# or in a user-supplied ``order_lists``.
DEFAULT_ORDER_LISTS: list[list[str]] = [
    ["poor", "average", "good"],
    ["low", "medium", "high"],
    ["easy", "moderate", "hard"],
    ["no", "yes"],
]

# Regression is not wired end-to-end yet (no regressor model definitions, and the
# ensembling/prediction path only handles classification). We fail fast -- both
# when ``task='regression'`` is set explicitly and when it is autodetected --
# rather than letting a run proceed to a confusing late failure.
REGRESSION_NOT_IMPLEMENTED = (
    "task='regression' is not implemented yet: this pipeline currently only "
    "supports classification end-to-end. Set task='classification' (or leave it "
    "unset for a classification target). Regression support is planned."
)


@dataclass
class Config:
    """All tunable parameters for a single competition run.

    Attributes are grouped roughly by concern: problem definition, feature
    engineering, the model search, cross-validation, runtime budget and I/O.
    """

    # --- Problem definition -------------------------------------------------
    # Each field below may be left as ``None`` to have it autodetected from the
    # training data once it is loaded (see :mod:`kaggle_pipeline.data.autodetect`).
    # A short message is printed for every value filled in this way, so a sparse
    # config still produces a reproducible, self-documenting run log.
    # The competition slug. Optional: on Kaggle, when ``data_dir`` is unset the
    # data directory is autodetected by scanning the attached inputs for the one
    # holding train/test CSVs (see :func:`~kaggle_pipeline.config.resolve_paths`).
    # Set ``competition`` only to disambiguate when several inputs match; it is
    # also used to derive the default Colab data path.
    competition: str | None = None
    # Target column(s). ``None`` -> the last (non-id) column of the train frame.
    target: list[str] | None = None
    id_col: list[str] = field(default_factory=lambda: ["id"])
    # 'classification' or 'regression'. ``None`` -> inferred from the target
    # dtype: non-numeric or low-cardinality integers give classification,
    # otherwise regression. This replaces the notebook's implicit (and buggy)
    # dtype sniffing; the working path is classification.
    task: str | None = None
    # Implemented so far: 'balanced_accuracy', 'roc_auc', 'neg_root_mean_squared_error'.
    # ``None`` -> 'roc_auc' for binary targets, 'balanced_accuracy' for multiclass.
    scoring: str | None = None
    # Implemented so far: 'category', 'probability'. ``None`` -> 'probability'
    # when the task is classification.
    prediction_aim: str | None = None

    # --- Feature engineering ------------------------------------------------
    # ``df.eval`` expressions applied by FeatureEngineer, e.g.
    #   "is_low = some_feature < 25".
    feature_expressions: list[str] = field(default_factory=list)
    order_lists: list[list[str]] = field(default_factory=lambda: list(DEFAULT_ORDER_LISTS))
    # How each categorical predictor is encoded *for models that cannot consume a
    # raw categorical column* (RandomForest, LogisticRegression). Maps a column
    # name to a strategy in ``ENCODING_STRATEGIES``; a column left out gets a
    # cardinality-based default (see ``onehot_max_cardinality``). Capability wins:
    # models that handle categoricals natively (CatBoost, XGBoost, LightGBM,
    # HistGB) always get the raw column and ignore this map (HistGB excepted above
    # its native cardinality cap).
    categorical_encoding: dict[str, str] = field(default_factory=dict)
    # Cardinality cut-off for the default encoding of a categorical not named in
    # ``categorical_encoding``: at or below this many distinct levels it defaults
    # to one-hot (cheap and lossless when narrow), above it to frequency. ``None``
    # (the default) falls back to ``encoders.ONEHOT_MAX_CARDINALITY`` (20).
    onehot_max_cardinality: int | None = None

    # --- Exploratory data analysis ------------------------------------------
    # Whether ``analyze`` renders the EDA suite (metadata, correlation/association
    # heatmaps, pairwise plots). Off by default: EDA is a standalone, interactive
    # step fully decoupled from training -- a ``run`` never touches it -- so it is
    # opt-in. Set to True (in the YAML or ``cfg.run_eda = True``) to render; when
    # False, ``analyze`` logs that it is disabled and returns without loading data.
    # The two fields below only tune those plots and have no effect when False.
    run_eda: bool = False
    # Max unique values for a numeric column to still be drawn as categorical
    # on EDA graphs. Only affects plotting.
    cat_cutoff: int = 5
    # Max distinct levels a categorical may have before EDA plots fold it to its
    # top-N most frequent levels plus an "Other" bucket. Prevents a high-cardinality
    # column (e.g. a driver with hundreds of levels) from producing an illegible
    # one-layer-per-level plot. Only affects plotting.
    max_plot_cats: int = 20

    # --- Feature pruning ----------------------------------------------------
    # Automatically drop predictors that are uncorrelated with the target or
    # redundant with another predictor; thresholds are inferred from the dataset
    # size (see :mod:`kaggle_pipeline.preprocessing.selection`). On by default.
    prune_features: bool = True
    # Significance level for the size-inferred thresholds: the irrelevance cut-off
    # tau(n) and the (1 - prune_alpha) redundancy confidence bound.
    prune_alpha: float = 0.05
    # Confident-association floor above which two predictors are deemed redundant.
    redundancy_floor: float = 0.90

    # --- Model search -------------------------------------------------------
    # Number of model-search batches to run. Set to ``None`` to instead run until
    # ``max_running_time`` is the only thing that stops the search (the Kaggle
    # notebook does this to use its whole 12h budget); the package keeps a small
    # finite default so a bare ``Config()`` run is bounded by step count, not time.
    n_steps: int | None = 10
    # Leaderboard capacity. Per-class lower/upper bounds given as floats are
    # read as fractions of this (see @register_model), so the search scales with it.
    num_models: int = 300
    step_batch_size: int = 32
    n_workers: int = -1
    ensemble_length: int = 30
    ensemble_min_repr: int = 1
    # --- Ensemble de-correlation -------------------------------------------
    # Before building the final ensemble, permanently drop any model whose
    # out-of-fold *residuals* (y - y_oof) are confidently correlated with a
    # better-scoring model's. Such a model makes the same mistakes and adds
    # nothing to the stack -- it is what makes "ensembling do nothing" when one
    # model class dominates the leaderboard with near-identical copies of itself.
    # On by default (see :mod:`kaggle_pipeline.search.decorrelation`).
    prune_correlated_models: bool = True
    # Redundancy threshold on the residual correlation's *lower confidence bound*
    # (Fisher z, one-sided 95%; the bound widens as the dataset shrinks). For a
    # large dataset this is effectively "drop models whose residuals correlate
    # above 0.98"; on smaller data a higher observed correlation is required.
    correlation_tau: float = 0.98

    # --- Cross-validation ---------------------------------------------------
    cv_splits: int = 5

    # --- Runtime ------------------------------------------------------------
    # Cut the dataset to speed things up; use for debugging only.
    speed_up: bool = False
    speed_up_train_rows: int = 1000
    speed_up_test_rows: int = 500
    # Global running-time limit in seconds (Kaggle kernels cap at 12h).
    max_running_time: int = 43200
    # The single random seed for the whole run: model hyperparameter sampling,
    # the leaderboard's class selection, the cross-validation fold shuffling and
    # the ensemble search all derive from it. Set it to an int to make a run
    # fully reproducible (same seed -> same folds, leaderboard and submission);
    # the default ``None`` leaves every random process unseeded (non-reproducible,
    # as in the original notebook).
    seed: int | None = None
    # How much the pipeline prints as it runs. One of 'quiet' (warnings/errors
    # only), 'normal' (stage progress + autodetect/prune summaries + each model's
    # score and timing) or 'verbose' (the default: adds the sampled per-model
    # parameters, the full leaderboard each step and the encoding plan). Mapped to
    # a logging level on the package logger; the CLI ``-v``/``-q`` flags override
    # it. See :mod:`kaggle_pipeline.logconfig`.
    verbosity: str = DEFAULT_VERBOSITY

    # --- I/O ----------------------------------------------------------------
    # When unset these are derived from the environment + ``competition``.
    data_dir: Path | None = None
    storage_dir: Path | None = None
    # Kaggle only: a previous notebook's output dir to warm-start (resume) the
    # leaderboard from. ``None`` (the default) -> the Kaggle input mount is
    # scanned automatically, so a re-run that has the prior output attached via
    # *Add Input* continues its leaderboard without hand-wiring the mount path.
    # Set this only to point at a specific directory or to disambiguate.
    previous_output_dir: Path | None = None
    # CSV filenames inside ``data_dir``. ``None`` -> found by searching the
    # directory for files whose names contain 'train' / 'test' / 'sample'.
    train_csv: str | None = None
    test_csv: str | None = None
    sample_csv: str | None = None
    submission_name: str = "submission"

    def __post_init__(self) -> None:
        # Be forgiving: accept a bare string for single-column target/id.
        if isinstance(self.target, str):
            self.target = [self.target]
        if isinstance(self.id_col, str):
            self.id_col = [self.id_col]
        for attr in ("data_dir", "storage_dir", "previous_output_dir"):
            value = getattr(self, attr)
            if value is not None and not isinstance(value, Path):
                setattr(self, attr, Path(value))
        # Order-list matching is case-insensitive, so normalise to lower case
        # here rather than requiring the user to write entries in lower case.
        self.order_lists = [[str(v).lower() for v in group] for group in self.order_lists]
        self._validate_categorical_encoding()
        if not 0.0 < self.prune_alpha < 1.0:
            raise ValueError(f"prune_alpha must be in (0, 1), got {self.prune_alpha}.")
        if not 0.0 <= self.redundancy_floor <= 1.0:
            raise ValueError(f"redundancy_floor must be in [0, 1], got {self.redundancy_floor}.")
        if not 0.0 <= self.correlation_tau <= 1.0:
            raise ValueError(f"correlation_tau must be in [0, 1], got {self.correlation_tau}.")
        if self.onehot_max_cardinality is not None and self.onehot_max_cardinality < 1:
            raise ValueError(
                f"onehot_max_cardinality must be a positive int or None, "
                f"got {self.onehot_max_cardinality}."
            )
        if self.verbosity not in VERBOSITY_LEVELS:
            raise ValueError(
                f"verbosity must be one of {sorted(VERBOSITY_LEVELS)}, got {self.verbosity!r}."
            )
        if self.task == "regression":
            raise NotImplementedError(REGRESSION_NOT_IMPLEMENTED)

    def _validate_categorical_encoding(self) -> None:
        """Fail loudly on an unknown encoding strategy (lazy import avoids a cycle)."""
        from kaggle_pipeline.preprocessing.encoders import ENCODING_STRATEGIES

        bad = {
            col: strategy
            for col, strategy in self.categorical_encoding.items()
            if strategy not in ENCODING_STRATEGIES
        }
        if bad:
            raise ValueError(
                f"Unknown categorical_encoding strategies {bad}; "
                f"each must be one of {sorted(ENCODING_STRATEGIES)}."
            )

    @property
    def target_is_num(self) -> bool:
        """Whether the target is treated as numeric (regression)."""
        return self.task == "regression"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build a Config from a plain dict, ignoring unknown keys gracefully."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a Config from a YAML file. See :mod:`kaggle_pipeline.config.loader`."""
        from kaggle_pipeline.config.loader import load_config

        return load_config(path)
