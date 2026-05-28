"""The :class:`Config` object -- every metaparameter the pipeline needs.

This is the *only* thing a user is expected to edit per competition. Values
live in one explicit, serialisable place so the pipeline can be driven from a
YAML file or a few lines of Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kaggle_pipeline.logconfig import DEFAULT_VERBOSITY, VERBOSITY_LEVELS

# Default ordering lists for categorical variables (case-insensitive: entries
# are lower-cased in Config.__post_init__). Used only by the EDA pass to draw
# ordered category axes.
DEFAULT_ORDER_LISTS: list[list[str]] = [
    ["poor", "average", "good"],
    ["low", "medium", "high"],
    ["easy", "moderate", "hard"],
    ["no", "yes"],
]

# Regression is not wired end-to-end yet. We fail fast -- both when
# ``task='regression'`` is set explicitly and when it is autodetected -- rather
# than letting a run proceed to a confusing late failure.
REGRESSION_NOT_IMPLEMENTED = (
    "task='regression' is not implemented yet: this pipeline currently only "
    "supports classification end-to-end. Set task='classification' (or leave it "
    "unset for a classification target). Regression support is planned."
)


@dataclass
class Config:
    """All tunable parameters for a single competition run."""

    # --- Problem definition -------------------------------------------------
    # Each field below may be left as ``None`` to have it autodetected from the
    # training data once it is loaded (see :mod:`kaggle_pipeline.data.autodetect`).
    competition: str | None = None
    # Target column(s). ``None`` -> the last (non-id) column of the train frame.
    target: list[str] | None = None
    id_col: list[str] = field(default_factory=lambda: ["id"])
    # 'classification' or 'regression'. ``None`` -> inferred from the target dtype.
    task: str | None = None
    # Implemented so far: 'balanced_accuracy', 'roc_auc', 'neg_root_mean_squared_error'.
    # ``None`` -> 'roc_auc' for binary targets, 'balanced_accuracy' for multiclass.
    scoring: str | None = None
    # 'category' or 'probability'. ``None`` -> 'probability' when classification.
    prediction_aim: str | None = None

    # --- Feature engineering ------------------------------------------------
    # ``df.eval`` expressions applied by FeatureEngineer, e.g.
    #   "is_low = some_feature < 25".
    feature_expressions: list[str] = field(default_factory=list)
    # Ordering hints for categorical levels (EDA only).
    order_lists: list[list[str]] = field(default_factory=lambda: list(DEFAULT_ORDER_LISTS))

    # --- Exploratory data analysis ------------------------------------------
    # Whether ``analyze`` renders the EDA suite. Off by default: EDA is opt-in.
    run_eda: bool = False
    # Max unique values for a numeric column to still be drawn as categorical
    # on EDA graphs.
    cat_cutoff: int = 5
    # Max distinct levels a categorical may have before EDA plots fold it to its
    # top-N most frequent levels plus an "Other" bucket.
    max_plot_cats: int = 20

    # --- Cross-validation ---------------------------------------------------
    cv_splits: int = 5

    # --- Runtime ------------------------------------------------------------
    # Cut the dataset to speed things up; use for debugging only.
    speed_up: bool = False
    speed_up_train_rows: int = 1000
    speed_up_test_rows: int = 500
    # Global running-time limit in seconds (Kaggle kernels cap at 12h).
    max_running_time: int = 43200
    # Single random seed for the whole run; ``None`` leaves randomness unseeded.
    seed: int | None = None
    verbosity: str = DEFAULT_VERBOSITY

    # --- I/O ----------------------------------------------------------------
    # When unset these are derived from the environment + ``competition``.
    data_dir: Path | None = None
    storage_dir: Path | None = None
    # CSV filenames inside ``data_dir``. ``None`` -> found by searching the
    # directory for files whose names contain 'train' / 'test' / 'sample'.
    train_csv: str | None = None
    test_csv: str | None = None
    sample_csv: str | None = None

    def __post_init__(self) -> None:
        # Be forgiving: accept a bare string for single-column target/id.
        if isinstance(self.target, str):
            self.target = [self.target]
        if isinstance(self.id_col, str):
            self.id_col = [self.id_col]
        for attr in ("data_dir", "storage_dir"):
            value = getattr(self, attr)
            if value is not None and not isinstance(value, Path):
                setattr(self, attr, Path(value))
        # Order-list matching is case-insensitive, so normalise to lower case.
        self.order_lists = [[str(v).lower() for v in group] for group in self.order_lists]
        if self.verbosity not in VERBOSITY_LEVELS:
            raise ValueError(
                f"verbosity must be one of {sorted(VERBOSITY_LEVELS)}, got {self.verbosity!r}."
            )
        if self.task == "regression":
            raise NotImplementedError(REGRESSION_NOT_IMPLEMENTED)

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
