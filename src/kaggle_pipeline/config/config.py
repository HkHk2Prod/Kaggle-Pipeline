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

# Default ordering lists for categorical variables. Whenever a categorical
# column's values are a subset of one of these lists it is treated as ordinal
# and encoded in this order. EVERYTHING SHOULD BE LOWER CASE.
DEFAULT_ORDER_LISTS: list[list[str]] = [
    ["poor", "average", "good"],
    ["low", "medium", "high"],
    ["easy", "moderate", "hard"],
    ["no", "yes"],
]


@dataclass
class Config:
    """All tunable parameters for a single competition run.

    Attributes are grouped roughly by concern: problem definition, feature
    engineering, the model search, cross-validation, runtime budget and I/O.
    """

    # --- Problem definition -------------------------------------------------
    competition: str = "playground-series-s6e4"
    target: list[str] = field(default_factory=lambda: ["target"])
    id_col: list[str] = field(default_factory=lambda: ["id"])
    # 'classification' or 'regression'. This replaces the notebook's implicit
    # (and buggy) dtype sniffing; the working path is classification.
    task: str = "classification"
    # Implemented so far: 'balanced_accuracy', 'roc_auc'.
    scoring: str = "balanced_accuracy"
    # Implemented so far: 'category', 'probability'.
    prediction_aim: str = "category"

    # --- Feature engineering ------------------------------------------------
    # ``df.eval`` expressions applied by FeatureEngineer, e.g.
    #   "soil_lt_25 = Soil_Moisture < 25".
    feature_expressions: list[str] = field(default_factory=list)
    order_lists: list[list[str]] = field(default_factory=lambda: list(DEFAULT_ORDER_LISTS))
    # Max unique values for a numeric column to still be drawn as categorical
    # on EDA graphs. Only affects plotting.
    cat_cutoff: int = 5

    # --- Model search -------------------------------------------------------
    n_steps: int = 10
    num_models: int = 100
    step_batch_size: int = 32
    n_workers: int = -1
    ensemble_length: int = 30
    ensemble_min_repr: int = 1

    # --- Cross-validation ---------------------------------------------------
    cv_splits: int = 5
    cv_seed: int = 42

    # --- Runtime ------------------------------------------------------------
    # Cut the dataset to speed things up; use for debugging only.
    speed_up: bool = False
    speed_up_train_rows: int = 1000
    speed_up_test_rows: int = 500
    # Global running-time limit in seconds (Kaggle kernels cap at 12h).
    max_running_time: int = 43200
    # Random seed. ``None`` means non-reproducible (matches the notebook default).
    seed: int | None = None

    # --- I/O ----------------------------------------------------------------
    # When unset these are derived from the environment + ``competition``.
    data_dir: Path | None = None
    storage_dir: Path | None = None
    # Kaggle only: a previous notebook's output dir to warm-start models from.
    previous_output_dir: Path | None = None
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    sample_csv: str = "sample_submission.csv"
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
