"""Fill in :class:`~kaggle_pipeline.config.Config` values left as ``None``.

The config lets the problem-definition fields (``target``, ``task``,
``prediction_aim``, ``scoring``) and the CSV filenames be left unset. This module
infers them from the loaded data and logs a short message for every value it
fills, so a run is reproducible from the log even when the config was sparse.

All functions mutate the passed ``Config`` in place and are idempotent: a field
that is already set is left untouched (and logs nothing), so calling ``analyze``
then ``run`` with the same config autodetects only once.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from kaggle_pipeline.config import REGRESSION_NOT_IMPLEMENTED, Config

logger = logging.getLogger(__name__)

# A numeric target with at most this many distinct integer values is treated as
# a classification label set rather than a continuous regression target.
MAX_AUTODETECT_CLASSES = 20


def _announce(field: str, value: object, reason: str) -> None:
    """Log one line explaining an autodetected value."""
    logger.info("[autodetect] %s = %r  (%s)", field, value, reason)


def resolve_csv_filenames(config: Config, data_dir: Path) -> None:
    """Fill ``train_csv`` / ``test_csv`` / ``sample_csv`` by searching ``data_dir``.

    Only fields left as ``None`` are filled; a name is chosen by looking for the
    first ``.csv`` whose (lower-cased) filename contains a matching keyword.
    """
    if all(v is not None for v in (config.train_csv, config.test_csv, config.sample_csv)):
        return
    names = sorted(p.name for p in Path(data_dir).glob("*.csv"))
    if config.train_csv is None:
        config.train_csv = _find_csv(names, ["train"], data_dir, "train_csv")
    if config.test_csv is None:
        config.test_csv = _find_csv(names, ["test"], data_dir, "test_csv")
    if config.sample_csv is None:
        config.sample_csv = _find_csv(
            names, ["sample_submission", "sample", "submission"], data_dir, "sample_csv"
        )


def _find_csv(names: list[str], keywords: Sequence[str], data_dir: Path, field: str) -> str:
    """Return the first CSV in ``names`` matching any keyword, or raise."""
    for keyword in keywords:
        for name in names:
            if keyword in name.lower():
                _announce(field, name, f"matched {keyword!r} in {data_dir}")
                return name
    raise FileNotFoundError(
        f"Could not autodetect {field}: no .csv in {data_dir} matched any of "
        f"{list(keywords)}. Found: {names or 'no CSV files'}. "
        f"Set config.{field} explicitly."
    )


def resolve_problem_definition(config: Config, train_df: pd.DataFrame) -> None:
    """Fill ``target`` / ``task`` / ``prediction_aim`` / ``scoring`` from the data.

    Only fields left as ``None`` are filled. ``target`` is the last non-id
    column; ``task`` follows the target dtype; ``prediction_aim`` defaults to
    probabilities for classification; ``scoring`` is a common metric for the task.
    """
    if config.target is None:
        config.target = [_detect_target(train_df, config.id_col)]
        _announce("target", config.target, "last non-id column of the train frame")

    target_col = config.target[0]
    y = train_df[target_col]

    if config.task is None:
        config.task = _detect_task(y)
        _announce(
            "task",
            config.task,
            f"{target_col!r} has dtype {y.dtype} with {y.nunique()} unique value(s)",
        )
        # Regression is not implemented end-to-end; fail fast on the inferred
        # value just as Config does for an explicitly set one, so a continuous
        # target does not silently proceed to a late failure.
        if config.task == "regression":
            raise NotImplementedError(REGRESSION_NOT_IMPLEMENTED)

    if config.prediction_aim is None:
        config.prediction_aim = "probability" if config.task == "classification" else "category"
        _announce("prediction_aim", config.prediction_aim, f"task is {config.task}")

    if config.scoring is None:
        config.scoring = _detect_scoring(config.task, y)
        _announce("scoring", config.scoring, f"common metric for {config.task}")


def _detect_target(train_df: pd.DataFrame, id_col: Sequence[str] | None) -> str:
    """The target is the last column that is not an id column."""
    id_cols = set(id_col or [])
    candidates = [c for c in train_df.columns if c not in id_cols]
    if not candidates:  # degenerate frame of nothing but id columns
        candidates = list(train_df.columns)
    return candidates[-1]


def _detect_task(y: pd.Series) -> str:
    """Classification for labels, regression for continuous numeric targets.

    Non-numeric and boolean columns are always classification. A numeric column
    is classification only when its values are integer-valued *and* there are at
    most :data:`MAX_AUTODETECT_CLASSES` of them (e.g. 0/1 labels); otherwise it
    is treated as a continuous regression target.
    """
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "classification"
    non_null = y.dropna()
    if pd.api.types.is_integer_dtype(non_null):
        integer_valued = True
    elif len(non_null):
        integer_valued = bool((non_null == non_null.round()).all())
    else:
        integer_valued = True
    if integer_valued and non_null.nunique() <= MAX_AUTODETECT_CLASSES:
        return "classification"
    return "regression"


def _detect_scoring(task: str, y: pd.Series) -> str:
    """A sensible default metric: ROC-AUC for binary, balanced accuracy else."""
    if task == "regression":
        return "neg_root_mean_squared_error"
    return "roc_auc" if y.nunique() == 2 else "balanced_accuracy"
