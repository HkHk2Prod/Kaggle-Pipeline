"""Data-preparation helpers used by :class:`KagglePipeline.fit`.

These functions are stateless: they take frames + config and return the
arrays/frames the pipeline needs. Pulling them out keeps ``pipeline.py``
focused on orchestration and makes each step independently testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kaggle_pipeline.evolution.features.recipe import CATEGORICAL, NUMERIC


def autodetect_problem(
    train_df: pd.DataFrame,
    target: str | None,
    task: str | None,
    scoring: str | None,
    prediction_aim: str | None,
    id_col: str,
) -> tuple[str, str, str, str]:
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
    # ``resolve_problem_definition`` guarantees these are filled in; the
    # typecast lets callers consume them without re-narrowing.
    assert cfg.target and cfg.task and cfg.scoring and cfg.prediction_aim
    return cfg.target[0], cfg.task, cfg.scoring, cfg.prediction_aim


def engineer_features(df: pd.DataFrame, feature_expressions: list[str] | None) -> pd.DataFrame:
    """Apply ``df.eval`` feature expressions (no encodings)."""
    if not feature_expressions:
        return df
    from kaggle_pipeline.preprocessing.transformers import FeatureEngineer

    return FeatureEngineer(expressions=feature_expressions).fit_transform(df.copy())


def build_search_sample(
    features: pd.DataFrame,
    y: np.ndarray,
    task: str,
    *,
    fraction: float,
    cv_splits: int,
    seed: int | None,
) -> tuple[pd.DataFrame, np.ndarray, bool]:
    """Return ``(features, y, used_subsample)`` for the evolutionary search.

    Falls back to the full data when the configured fraction is out of range
    or would leave too few rows to cross-validate cleanly. The third tuple
    element lets the caller log whether a subsample was actually taken.
    """
    n = len(features)
    if not (0.0 < fraction < 1.0):
        return features, y, False
    n_sample = int(round(n * fraction))
    min_rows = max(2 * cv_splits, 30)
    if n_sample < min_rows or n_sample >= n:
        return features, y, False
    from sklearn.model_selection import train_test_split

    stratify = y if task == "classification" else None
    try:
        idx, _ = train_test_split(
            np.arange(n), train_size=n_sample, random_state=seed, stratify=stratify
        )
    except ValueError:  # a class too rare to stratify -- sample without it
        idx, _ = train_test_split(np.arange(n), train_size=n_sample, random_state=seed)
    idx = np.sort(idx)
    sampled = features.iloc[idx].reset_index(drop=True)
    return sampled, np.asarray(y)[idx], True


def infer_feature_type(column: str, frame: pd.DataFrame, overrides: dict[str, str] | None) -> str:
    """Resolve a column's feature type, honoring caller-supplied overrides."""
    if overrides and column in overrides:
        return overrides[column]
    return NUMERIC if pd.api.types.is_numeric_dtype(frame[column]) else CATEGORICAL
