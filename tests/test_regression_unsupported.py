"""Regression is not implemented; it must fail fast, not run a broken pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from kaggle_pipeline import Config
from kaggle_pipeline.data.autodetect import resolve_problem_definition


def test_explicit_regression_task_raises():
    with pytest.raises(NotImplementedError, match="regression"):
        Config(competition="x", target="y", task="regression")


def test_autodetected_regression_task_raises():
    # A continuous numeric target autodetects as regression.
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [0.13, 1.7, 3.4, 9.1, 12.5],
        }
    )
    cfg = Config(competition="x")  # task left unset -> autodetect
    with pytest.raises(NotImplementedError, match="regression"):
        resolve_problem_definition(cfg, df)


def test_classification_task_is_unaffected():
    # Sanity: a categorical target still resolves normally (no false positives).
    df = pd.DataFrame({"id": [0, 1, 2, 3], "y": ["no", "yes", "no", "yes"]})
    cfg = Config(competition="x")
    resolve_problem_definition(cfg, df)
    assert cfg.task == "classification"
