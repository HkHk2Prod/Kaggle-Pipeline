"""Shared fixtures: a small synthetic tabular competition on disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_pipeline.config import Config

N_TRAIN = 200
N_TEST = 80
TARGET = "y"


def _make_frame(n: int, rng: np.random.Generator, *, with_target: bool) -> pd.DataFrame:
    num1 = rng.normal(size=n)
    num2 = rng.uniform(0, 100, size=n)
    cat1 = rng.choice(["low", "medium", "high"], size=n)
    cat2 = rng.choice(["no", "yes"], size=n)
    data = {"num1": num1, "num2": num2, "cat1": cat1, "cat2": cat2}
    if with_target:
        # A target that genuinely depends on the features, so models can learn.
        logit = 1.5 * num1 + 0.02 * (num2 - 50) + (cat2 == "yes").astype(float)
        data[TARGET] = np.where(logit + rng.normal(scale=0.5, size=n) > 0, "yes", "no")
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    """Write train/test/sample CSVs into a temp dir and return it."""
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    train = _make_frame(N_TRAIN, rng, with_target=True)
    train.insert(0, "id", range(N_TRAIN))
    train.to_csv(data_dir / "train.csv", index=False)

    test = _make_frame(N_TEST, rng, with_target=False)
    test.insert(0, "id", range(N_TEST))
    test.to_csv(data_dir / "test.csv", index=False)

    sample = pd.DataFrame({"id": range(N_TEST), TARGET: ["no"] * N_TEST})
    sample.to_csv(data_dir / "sample_submission.csv", index=False)
    return data_dir


@pytest.fixture
def smoke_config(synthetic_data_dir: Path, tmp_path: Path) -> Config:
    """A tiny, fast, reproducible config pointing at the synthetic data."""
    return Config(
        competition="synthetic",
        target=TARGET,
        id_col="id",
        task="classification",
        scoring="balanced_accuracy",
        prediction_aim="category",
        feature_expressions=[],
        n_steps=1,
        num_models=20,
        step_batch_size=8,
        n_workers=1,
        ensemble_length=8,
        ensemble_min_repr=1,
        cv_splits=3,
        cv_seed=42,
        seed=0,
        data_dir=synthetic_data_dir,
        storage_dir=tmp_path / "models",
    )
