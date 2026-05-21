"""Load the train / test / sample-submission CSVs for a competition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kaggle_pipeline.config import Config


@dataclass
class Datasets:
    """The three dataframes every Kaggle tabular run starts from."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame


def load_datasets(config: Config, data_dir: Path) -> Datasets:
    """Read the three CSVs and, if ``config.speed_up``, subsample for debugging.

    Subsampling mirrors the notebook: the first N rows of each frame so a full
    run completes in seconds while you iterate on the pipeline itself.
    """
    train = pd.read_csv(data_dir / config.train_csv)
    test = pd.read_csv(data_dir / config.test_csv)
    sample = pd.read_csv(data_dir / config.sample_csv)

    if config.speed_up:
        train = train[: config.speed_up_train_rows].copy()
        test = test[: config.speed_up_test_rows].copy()
        sample = sample[: config.speed_up_test_rows].copy()

    _check_no_nulls(train, test)
    return Datasets(train=train, test=test, sample=sample)


def _check_no_nulls(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """The pipeline has no imputer yet; fail early if data has missing values."""
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError(
            "Input data contains nulls but no imputer is implemented yet. "
            "Add imputation before running the pipeline."
        )
