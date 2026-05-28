"""Load the train / test / sample-submission CSVs for a competition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kaggle_pipeline.config import Config
from kaggle_pipeline.data.autodetect import resolve_csv_filenames, resolve_problem_definition


@dataclass
class Datasets:
    """The three dataframes every Kaggle tabular run starts from."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample: pd.DataFrame


def load_datasets(config: Config, data_dir: Path, *, check_nulls: bool = True) -> Datasets:
    """Read the three CSVs and, if ``config.speed_up``, subsample for debugging.

    Any CSV filename or problem-definition field left as ``None`` on the config
    is autodetected here (and the choice is printed); see
    :mod:`kaggle_pipeline.data.autodetect`. The detection runs on the full train
    frame before any ``speed_up`` subsampling so it sees all classes.

    Subsampling mirrors the notebook: the first N rows of each frame so a full
    run completes in seconds while you iterate on the pipeline itself.

    The v1 ``run`` flow has no imputer, so by default a null in train/test raises
    early. Callers that *do* impute (the evolutionary ``KagglePipeline`` imputes
    numerics and treats missing categoricals as their own level) can pass
    ``check_nulls=False`` to load null-bearing data.
    """
    resolve_csv_filenames(config, data_dir)
    # resolve_csv_filenames fills any unset filename (or raises), so all three
    # are concrete here.
    assert config.train_csv is not None
    assert config.test_csv is not None
    assert config.sample_csv is not None
    train = pd.read_csv(data_dir / config.train_csv)
    test = pd.read_csv(data_dir / config.test_csv)
    sample = pd.read_csv(data_dir / config.sample_csv)

    resolve_problem_definition(config, train)

    if config.speed_up:
        train = train[: config.speed_up_train_rows].copy()
        test = test[: config.speed_up_test_rows].copy()
        sample = sample[: config.speed_up_test_rows].copy()

    if check_nulls:
        _check_no_nulls(train, test)
    return Datasets(train=train, test=test, sample=sample)


def _check_no_nulls(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """The pipeline has no imputer yet; fail early if data has missing values."""
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError(
            "Input data contains nulls but no imputer is implemented yet. "
            "Add imputation before running the pipeline."
        )
