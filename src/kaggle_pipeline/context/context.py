"""The :class:`PipelineContext` -- everything derived once and threaded through.

The notebook kept the fitted state (transformed frames, column splits, target
transforms, the scoring function, the RNG seed sequence) in module globals that
many classes read directly. Packaging that cleanly means turning those globals
into one object that is built after preprocessing and passed explicitly to the
search, models and submission code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_pipeline.config import Config, ResolvedPaths
from kaggle_pipeline.data import Datasets
from kaggle_pipeline.preprocessing import (
    CATEGORICAL_TYPER_STEP,
    TargetTransforms,
    build_pretrain_pipeline,
    build_target_transforms,
    get_columns,
    get_predictor_names,
    resolve_encoding_plan,
    split_num_cat,
)
from kaggle_pipeline.scoring import ScoringFn, resolve_scoring


@dataclass
class PipelineContext:
    """Fitted, run-wide state shared by the search, models and submission code."""

    config: Config
    paths: ResolvedPaths

    # Frames after the pre-training pipeline (sample is the raw submission frame).
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    sample_df: pd.DataFrame

    # Learned category orderings, keyed by column.
    ordered_cats: dict[str, list]

    # Column groupings.
    all_columns: list[str]
    predictor_columns: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    num_cols_x: list[str]
    cat_cols_x: list[str]

    # Resolved per-column encoding strategy for the categorical predictors, used
    # by models without native categorical support (default: frequency).
    categorical_encoding: dict[str, str]

    # Target handling.
    target_is_num: bool
    target_transforms: TargetTransforms

    # Scoring + RNG.
    scoring_fn: ScoringFn
    seed_seq: np.random.SeedSequence = field(repr=False)

    # --- convenience accessors ------------------------------------------------
    @property
    def target(self) -> list[str]:
        # Resolved by autodetect before the context is built (never None here).
        assert self.config.target is not None
        return self.config.target

    @property
    def id_col(self) -> list[str]:
        return self.config.id_col

    @property
    def seed(self) -> int | None:
        return self.config.seed

    @property
    def storage_dir(self) -> Path:
        return self.paths.storage_dir

    @property
    def target_width(self) -> int:
        return self.target_transforms.width


def build_context(config: Config, datasets: Datasets, paths: ResolvedPaths) -> PipelineContext:
    """Fit preprocessing on the training data and derive all run-wide state.

    ``datasets`` should hold the *raw* frames (post any speed-up subsampling).
    The pre-training pipeline is fitted on train and applied to test; column
    splits, target transforms, the scoring function and the seed sequence are
    all computed from the transformed training frame.
    """
    # These are autodetected (or set) by load_datasets before we get here.
    assert config.target is not None
    assert config.scoring is not None
    assert config.prediction_aim is not None
    pretrain = build_pretrain_pipeline(config)
    train_df = pretrain.fit_transform(datasets.train)
    test_df = pretrain.transform(datasets.test)
    ordered_cats = pretrain.named_steps[CATEGORICAL_TYPER_STEP].ordered_cats_

    all_columns = list(get_columns(train_df, config.id_col))
    predictor_columns = list(get_predictor_names(train_df, config.target, config.id_col))
    num_cols, cat_cols = split_num_cat(all_columns, train_df, cat_cutoff=config.cat_cutoff)
    num_cols_x, cat_cols_x = split_num_cat(
        predictor_columns, train_df, cat_cutoff=config.cat_cutoff
    )
    categorical_encoding = resolve_encoding_plan(config.categorical_encoding, train_df, cat_cols_x)

    target_transforms = build_target_transforms(
        train_df,
        target=config.target,
        target_is_num=config.target_is_num,
        ordered_cats=ordered_cats,
        prediction_aim=config.prediction_aim,
    )

    return PipelineContext(
        config=config,
        paths=paths,
        train_df=train_df,
        test_df=test_df,
        sample_df=datasets.sample,
        ordered_cats=ordered_cats,
        all_columns=all_columns,
        predictor_columns=predictor_columns,
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_cols_x=num_cols_x,
        cat_cols_x=cat_cols_x,
        categorical_encoding=categorical_encoding,
        target_is_num=config.target_is_num,
        target_transforms=target_transforms,
        scoring_fn=resolve_scoring(config.scoring),
        seed_seq=np.random.SeedSequence(config.seed),
    )
