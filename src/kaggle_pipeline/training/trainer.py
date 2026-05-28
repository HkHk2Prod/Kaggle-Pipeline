"""The training loop: run search steps under a time budget, then ensemble."""

from __future__ import annotations

import itertools
import logging
import time

import numpy as np

from kaggle_pipeline.context import PipelineContext
from kaggle_pipeline.search import Judge
from kaggle_pipeline.search.cv import make_cv_splitter

logger = logging.getLogger(__name__)


def run_training(ctx: PipelineContext) -> np.ndarray:
    """Search for models step-by-step, then return ensembled test predictions.

    Each step is cross-validated and the leaderboard is checkpointed to disk, so
    a Kaggle kernel that is interrupted can resume from the last saved board.
    The loop stops early if another step would risk exceeding
    ``config.max_running_time``. With ``config.n_steps`` set to ``None`` the
    search runs until that time budget is the only thing that stops it.
    """
    config = ctx.config
    start_time = time.perf_counter()

    cv = make_cv_splitter(n_splits=config.cv_splits, seed=config.seed, task=config.task)
    judge = Judge(ctx, cv)
    judge.load()

    steps = itertools.count() if config.n_steps is None else range(config.n_steps)
    for i in steps:
        compute_time = judge.step()
        judge.save()
        if config.n_steps is None:
            logger.info("%d steps done.\n", i + 1)
        else:
            logger.info("%d steps done out of %d.\n", i + 1, config.n_steps)
        elapsed = time.perf_counter() - start_time
        if elapsed + 3 * compute_time > config.max_running_time:
            logger.info("We are low on time and stop the training cycle.")
            break

    return judge.predict()
