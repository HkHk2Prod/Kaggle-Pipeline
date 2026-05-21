"""The training loop: run search steps under a time budget, then ensemble."""

from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from kaggle_pipeline.context import PipelineContext
from kaggle_pipeline.search import Judge


def run_training(ctx: PipelineContext) -> np.ndarray:
    """Search for models step-by-step, then return ensembled test predictions.

    Each step is cross-validated and the leaderboard is checkpointed to disk, so
    a Kaggle kernel that is interrupted can resume from the last saved board.
    The loop stops early if another step would risk exceeding
    ``config.max_running_time``.
    """
    config = ctx.config
    start_time = time.perf_counter()

    cv = StratifiedKFold(n_splits=config.cv_splits, shuffle=True, random_state=config.cv_seed)
    judge = Judge(ctx, cv)
    judge.load()

    for i in range(config.n_steps):
        compute_time = judge.step()
        judge.save()
        print(f"{i + 1} steps done out of {config.n_steps}.\n")
        elapsed = time.perf_counter() - start_time
        if elapsed + 3 * compute_time > config.max_running_time:
            print("We are low on time and stop the training cycle.")
            break

    return judge.predict()
