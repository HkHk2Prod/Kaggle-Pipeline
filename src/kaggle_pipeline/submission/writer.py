"""Write predictions into the competition's submission format."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from kaggle_pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


def write_submission(ctx: PipelineContext, y: np.ndarray) -> Path:
    """Drop predictions into a copy of the sample submission and save the CSV.

    The target column of the sample submission is overwritten with ``y`` and the
    original column order is preserved, so the output matches exactly what the
    competition expects. The file is written to the working directory as
    ``<submission_name>.csv``.
    """
    result = ctx.sample_df.copy()
    result[ctx.target[0]] = y
    result = result[list(ctx.sample_df.columns)]

    out_path = ctx.paths.working_dir / f"{ctx.config.submission_name}.csv"
    result.to_csv(out_path, index=False)
    logger.debug("Submission preview:\n%s", result.head(10))
    logger.info("Submission written to %s", out_path)
    return out_path
