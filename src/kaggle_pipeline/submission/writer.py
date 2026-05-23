"""Write predictions into the competition's submission format."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from kaggle_pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


def write_submission(ctx: PipelineContext, y: np.ndarray) -> Path:
    """Join predictions onto the sample submission by ``id`` and save the CSV.

    ``y`` is in ``ctx.test_df`` row order. ``test.csv`` and the sample submission
    are *not* guaranteed to share a row order, so predictions are matched to the
    sample's rows on the id column(s) rather than by position. Aligning by
    position (the previous behaviour) silently scrambles every prediction
    whenever the two files are sorted differently -- the submission still looks
    well-formed but scores at chance. The id sets are asserted equal first so a
    mismatch fails loudly instead of yielding a submission full of nulls. The
    sample's original column order is preserved.
    """
    id_cols = list(ctx.id_col)
    target = ctx.target[0]

    # Tag the predictions with the test ids (same row order as ``y``). The
    # pre-training pipeline can retype an id column (e.g. string -> category), so
    # coerce each id back to the sample submission's dtype before joining.
    preds = ctx.test_df[id_cols].copy()
    for col in id_cols:
        preds[col] = preds[col].astype(ctx.sample_df[col].dtype)
    preds[target] = y

    sample_ids = set(map(tuple, ctx.sample_df[id_cols].to_numpy().tolist()))
    pred_ids = set(map(tuple, preds[id_cols].to_numpy().tolist()))
    if sample_ids != pred_ids:
        raise ValueError(
            f"Test ids and sample-submission ids do not match on {id_cols}: "
            f"{len(pred_ids)} test id(s) vs {len(sample_ids)} sample id(s), "
            f"{len(pred_ids & sample_ids)} in common. Cannot build a submission."
        )

    result = ctx.sample_df.drop(columns=[target]).merge(preds, on=id_cols, how="left")
    if len(result) != len(ctx.sample_df):
        raise ValueError(
            f"Joining predictions on {id_cols} changed the row count "
            f"({len(ctx.sample_df)} -> {len(result)}); id column(s) are not unique."
        )
    result = result[list(ctx.sample_df.columns)]

    out_path = ctx.paths.working_dir / f"{ctx.config.submission_name}.csv"
    result.to_csv(out_path, index=False)
    logger.debug("Submission preview:\n%s", result.head(10))
    logger.info("Submission written to %s", out_path)
    return out_path
