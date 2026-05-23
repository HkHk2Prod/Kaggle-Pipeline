"""Submission writing: predictions must be joined to the sample rows by ``id``.

The previous writer dropped predictions into the sample submission by row
position, which silently scrambles every prediction when ``test.csv`` and the
sample submission are sorted differently -- the file still looks well-formed but
scores at chance. These tests pin the id-join behaviour and the loud failure on
an id mismatch.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kaggle_pipeline import Config, build_pipeline
from kaggle_pipeline.submission import write_submission


def test_submission_joins_on_id_not_position(smoke_config: Config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx, _ = build_pipeline(smoke_config)

    id_col, target = ctx.id_col[0], ctx.target[0]
    # Use each row's own id as its (stand-in) prediction so we can recover, from
    # the written file, which test id every value actually came from.
    y = ctx.test_df[id_col].to_numpy().astype(float)
    # Scramble the sample submission's row order: positional alignment would now
    # attach each prediction to the wrong id, but an id-join must not.
    ctx.sample_df = ctx.sample_df.iloc[::-1].reset_index(drop=True)

    out_path = write_submission(ctx, y)
    sub = pd.read_csv(out_path)

    # Every row's prediction equals its own id -> joined by id, not by position.
    assert (sub[target].to_numpy() == sub[id_col].to_numpy().astype(float)).all()
    # The sample's (scrambled) row order is preserved in the output.
    assert list(sub[id_col]) == list(ctx.sample_df[id_col])


def test_submission_raises_on_id_mismatch(smoke_config: Config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx, _ = build_pipeline(smoke_config)

    id_col = ctx.id_col[0]
    y = ctx.test_df[id_col].to_numpy().astype(float)
    # Make one sample id absent from the test ids: the writer must refuse rather
    # than emit a submission with a silently-null prediction.
    ctx.sample_df = ctx.sample_df.copy()
    ctx.sample_df.loc[0, id_col] = 999_999

    with pytest.raises(ValueError, match="do not match"):
        write_submission(ctx, y)
