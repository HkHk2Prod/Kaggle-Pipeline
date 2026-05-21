"""End-to-end smoke tests on the synthetic dataset.

These run the *whole* pipeline on tiny data to prove the wiring works: data
loading, preprocessing, context construction, the model search and the stacked
ensemble, then submission writing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kaggle_pipeline import Config, build_pipeline, run


def test_build_context_column_splits(smoke_config: Config):
    ctx, paths = build_pipeline(smoke_config)
    assert ctx.target_is_num is False
    assert ctx.target_width == 2
    # Ordinal cats (cat1/cat2) become integer-encoded -> numeric predictors.
    assert set(ctx.num_cols_x) == {"num1", "num2", "cat1", "cat2"}
    assert ctx.cat_cols_x == []
    assert "id" not in ctx.predictor_columns
    assert "y" not in ctx.predictor_columns


def test_run_writes_submission(smoke_config: Config, tmp_path, monkeypatch):
    # The submission is written to the working dir (cwd locally); isolate it.
    monkeypatch.chdir(tmp_path)
    out_path = run(smoke_config)

    assert Path(out_path).exists()
    submission = pd.read_csv(out_path)
    assert len(submission) == 80
    assert list(submission.columns) == ["id", "y"]
    assert set(submission["y"]).issubset({"no", "yes"})


def test_example_yaml_config_loads():
    cfg = Config.from_yaml("configs/playground-s6e4.yaml")
    assert cfg.target == ["Irrigation_Need"]
    assert cfg.scoring == "balanced_accuracy"
    assert len(cfg.feature_expressions) == 4
