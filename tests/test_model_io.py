"""Model persistence: ``save`` then ``load`` must rebuild the *tuned* model.

The original notebook rebuilt a loaded model's pipeline with fresh random
hyperparameters and never re-applied the saved ones, so reloaded ensemble
members were refit incorrectly. ``Model.load`` rebuilds from the saved
``_param``; these tests pin that behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from kaggle_pipeline import Config, build_pipeline
from kaggle_pipeline.models import Model, registry


@pytest.fixture
def fitted_ctx(smoke_config: Config):
    ctx, _ = build_pipeline(smoke_config)
    return ctx


def _xy(ctx):
    X = ctx.train_df[ctx.predictor_columns]
    y = ctx.target_transforms.forward(ctx.train_df[ctx.target])
    return X, y


def test_save_load_round_trip_preserves_params_and_predictions(fitted_ctx, tmp_path):
    ctx = fitted_ctx
    X, y = _xy(ctx)

    model = registry["LogisticRegression"](ctx, complexity=1.0)
    model.fit(X, y)
    original_pred = model.predict(X)

    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = Model.load(path, ctx)

    # The tuned hyperparameters survive the round trip...
    assert loaded.params == model.params
    # ...and the rebuilt pipeline, refit on the same data, predicts identically
    # (deterministic because random_state is threaded from config.seed).
    loaded.fit(X, y)
    np.testing.assert_allclose(loaded.predict(X), original_pred)


def test_load_missing_file_raises(fitted_ctx, tmp_path):
    with pytest.raises(ValueError, match="missing data file"):
        Model.load(tmp_path / "does_not_exist.pkl", fitted_ctx)
