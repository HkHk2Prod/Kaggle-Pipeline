"""A fixed ``seed`` must reproduce the search and the final predictions.

This guards the reproducibility fix: the leaderboard's class selection and the
ensemble search both draw from the run's seed sequence / ``config.seed`` rather
than the global RNG, so two runs with the same seed are bit-for-bit identical.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from kaggle_pipeline import Config, predict


def test_same_seed_gives_identical_predictions(smoke_config: Config, tmp_path):
    # Distinct storage dirs so the second run starts fresh instead of resuming
    # the first run's checkpointed leaderboard.
    pred_a = predict(replace(smoke_config, storage_dir=tmp_path / "run_a"))
    pred_b = predict(replace(smoke_config, storage_dir=tmp_path / "run_b"))
    assert np.array_equal(pred_a, pred_b)


def test_different_seed_changes_the_search(smoke_config: Config, tmp_path):
    # The seed must actually influence the run: with different seeds the sampled
    # hyperparameters (and thus the ensemble's probabilities) differ. Compare the
    # continuous probability output -- far less likely than hard labels to
    # coincide by chance on tiny data.
    base = replace(smoke_config, prediction_aim="probability")
    prob_a = predict(replace(base, seed=1, storage_dir=tmp_path / "s1"))
    prob_b = predict(replace(base, seed=2, storage_dir=tmp_path / "s2"))
    assert prob_a.shape == prob_b.shape
    assert not np.array_equal(prob_a, prob_b)
