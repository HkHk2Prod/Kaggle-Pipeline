"""Train-size cap wrapper: subsample-on-fit, pass-through on predict, one-shot warning."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from kaggle_pipeline.evolution.models.training import (
    _stratified_subsample_indices,
    _TrainSizeCappedEstimator,
    reset_train_size_cap_warnings,
)


class _RecordingEstimator:
    """Sklearn-shaped stub that remembers the size of the data ``fit`` saw."""

    def __init__(self) -> None:
        self.fit_n: int | None = None
        self.predict_n: int | None = None

    def fit(self, X, y):  # noqa: ANN001 - duck-typed sklearn API
        self.fit_n = len(X)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):  # noqa: ANN001
        self.predict_n = len(X)
        return np.zeros(len(X))

    def predict_proba(self, X):  # noqa: ANN001
        return np.full((len(X), 2), 0.5)


@pytest.fixture(autouse=True)
def _reset_warnings():
    reset_train_size_cap_warnings()
    yield
    reset_train_size_cap_warnings()


def test_below_cap_passes_through_without_subsampling(caplog):
    inner = _RecordingEstimator()
    wrapped = _TrainSizeCappedEstimator(inner, max_rows=1000, family_name="dummy", seed=0)
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(500, 4)))
    y = (np.arange(500) % 2).astype(int)

    with caplog.at_level(logging.WARNING):
        wrapped.fit(X, y)

    assert inner.fit_n == 500  # untouched
    assert not any("capped" in r.getMessage() for r in caplog.records)


def test_above_cap_subsamples_and_logs_warning_once(caplog):
    inner = _RecordingEstimator()
    wrapped = _TrainSizeCappedEstimator(inner, max_rows=100, family_name="mlp", seed=0)
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(500, 4)))
    y = (np.arange(500) % 2).astype(int)

    with caplog.at_level(logging.WARNING, logger="kaggle_pipeline.evolution.models.training"):
        wrapped.fit(X, y)

    assert inner.fit_n == 100  # cap applied
    msgs = [r.getMessage() for r in caplog.records if "capped" in r.getMessage()]
    assert len(msgs) == 1
    # The user-facing fields must be present so the warning is actionable.
    assert "100 / 500" in msgs[0]
    assert "measured" in msgs[0] and "estimated full-data" in msgs[0]


def test_warning_dedup_is_per_family(caplog):
    # Two wrappers, same family -> one warning total. Different family ->
    # second warning fires.
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(500, 4)))
    y = (np.arange(500) % 2).astype(int)

    with caplog.at_level(logging.WARNING, logger="kaggle_pipeline.evolution.models.training"):
        _TrainSizeCappedEstimator(_RecordingEstimator(), max_rows=100, family_name="mlp").fit(X, y)
        _TrainSizeCappedEstimator(_RecordingEstimator(), max_rows=100, family_name="mlp").fit(X, y)
        _TrainSizeCappedEstimator(_RecordingEstimator(), max_rows=100, family_name="knn").fit(X, y)

    families = {r.args[0] for r in caplog.records if "capped" in r.getMessage()}
    assert families == {"mlp", "knn"}


def test_predict_uses_full_input_not_capped_subsample():
    inner = _RecordingEstimator()
    wrapped = _TrainSizeCappedEstimator(inner, max_rows=100, family_name="mlp", seed=0)
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(500, 4)))
    y = (np.arange(500) % 2).astype(int)
    wrapped.fit(X, y)

    X_test = pd.DataFrame(np.random.default_rng(1).normal(size=(800, 4)))
    proba = wrapped.predict_proba(X_test)
    assert proba.shape == (800, 2)  # full input width preserved


def test_stratified_subsample_preserves_class_balance_roughly():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(900), np.ones(100)]).astype(int)
    idx = _stratified_subsample_indices(y, max_rows=200, rng=rng)
    sampled = y[idx]
    assert len(sampled) == 200
    # 90/10 in -> ~180/20 out, allow some rounding slack.
    assert 170 <= int((sampled == 0).sum()) <= 190
    assert 10 <= int((sampled == 1).sum()) <= 30


def test_stratified_falls_back_to_random_for_regression_targets():
    rng = np.random.default_rng(0)
    y = rng.uniform(size=500)  # many unique values -> treated as regression
    idx = _stratified_subsample_indices(y, max_rows=100, rng=rng)
    assert idx.shape == (100,)
    # Plain random sample -> no class structure to preserve.
    assert len(np.unique(idx)) == 100
