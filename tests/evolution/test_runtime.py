"""RuntimeManager: budget carving, early-stop and ensemble-reserve logic."""

from __future__ import annotations

from kaggle_pipeline.evolution.runtime import RuntimeManager


def _rt(**kw) -> RuntimeManager:
    base = dict(
        max_runtime_seconds=1000,
        safety_margin_seconds=10,
        checkpoint_time_reserve_seconds=5,
        ensemble_time_reserve_seconds=100,
        finalization_time_reserve_seconds=20,
        enable_ensembling=True,
    )
    base.update(kw)
    return RuntimeManager(**base)


def test_training_deadline_reserves_for_ensemble():
    rt = _rt()
    # training window = 1000 - (10 + 5 + 20 + 100) = 865
    assert 855 < rt.remaining_training_seconds() <= 866
    assert 985 < rt.remaining_seconds() <= 991


def test_disabling_ensembling_frees_training_time():
    on = _rt(enable_ensembling=True)
    off = _rt(enable_ensembling=False)
    assert off.remaining_training_seconds() > on.remaining_training_seconds()
    assert not off.has_time_for_ensemble()
    assert on.has_time_for_ensemble()


def test_should_stop_and_cannot_start_past_deadline():
    rt = _rt()
    rt.training_deadline = rt.start_time - 1.0  # force past the training deadline
    assert rt.should_stop_training()
    assert not rt.can_start_batch(1.0)
    assert not rt.can_start_model_training(1.0)


def test_can_start_within_budget():
    rt = _rt()
    assert rt.can_start_batch(10.0)
    assert rt.can_start_model_training(10.0)
