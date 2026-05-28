"""RuntimeManager: budget carving, early-stop and ensemble/submission reserves."""

from __future__ import annotations

from kaggle_pipeline.evolution.runtime import RuntimeManager


def _rt(**kw) -> RuntimeManager:
    base = dict(
        max_runtime_seconds=1000,
        safety_margin_seconds=10,
        checkpoint_time_reserve_seconds=5,
        ensemble_time_reserve_seconds=100,
        finalization_time_reserve_seconds=20,
        submission_time_reserve_seconds=0.0,  # disabled by default; tests opt in
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


def test_submission_reserve_only_applies_when_positive():
    # The reserve is gated by ``submission_time_reserve_seconds > 0`` -- no
    # separate boolean. 0 means "no submission planned".
    off = _rt(submission_time_reserve_seconds=0.0)
    on = _rt(submission_time_reserve_seconds=50.0)
    diff = off.remaining_training_seconds() - on.remaining_training_seconds()
    assert 49 < diff < 51
    assert on.has_time_for_submission()
    assert not off.has_time_for_submission()


def test_submission_reserve_is_mutable_and_recomputes_deadline():
    # A later, refined estimate must take effect on the next query -- the
    # training deadline is derived per-call, not cached.
    rt = _rt(submission_time_reserve_seconds=50.0)
    baseline = rt.remaining_training_seconds()
    rt.submission_time_reserve_seconds = 150.0  # 100s more held back
    tightened = rt.remaining_training_seconds()
    assert 99 < (baseline - tightened) < 101


def test_submission_reserve_fits_within_final_deadline():
    rt = _rt(submission_time_reserve_seconds=50.0)
    # Once less time remains than the reserve, the query must say no.
    rt.final_deadline = rt.start_time + 40.0
    assert not rt.has_time_for_submission()
