"""Unit tests for RuntimeEstimator cost heuristics."""

from __future__ import annotations

from types import SimpleNamespace

from kaggle_pipeline.evolution.runtime_estimator import RuntimeEstimator


def _settings(**overrides):
    defaults = dict(
        models_per_batch=4,
        search_sample_fraction=0.25,
        ensemble_max_models=5,
        submission_time_reserve_seconds=120.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _controller(times: list[float]):
    completed = [SimpleNamespace(score_set=SimpleNamespace(compute_time=t)) for t in times]
    return SimpleNamespace(population=SimpleNamespace(completed=lambda: completed))


def test_model_seconds_uses_bootstrap_when_no_history():
    est = RuntimeEstimator(_settings(), None, model_workers=2)
    assert est.model_seconds() == 5.0


def test_model_seconds_scales_median_by_safety_factor():
    est = RuntimeEstimator(_settings(), _controller([1.0, 2.0, 3.0]), model_workers=2)
    assert est.model_seconds() == 2.0 * 1.2  # median * 1.2


def test_completed_compute_times_filters_out_zero_and_missing():
    controller = SimpleNamespace(
        population=SimpleNamespace(
            completed=lambda: [
                SimpleNamespace(score_set=None),
                SimpleNamespace(score_set=SimpleNamespace(compute_time=0)),
                SimpleNamespace(score_set=SimpleNamespace(compute_time=1.5)),
            ]
        )
    )
    est = RuntimeEstimator(_settings(), controller, model_workers=1)
    assert est.completed_compute_times() == [1.5]


def test_batch_seconds_divides_by_workers_and_adds_overhead():
    est = RuntimeEstimator(_settings(models_per_batch=4), _controller([10.0]), model_workers=2)
    # per_model = 10 * 1.2 = 12 ; batch = 4*12/2 + 5 = 24 + 5 = 29
    assert est.batch_seconds() == 4 * 12.0 / 2 + 5.0


def test_submission_seconds_returns_bootstrap_when_no_history():
    est = RuntimeEstimator(_settings(submission_time_reserve_seconds=99.0), None, model_workers=1)
    assert est.submission_seconds() == 99.0


def test_submission_seconds_scales_by_full_train_fraction():
    est = RuntimeEstimator(
        _settings(search_sample_fraction=0.25, ensemble_max_models=3),
        _controller([2.0]),
        model_workers=1,
    )
    # per_refit = median(2.0) * (1/0.25) * 1.3 = 10.4; * 3 members = 31.2
    assert est.submission_seconds() == 2.0 * 4.0 * 1.3 * 3


def test_model_workers_clamped_to_at_least_one():
    est = RuntimeEstimator(_settings(), None, model_workers=0)
    assert est.model_workers == 1
