"""Unit tests for the verbosity-tiered log helpers."""

from __future__ import annotations

from types import SimpleNamespace

from kaggle_pipeline.evolution.logging_utils import Verbosity
from kaggle_pipeline.evolution.pipeline_log import (
    log_feature_generation,
    log_runtime_budget,
)


def _capture_log():
    """Return ``(log_fn, captured)`` -- a sink that appends ``(message, level)``."""
    captured: list[tuple[str, int]] = []

    def log_fn(message, *, level=Verbosity.NORMAL):
        captured.append((message, level))

    return log_fn, captured


def _settings(**overrides):
    defaults = dict(
        verbosity=Verbosity.DETAILED,
        max_runtime_seconds=3600.0,
        safety_margin_seconds=30.0,
        checkpoint_time_reserve_seconds=20.0,
        finalization_time_reserve_seconds=10.0,
        ensemble_time_reserve_seconds=15.0,
        enable_ensembling=True,
        make_submission_on_run=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _runtime(**overrides):
    defaults = dict(
        submission_time_reserve_seconds=50.0,
        remaining_training_seconds=lambda: 1000.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_log_runtime_budget_silent_below_detailed():
    log_fn, captured = _capture_log()
    log_runtime_budget(log_fn, runtime=_runtime(), settings=_settings(verbosity=Verbosity.NORMAL))
    assert captured == []


def test_log_runtime_budget_emits_at_detailed():
    log_fn, captured = _capture_log()
    log_runtime_budget(log_fn, runtime=_runtime(), settings=_settings())
    assert len(captured) == 1
    message, level = captured[0]
    assert level == Verbosity.DETAILED
    assert "total=3600s" in message
    assert "training_window=1000s" in message


def test_log_runtime_budget_flags_disabled_ensemble_and_submission():
    log_fn, captured = _capture_log()
    log_runtime_budget(
        log_fn,
        runtime=_runtime(submission_time_reserve_seconds=0.0),
        settings=_settings(enable_ensembling=False, make_submission_on_run=False),
    )
    message, _ = captured[0]
    assert "ensemble=15s (off)" in message
    assert "submission=0s (off)" in message


def test_log_feature_generation_no_op_for_empty_summary():
    log_fn, captured = _capture_log()
    log_feature_generation(
        log_fn,
        SimpleNamespace(generated_feature_names=[], n_features_active=0),
        controller=None,
        verbosity=Verbosity.SUMMARY,
    )
    assert captured == []


def test_log_feature_generation_summary_and_detail_tiers():
    log_fn, captured = _capture_log()
    log_feature_generation(
        log_fn,
        SimpleNamespace(generated_feature_names=["foo", "bar"], n_features_active=10),
        controller=None,
        verbosity=Verbosity.DETAILED,
    )
    levels = [lvl for _, lvl in captured]
    assert Verbosity.SUMMARY in levels and Verbosity.DETAILED in levels
    assert any("+2 new (10 active)" in m for m, _ in captured)


def test_log_feature_generation_debug_tier_emits_depth_and_utility():
    log_fn, captured = _capture_log()
    feature = SimpleNamespace(human_name="foo", depth=3, utility=0.42)
    controller = SimpleNamespace(registry=SimpleNamespace(all_features=lambda: [feature]))
    log_feature_generation(
        log_fn,
        SimpleNamespace(generated_feature_names=["foo"], n_features_active=5),
        controller=controller,
        verbosity=Verbosity.DEBUG,
    )
    debug_lines = [m for m, lvl in captured if lvl == Verbosity.DEBUG]
    assert any("foo(depth=3, util=0.420)" in m for m in debug_lines)
