"""The search loop in :func:`kaggle_pipeline.training.run_training`.

These tests stub out :class:`~kaggle_pipeline.search.Judge` so they exercise the
loop's control flow (step count vs. time budget) without running a real search.
"""

from __future__ import annotations

import types

import numpy as np

from kaggle_pipeline.config import Config
from kaggle_pipeline.training import trainer as trainer_module
from kaggle_pipeline.training.trainer import run_training

SENTINEL = np.array([0.0, 1.0])


class FakeJudge:
    """A Judge stand-in whose ``step`` returns scripted compute times (seconds)."""

    def __init__(self, ctx, cv, compute_times):
        self._compute_times = list(compute_times)
        self.steps_run = 0

    def load(self) -> None:
        pass

    def save(self) -> None:
        pass

    def step(self) -> float:
        # Return the next scripted compute time, repeating the last value once the
        # script is exhausted so a finite n_steps run never runs off the end.
        value = self._compute_times[min(self.steps_run, len(self._compute_times) - 1)]
        self.steps_run += 1
        return value

    def predict(self) -> np.ndarray:
        return SENTINEL


def _ctx(**overrides):
    return types.SimpleNamespace(config=Config(**overrides))


def _install_fake_judge(monkeypatch, compute_times):
    holder = {}

    def factory(ctx, cv):
        holder["judge"] = FakeJudge(ctx, cv, compute_times)
        return holder["judge"]

    monkeypatch.setattr(trainer_module, "Judge", factory)
    return holder


def test_none_n_steps_runs_until_time_budget(monkeypatch):
    """n_steps=None loops until a step would risk exceeding max_running_time."""
    # First three steps are ~free, then one reports a compute time large enough
    # that 3x it overruns the 10s budget, so the loop breaks after step 4.
    holder = _install_fake_judge(monkeypatch, [0.0, 0.0, 0.0, 100.0])
    result = run_training(_ctx(n_steps=None, max_running_time=10))

    assert holder["judge"].steps_run == 4
    assert result is SENTINEL


def test_fixed_n_steps_stops_at_count(monkeypatch):
    """A finite n_steps stops at the count when the time budget is generous."""
    holder = _install_fake_judge(monkeypatch, [0.0])
    result = run_training(_ctx(n_steps=3, max_running_time=10_000))

    assert holder["judge"].steps_run == 3
    assert result is SENTINEL


def test_fixed_n_steps_can_still_stop_early_on_time(monkeypatch):
    """Even with a finite n_steps, the time-budget guard can break first."""
    holder = _install_fake_judge(monkeypatch, [100.0])
    result = run_training(_ctx(n_steps=50, max_running_time=10))

    assert holder["judge"].steps_run == 1
    assert result is SENTINEL
