"""Parallel ensemble refit: each member runs concurrently, results align with weights."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from kaggle_pipeline.evolution.ensemble.manager import EnsembleManager, EnsembleResult


class _RecordingTrainer:
    """Returns a deterministic probability per model id and records the calling thread."""

    def __init__(self, n_rows: int = 4) -> None:
        self.n_rows = n_rows
        # Each member gets a distinct constant probability so weight-vs-member
        # alignment errors surface immediately as a wrong mean.
        self.probas = {
            "m_a": np.tile([0.9, 0.1], (n_rows, 1)),
            "m_b": np.tile([0.6, 0.4], (n_rows, 1)),
            "m_c": np.tile([0.2, 0.8], (n_rows, 1)),
        }
        self.threads: list[str] = []
        self._lock = threading.Lock()

    def fit_predict_test(self, genome, *, train_frame, y, test_frame, task, seed):
        with self._lock:
            self.threads.append(threading.current_thread().name)
        return self.probas[genome.model_id]


class _StubPopulation:
    def __init__(self) -> None:
        self.genomes = {mid: _StubGenome(mid) for mid in ("m_a", "m_b", "m_c")}

    def get(self, mid: str) -> _StubGenome:
        return self.genomes[mid]


class _StubGenome:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id


def _settings_stub():
    class S:
        pass

    return S()


def _result() -> EnsembleResult:
    return EnsembleResult(
        status="greedy",
        member_ids=["m_a", "m_b", "m_c"],
        weights={"m_a": 0.5, "m_b": 0.3, "m_c": 0.2},
        n_members=3,
    )


def _expected_blend(probas, weights) -> np.ndarray:
    # Weighted average normalised by total weight (mirrors weighted_average).
    total = sum(weights.values())
    out = np.zeros_like(probas[next(iter(probas))])
    for mid, w in weights.items():
        out += probas[mid] * (w / total)
    return out


def test_predict_is_sequential_when_no_executor():
    trainer = _RecordingTrainer()
    blended = EnsembleManager(_settings_stub()).predict(
        _result(),
        trainer=trainer,
        population=_StubPopulation(),
        train_frame=pd.DataFrame(),
        y=np.zeros(4),
        test_frame=pd.DataFrame(),
        executor=None,
    )

    np.testing.assert_allclose(blended, _expected_blend(trainer.probas, _result().weights))
    # All calls came from the main thread.
    assert all(t == threading.current_thread().name for t in trainer.threads)


def test_predict_is_parallel_with_executor():
    trainer = _RecordingTrainer()
    # 3 workers, 3 members -> all three should run on the pool threads, not
    # the main thread. We don't assert simultaneity (flaky on busy CI) -- we
    # just verify the work was dispatched to the executor.
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="evo-test") as pool:
        blended = EnsembleManager(_settings_stub()).predict(
            _result(),
            trainer=trainer,
            population=_StubPopulation(),
            train_frame=pd.DataFrame(),
            y=np.zeros(4),
            test_frame=pd.DataFrame(),
            executor=pool,
        )

    # Same blended output regardless of execution mode.
    np.testing.assert_allclose(blended, _expected_blend(trainer.probas, _result().weights))
    # Every refit ran on a worker thread, never on the caller.
    assert all(t.startswith("evo-test") for t in trainer.threads)
    assert len(trainer.threads) == 3
