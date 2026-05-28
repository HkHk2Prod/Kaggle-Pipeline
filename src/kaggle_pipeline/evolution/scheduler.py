"""Thread pools for parallel feature/model work.

Separate pools for feature work and model training keep the two from starving
each other. The orchestrator submits *pure* tasks (e.g. ``ModelTrainer.train``,
which reads but never mutates shared state) to ``model_executor`` and applies the
returned results on the main thread, so there are no races on the registries.

Avoids CPU oversubscription by training models with ``n_jobs=1`` (the estimator
builders set this) while several models train in parallel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


class TaskScheduler:
    """Owns the feature and model thread pools and shuts them down cleanly."""

    def __init__(self, *, model_workers: int = 1, feature_workers: int = 1):
        self.model_workers = max(1, model_workers)
        self.feature_workers = max(1, feature_workers)
        self.model_executor = ThreadPoolExecutor(
            max_workers=self.model_workers, thread_name_prefix="evo-model"
        )
        self.feature_executor = ThreadPoolExecutor(
            max_workers=self.feature_workers, thread_name_prefix="evo-feature"
        )
        self._shut = False

    @property
    def parallel_models(self) -> bool:
        return self.model_workers > 1

    def model_pool(self) -> ThreadPoolExecutor | None:
        """The executor to use for model training, or ``None`` to run sequentially."""
        return self.model_executor if self.parallel_models else None

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        if self._shut:
            return
        self._shut = True
        self.model_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        self.feature_executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> TaskScheduler:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown(wait=True)
