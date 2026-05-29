"""Per-batch / per-model / per-submission cost estimates from observed timings.

The pipeline uses these estimates to decide whether to start another batch and
to size the submission reserve carved out of the runtime budget. Keeping them
in their own module makes the calibration math (median, scale factors,
bootstrap defaults) easy to test in isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kaggle_pipeline.evolution.config import KagglePipelineSettings
    from kaggle_pipeline.evolution.controllers.evolution_controller import EvolutionController


class RuntimeEstimator:
    """Bundle the cost-estimation heuristics used by the run loop.

    The estimator reads timings off the controller's completed genomes and the
    pipeline's settings; it owns no state of its own, so a fresh instance can
    be constructed cheaply each time an estimate is needed.
    """

    def __init__(
        self,
        settings: KagglePipelineSettings,
        controller: EvolutionController | None,
        *,
        model_workers: int,
    ) -> None:
        self.settings = settings
        self.controller = controller
        self.model_workers = max(1, model_workers)

    def completed_compute_times(self) -> list[float]:
        if self.controller is None:
            return []
        return [
            g.score_set.compute_time
            for g in self.controller.population.completed()
            if g.score_set is not None and g.score_set.compute_time > 0
        ]

    def model_seconds(self) -> float:
        times = self.completed_compute_times()
        if times:
            # Slight safety factor over the observed median.
            return float(np.median(times)) * 1.2
        return 5.0  # optimistic bootstrap before any timing history exists

    def batch_seconds(self) -> float:
        per_model = self.model_seconds()
        return (self.settings.models_per_batch * per_model) / self.model_workers + 5.0

    def submission_seconds(self) -> float:
        """Estimate make_submission cost from measured per-model search times.

        Each ensemble member is refit on the FULL training data, so its refit
        cost is roughly the median search-time per model scaled by
        ``1 / search_sample_fraction`` (linear in row count for the trees and
        gradient boosters we use). The submission step refits up to
        ``ensemble_max_models`` members and predicts test. A 1.3x safety
        multiplier over the median keeps the estimate conservative; CSV write
        is microseconds and not worth modelling. Falls back to the bootstrap
        default when there is no timing history yet.
        """
        times = self.completed_compute_times()
        if not times:
            return self.settings.submission_time_reserve_seconds
        median_search = float(np.median(times))
        scale_full = 1.0 / max(0.01, self.settings.search_sample_fraction)
        per_refit = median_search * scale_full * 1.3
        return per_refit * self.settings.ensemble_max_models
