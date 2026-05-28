"""The :class:`RuntimeManager` -- the 12-hour budget and early-stop logic.

Uses ``time.monotonic`` (never wall-clock) so clock adjustments cannot corrupt
the budget. It carves the total runtime into a training window and reserves for
checkpointing, ensembling (when enabled), finalization, and -- when the
orchestrator will write a submission itself -- the submission step (refit each
ensemble member on the full train set, predict test, write CSV). It answers the
questions the orchestrator asks: can a batch/model still start, should training
stop, should we checkpoint, should ensembling begin, does submission still fit.

The submission reserve is *mutable*: a fixed default is conservative at startup,
and the orchestrator overwrites it with a data-aware estimate as soon as it has
measured per-model timings (see ``KagglePipeline._estimated_submission_seconds``).
Each query recomputes the training deadline from the current reserve, so a
shrinking estimate frees training time and a growing one stops the loop sooner.

The pipeline must stop *itself* before the deadline -- it does not rely on an
external timeout killing the process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Conservative default when we have no measured model time yet (seconds).
DEFAULT_MODEL_TIME_ESTIMATE = 30.0
DEFAULT_BATCH_TIME_ESTIMATE = 120.0


@dataclass
class RuntimeManager:
    """Tracks elapsed monotonic time against a training deadline and reserves."""

    max_runtime_seconds: float = 12 * 60 * 60
    safety_margin_seconds: float = 10 * 60
    checkpoint_time_reserve_seconds: float = 2 * 60
    ensemble_time_reserve_seconds: float = 30 * 60
    finalization_time_reserve_seconds: float = 5 * 60
    # Time held back for the submission step (refit each ensemble member on the
    # full train data, predict test, write CSV). ``0.0`` means "no submission
    # planned" -- the orchestrator is responsible for fitting any later
    # ``make_submission`` call into whatever time is left. The orchestrator may
    # overwrite this field at any point (e.g. after a batch, once it has real
    # timings) to refine the reserve.
    submission_time_reserve_seconds: float = 0.0
    enable_ensembling: bool = True
    start_time: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        # The final deadline is the hard wall -- nothing recomputes it; the
        # training deadline is computed *per-query* off the current submission
        # reserve, so an updated estimate takes effect immediately.
        self.final_deadline = (
            self.start_time + self.max_runtime_seconds - self.safety_margin_seconds
        )

    def reset(self) -> None:
        """Restart the clock (e.g. after restoring from a checkpoint)."""
        self.start_time = time.monotonic()
        self.__post_init__()

    # --- internals ----------------------------------------------------------
    def _reserves_total(self) -> float:
        return (
            self.safety_margin_seconds
            + self.checkpoint_time_reserve_seconds
            + self.finalization_time_reserve_seconds
            + (self.ensemble_time_reserve_seconds if self.enable_ensembling else 0.0)
            + max(0.0, self.submission_time_reserve_seconds)
        )

    def _training_deadline(self) -> float:
        return self.start_time + self.max_runtime_seconds - self._reserves_total()

    # --- queries ------------------------------------------------------------
    @property
    def training_deadline(self) -> float:
        """Backwards-compatible read of the current (dynamic) training deadline."""
        return self._training_deadline()

    @training_deadline.setter
    def training_deadline(self, value: float) -> None:
        # Tests force the deadline past `now` to simulate exhaustion. Re-derive
        # an effective ``submission_time_reserve_seconds`` so a later mutation
        # of the actual reserve still leaves the deadline in the past.
        baseline = self.start_time + self.max_runtime_seconds - self._reserves_total()
        self.submission_time_reserve_seconds += baseline - value

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def remaining_seconds(self) -> float:
        return max(0.0, self.final_deadline - time.monotonic())

    def remaining_training_seconds(self) -> float:
        return max(0.0, self._training_deadline() - time.monotonic())

    def remaining_finalization_seconds(self) -> float:
        return max(0.0, self.final_deadline - time.monotonic())

    def remaining_submission_seconds(self) -> float:
        """How much wall time the submission step still has before the deadline."""
        return max(0.0, self.final_deadline - time.monotonic())

    def is_close_to_deadline(self) -> bool:
        return self.remaining_training_seconds() <= 0.0

    def should_stop_training(self) -> bool:
        return self.remaining_training_seconds() <= 0.0

    def can_start_batch(self, estimated_batch_seconds: float = DEFAULT_BATCH_TIME_ESTIMATE) -> bool:
        return self.remaining_training_seconds() >= estimated_batch_seconds

    def can_start_model_training(
        self, estimated_model_seconds: float = DEFAULT_MODEL_TIME_ESTIMATE
    ) -> bool:
        return self.remaining_training_seconds() >= estimated_model_seconds

    def should_checkpoint(self, last_checkpoint_time: float, interval_seconds: float) -> bool:
        return (time.monotonic() - last_checkpoint_time) >= interval_seconds

    def has_time_for_ensemble(self) -> bool:
        if not self.enable_ensembling:
            return False
        return self.remaining_finalization_seconds() >= self.finalization_time_reserve_seconds

    def has_time_for_submission(self) -> bool:
        """True iff the current submission reserve still fits before the deadline.

        Returns False when ``submission_time_reserve_seconds <= 0`` (no
        submission planned) or when there is less wall time left than the reserve
        currently estimates the submission step will need.
        """
        if self.submission_time_reserve_seconds <= 0:
            return False
        return self.remaining_submission_seconds() >= self.submission_time_reserve_seconds

    def time_summary(self) -> dict[str, float]:
        return {
            "elapsed": self.elapsed_seconds(),
            "remaining": self.remaining_seconds(),
            "remaining_training": self.remaining_training_seconds(),
            "ensemble_reserved": self.ensemble_time_reserve_seconds
            if self.enable_ensembling
            else 0.0,
            "submission_reserved": max(0.0, self.submission_time_reserve_seconds),
            "close_to_deadline": float(self.is_close_to_deadline()),
        }
