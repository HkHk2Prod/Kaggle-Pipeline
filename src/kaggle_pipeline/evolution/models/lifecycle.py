"""Model lifecycle statuses and failure reasons.

Kept as plain string constants (grouped on :class:`ModelStatus` /
:class:`FailureReason`) so they serialise trivially and new states can be added
without touching call sites.
"""

from __future__ import annotations


class ModelStatus:
    """The lifecycle states a model genome moves through."""

    CREATED = "created"
    QUEUED = "queued"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"
    PROMOTED = "promoted"
    ARCHIVED = "archived"
    MUTATED = "mutated"

    ALL = frozenset(
        {CREATED, QUEUED, TRAINING, COMPLETED, FAILED, PRUNED, PROMOTED, ARCHIVED, MUTATED}
    )


class FailureReason:
    """Why a model training attempt failed (recorded on the result)."""

    INVALID_PARAMETER = "invalid_parameter"
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    NAN_PREDICTIONS = "nan_predictions"
    CONSTANT_PREDICTIONS = "constant_predictions"
    METRIC_ERROR = "metric_error"
    TRAINING_EXCEPTION = "training_exception"
