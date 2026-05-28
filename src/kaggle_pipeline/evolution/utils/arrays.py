"""Small array helpers shared across the feature layer.

Centralises the "is this categorical value missing?" idiom -- a missing marker in
an object array is either ``None`` or a float ``NaN`` -- so it is defined once
rather than re-spelled at every call site.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def is_missing(value: Any) -> bool:
    """True if ``value`` is a missing marker (``None`` or float ``NaN``)."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def missing_mask(values: Any) -> np.ndarray:
    """Boolean mask of missing entries over the raveled values."""
    raveled = np.asarray(values, dtype=object).ravel()
    return np.array([is_missing(v) for v in raveled], dtype=bool)
