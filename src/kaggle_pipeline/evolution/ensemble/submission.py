"""Writing a submission CSV from final predictions.

Kept decoupled from any particular target encoding: pass a ``decode`` callable
(e.g. the v1 ``TargetTransforms.inverse``) to turn the prediction matrix into the
submission column(s). Without one, a sensible default is used (positive-class
probability for binary, arg-max label for multiclass, raw values otherwise).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _default_decode(predictions: np.ndarray) -> np.ndarray:
    arr = np.asarray(predictions)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1]
    if arr.ndim == 2 and arr.shape[1] > 2:
        return arr.argmax(axis=1)
    return arr.ravel()


def write_submission(
    path: str | Path,
    *,
    ids: Any,
    predictions: np.ndarray,
    id_col: str = "id",
    target_col: str = "target",
    decode: Callable[[np.ndarray], Any] | None = None,
) -> Path:
    """Write ``id_col,target_col`` rows to ``path`` and return it."""
    decoded = (decode or _default_decode)(predictions)
    decoded = np.asarray(decoded)
    frame = pd.DataFrame({id_col: np.asarray(ids)})
    if decoded.ndim == 2 and decoded.shape[1] > 1:
        for i in range(decoded.shape[1]):
            frame[f"{target_col}_{i}"] = decoded[:, i]
    else:
        frame[target_col] = decoded.ravel()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out
