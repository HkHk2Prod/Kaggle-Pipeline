"""Locate a previous run's checkpoint directory for warm-start.

The orchestrator writes checkpoints under ``state_dir/checkpoints/``. On Kaggle
that directory ships as the notebook's *output*; the next run attaches the
previous output as an input dataset, which mounts under ``/kaggle/input/``.
``find_previous_state_dir`` resolves where to load from:

1. An explicit ``previous_state_dir`` always wins (used in offline / local runs).
2. Otherwise scan ``/kaggle/input/`` a couple of levels deep for any directory
   named ``state_dir`` (default ``kagglepipeline_state``) that holds a
   non-empty ``checkpoints/`` folder, and return the one whose newest
   checkpoint is the most recent.

Returns ``None`` when nothing is found -- the caller starts fresh.
"""

from __future__ import annotations

from pathlib import Path

KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def find_previous_state_dir(
    *,
    previous_state_dir: str | Path | None,
    state_dir_name: str = "kagglepipeline_state",
    kaggle_root: Path = KAGGLE_INPUT_ROOT,
) -> Path | None:
    """Return the state directory of a previous run, or ``None``."""
    if previous_state_dir is not None:
        path = Path(previous_state_dir)
        return path if _has_checkpoints(path) else None

    if not kaggle_root.is_dir():
        return None
    candidates = [
        match
        for pattern in (f"*/{state_dir_name}", f"*/*/{state_dir_name}")
        for match in kaggle_root.glob(pattern)
        if _has_checkpoints(match)
    ]
    if not candidates:
        return None
    return max(candidates, key=_latest_checkpoint_mtime)


def _has_checkpoints(state_dir: Path) -> bool:
    checkpoints = state_dir / "checkpoints"
    return checkpoints.is_dir() and any(checkpoints.glob("checkpoint_*"))


def _latest_checkpoint_mtime(state_dir: Path) -> float:
    checkpoints = list((state_dir / "checkpoints").glob("checkpoint_*"))
    return max((c.stat().st_mtime for c in checkpoints), default=0.0)
