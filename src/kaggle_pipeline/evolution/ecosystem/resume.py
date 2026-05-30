"""Locate a previous run's checkpoint directory for warm-start.

The orchestrator writes checkpoints under ``state_dir/checkpoints/``. On Kaggle
that directory ships as the notebook's *output*; the next run attaches the
previous output as an input dataset, which mounts under ``/kaggle/input/``.
``find_previous_state_dir`` resolves where to load from:

1. An explicit ``previous_state_dir`` always wins (used in offline / local runs).
2. Otherwise scan ``/kaggle/input/`` recursively for any directory named
   ``state_dir`` (default ``kagglepipeline_state``) that holds a non-empty
   ``checkpoints/`` folder, and return the one whose newest checkpoint is the
   most recent. The recursive walk handles every Kaggle layout: attached
   datasets (``/kaggle/input/<dataset>/...``) and notebook outputs
   (``/kaggle/input/notebooks/<user>/<slug>/...``).

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
    candidates = find_all_state_dirs(
        previous_state_dir=previous_state_dir,
        state_dir_name=state_dir_name,
        kaggle_root=kaggle_root,
    )
    if not candidates:
        return None
    return max(candidates, key=_latest_checkpoint_mtime)


def find_all_state_dirs(
    *,
    previous_state_dir: str | Path | None,
    state_dir_name: str = "kagglepipeline_state",
    kaggle_root: Path = KAGGLE_INPUT_ROOT,
) -> list[Path]:
    """Return *every* resumable state directory, for merging parallel ecosystems.

    Parallel notebooks each emit their own ``state_dir`` as an output; the merge
    notebook attaches all of them as inputs, so several mount under
    ``/kaggle/input/``. This returns each distinct one holding a non-empty
    ``checkpoints/`` folder (newest-checkpoint first). An explicit
    ``previous_state_dir`` short-circuits to just that directory. Empty when
    nothing is found.
    """
    if previous_state_dir is not None:
        path = Path(previous_state_dir)
        return [path] if _has_checkpoints(path) else []

    if not kaggle_root.is_dir():
        return []
    seen: set[Path] = set()
    candidates: list[Path] = []
    for match in kaggle_root.rglob(state_dir_name):
        resolved = match.resolve()
        if resolved in seen or not _has_checkpoints(match):
            continue
        seen.add(resolved)
        candidates.append(match)
    candidates.sort(key=_latest_checkpoint_mtime, reverse=True)
    return candidates


def _has_checkpoints(state_dir: Path) -> bool:
    checkpoints = state_dir / "checkpoints"
    return checkpoints.is_dir() and any(checkpoints.glob("checkpoint_*"))


def _latest_checkpoint_mtime(state_dir: Path) -> float:
    checkpoints = list((state_dir / "checkpoints").glob("checkpoint_*"))
    return max((c.stat().st_mtime for c in checkpoints), default=0.0)
