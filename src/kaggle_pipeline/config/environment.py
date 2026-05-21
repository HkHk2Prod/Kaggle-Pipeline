"""Detect where we are running (Kaggle / Colab / local) and resolve paths.

The original notebook hard-coded a Kaggle-vs-Colab ``if`` block. Here the same
logic is isolated so the rest of the package never touches ``os.environ`` and
so paths can always be overridden explicitly via :class:`Config`.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from kaggle_pipeline.config.config import Config

KAGGLE = "kaggle"
COLAB = "colab"
LOCAL = "local"


def detect_environment() -> str:
    """Return one of ``"kaggle"``, ``"colab"`` or ``"local"``."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return KAGGLE
    if "COLAB_RELEASE_TAG" in os.environ:
        return COLAB
    return LOCAL


@dataclass
class ResolvedPaths:
    """Concrete filesystem locations for a run."""

    data_dir: Path
    storage_dir: Path
    working_dir: Path


def resolve_paths(config: Config, env: str | None = None) -> ResolvedPaths:
    """Work out data/storage directories from the environment and config.

    Explicit ``config.data_dir`` / ``config.storage_dir`` always win. Otherwise
    we fall back to the conventional Kaggle and Colab locations. On Kaggle, if
    ``config.previous_output_dir`` exists it is copied into the working dir to
    warm-start the leaderboard from a prior run.
    """
    env = env or detect_environment()

    if env == KAGGLE:
        data_dir = config.data_dir or Path("/kaggle/input/competitions") / config.competition
        working_dir = Path("/kaggle/working")
        storage_dir = config.storage_dir or working_dir / "Models"
        _warm_start_from_previous_output(config, working_dir)
    elif env == COLAB:
        _mount_drive()
        base = Path("/content/drive/MyDrive/Colab Notebooks/Data") / config.competition
        data_dir = config.data_dir or base
        storage_dir = config.storage_dir or base / "Models"
        working_dir = Path.cwd()
    else:  # local
        if config.data_dir is None:
            raise ValueError(
                "Running locally but config.data_dir is not set. Point it at the "
                "directory containing train.csv / test.csv / sample_submission.csv."
            )
        data_dir = config.data_dir
        storage_dir = config.storage_dir or data_dir / "Models"
        working_dir = Path.cwd()

    storage_dir.mkdir(parents=True, exist_ok=True)
    return ResolvedPaths(data_dir=data_dir, storage_dir=storage_dir, working_dir=working_dir)


def _warm_start_from_previous_output(config: Config, working_dir: Path) -> None:
    """Copy a previous Kaggle notebook's output into the working dir (if present)."""
    previous = config.previous_output_dir
    if previous is None:
        return
    if os.path.exists(previous):
        shutil.copytree(previous, working_dir, dirs_exist_ok=True)
        print(f"Successfully copied previous output to: {working_dir}")
    else:
        print(f"Previous output dir not found: {previous}. Check your path!")


def _mount_drive() -> None:
    """Mount Google Drive when running on Colab."""
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount("/content/drive")
