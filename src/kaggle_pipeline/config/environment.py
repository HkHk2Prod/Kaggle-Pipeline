"""Detect where we are running (Kaggle / Colab / local) and resolve paths.

The original notebook hard-coded a Kaggle-vs-Colab ``if`` block. Here the same
logic is isolated so the rest of the package never touches ``os.environ`` and
so paths can always be overridden explicitly via :class:`Config`.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from kaggle_pipeline.config.config import Config

logger = logging.getLogger(__name__)

KAGGLE = "kaggle"
COLAB = "colab"
LOCAL = "local"

# Where Kaggle mounts attached datasets / competition data (one sub-dir each).
# Module-level so tests can point it at a temporary directory.
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


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
        working_dir = Path("/kaggle/working")
        # storage_dir derives from the (writable) working dir, not data_dir, since
        # the input mount is read-only -- so resolve it independently of data_dir.
        storage_dir = config.storage_dir or working_dir / "Models"
        data_dir = config.data_dir or _resolve_kaggle_data_dir(config)
        _warm_start_from_previous_output(config, working_dir)
    elif env == COLAB:
        _mount_drive()
        working_dir = Path.cwd()
        if config.data_dir is not None:
            data_dir = config.data_dir
            storage_dir = config.storage_dir or data_dir / "Models"
        elif config.competition is not None:
            base = Path("/content/drive/MyDrive/Colab Notebooks/Data") / config.competition
            data_dir = base
            storage_dir = config.storage_dir or base / "Models"
        else:
            raise ValueError(
                "Running on Colab but neither config.data_dir nor config.competition "
                "is set, so the data location is unknown. Set one of them."
            )
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


def autodetect_data_dir(input_root: Path | str, competition: str | None = None) -> Path | None:
    """Find the attached input directory that holds the competition CSVs.

    Scans the immediate sub-directories of ``input_root`` (on Kaggle each
    attached dataset / competition mounts as one such directory) for a folder
    with both a *train* and a *test* CSV at its top level. Returns:

    * the single matching directory, or
    * the one whose name equals ``competition`` when several match, or
    * ``None`` when nothing matches (the caller then falls back / errors).

    Raises :class:`FileNotFoundError` when several inputs match and
    ``competition`` does not pick one, since guessing risks silently loading the
    wrong data. The chosen directory is logged as an ``[autodetect]`` line.
    """
    root = Path(input_root)
    if not root.is_dir():
        return None
    candidates = [d for d in sorted(root.iterdir()) if d.is_dir() and _has_train_and_test(d)]
    if not candidates:
        return None
    if len(candidates) == 1:
        _announce_data_dir(candidates[0], f"only input under {root} with train/test CSVs")
        return candidates[0]
    if competition is not None:
        for directory in candidates:
            if directory.name == competition:
                _announce_data_dir(
                    directory, f"matched competition {competition!r} among {len(candidates)} inputs"
                )
                return directory
    raise FileNotFoundError(
        f"Multiple inputs under {root} contain train/test CSVs "
        f"({[d.name for d in candidates]}). Set config.competition to one of them, "
        "or set config.data_dir explicitly."
    )


def _has_train_and_test(directory: Path) -> bool:
    """True if ``directory`` has at least one train-like and one test-like CSV."""
    names = [p.name.lower() for p in directory.glob("*.csv")]
    return any("train" in name for name in names) and any("test" in name for name in names)


def _announce_data_dir(data_dir: Path, reason: str) -> None:
    logger.info("[autodetect] data_dir = %r  (%s)", str(data_dir), reason)


def _resolve_kaggle_data_dir(config: Config) -> Path:
    """Locate the Kaggle competition data when ``config.data_dir`` is unset."""
    detected = autodetect_data_dir(KAGGLE_INPUT_ROOT, config.competition)
    if detected is not None:
        return detected
    if config.competition is not None:
        # Legacy convention, kept as a fall-back when the scan finds nothing.
        return KAGGLE_INPUT_ROOT / "competitions" / config.competition
    raise FileNotFoundError(
        f"Could not locate competition data under {KAGGLE_INPUT_ROOT}: no attached "
        "input has both a train and a test CSV. Attach the competition data, or set "
        "config.data_dir / config.competition explicitly."
    )


def _warm_start_from_previous_output(config: Config, working_dir: Path) -> None:
    """Copy a previous Kaggle notebook's output into the working dir (if present)."""
    previous = config.previous_output_dir
    if previous is None:
        return
    if os.path.exists(previous):
        shutil.copytree(previous, working_dir, dirs_exist_ok=True)
        logger.info("Successfully copied previous output to: %s", working_dir)
    else:
        logger.warning("Previous output dir not found: %s. Check your path!", previous)


def _mount_drive() -> None:
    """Mount Google Drive when running on Colab."""
    from google.colab import drive  # Colab-only module; absent elsewhere.

    drive.mount("/content/drive")
