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

# Where Kaggle mounts attached datasets / competition data. Module-level so tests
# can point it at a temporary directory.
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
# How many levels below the input root to look for the data directory. Kaggle
# nests competition data as /kaggle/input/competitions/<slug>/ (depth 2) and
# datasets as /kaggle/input/<slug>/ (depth 1), so scan a few levels rather than
# only the immediate children.
DATA_DIR_SEARCH_DEPTH = 3
# How many levels below the input root to look for a previous run's saved
# leaderboard, for warm-starting. A notebook's own prior output mounts under
# /kaggle/input/notebooks/<user>/<slug>/ (depth 3) and the pipeline saves the
# board one level deeper in a Models/ dir, so search comfortably past that.
PREVIOUS_OUTPUT_SEARCH_DEPTH = 6


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
        _warm_start_from_previous_output(config, storage_dir)
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


def autodetect_data_dir(
    input_root: Path | str,
    competition: str | None = None,
    max_depth: int = DATA_DIR_SEARCH_DEPTH,
) -> Path | None:
    """Find the directory under ``input_root`` that holds the competition CSVs.

    Walks ``input_root`` and its sub-directories down to ``max_depth`` levels
    (Kaggle nests competition data under ``competitions/<slug>/``) for a folder
    with both a *train* and a *test* CSV at its top level. Returns:

    * the one whose name equals ``competition`` when set and matched, else
    * the single matching directory, or
    * ``None`` when nothing matches (the caller then falls back / errors).

    Raises :class:`FileNotFoundError` when several directories match and
    ``competition`` does not pick one, since guessing risks silently loading the
    wrong data. The chosen directory is logged as an ``[autodetect]`` line.
    """
    root = Path(input_root)
    if not root.is_dir():
        return None
    candidates = [d for d in _dirs_to_depth(root, max_depth) if _has_train_and_test(d)]
    if not candidates:
        return None
    if competition is not None:
        for directory in candidates:
            if directory.name == competition:
                _announce_data_dir(directory, f"matched competition {competition!r}")
                return directory
    if len(candidates) == 1:
        _announce_data_dir(candidates[0], f"only directory under {root} with train/test CSVs")
        return candidates[0]
    raise FileNotFoundError(
        f"Multiple directories under {root} contain train/test CSVs "
        f"({[str(d) for d in candidates]}). Set config.competition to one of them, "
        "or set config.data_dir explicitly."
    )


def _dirs_to_depth(root: Path, max_depth: int):
    """Yield ``root`` and its sub-directories down to ``max_depth`` levels deep."""
    yield root
    for depth in range(1, max_depth + 1):
        for path in sorted(root.glob("/".join(["*"] * depth))):
            if path.is_dir():
                yield path


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
        # Direct fall-back paths for the named competition (the scan may miss it
        # if the CSVs sit deeper than the search depth or are named unusually).
        for candidate in (
            KAGGLE_INPUT_ROOT / "competitions" / config.competition,
            KAGGLE_INPUT_ROOT / config.competition,
        ):
            if candidate.is_dir():
                return candidate
    found = (
        sorted(str(p) for p in KAGGLE_INPUT_ROOT.rglob("*.csv"))
        if KAGGLE_INPUT_ROOT.is_dir()
        else []
    )
    raise FileNotFoundError(
        f"Could not locate competition data under {KAGGLE_INPUT_ROOT}: no directory "
        f"has both a train and a test CSV. CSVs found: {found or 'none'}. Attach the "
        "competition (Add Input), or set config.data_dir to the directory containing "
        "the train/test CSVs."
    )


def _warm_start_from_previous_output(config: Config, storage_dir: Path) -> None:
    """Resume a prior run by copying its saved leaderboard into ``storage_dir``.

    The search root is ``config.previous_output_dir`` when set, otherwise the
    whole Kaggle input mount: a notebook's own previous output, re-attached via
    *Add Input*, lands under ``/kaggle/input/notebooks/<user>/<slug>/`` and is
    found automatically, so a re-run continues the previous leaderboard without
    the user hand-wiring the (non-obvious) mount path. The directory holding the
    ``LeaderBoard`` file is copied wholesale -- the board plus every model pickle
    beside it -- into ``storage_dir``, where :meth:`Judge.load` then picks it up.
    """
    explicit = config.previous_output_dir
    search_root = Path(explicit) if explicit is not None else KAGGLE_INPUT_ROOT
    if not search_root.exists():
        if explicit is not None:
            logger.warning("previous_output_dir not found: %s. Check your path!", explicit)
        return

    source = _find_previous_leaderboard_dir(search_root)
    if source is None:
        # An explicit dir that holds no board is a likely mistake worth warning
        # about; finding none under the whole input mount is just a first run.
        if explicit is not None:
            logger.warning("No saved leaderboard found under previous_output_dir %s.", explicit)
        else:
            logger.info("[warm-start] no previous leaderboard found under %s.", search_root)
        return

    if source.resolve() == storage_dir.resolve():
        return  # Already the active board (e.g. a same-session re-run).
    storage_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, storage_dir, dirs_exist_ok=True)
    logger.info("[warm-start] resumed previous leaderboard from %s", source)


def _find_previous_leaderboard_dir(
    root: Path, max_depth: int = PREVIOUS_OUTPUT_SEARCH_DEPTH
) -> Path | None:
    """Return the directory holding a saved ``LeaderBoard`` under ``root``.

    Looks at ``root`` itself and its sub-directories down to ``max_depth`` levels
    for the pickle written by :meth:`LeaderBoard.save`. When more than one prior
    output is attached, the most recently modified board wins.
    """
    from kaggle_pipeline.search.leaderboard import LEADERBOARD_FILENAME

    found: list[Path] = []
    if (root / LEADERBOARD_FILENAME).is_file():
        found.append(root / LEADERBOARD_FILENAME)
    for depth in range(1, max_depth + 1):
        pattern = "/".join(["*"] * depth + [LEADERBOARD_FILENAME])
        found.extend(p for p in root.glob(pattern) if p.is_file())
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime).parent


def _mount_drive() -> None:
    """Mount Google Drive when running on Colab."""
    from google.colab import drive  # Colab-only module; absent elsewhere.

    drive.mount("/content/drive")
