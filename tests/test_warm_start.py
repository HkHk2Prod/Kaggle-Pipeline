"""Warm-start / resume: finding and loading a previous run's saved leaderboard.

A second Kaggle run continues the previous leaderboard if that run's output is
re-attached as an input. These tests cover the detection of the prior board
under the input mount and the copy into the active storage dir, simulating the
Kaggle layout (``/kaggle/input/notebooks/<user>/<slug>/Models/LeaderBoard``) on
a temp directory.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from kaggle_pipeline import Config
from kaggle_pipeline.config import environment as envmod
from kaggle_pipeline.config.environment import (
    _find_previous_leaderboard_dir,
    resolve_paths,
)
from kaggle_pipeline.search.leaderboard import LEADERBOARD_FILENAME


def _make_prior_output(mount: Path) -> Path:
    """Write a fake previous run's Models/ dir (board + a model file). Return it."""
    models = mount / "Models"
    models.mkdir(parents=True)
    (models / LEADERBOARD_FILENAME).write_text("board")
    (models / "LogisticRegression_x").write_text("model")
    return models


def test_find_previous_leaderboard_dir_kaggle_layout(tmp_path: Path):
    # The real Kaggle mount for a re-attached notebook output is nested
    # /kaggle/input/notebooks/<user>/<slug>/Models/LeaderBoard.
    inp = tmp_path / "input"
    models = _make_prior_output(inp / "notebooks" / "hkhk2prod" / "predicting-f1-pit-stops")
    assert _find_previous_leaderboard_dir(inp) == models


def test_find_previous_leaderboard_dir_at_root(tmp_path: Path):
    (tmp_path / LEADERBOARD_FILENAME).write_text("board")
    assert _find_previous_leaderboard_dir(tmp_path) == tmp_path


def test_find_previous_leaderboard_dir_none_when_absent(tmp_path: Path):
    (tmp_path / "some_dataset").mkdir()
    assert _find_previous_leaderboard_dir(tmp_path) is None


def test_find_previous_leaderboard_dir_picks_most_recent(tmp_path: Path):
    older = _make_prior_output(tmp_path / "input" / "run-a")
    newer = _make_prior_output(tmp_path / "input" / "run-b")
    # Make run-b's board newer than run-a's.
    os.utime(older / LEADERBOARD_FILENAME, (1, 1))
    os.utime(newer / LEADERBOARD_FILENAME, (time.time(), time.time()))
    assert _find_previous_leaderboard_dir(tmp_path / "input") == newer


def test_resolve_paths_kaggle_warm_starts_from_attached_output(tmp_path: Path, monkeypatch):
    """resolve_paths copies an attached prior board into the working storage dir."""
    inp = tmp_path / "input"
    # The competition data (so data_dir resolves) and a prior notebook output.
    comp = inp / "playground"
    comp.mkdir(parents=True)
    for csv in ("train.csv", "test.csv"):
        (comp / csv).write_text("a,b\n1,2\n")
    _make_prior_output(inp / "notebooks" / "me" / "playground")
    monkeypatch.setattr(envmod, "KAGGLE_INPUT_ROOT", inp)

    storage = tmp_path / "working" / "Models"
    cfg = Config(competition="playground", storage_dir=storage)  # previous_output_dir unset
    resolve_paths(cfg, env="kaggle")

    # The board and its model file were copied into the active storage dir.
    assert (storage / LEADERBOARD_FILENAME).is_file()
    assert (storage / "LogisticRegression_x").is_file()


def test_resolve_paths_kaggle_no_prior_board_is_quiet(tmp_path: Path, monkeypatch):
    """With no attached prior output, warm-start is a no-op (no board appears)."""
    inp = tmp_path / "input"
    comp = inp / "playground"
    comp.mkdir(parents=True)
    for csv in ("train.csv", "test.csv"):
        (comp / csv).write_text("a,b\n1,2\n")
    monkeypatch.setattr(envmod, "KAGGLE_INPUT_ROOT", inp)

    storage = tmp_path / "working" / "Models"
    cfg = Config(competition="playground", storage_dir=storage)
    resolve_paths(cfg, env="kaggle")
    assert not (storage / LEADERBOARD_FILENAME).exists()


def test_explicit_previous_output_dir_resolves_within(tmp_path: Path):
    """An explicit previous_output_dir pointing at the mount root still resolves."""
    mount = tmp_path / "input" / "notebooks" / "me" / "slug"
    _make_prior_output(mount)
    # Point at the output root (which contains Models/), not the Models dir itself.
    assert _find_previous_leaderboard_dir(mount) == mount / "Models"
