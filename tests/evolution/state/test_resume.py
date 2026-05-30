"""Warm-start resolution: find a previous run's state dir to load from."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kaggle_pipeline.evolution.ecosystem.resume import (
    find_all_state_dirs,
    find_previous_state_dir,
)


def _make_state(tmp: Path, mtime: float | None = None) -> Path:
    checkpoints = tmp / "checkpoints" / "checkpoint_000001"
    checkpoints.mkdir(parents=True)
    (checkpoints / "state.pkl").write_bytes(b"x")
    if mtime is not None:
        os.utime(checkpoints, (mtime, mtime))
    return tmp


def test_returns_none_when_nothing_to_resume(tmp_path):
    out = find_previous_state_dir(previous_state_dir=None, kaggle_root=tmp_path / "no_kaggle_here")
    assert out is None


def test_explicit_previous_state_dir_wins(tmp_path):
    explicit = _make_state(tmp_path / "explicit")
    out = find_previous_state_dir(previous_state_dir=str(explicit), kaggle_root=tmp_path / "absent")
    assert out == explicit


def test_explicit_path_without_checkpoints_returns_none(tmp_path):
    (tmp_path / "empty").mkdir()
    out = find_previous_state_dir(
        previous_state_dir=str(tmp_path / "empty"), kaggle_root=tmp_path / "absent"
    )
    assert out is None


def test_scans_kaggle_inputs_for_state_dir(tmp_path):
    kaggle_root = tmp_path / "kaggle_input"
    kaggle_root.mkdir()
    _make_state(kaggle_root / "prev_output" / "kagglepipeline_state")
    out = find_previous_state_dir(previous_state_dir=None, kaggle_root=kaggle_root)
    assert out == kaggle_root / "prev_output" / "kagglepipeline_state"


def test_scans_one_level_deeper(tmp_path):
    kaggle_root = tmp_path / "kaggle_input"
    kaggle_root.mkdir()
    _make_state(kaggle_root / "prev_output" / "subdir" / "kagglepipeline_state")
    out = find_previous_state_dir(previous_state_dir=None, kaggle_root=kaggle_root)
    assert out == kaggle_root / "prev_output" / "subdir" / "kagglepipeline_state"


def test_scans_notebook_output_layout(tmp_path):
    """Kaggle's notebook-output layout: notebooks/<user>/<slug>/state_dir."""
    kaggle_root = tmp_path / "kaggle_input"
    kaggle_root.mkdir()
    nb_state = kaggle_root / "notebooks" / "hkhk2prod" / "predicting-f1" / "kagglepipeline_state"
    _make_state(nb_state)
    out = find_previous_state_dir(previous_state_dir=None, kaggle_root=kaggle_root)
    assert out == nb_state


def test_scans_arbitrarily_deep(tmp_path):
    """The walk should find a state dir at any depth under the kaggle root."""
    kaggle_root = tmp_path / "kaggle_input"
    kaggle_root.mkdir()
    deep = kaggle_root / "a" / "b" / "c" / "d" / "e" / "kagglepipeline_state"
    _make_state(deep)
    out = find_previous_state_dir(previous_state_dir=None, kaggle_root=kaggle_root)
    assert out == deep


def test_picks_freshest_checkpoint_when_multiple_match(tmp_path):
    kaggle_root = tmp_path / "kaggle_input"
    kaggle_root.mkdir()
    old = _make_state(kaggle_root / "older_output" / "kagglepipeline_state", mtime=1_000.0)
    new = _make_state(kaggle_root / "newer_output" / "kagglepipeline_state", mtime=2_000_000.0)
    out = find_previous_state_dir(previous_state_dir=None, kaggle_root=kaggle_root)
    assert out == new
    assert out != old


def test_skips_state_dir_with_no_checkpoints(tmp_path):
    kaggle_root = tmp_path / "kaggle_input"
    (kaggle_root / "prev_output" / "kagglepipeline_state" / "checkpoints").mkdir(parents=True)
    out = find_previous_state_dir(previous_state_dir=None, kaggle_root=kaggle_root)
    assert out is None


def test_custom_state_dir_name(tmp_path):
    kaggle_root = tmp_path / "kaggle_input"
    kaggle_root.mkdir()
    _make_state(kaggle_root / "prev_output" / "alt_state")
    out_default = find_previous_state_dir(previous_state_dir=None, kaggle_root=kaggle_root)
    assert out_default is None  # default name doesn't match
    out_named = find_previous_state_dir(
        previous_state_dir=None, kaggle_root=kaggle_root, state_dir_name="alt_state"
    )
    assert out_named == kaggle_root / "prev_output" / "alt_state"


def test_returns_none_when_kaggle_root_missing(tmp_path):
    out = find_previous_state_dir(
        previous_state_dir=None, kaggle_root=tmp_path / "nope" / "still_nope"
    )
    assert out is None


@pytest.fixture
def isolate_kaggle_input(monkeypatch, tmp_path):
    """Make sure no host /kaggle/input mount bleeds into the default kaggle_root."""
    fake = tmp_path / "no_kaggle"
    monkeypatch.setattr("kaggle_pipeline.evolution.ecosystem.resume.KAGGLE_INPUT_ROOT", fake)
    return fake


def test_default_kaggle_root_used_when_not_passed(isolate_kaggle_input):
    assert find_previous_state_dir(previous_state_dir=None) is None


# --- find_all_state_dirs: multi-ecosystem discovery for merging -------------
def test_find_all_returns_every_input_ecosystem(tmp_path):
    kaggle_root = tmp_path / "kaggle_input"
    kaggle_root.mkdir()
    a = _make_state(kaggle_root / "worker_a" / "kagglepipeline_state", mtime=1_000.0)
    b = _make_state(kaggle_root / "worker_b" / "kagglepipeline_state", mtime=2_000.0)
    c = _make_state(kaggle_root / "worker_c" / "kagglepipeline_state", mtime=3_000.0)
    out = find_all_state_dirs(previous_state_dir=None, kaggle_root=kaggle_root)
    assert set(out) == {a, b, c}
    # Newest-checkpoint first.
    assert out == [c, b, a]


def test_find_all_empty_when_nothing_found(tmp_path):
    assert find_all_state_dirs(previous_state_dir=None, kaggle_root=tmp_path / "nope") == []


def test_find_all_explicit_dir_short_circuits(tmp_path):
    explicit = _make_state(tmp_path / "explicit")
    out = find_all_state_dirs(previous_state_dir=str(explicit), kaggle_root=tmp_path / "absent")
    assert out == [explicit]


def test_find_all_skips_dirs_without_checkpoints(tmp_path):
    kaggle_root = tmp_path / "kaggle_input"
    good = _make_state(kaggle_root / "worker_a" / "kagglepipeline_state")
    (kaggle_root / "worker_b" / "kagglepipeline_state" / "checkpoints").mkdir(parents=True)
    out = find_all_state_dirs(previous_state_dir=None, kaggle_root=kaggle_root)
    assert out == [good]
