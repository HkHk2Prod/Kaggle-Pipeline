"""Autodetection of CSV filenames and problem-definition fields left as None."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kaggle_pipeline import Config
from kaggle_pipeline.config import environment as envmod
from kaggle_pipeline.config.environment import autodetect_data_dir, resolve_paths
from kaggle_pipeline.data.autodetect import (
    _detect_task,
    resolve_csv_filenames,
    resolve_problem_definition,
)


def _make_input_dir(root: Path, name: str, csvs=("train.csv", "test.csv", "sample_submission.csv")):
    """Create root/name/<csvs> with trivial content; return the created dir."""
    directory = root / name
    directory.mkdir(parents=True)
    for csv in csvs:
        (directory / csv).write_text("a,b\n1,2\n")
    return directory


def test_resolve_csv_filenames_from_search(synthetic_data_dir: Path):
    cfg = Config(competition="synthetic", data_dir=synthetic_data_dir)
    resolve_csv_filenames(cfg, synthetic_data_dir)
    assert cfg.train_csv == "train.csv"
    assert cfg.test_csv == "test.csv"
    assert cfg.sample_csv == "sample_submission.csv"


def test_resolve_csv_filenames_keeps_explicit(synthetic_data_dir: Path):
    cfg = Config(competition="synthetic", train_csv="train.csv")
    resolve_csv_filenames(cfg, synthetic_data_dir)
    assert cfg.train_csv == "train.csv"  # untouched


def test_resolve_csv_filenames_missing_raises(tmp_path: Path):
    cfg = Config(competition="synthetic")
    with pytest.raises(FileNotFoundError, match="train_csv"):
        resolve_csv_filenames(cfg, tmp_path)


def test_resolve_problem_definition_classification():
    df = pd.DataFrame(
        {"id": [0, 1, 2, 3], "x": [1.0, 2.0, 3.0, 4.0], "y": ["no", "yes", "no", "yes"]}
    )
    cfg = Config(competition="synthetic")  # everything None
    resolve_problem_definition(cfg, df)
    assert cfg.target == ["y"]  # last non-id column
    assert cfg.task == "classification"  # object dtype
    assert cfg.prediction_aim == "probability"  # cat target
    assert cfg.scoring == "roc_auc"  # binary


def test_resolve_problem_definition_keeps_explicit():
    df = pd.DataFrame({"id": [0, 1], "y": ["a", "b"]})
    cfg = Config(
        competition="synthetic", target="y", task="classification", scoring="balanced_accuracy"
    )
    resolve_problem_definition(cfg, df)
    assert cfg.target == ["y"]
    assert cfg.scoring == "balanced_accuracy"  # not overwritten with the binary default


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["no", "yes", "no"], "classification"),  # non-numeric
        ([True, False, True], "classification"),  # boolean
        ([0, 1, 1, 0], "classification"),  # low-cardinality integers
        ([1.0, 2.0, 3.0], "classification"),  # integer-valued floats
        ([0.13, 1.7, 3.4, 9.1], "regression"),  # continuous
        (list(range(50)), "regression"),  # high-cardinality integers
    ],
)
def test_detect_task(values, expected):
    assert _detect_task(pd.Series(values)) == expected


def test_config_competition_is_optional():
    cfg = Config()  # no competition, no data_dir -> must not raise at construction
    assert cfg.competition is None


def test_autodetect_data_dir_single_candidate(tmp_path: Path):
    inp = tmp_path / "input"
    comp = _make_input_dir(inp, "some-competition")
    (inp / "helper-package").mkdir(parents=True)  # no CSVs -> ignored (e.g. our pkg dataset)
    assert autodetect_data_dir(inp) == comp


def test_autodetect_data_dir_requires_both_train_and_test(tmp_path: Path):
    inp = tmp_path / "input"
    _make_input_dir(inp, "only-train", csvs=("train.csv",))  # missing a test CSV
    assert autodetect_data_dir(inp) is None


def test_autodetect_data_dir_missing_root_returns_none(tmp_path: Path):
    assert autodetect_data_dir(tmp_path / "does-not-exist") is None


def test_autodetect_data_dir_finds_nested_competition_layout(tmp_path: Path):
    # Real Kaggle layout: /kaggle/input/competitions/<slug>/{train,test}.csv,
    # i.e. the data dir is two levels below the input root, not an immediate child.
    inp = tmp_path / "input"
    target = _make_input_dir(inp / "competitions", "playground-series-s6e5")
    assert autodetect_data_dir(inp) == target


def test_autodetect_data_dir_disambiguates_by_competition(tmp_path: Path):
    inp = tmp_path / "input"
    _make_input_dir(inp, "comp-a")
    target = _make_input_dir(inp, "comp-b")
    assert autodetect_data_dir(inp, competition="comp-b") == target


def test_autodetect_data_dir_ambiguous_raises(tmp_path: Path):
    inp = tmp_path / "input"
    _make_input_dir(inp, "comp-a")
    _make_input_dir(inp, "comp-b")
    with pytest.raises(FileNotFoundError, match="Multiple directories"):
        autodetect_data_dir(inp)  # two candidates, nothing to disambiguate


def test_resolve_paths_kaggle_autodetects_data_dir(tmp_path: Path, monkeypatch):
    inp = tmp_path / "input"
    comp = _make_input_dir(inp, "some-competition")
    monkeypatch.setattr(envmod, "KAGGLE_INPUT_ROOT", inp)
    # competition + data_dir both unset; storage_dir set so mkdir stays in tmp.
    cfg = Config(storage_dir=tmp_path / "models")
    paths = resolve_paths(cfg, env="kaggle")
    assert paths.data_dir == comp
    assert paths.storage_dir == tmp_path / "models"


def test_autodetect_through_data_pipeline(synthetic_data_dir: Path, tmp_path: Path):
    """A sparse Config + data load triggers full autodetect of problem definition."""
    from kaggle_pipeline.data import load_datasets

    cfg = Config(
        competition="synthetic",
        data_dir=synthetic_data_dir,
        storage_dir=tmp_path / "models",
    )
    resolve_paths(cfg, env="local")
    load_datasets(cfg, synthetic_data_dir, check_nulls=False)
    assert cfg.target == ["y"]
    assert cfg.task == "classification"
    assert cfg.prediction_aim == "probability"
    assert cfg.scoring == "roc_auc"
    assert cfg.train_csv == "train.csv"
