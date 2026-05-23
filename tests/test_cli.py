"""The ``kaggle-pipeline`` CLI: argument parsing and the ``run`` command."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kaggle_pipeline.cli import build_parser, main


def _write_config(path: Path, data_dir: Path, storage_dir: Path) -> Path:
    cfg = {
        "competition": "synthetic",
        "target": "y",
        "id_col": "id",
        "task": "classification",
        "scoring": "balanced_accuracy",
        "prediction_aim": "category",
        "n_steps": 1,
        "num_models": 8,
        "step_batch_size": 4,
        "n_workers": 1,
        "ensemble_length": 4,
        "ensemble_min_repr": 1,
        "cv_splits": 3,
        "seed": 0,
        "data_dir": str(data_dir),
        "storage_dir": str(storage_dir),
    }
    cfg_path = path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_run_command_writes_submission(synthetic_data_dir: Path, tmp_path: Path, monkeypatch):
    cfg_path = _write_config(tmp_path, synthetic_data_dir, tmp_path / "models")
    monkeypatch.chdir(tmp_path)  # submission lands in the working dir

    rc = main(["run", "--config", str(cfg_path)])

    assert rc == 0
    assert (tmp_path / "submission.csv").exists()


def test_parser_requires_a_subcommand():
    # argparse exits with code 2 when the required subcommand is missing.
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_config_is_optional():
    # --config may be omitted (defaults to None); the subcommand still parses.
    assert build_parser().parse_args(["run"]).config is None


def test_run_without_config_uses_autodetecting_default(monkeypatch):
    # With no --config, main() builds a bare Config() whose problem-definition
    # fields are all left for autodetection (notebook style).
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return Path("submission.csv")

    monkeypatch.setattr("kaggle_pipeline.cli.run", fake_run)
    rc = main(["run"])

    assert rc == 0
    cfg = captured["config"]
    assert cfg.competition is None
    assert cfg.target is None
    assert cfg.scoring is None
    assert cfg.prediction_aim is None
