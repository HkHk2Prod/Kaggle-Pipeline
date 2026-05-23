"""Verbosity: the level mapping, Config validation and the CLI -v/-q override."""

from __future__ import annotations

import logging

import pytest

from kaggle_pipeline.cli import _apply_cli_verbosity, build_parser
from kaggle_pipeline.config import Config
from kaggle_pipeline.logconfig import VERBOSITY_LEVELS, level_for_verbosity


def test_level_for_verbosity_maps_each_name():
    assert level_for_verbosity("quiet") == logging.WARNING
    assert level_for_verbosity("normal") == logging.INFO
    assert level_for_verbosity("verbose") == logging.DEBUG
    # The mapping is the single source of truth for the valid names.
    assert set(VERBOSITY_LEVELS) == {"quiet", "normal", "verbose"}


def test_level_for_verbosity_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown verbosity"):
        level_for_verbosity("loud")


def test_config_defaults_to_verbose():
    assert Config().verbosity == "verbose"


def test_config_rejects_unknown_verbosity():
    with pytest.raises(ValueError, match="verbosity must be one of"):
        Config(verbosity="loud")


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["run", "-c", "cfg.yaml"], "normal"),  # no flag -> keep the config value
        (["run", "-c", "cfg.yaml", "-v"], "verbose"),
        (["run", "-c", "cfg.yaml", "--verbose"], "verbose"),
        (["run", "-c", "cfg.yaml", "-q"], "quiet"),
        (["analyze", "-c", "cfg.yaml", "-q"], "quiet"),
    ],
)
def test_cli_flags_override_config_verbosity(argv, expected):
    args = build_parser().parse_args(argv)
    config = Config(verbosity="normal")
    _apply_cli_verbosity(config, args)
    assert config.verbosity == expected


def test_cli_verbose_and_quiet_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "-c", "cfg.yaml", "-v", "-q"])
