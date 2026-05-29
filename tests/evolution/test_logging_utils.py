"""Phase / batch banner formatters and the verbosity-to-level mapping."""

from __future__ import annotations

import logging

from kaggle_pipeline.evolution.logging_utils import (
    Verbosity,
    format_batch_banner,
    format_duration,
    format_phase_banner,
    verbosity_to_logging_level,
)


def test_format_duration_seconds_minutes_hours():
    assert format_duration(0) == "0s"
    assert format_duration(45) == "45s"
    assert format_duration(125) == "2m05s"
    assert format_duration(3700) == "1h01m"


def test_format_duration_clamps_negative_to_zero():
    assert format_duration(-10) == "0s"


def test_verbosity_to_logging_level_maps_each_level():
    assert verbosity_to_logging_level(Verbosity.SILENT) == logging.CRITICAL
    assert verbosity_to_logging_level(Verbosity.SUMMARY) == logging.INFO
    assert verbosity_to_logging_level(Verbosity.DEBUG) == logging.DEBUG
    # Unknown integers fall back to INFO rather than raising.
    assert verbosity_to_logging_level(99) == logging.INFO


def test_format_phase_banner_contains_uppercased_label_and_three_rules():
    text = format_phase_banner("training")
    # Leading blank line + three rule lines so phases visually break apart from
    # the per-batch lines around them.
    lines = text.split("\n")
    assert lines[0] == ""  # leading newline so the banner is preceded by space
    assert len(lines) == 4
    bar = lines[1]
    label = lines[2]
    assert set(bar) == {"="}
    assert "PHASE: TRAINING" in label
    assert label.startswith("=") and label.endswith("=")
    assert lines[3] == bar


def test_format_batch_banner_marks_start_and_end_distinctly():
    start = format_batch_banner(7)
    end = format_batch_banner(7, end=True)
    assert "batch 7" in start
    assert "batch 7 end" in end
    # Both lines are dash-padded so they line up vertically when streamed.
    assert start.startswith("-") and start.endswith("-")
    assert end.startswith("-") and end.endswith("-")
    assert len(start) == len(end)
