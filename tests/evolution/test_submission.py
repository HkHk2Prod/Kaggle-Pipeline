"""Unit tests for the submission helpers extracted from KagglePipeline."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kaggle_pipeline.evolution.logging_utils import Verbosity
from kaggle_pipeline.evolution.submission import (
    SubmissionWriter,
    submission_skip_reason,
    submission_summary_lines,
)


def _writer(**overrides):
    defaults = dict(
        task="classification",
        classes=np.array([0, 1]),
        prediction_aim="probability",
        id_col="id",
        test_ids=np.arange(3),
        test_has_ids=True,
    )
    defaults.update(overrides)
    return SubmissionWriter(**defaults)


def test_decode_single_returns_positive_class_for_binary_probability():
    proba = np.array([[0.9, 0.1], [0.2, 0.8]])
    decoded = _writer().decode_single(proba)
    assert decoded.tolist() == pytest.approx([0.1, 0.8])


def test_decode_single_returns_argmax_label_for_category_aim():
    proba = np.array([[0.7, 0.3], [0.1, 0.9]])
    writer = _writer(prediction_aim="category", classes=np.array(["A", "B"]))
    assert writer.decode_single(proba).tolist() == ["A", "B"]


def test_decode_single_passes_regression_proba_through():
    writer = _writer(task="regression", classes=None)
    out = writer.decode_single(np.array([0.4, -1.0, 7.5]))
    assert out.tolist() == [0.4, -1.0, 7.5]


def test_build_frame_uses_sample_columns_and_aligns_by_id():
    sample = pd.DataFrame({"id": [2, 0, 1], "target": [0.0, 0.0, 0.0]})
    writer = _writer(test_ids=np.array([0, 1, 2]))
    predictions = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    frame = writer.build_frame(predictions, sample)
    # Sample's id order is preserved; target is the positive-class probability
    # joined back on id, not on row position.
    assert frame["id"].tolist() == [2, 0, 1]
    assert frame["target"].tolist() == pytest.approx([0.8, 0.1, 0.5])


def test_build_frame_raises_on_mismatched_ids():
    sample = pd.DataFrame({"id": [5, 6, 7], "target": [0.0, 0.0, 0.0]})
    writer = _writer(test_ids=np.array([0, 1, 2]))
    predictions = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    with pytest.raises(ValueError, match="do not match"):
        writer.build_frame(predictions, sample)


def test_build_frame_falls_back_to_id_target_when_no_sample():
    writer = _writer(test_ids=np.array([10, 11, 12]))
    predictions = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    frame = writer.build_frame(predictions, sample=None)
    assert list(frame.columns) == ["id", "target"]
    assert frame["id"].tolist() == [10, 11, 12]
    assert frame["target"].tolist() == pytest.approx([0.1, 0.5, 0.8])


def test_build_frame_uses_positional_alignment_when_test_lacks_ids():
    sample = pd.DataFrame({"id": [2, 0, 1], "target": [0.0, 0.0, 0.0]})
    writer = _writer(test_ids=np.arange(3), test_has_ids=False)
    predictions = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    frame = writer.build_frame(predictions, sample)
    # No real ids => keep test order; sample's ids are zipped positionally.
    assert frame["target"].tolist() == pytest.approx([0.1, 0.5, 0.8])


def test_build_frame_per_class_columns_for_multiclass_sample():
    sample = pd.DataFrame({"id": [0, 1, 2], "class_a": [0.0, 0.0, 0.0], "class_b": [0.0, 0.0, 0.0]})
    writer = _writer(test_ids=np.array([0, 1, 2]))
    predictions = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    frame = writer.build_frame(predictions, sample)
    assert frame["class_a"].tolist() == pytest.approx([0.9, 0.5, 0.2])
    assert frame["class_b"].tolist() == pytest.approx([0.1, 0.5, 0.8])


def test_summary_lines_for_disabled_ensemble():
    lines, composition = submission_summary_lines(
        Path("submission.csv"),
        pd.DataFrame({"id": [1], "target": [0.5]}),
        np.array([0.5]),
        ensemble_result=None,
    )
    assert composition is None
    assert len(lines) == 1
    message, level = lines[0]
    assert level == Verbosity.SUMMARY
    assert "ensemble=disabled" in message


def test_summary_lines_include_composition_when_lookup_returns_genome():
    ensemble = SimpleNamespace(
        oof_score=0.91,
        status="weighted",
        n_members=1,
        note="",
        member_ids=["m1"],
        weights={"m1": 1.0},
    )
    genome = SimpleNamespace(
        family="lgbm",
        score_set=SimpleNamespace(score=0.91, score_std=0.01, compute_time=2.5),
    )
    lines, composition = submission_summary_lines(
        Path("submission.csv"),
        pd.DataFrame({"id": [1], "target": [0.5]}),
        np.array([0.5]),
        ensemble_result=ensemble,
        population_lookup=lambda mid: genome if mid == "m1" else None,
    )
    assert any("submission summary" in m for m, _ in lines)
    assert any("strategy=weighted" in m for m, _ in lines)
    assert composition is not None and "m1" in composition and "lgbm" in composition


def test_skip_reason_silently_skips_when_flag_off():
    assert (
        submission_skip_reason(make_submission_on_run=False, has_test_features=True, runtime=None)
        == ""
    )


def test_skip_reason_reports_missing_test():
    msg = submission_skip_reason(make_submission_on_run=True, has_test_features=False, runtime=None)
    assert msg and "no test data" in msg


def test_skip_reason_reports_no_budget():
    runtime = SimpleNamespace(
        has_time_for_submission=lambda: False,
        remaining_submission_seconds=lambda: 0.0,
        submission_time_reserve_seconds=42.0,
    )
    msg = submission_skip_reason(
        make_submission_on_run=True, has_test_features=True, runtime=runtime
    )
    assert msg and "not enough time" in msg and "42" in msg


def test_skip_reason_none_when_ok():
    runtime = SimpleNamespace(
        has_time_for_submission=lambda: True,
        remaining_submission_seconds=lambda: 60.0,
        submission_time_reserve_seconds=10.0,
    )
    assert (
        submission_skip_reason(make_submission_on_run=True, has_test_features=True, runtime=runtime)
        is None
    )
