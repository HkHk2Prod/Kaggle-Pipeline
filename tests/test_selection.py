"""Tests for size-inferred correlation-based predictor pruning."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kaggle_pipeline import Config
from kaggle_pipeline.pipeline import build_pipeline
from kaggle_pipeline.preprocessing.selection import (
    irrelevance_threshold,
    plan_pruning,
    redundancy_lower_bound,
)

ALPHA = 0.05
FLOOR = 0.90


def _matrix(cols: list[str], pairs: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Symmetric association matrix (unit diagonal) from explicit pair values."""
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols, dtype=float)
    for (a, b), v in pairs.items():
        m.loc[a, b] = v
        m.loc[b, a] = v
    return m


# --------------------------------------------------------------------------- #
# Size-inferred thresholds
# --------------------------------------------------------------------------- #
def test_irrelevance_threshold_shrinks_with_n():
    assert irrelevance_threshold(30, ALPHA) > irrelevance_threshold(100_000, ALPHA)
    assert 0.0 < irrelevance_threshold(100_000, ALPHA) < 0.05
    assert irrelevance_threshold(2, ALPHA) == 1.0  # df <= 0 -> degenerate


def test_redundancy_lower_bound_is_size_aware():
    # The same observed association is "redundant" only with enough data to be
    # confident the true association clears the floor.
    assert redundancy_lower_bound(0.92, 30, ALPHA) < FLOOR  # tiny n -> wide CI
    assert redundancy_lower_bound(0.92, 100_000, ALPHA) > FLOOR  # large n -> tight CI
    assert redundancy_lower_bound(0.92, 3, ALPHA) == 0.0  # too small to form an interval


# --------------------------------------------------------------------------- #
# plan_pruning decision logic (hand-built association matrices)
# --------------------------------------------------------------------------- #
def test_drops_predictor_uncorrelated_with_target():
    m = _matrix(
        ["good", "noise", "y"], {("good", "y"): 0.6, ("noise", "y"): 0.01, ("good", "noise"): 0.02}
    )
    result = plan_pruning(m, "y", 1000, alpha=ALPHA, redundancy_floor=FLOOR)
    assert result.dropped == ["noise"]
    assert "uncorrelated" in result.reasons["noise"]
    assert result.anomalies == []


def test_drops_weaker_member_of_a_redundant_pair():
    m = _matrix(
        ["strong", "weak", "y"],
        {("strong", "y"): 0.6, ("weak", "y"): 0.3, ("strong", "weak"): 0.98},
    )
    result = plan_pruning(m, "y", 1000, alpha=ALPHA, redundancy_floor=FLOOR)
    assert result.dropped == ["weak"]  # keep the more target-correlated "strong"
    assert "redundant with 'strong'" in result.reasons["weak"]


def test_keeps_and_warns_on_transitivity_anomaly():
    # x is unrelated to the target yet ~duplicates y_rel, which IS related.
    m = _matrix(
        ["x", "y_rel", "y"],
        {("x", "y"): 0.01, ("y_rel", "y"): 0.6, ("x", "y_rel"): 0.98},
    )
    # Capture via a handler on the module logger directly: robust to the package's
    # logging config, which may disable propagation to the root (where caplog sits).
    messages: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())  # type: ignore[method-assign]
    sel_logger = logging.getLogger("kaggle_pipeline.preprocessing.selection")
    sel_logger.addHandler(handler)
    previous_level = sel_logger.level
    sel_logger.setLevel(logging.WARNING)
    try:
        result = plan_pruning(m, "y", 1000, alpha=ALPHA, redundancy_floor=FLOOR)
    finally:
        sel_logger.removeHandler(handler)
        sel_logger.setLevel(previous_level)

    assert result.dropped == []  # x is protected, not dropped
    assert result.anomalies == [("x", "y_rel")]  # structured record of the anomaly
    assert any("SUSPICIOUS" in msg for msg in messages)  # loud warning was emitted


def test_safeguard_never_drops_every_predictor():
    m = _matrix(["a", "b", "y"], {("a", "y"): 0.01, ("b", "y"): 0.02, ("a", "b"): 0.01})
    result = plan_pruning(m, "y", 1000, alpha=ALPHA, redundancy_floor=FLOOR)
    assert result.dropped == ["a"]  # keeps the most target-correlated ("b")


# --------------------------------------------------------------------------- #
# End-to-end through the pretrain pipeline
# --------------------------------------------------------------------------- #
def _write_pruning_competition(data_dir: Path, n_train: int = 400, n_test: int = 100) -> None:
    rng = np.random.default_rng(0)

    def frame(n: int, *, with_target: bool) -> pd.DataFrame:
        strong = rng.normal(size=n)
        data = {
            "id": range(n),
            "num_strong": strong,
            "num_dup": strong + rng.normal(scale=0.02, size=n),  # ~duplicate of num_strong
            "num_noise": rng.normal(size=n),  # independent of the target
        }
        if with_target:
            data["y"] = np.where(strong + rng.normal(scale=0.3, size=n) > 0, "yes", "no")
        return pd.DataFrame(data)

    frame(n_train, with_target=True).to_csv(data_dir / "train.csv", index=False)
    frame(n_test, with_target=False).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": range(n_test), "y": ["no"] * n_test}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )


def test_build_pipeline_prunes_noise_and_redundant_columns(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_pruning_competition(data_dir)
    cfg = Config(
        competition="synthetic",
        target="y",
        id_col="id",
        task="classification",
        scoring="balanced_accuracy",
        prediction_aim="category",
        prune_features=True,
        seed=0,
        data_dir=data_dir,
        storage_dir=tmp_path / "models",
    )
    ctx, _ = build_pipeline(cfg)

    predictors = set(ctx.predictor_columns)
    assert "num_noise" not in predictors  # irrelevant -> dropped
    # The near-duplicate pair collapses to exactly one survivor.
    assert len(predictors & {"num_strong", "num_dup"}) == 1


def test_prune_features_false_keeps_all_predictors(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_pruning_competition(data_dir)
    cfg = Config(
        competition="synthetic",
        target="y",
        id_col="id",
        task="classification",
        scoring="balanced_accuracy",
        prediction_aim="category",
        prune_features=False,
        seed=0,
        data_dir=data_dir,
        storage_dir=tmp_path / "models",
    )
    ctx, _ = build_pipeline(cfg)
    assert {"num_strong", "num_dup", "num_noise"} <= set(ctx.predictor_columns)


def test_config_rejects_bad_prune_params():
    with pytest.raises(ValueError, match="prune_alpha"):
        Config(competition="x", prune_alpha=0.0)
    with pytest.raises(ValueError, match="redundancy_floor"):
        Config(competition="x", redundancy_floor=1.5)
