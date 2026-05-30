"""Per-family compute-spend breakdown used at end of a training cycle."""

from __future__ import annotations

from typing import Any

import pytest

from kaggle_pipeline.evolution.ecosystem.compute_waste import (
    BUCKETS,
    build_compute_waste_summary,
    format_compute_waste_summary,
)
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus


class _StubScoreSet:
    def __init__(self, compute_time: float | None) -> None:
        self.compute_time = compute_time


class _StubGenome:
    def __init__(
        self,
        *,
        model_id: str,
        family: str,
        compute_time: float | None,
        parent_model_id: str | None = None,
        was_elite: bool = False,
        status: str = ModelStatus.COMPLETED,
    ) -> None:
        self.model_id = model_id
        self.family = family
        self.parent_model_id = parent_model_id
        self.was_elite = was_elite
        self.status = status
        self.score_set: Any = _StubScoreSet(compute_time) if compute_time is not None else None


class _StubPopulation:
    def __init__(self, genomes: list[_StubGenome]) -> None:
        self._genomes = list(genomes)

    def all_genomes(self) -> list[_StubGenome]:
        return list(self._genomes)


class _StubEnsembleResult:
    def __init__(self, member_ids: list[str]) -> None:
        self.weights = {mid: 1.0 / len(member_ids) for mid in member_ids}


def test_buckets_cover_three_outcomes_times_two_origins():
    # Sanity: the bucket grid is exactly 3*2 in the order the formatter relies on.
    assert len(BUCKETS) == 6
    assert {b[0] for b in BUCKETS} == {"win", "partial", "waste"}
    assert {b[1] for b in BUCKETS} == {"new", "mutation"}


def test_summary_aggregates_by_family_outcome_origin():
    # lightgbm: one win-new (10s), one waste-mut (40s)  -> share 50/120 = 41.7%
    # xgboost: one partial-new (20s), one partial-mut (50s) -> share 70/120 = 58.3%
    population = _StubPopulation(
        [
            _StubGenome(
                model_id="m_lg_win",
                family="lightgbm",
                compute_time=10.0,
                parent_model_id=None,
            ),
            _StubGenome(
                model_id="m_lg_waste",
                family="lightgbm",
                compute_time=40.0,
                parent_model_id="m_lg_win",  # mutation
            ),
            _StubGenome(
                model_id="m_xg_partial_a",
                family="xgboost",
                compute_time=20.0,
                parent_model_id=None,
                was_elite=True,
            ),
            _StubGenome(
                model_id="m_xg_partial_b",
                family="xgboost",
                compute_time=50.0,
                parent_model_id="m_xg_partial_a",
                was_elite=True,
            ),
        ]
    )
    summary = build_compute_waste_summary(
        population,
        _StubEnsembleResult(["m_lg_win"]),  # only the lightgbm win is in ensemble
    )

    assert summary.grand_total_seconds == pytest.approx(120.0)
    rows = {row.family: row for row in summary.rows}

    lg = rows["lightgbm"]
    assert lg.total_seconds == pytest.approx(50.0)
    assert lg.share_of(summary.grand_total_seconds) == pytest.approx(50 / 120)
    assert lg.fraction("win", "new") == pytest.approx(10 / 50)
    assert lg.fraction("waste", "mutation") == pytest.approx(40 / 50)
    # Within-family distribution must total 100% (within rounding).
    assert sum(lg.fraction(o, p) for o, p in BUCKETS) == pytest.approx(1.0)

    xg = rows["xgboost"]
    assert xg.fraction("partial", "new") == pytest.approx(20 / 70)
    assert xg.fraction("partial", "mutation") == pytest.approx(50 / 70)
    assert xg.fraction("win", "new") == 0.0  # nothing landed in the ensemble


def test_untrained_genomes_are_ignored():
    # No score_set => no measured compute => not in the report.
    population = _StubPopulation(
        [
            _StubGenome(
                model_id="m_ghost", family="lightgbm", compute_time=None, status=ModelStatus.CREATED
            ),
        ]
    )
    summary = build_compute_waste_summary(population, None)
    assert summary.rows == []
    assert summary.grand_total_seconds == 0.0


def test_no_ensemble_means_no_wins():
    # Without an ensemble result, an elite-but-no-win genome should be ``partial``.
    population = _StubPopulation(
        [
            _StubGenome(
                model_id="m_a",
                family="lightgbm",
                compute_time=10.0,
                was_elite=True,
                parent_model_id=None,
            ),
        ]
    )
    summary = build_compute_waste_summary(population, None)
    row = summary.rows[0]
    assert row.fraction("partial", "new") == pytest.approx(1.0)


def test_rows_sorted_by_share_descending():
    population = _StubPopulation(
        [
            _StubGenome(model_id="m_x", family="xgboost", compute_time=5.0),
            _StubGenome(model_id="m_l", family="lightgbm", compute_time=20.0),
            _StubGenome(model_id="m_c", family="catboost", compute_time=10.0),
        ]
    )
    summary = build_compute_waste_summary(population, None)
    assert [r.family for r in summary.rows] == ["lightgbm", "catboost", "xgboost"]


def test_formatter_renders_table_with_one_row_per_family_summing_to_100pct():
    population = _StubPopulation(
        [
            _StubGenome(
                model_id="m_win", family="lightgbm", compute_time=10.0, parent_model_id=None
            ),
            _StubGenome(
                model_id="m_mut_waste",
                family="lightgbm",
                compute_time=30.0,
                parent_model_id="m_win",
            ),
        ]
    )
    summary = build_compute_waste_summary(population, _StubEnsembleResult(["m_win"]))
    rendered = format_compute_waste_summary(summary)

    # Header is present.
    assert "family" in rendered and "share" in rendered
    # All six bucket headers appear.
    for label in ("win:new", "win:mut", "part:new", "part:mut", "wst:new", "wst:mut"):
        assert label in rendered
    # The legend explains the abbreviations before the table.
    for legend_term in (
        "win = in final ensemble",
        "part = was on leaderboard",
        "wst = never on leaderboard",
        "new = factory-generated",
        "mut = mutation",
    ):
        assert legend_term in rendered
    # Legend appears above the column headers so readers see it first.
    assert rendered.index("win = in final ensemble") < rendered.index("win:new")
    # The one row covers lightgbm.
    assert "lightgbm" in rendered
    # Win:new is 25% (10s of 40s); waste:mut is 75%.
    assert " 25.0%" in rendered and " 75.0%" in rendered


def test_formatter_handles_empty_summary_gracefully():
    population = _StubPopulation([])
    rendered = format_compute_waste_summary(build_compute_waste_summary(population, None))
    assert "no completed models" in rendered
