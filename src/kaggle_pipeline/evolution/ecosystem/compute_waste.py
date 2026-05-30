"""End-of-cycle compute-waste summary, grouped by model family.

Every trained genome consumed wall-time (``score_set.compute_time``). At the
end of a run we want to know where that time went, sliced two ways:

* **Outcome** -- did the model's training pay off?
    * ``win``     -- model is a final ensemble member.
    * ``partial`` -- model entered the leaderboard at some point but is not
      a final member.
    * ``waste``   -- model never entered the leaderboard.

* **Origin** -- where the genome came from.
    * ``new``      -- factory-generated (``parent_model_id is None``).
    * ``mutation`` -- spawned from an existing genome.

That's six buckets. The table reports one row per family with a leading
column for the family's share of the *grand* compute total, then six
percentages that distribute that family's spend across the buckets and sum
to 100% (within rounding). A reader can scan the first column to find the
heavy hitters and the next six to see how productive each family was.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaggle_pipeline.evolution.ensemble.manager import EnsembleResult
    from kaggle_pipeline.evolution.models.genome import ModelGenome
    from kaggle_pipeline.evolution.models.registry import ModelPopulation


OUTCOMES = ("win", "partial", "waste")
ORIGINS = ("new", "mutation")
# Iteration order also fixes column order in the rendered table.
BUCKETS = tuple((outcome, origin) for outcome in OUTCOMES for origin in ORIGINS)


@dataclass
class FamilyComputeRow:
    """One family's compute spend split across the six outcome × origin buckets."""

    family: str
    total_seconds: float
    by_bucket: dict[tuple[str, str], float]  # (outcome, origin) -> seconds

    def share_of(self, grand_total: float) -> float:
        return self.total_seconds / grand_total if grand_total > 0 else 0.0

    def fraction(self, outcome: str, origin: str) -> float:
        if self.total_seconds <= 0:
            return 0.0
        return self.by_bucket.get((outcome, origin), 0.0) / self.total_seconds


@dataclass
class ComputeWasteSummary:
    """The whole report: per-family rows plus the grand compute total."""

    rows: list[FamilyComputeRow]
    grand_total_seconds: float
    ensemble_member_ids: frozenset[str]


def classify(
    g: ModelGenome,
    ensemble_member_ids: frozenset[str],
) -> tuple[str, str] | None:
    """Return ``(outcome, origin)`` for a genome, or ``None`` if it shouldn't count.

    Genomes that never finished training (no ``score_set``) consumed no
    measurable compute and are dropped from the report.
    """
    if g.score_set is None or g.score_set.compute_time is None:
        return None
    if g.model_id in ensemble_member_ids:
        outcome = "win"
    elif getattr(g, "was_elite", False):
        outcome = "partial"
    else:
        outcome = "waste"
    origin = "new" if g.parent_model_id is None else "mutation"
    return outcome, origin


def build_compute_waste_summary(
    population: ModelPopulation,
    ensemble_result: EnsembleResult | None,
) -> ComputeWasteSummary:
    """Aggregate compute spend by ``(family, outcome, origin)`` across the population."""
    members = frozenset(ensemble_result.weights.keys()) if ensemble_result else frozenset()
    by_family: dict[str, dict[tuple[str, str], float]] = defaultdict(
        lambda: dict.fromkeys(BUCKETS, 0.0)
    )
    grand_total = 0.0
    for g in population.all_genomes():
        bucket = classify(g, members)
        if bucket is None:
            continue
        # ``score_set`` and ``compute_time`` are non-None per classify's guard.
        assert g.score_set is not None and g.score_set.compute_time is not None
        elapsed = float(g.score_set.compute_time)
        by_family[g.family][bucket] += elapsed
        grand_total += elapsed
    rows = [
        FamilyComputeRow(
            family=name,
            total_seconds=sum(buckets.values()),
            by_bucket=buckets,
        )
        for name, buckets in by_family.items()
    ]
    # Sort by share desc so heavy hitters lead the table.
    rows.sort(key=lambda r: r.total_seconds, reverse=True)
    return ComputeWasteSummary(
        rows=rows,
        grand_total_seconds=grand_total,
        ensemble_member_ids=members,
    )


def _pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def format_compute_waste_summary(summary: ComputeWasteSummary) -> str:
    """Render the report as a fixed-width text table for the log."""
    if not summary.rows or summary.grand_total_seconds <= 0:
        return "compute-waste summary: no completed models with measured time"

    header_row = (
        f"{'family':<14} {'share':>6} "
        f"| {'win:new':>7} {'win:mut':>7} "
        f"| {'part:new':>8} {'part:mut':>8} "
        f"| {'wst:new':>7} {'wst:mut':>7}"
    )
    sep = "-" * len(header_row)

    lines = [
        f"compute-waste summary (total measured fit time: "
        f"{summary.grand_total_seconds:.0f}s; "
        f"{len(summary.ensemble_member_ids)} ensemble members)",
        # Legend so the abbreviated column headers are self-documenting.
        "  win = in final ensemble    "
        "part = was on leaderboard, evicted later    "
        "wst = never on leaderboard",
        "  new = factory-generated    mut = mutation of an existing model",
        sep,
        header_row,
        sep,
    ]
    for row in summary.rows:
        cells = [_pct(row.fraction(o, p)) for o, p in BUCKETS]
        lines.append(
            f"{row.family:<14} {_pct(row.share_of(summary.grand_total_seconds)):>6} "
            f"| {cells[0]:>7} {cells[1]:>7} "
            f"| {cells[2]:>8} {cells[3]:>8} "
            f"| {cells[4]:>7} {cells[5]:>7}"
        )
    lines.append(sep)
    return "\n".join(lines)
