"""DeletionPolicy: usage credit is a per-batch *rate*, not a raw count.

Guards Fix A from the feature-pool admission bias: a long-lived feature must
not accumulate an unbounded head start over a newcomer just by surviving.
"""

from __future__ import annotations

from kaggle_pipeline.evolution.features.deletion import DeletionPolicy
from kaggle_pipeline.evolution.features.genome import FeatureGenome
from kaggle_pipeline.evolution.features.recipe import NUMERIC, FeatureRecipe


def _make(*, utility: float, completed: int, elite: int, created_at: int) -> FeatureGenome:
    g = FeatureGenome(
        recipe=FeatureRecipe("log1p", ("orig::x",), {}, NUMERIC),
        human_name="t",
        created_at_batch=created_at,
    )
    g.score_set.utility = utility
    g.usage_stats.times_in_completed_model = completed
    g.usage_stats.times_in_elite_model = elite
    return g


def test_same_rate_same_score_regardless_of_age():
    policy = DeletionPolicy()
    young = _make(utility=0.2, completed=2, elite=0, created_at=8)  # age=2 -> rate 1.0
    old = _make(utility=0.2, completed=10, elite=0, created_at=0)  # age=10 -> rate 1.0
    assert policy.score(young, current_batch=10) == policy.score(old, current_batch=10)


def test_usage_bonus_is_capped_by_rate():
    policy = DeletionPolicy()
    # Even after 20 batches with usage every batch, the bonus tops out at
    # weight * 1 = 0.10 (completed) + 0.30 * elite_rate, not weight * count.
    old = _make(utility=0.2, completed=20, elite=4, created_at=0)
    score = policy.score(old, current_batch=20)
    # 0.2 + 0.10 * (20/20) + 0.30 * (4/20) = 0.2 + 0.10 + 0.06 = 0.36
    assert abs(score - 0.36) < 1e-9


def test_newcomer_can_beat_idle_incumbent():
    policy = DeletionPolicy()
    incumbent = _make(utility=0.2, completed=0, elite=0, created_at=0)  # 10 batches, never used
    newcomer = _make(utility=0.3, completed=0, elite=0, created_at=10)
    assert policy.score(newcomer, 10) > policy.score(incumbent, 10)


def test_weakest_picks_lowest_score():
    policy = DeletionPolicy()
    a = _make(utility=0.4, completed=2, elite=0, created_at=8)  # higher
    b = _make(utility=0.1, completed=0, elite=0, created_at=0)  # lowest
    c = _make(utility=0.3, completed=1, elite=0, created_at=9)
    assert policy.weakest([a, b, c], current_batch=10) is b


def test_weakest_of_empty_list_is_none():
    assert DeletionPolicy().weakest([], current_batch=0) is None


def test_zero_age_does_not_divide_by_zero():
    """A feature created in the *current* batch (age=0) still gets a finite score."""
    policy = DeletionPolicy()
    fresh = _make(utility=0.5, completed=0, elite=0, created_at=10)
    # Should not raise; max(1, age) clamps the denominator.
    score = policy.score(fresh, current_batch=10)
    assert score == 0.5  # no usage, no redundancy, no cost -> just utility
