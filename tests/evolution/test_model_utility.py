"""Model scoring/utility: metric conversion, cost-awareness, failed-result storage."""

from __future__ import annotations

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.evaluation.metrics import metric_higher_is_better
from kaggle_pipeline.evolution.models.lifecycle import FailureReason, ModelStatus
from kaggle_pipeline.evolution.models.scoring import (
    ModelScoreSet,
    ModelUtility,
    comparable_stats,
    to_internal,
)
from kaggle_pipeline.evolution.models.training import ModelResult


def test_lower_is_better_metric_is_negated():
    # An RMSE-like metric is converted so larger is always better internally.
    assert to_internal(2.0, higher_is_better=False) == -2.0
    assert to_internal(0.9, higher_is_better=True) == 0.9


def test_neg_rmse_already_higher_is_better():
    assert metric_higher_is_better("neg_root_mean_squared_error")


def test_adj_score_penalises_instability():
    settings = EvolutionSettings()
    stable = ModelScoreSet(score=0.9, score_std=0.01)
    shaky = ModelScoreSet(score=0.9, score_std=0.10)
    p = settings.score_std_penalty
    assert stable.adj_score(p) > shaky.adj_score(p)


def test_utility_is_cost_aware():
    settings = EvolutionSettings()
    util = ModelUtility(settings)
    cheap = ModelScoreSet(score=0.9, score_std=0.0, compute_time=1.0)
    pricey = ModelScoreSet(score=0.9, score_std=0.0, compute_time=100.0)
    stats = comparable_stats([cheap, pricey], std_penalty=settings.score_std_penalty)
    # Equal adj-score, so the cheaper model has the higher (less divided) utility.
    assert util.utility(cheap, stats) >= util.utility(pricey, stats)


def test_failed_result_can_be_stored():
    result = ModelResult(
        model_id="m_x",
        status=ModelStatus.FAILED,
        failure_reason=FailureReason.TRAINING_EXCEPTION,
        error_message="boom",
    )
    assert result.status == ModelStatus.FAILED
    assert result.score_set is None
    assert result.failure_reason == FailureReason.TRAINING_EXCEPTION
