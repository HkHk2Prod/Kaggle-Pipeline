"""Verbosity-aware rendering of the ecosystem summary."""

from __future__ import annotations

from kaggle_pipeline.evolution.ecosystem.summary import format_summary
from kaggle_pipeline.evolution.logging_utils import Verbosity


def _summary_with_one_top_model(**model_overrides):
    """Minimal-but-complete summary dict for the renderer."""
    base_model = {
        "model_id": "m1",
        "family": "lightgbm",
        "score": 0.9123,
        "score_std": 0.0042,
        "compute_time": 1.23,
        "n_features": 5,
        "utility": 0.4,
        "genes": ["gene_a", "gene_b"],
    }
    base_model.update(model_overrides)
    return {
        "batch_index": 1,
        "runtime": {"elapsed": 10, "remaining": 90},
        "features": {
            "active_count": 6,
            "inactive_count": 0,
            "protected_count": 2,
            "generated_count": 1,
            "top_features": [{"name": "num1", "utility": 0.5}],
            "weakest_removable_feature": None,
        },
        "models": {
            "completed": 1,
            "failed": 0,
            "pruned": 0,
            "active_population_size": 1,
            "elite_archive_size": 1,
            "best_score": 0.9123,
            "best_utility": 0.4,
            "top_models": [base_model],
        },
        "families": {"lightgbm": {"best_score": 0.9123, "count": 1}},
        "mutations": {
            "models_mutated_count": 0,
            "scored_mutations": 0,
            "recent_mutation_success_rate": None,
            "mutation_types": {},
            "successful_mutation_types": {},
        },
        "batch": {},
        "ensemble": {"status": "none"},
    }


def test_top_model_line_contains_time_std_and_features():
    text = format_summary(_summary_with_one_top_model(), Verbosity.DETAILED)
    # First line for the model carries score, std, training time and feature count.
    model_line = next(line for line in text.split("\n") if "model m1" in line)
    assert "score=0.9123" in model_line
    assert "±0.0042" in model_line
    assert "time=1.23s" in model_line
    assert "feats=5" in model_line


def test_top_model_genes_render_directly_under_score_line():
    text = format_summary(_summary_with_one_top_model(), Verbosity.DETAILED)
    lines = text.split("\n")
    score_idx = next(i for i, line in enumerate(lines) if "model m1" in line)
    # The very next line is the genes line -- no empty line, no other block in between.
    assert "genes:" in lines[score_idx + 1]
    assert lines[score_idx + 1].strip() != ""


def test_top_model_line_handles_missing_score_set_fields():
    summary = _summary_with_one_top_model(
        score=None, score_std=None, compute_time=None, n_features=None, utility=None
    )
    # Renderer must not crash when the genome never finished scoring.
    text = format_summary(summary, Verbosity.DETAILED)
    assert "model m1" in text
