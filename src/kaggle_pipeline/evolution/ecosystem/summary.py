"""Structured ecosystem summaries and their verbosity-aware text formatting.

``build_ecosystem_summary`` produces a JSON-serialisable dict (used by
``KagglePipeline.summarize_state`` and saved alongside checkpoints).
``format_summary`` renders that dict to text at a detail level (1..4), matching
the verbosity contract. Neither prints huge arrays/frames -- only compact
aggregates and the top handful of items.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from kaggle_pipeline.evolution.logging_utils import format_duration
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus

if TYPE_CHECKING:
    from kaggle_pipeline.evolution.features.registry import FeatureRegistry
    from kaggle_pipeline.evolution.models.registry import ModelPopulation
    from kaggle_pipeline.evolution.runtime import RuntimeManager

_TOP_N = 5


def build_ecosystem_summary(
    registry: FeatureRegistry,
    population: ModelPopulation,
    runtime: RuntimeManager | None = None,
    *,
    batch_index: int = 0,
    last_batch: Any | None = None,
    ensemble: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a structured, serialisable summary of the current ecosystem."""
    return {
        "batch_index": batch_index,
        "runtime": _runtime_section(runtime),
        "features": _features_section(registry),
        "models": _models_section(population),
        "families": _families_section(population),
        "mutations": _mutations_section(population),
        "batch": _batch_section(last_batch),
        "ensemble": ensemble or {"enabled": False, "status": "none"},
    }


def _runtime_section(runtime: RuntimeManager | None) -> dict[str, Any]:
    if runtime is None:
        return {}
    s = runtime.time_summary()
    return {
        "elapsed": s["elapsed"],
        "remaining": s["remaining"],
        "remaining_training": s["remaining_training"],
        "ensemble_reserved": s["ensemble_reserved"],
        "close_to_deadline": bool(s["close_to_deadline"]),
    }


def _features_section(registry: FeatureRegistry) -> dict[str, Any]:
    active = registry.get_active_features()
    generated_active = [g for g in active if not g.is_original]
    all_feats = registry.all_features()
    weakest = registry.find_weakest_removable_feature()
    top = sorted(active, key=lambda g: g.utility, reverse=True)[:_TOP_N]
    return {
        "original_count": registry.n_original,
        "generated_count": len(generated_active),
        "active_count": len(active),
        "inactive_count": sum(1 for g in all_feats if not g.active),
        "protected_count": len(registry.get_protected_features()),
        "weakest_removable_feature": weakest.human_name if weakest else None,
        "top_features": [{"name": g.human_name, "utility": round(g.utility, 4)} for g in top],
    }


def _models_section(population: ModelPopulation) -> dict[str, Any]:
    counts = population.status_counts()
    ranking = population.absolute_score_ranking()
    best = ranking[0] if ranking else None
    top = ranking[:_TOP_N]
    return {
        "total_models": len(population.all_genomes()),
        "completed": counts.get(ModelStatus.COMPLETED, 0),
        "failed": counts.get(ModelStatus.FAILED, 0),
        "pruned": counts.get(ModelStatus.PRUNED, 0),
        "active_population_size": len(population.active),
        "elite_archive_size": len(population.elite),
        "best_model_id": best.model_id if best else None,
        "best_score": round(best.score_set.score, 4) if best and best.score_set else None,
        "best_utility": round(best.utility, 4) if best and best.utility is not None else None,
        "top_models": [
            {
                "model_id": g.model_id,
                "family": g.family,
                "score": round(g.score_set.score, 4) if g.score_set else None,
                "utility": round(g.utility, 4) if g.utility is not None else None,
            }
            for g in top
        ],
    }


def _families_section(population: ModelPopulation) -> dict[str, Any]:
    completed = [g for g in population.completed() if g.score_set is not None]
    by_family: dict[str, list] = {}
    for g in completed:
        by_family.setdefault(g.family, []).append(g)
    return {
        family: {
            "count": len(genomes),
            "best_score": round(max(g.score_set.score for g in genomes), 4),
            "avg_utility": round(sum((g.utility or 0.0) for g in genomes) / len(genomes), 4),
        }
        for family, genomes in by_family.items()
    }


def _mutations_section(population: ModelPopulation) -> dict[str, Any]:
    records = population.mutation_records
    scored = [r for r in records if r.delta_utility is not None]
    improved = [r for r in scored if (r.delta_utility or 0.0) > 0]
    by_type: Counter = Counter(r.mutation_type for r in records)
    success_by_type: Counter = Counter(
        r.mutation_type for r in scored if (r.delta_utility or 0.0) > 0
    )
    return {
        "models_mutated_count": len(records),
        "scored_mutations": len(scored),
        "recent_mutation_success_rate": round(len(improved) / len(scored), 3) if scored else None,
        "mutation_types": dict(by_type),
        "successful_mutation_types": dict(success_by_type),
    }


def _batch_section(last_batch: Any | None) -> dict[str, Any]:
    if last_batch is None:
        return {}
    return {
        "batch_index": last_batch.batch,
        "generated": last_batch.n_generated,
        "mutated": last_batch.n_mutated,
        "completed": last_batch.n_completed,
        "failed": last_batch.n_failed,
        "skipped": last_batch.n_skipped,
        "features_active": last_batch.n_features_active,
        "features_generated": last_batch.n_features_generated,
        "promoted": list(last_batch.promoted),
    }


def format_summary(summary: dict[str, Any], level: int) -> str:
    """Render ``summary`` to text at the given detail ``level`` (1..4)."""
    if level <= 0:
        return ""
    rt = summary.get("runtime", {})
    feats = summary["features"]
    models = summary["models"]
    lines: list[str] = []

    elapsed = format_duration(rt.get("elapsed", 0)) if rt else "?"
    remaining = format_duration(rt.get("remaining", 0)) if rt else "?"
    lines.append(
        f"[batch {summary['batch_index']}] elapsed={elapsed} left={remaining} "
        f"active_feats={feats['active_count']} models={models['completed']} "
        f"failed={models['failed']} best={models['best_score']} util={models['best_utility']} "
        f"ensemble={summary['ensemble'].get('status', 'none')}"
    )
    if level == 1:
        return lines[0]

    # level >= 2
    batch = summary.get("batch") or {}
    lines.append(
        f"  models: completed={models['completed']} failed={models['failed']} "
        f"pruned={models['pruned']} active_pop={models['active_population_size']} "
        f"elite={models['elite_archive_size']}"
    )
    if models["top_models"]:
        top = ", ".join(f"{m['family']}:{m['score']}" for m in models["top_models"][:_TOP_N])
        lines.append(f"  top models: {top}")
    if batch:
        lines.append(
            f"  this batch: gen={batch.get('generated')} mut={batch.get('mutated')} "
            f"done={batch.get('completed')} fail={batch.get('failed')} "
            f"skip={batch.get('skipped')} promoted={len(batch.get('promoted', []))}"
        )

    if level >= 3:
        top_feats = ", ".join(f"{f['name']}({f['utility']})" for f in feats["top_features"])
        lines.append(f"  top features: {top_feats}")
        lines.append(f"  weakest removable: {feats['weakest_removable_feature']}")
        if summary["families"]:
            fam = ", ".join(
                f"{name}:{info['best_score']}(n={info['count']})"
                for name, info in summary["families"].items()
            )
            lines.append(f"  families: {fam}")
        mut = summary["mutations"]
        lines.append(
            f"  mutations: n={mut['models_mutated_count']} "
            f"success_rate={mut['recent_mutation_success_rate']}"
        )

    if level >= 4:
        lines.append(f"  mutation types: {summary['mutations']['mutation_types']}")
        lines.append(f"  successful types: {summary['mutations']['successful_mutation_types']}")
        lines.append(
            f"  features: active={feats['active_count']} inactive={feats['inactive_count']} "
            f"protected={feats['protected_count']} generated={feats['generated_count']}"
        )

    return "\n".join(lines)
