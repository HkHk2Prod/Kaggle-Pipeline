"""The global :class:`FeatureGenome` and its :class:`FeatureUsageStats`.

A ``FeatureGenome`` is a *global, logical* feature definition. Its identity
fields (recipe, id, name, lineage, depth) are immutable: "mutating" a feature
means deriving a *new* genome from a transform, never editing this one. Only
runtime state owned by the registry -- ``active``, ``score_set``, ``usage_stats``
-- changes over a run. Models reference a genome by :attr:`feature_id`, never by
:attr:`human_name` (which is for humans only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaggle_pipeline.evolution.features.recipe import FeatureRecipe
from kaggle_pipeline.evolution.features.scoring import FeatureScoreSet
from kaggle_pipeline.evolution.storage.hashing import short_hash


def derive_feature_id(recipe: FeatureRecipe) -> str:
    """Deterministic feature id from a recipe.

    Identical canonical recipes therefore share a feature id (and deduplicate).
    Originals get a readable ``orig::<column>`` id; generated features get
    ``gen::<short recipe hash>``. The id is stable but is *not* the source of
    truth -- the recipe hash is.
    """
    if recipe.is_original:
        return f"orig::{recipe.parameters['source_column']}"
    return f"gen::{short_hash(recipe.recipe_hash, 16)}"


@dataclass
class FeatureUsageStats:
    """Running downstream-usage accumulators for one feature.

    Averages are exposed as properties over running sums/counts so updates are
    O(1) and order-independent. ``downstream_observation_count`` drives the
    confidence weighting between intrinsic and downstream scores.
    """

    times_selected_by_model: int = 0
    times_in_completed_model: int = 0
    times_in_elite_model: int = 0
    times_added_by_mutation: int = 0
    times_removed_by_mutation: int = 0
    sum_model_utility: float = 0.0
    n_model_utility: int = 0
    sum_add_delta: float = 0.0
    n_add_delta: int = 0
    sum_remove_delta: float = 0.0
    n_remove_delta: int = 0
    sum_importance: float = 0.0
    n_importance: int = 0
    model_family_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    # --- updates ------------------------------------------------------------
    def record_selected(self) -> None:
        self.times_selected_by_model += 1

    def record_completed(
        self, model_utility: float, *, family: str | None = None, importance: float | None = None
    ) -> None:
        self.times_in_completed_model += 1
        self.sum_model_utility += float(model_utility)
        self.n_model_utility += 1
        if importance is not None:
            self.sum_importance += float(importance)
            self.n_importance += 1
        if family is not None:
            fam = self.model_family_stats.setdefault(family, {"n": 0.0, "sum_utility": 0.0})
            fam["n"] += 1.0
            fam["sum_utility"] += float(model_utility)

    def record_elite(self) -> None:
        self.times_in_elite_model += 1

    def record_added(self, delta_utility: float) -> None:
        self.times_added_by_mutation += 1
        self.sum_add_delta += float(delta_utility)
        self.n_add_delta += 1

    def record_removed(self, delta_utility: float) -> None:
        self.times_removed_by_mutation += 1
        self.sum_remove_delta += float(delta_utility)
        self.n_remove_delta += 1

    # --- derived averages ---------------------------------------------------
    @property
    def avg_model_utility_when_used(self) -> float:
        return self.sum_model_utility / self.n_model_utility if self.n_model_utility else 0.0

    @property
    def avg_add_delta(self) -> float:
        return self.sum_add_delta / self.n_add_delta if self.n_add_delta else 0.0

    @property
    def avg_remove_delta(self) -> float:
        return self.sum_remove_delta / self.n_remove_delta if self.n_remove_delta else 0.0

    @property
    def avg_importance(self) -> float:
        return self.sum_importance / self.n_importance if self.n_importance else 0.0

    @property
    def elite_usage_rate(self) -> float:
        if not self.times_in_completed_model:
            return 0.0
        return self.times_in_elite_model / self.times_in_completed_model

    @property
    def downstream_observation_count(self) -> int:
        """Total downstream evidence accumulated (drives confidence weighting)."""
        return self.n_add_delta + self.n_remove_delta + self.n_model_utility

    def to_serializable(self) -> dict[str, Any]:
        return {
            "times_selected_by_model": self.times_selected_by_model,
            "times_in_completed_model": self.times_in_completed_model,
            "times_in_elite_model": self.times_in_elite_model,
            "times_added_by_mutation": self.times_added_by_mutation,
            "times_removed_by_mutation": self.times_removed_by_mutation,
            "avg_model_utility_when_used": self.avg_model_utility_when_used,
            "avg_add_delta": self.avg_add_delta,
            "avg_remove_delta": self.avg_remove_delta,
            "avg_importance": self.avg_importance,
            "elite_usage_rate": self.elite_usage_rate,
            "downstream_observation_count": self.downstream_observation_count,
            "model_family_stats": {k: dict(v) for k, v in self.model_family_stats.items()},
        }


@dataclass
class FeatureGenome:
    """A global logical feature: an immutable recipe plus mutable runtime state."""

    recipe: FeatureRecipe
    human_name: str
    is_original: bool = False
    protected: bool = False
    created_at_batch: int = 0
    parent_genome_id: str | None = None
    depth: int = 0
    complexity: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    # --- runtime state owned by the FeatureRegistry ------------------------
    active: bool = True
    score_set: FeatureScoreSet = field(default_factory=FeatureScoreSet)
    usage_stats: FeatureUsageStats = field(default_factory=FeatureUsageStats)
    # Cached id (derived from the recipe); set in __post_init__.
    feature_id: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if not self.feature_id:
            self.feature_id = derive_feature_id(self.recipe)

    # --- convenience views onto the recipe (the source of truth) -----------
    @property
    def recipe_hash(self) -> str:
        return self.recipe.recipe_hash

    @property
    def transform_name(self) -> str:
        return self.recipe.transform_name

    @property
    def parent_feature_ids(self) -> tuple[str, ...]:
        return self.recipe.parent_feature_ids

    @property
    def output_type(self) -> str:
        return self.recipe.output_type

    @property
    def uses_target(self) -> bool:
        return self.recipe.uses_target

    @property
    def requires_oof(self) -> bool:
        return self.recipe.requires_oof

    @property
    def utility(self) -> float:
        return self.score_set.utility

    # --- factories ----------------------------------------------------------
    @classmethod
    def original(
        cls,
        column: str,
        output_type: str,
        *,
        created_at_batch: int = 0,
        protected: bool = True,
    ) -> FeatureGenome:
        """Build the genome for a raw input column (an identity recipe)."""
        recipe = FeatureRecipe(
            transform_name="identity",
            parent_feature_ids=(),
            parameters={"source_column": column},
            output_type=output_type,
        )
        return cls(
            recipe=recipe,
            human_name=column,
            is_original=True,
            protected=protected,
            created_at_batch=created_at_batch,
            depth=0,
            complexity=0.0,
        )

    def to_serializable(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "human_name": self.human_name,
            "recipe": self.recipe.to_serializable(),
            "recipe_hash": self.recipe_hash,
            "is_original": self.is_original,
            "protected": self.protected,
            "active": self.active,
            "created_at_batch": self.created_at_batch,
            "parent_genome_id": self.parent_genome_id,
            "depth": self.depth,
            "complexity": self.complexity,
            "score_set": self.score_set.to_serializable(),
            "usage_stats": self.usage_stats.to_serializable(),
            "metadata": dict(self.metadata),
        }
