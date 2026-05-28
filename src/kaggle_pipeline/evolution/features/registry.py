"""The global :class:`FeatureRegistry` -- the source of truth for features.

Stores every original and generated :class:`FeatureGenome`, tracks which are
active/protected, deduplicates by recipe hash, scores features at the registry
level (target correlation, redundancy via :class:`FeatureSimilarity`, tree
importance), turns utilities into selection probabilities, and enforces the active
cap by evicting the weakest *removable* generated feature. It never trains models;
it delegates materialization to :class:`FeatureMaterializer` and scoring to
:class:`FeatureScorer`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.features.deletion import DeletionPolicy
from kaggle_pipeline.evolution.features.genome import FeatureGenome
from kaggle_pipeline.evolution.features.materialization import (
    FeatureMaterializer,
    MaterializationContext,
)
from kaggle_pipeline.evolution.features.recipe import CATEGORICAL, NUMERIC
from kaggle_pipeline.evolution.features.scoring import (
    GENERATION_COST,
    MISSINGNESS,
    REDUNDANCY,
    TARGET_CORRELATION,
    TREE_IMPORTANCE,
    FeatureScorer,
    FeatureUtility,
    rank_normalize,
)
from kaggle_pipeline.evolution.features.similarity import FeatureSimilarity
from kaggle_pipeline.evolution.features.transformations import (
    TransformationRegistry,
    build_default_registry,
)
from kaggle_pipeline.evolution.utils.arrays import missing_mask
from kaggle_pipeline.evolution.utils.logging import get_logger
from kaggle_pipeline.evolution.utils.random import (
    softmax_with_exploration,
    spawn_rng,
    weighted_choice,
)

logger = get_logger(__name__)

# Insertion outcomes for a generated candidate.
INSERTED = "inserted"
REPLACED = "replaced"
STORED_INACTIVE = "stored_inactive"
DUPLICATE = "duplicate"

# Score names normalised across the active pool before combining into utility.
_NORMALIZED_SCORES = (
    TARGET_CORRELATION,
    TREE_IMPORTANCE,
    REDUNDANCY,
    GENERATION_COST,
    MISSINGNESS,
)


@dataclass
class InsertionResult:
    """Outcome of trying to insert a generated feature into the active pool."""

    genome: FeatureGenome
    status: str
    evicted: FeatureGenome | None = None


class FeatureRegistry:
    """Holds all features and manages activation, scoring, selection and deletion."""

    def __init__(
        self,
        settings: EvolutionSettings,
        *,
        transformations: TransformationRegistry | None = None,
        similarity_top_k: int = 5,
    ):
        self.settings = settings
        self.transformations = transformations or build_default_registry()
        self.materializer = FeatureMaterializer(self, self.transformations)
        self.similarity = FeatureSimilarity(top_k=similarity_top_k)
        self.scorer = FeatureScorer()
        self.utility = FeatureUtility(settings)
        self.deletion = DeletionPolicy()
        self.current_batch = 0
        self._features: dict[str, FeatureGenome] = {}
        self._by_recipe_hash: dict[str, str] = {}

    # --- registration -------------------------------------------------------
    def add_original_feature(
        self, column: str, output_type: str, *, protected: bool = True
    ) -> FeatureGenome:
        """Register and activate a raw input column. Originals are protected."""
        genome = FeatureGenome.original(
            column, output_type, created_at_batch=self.current_batch, protected=protected
        )
        if self.has_recipe_hash(genome.recipe_hash):
            return self.get_feature(self._by_recipe_hash[genome.recipe_hash])
        self._store(genome)
        genome.active = True
        logger.debug("registered original feature %s (%s)", genome.feature_id, output_type)
        return genome

    def add_generated_feature(self, genome: FeatureGenome) -> FeatureGenome | None:
        """Store a generated feature (inactive). Returns ``None`` on a duplicate recipe."""
        if self.has_recipe_hash(genome.recipe_hash):
            return None
        self._store(genome)
        genome.active = False
        return genome

    def _store(self, genome: FeatureGenome) -> None:
        if genome.feature_id in self._features:
            raise ValueError(f"feature id {genome.feature_id!r} already registered")
        self._features[genome.feature_id] = genome
        self._by_recipe_hash[genome.recipe_hash] = genome.feature_id
        # Attach a generation-cost score so it can be normalised and penalised.
        cost = 0.0
        if not genome.is_original:
            try:
                cost = self.transformations.get(genome.transform_name).cost_estimate
            except KeyError:
                cost = 1.0
        genome.score_set.set(GENERATION_COST, cost, higher_is_better=False)

    # --- lookup -------------------------------------------------------------
    def get_feature(self, feature_id: str) -> FeatureGenome:
        if feature_id not in self._features:
            raise KeyError(f"feature {feature_id!r} not in registry")
        return self._features[feature_id]

    def has_recipe_hash(self, recipe_hash: str) -> bool:
        return recipe_hash in self._by_recipe_hash

    def all_features(self) -> list[FeatureGenome]:
        return list(self._features.values())

    def get_active_features(self) -> list[FeatureGenome]:
        return [g for g in self._features.values() if g.active]

    def get_protected_features(self) -> list[FeatureGenome]:
        return [g for g in self._features.values() if g.protected]

    def get_original_features(self) -> list[FeatureGenome]:
        return [g for g in self._features.values() if g.is_original]

    @property
    def n_original(self) -> int:
        return sum(1 for g in self._features.values() if g.is_original)

    @property
    def effective_max_active_features(self) -> int:
        return self.settings.effective_max_active_features(self.n_original)

    def is_removable(self, genome: FeatureGenome) -> bool:
        """Generated, active, unprotected, and past the creation cooldown."""
        if genome.is_original or genome.protected or not genome.active:
            return False
        age = self.current_batch - genome.created_at_batch
        return age >= self.settings.feature_deletion_cooldown_batches

    def get_removable_features(self) -> list[FeatureGenome]:
        return [g for g in self._features.values() if self.is_removable(g)]

    # --- activation ---------------------------------------------------------
    def activate_feature(self, feature_id: str) -> None:
        self.get_feature(feature_id).active = True

    def deactivate_feature(self, feature_id: str) -> None:
        genome = self.get_feature(feature_id)
        if genome.protected or genome.is_original:
            raise ValueError(f"cannot deactivate protected/original feature {feature_id!r}")
        genome.active = False

    # --- materialization ----------------------------------------------------
    def materialize(self, feature_id: str, context: MaterializationContext) -> np.ndarray:
        return self.materializer.materialize(feature_id, context)

    # --- scoring ------------------------------------------------------------
    def score_feature(
        self,
        feature_id: str,
        *,
        context: MaterializationContext,
        y: np.ndarray,
        task: str = "classification",
        update_similarity: bool = True,
    ) -> None:
        """Compute intrinsic scores for one feature on the evaluation context."""
        genome = self.get_feature(feature_id)
        if genome.uses_target:
            return  # OOF scoring path is a TODO (see materialization).
        values = self.materialize(feature_id, context)
        genome.score_set.set(
            TARGET_CORRELATION, self.scorer.target_correlation(values, y, task=task)
        )
        genome.score_set.set(MISSINGNESS, self.scorer.missingness(values), higher_is_better=False)
        if update_similarity and genome.output_type in (NUMERIC, "boolean"):
            redundancy = self.similarity.update_for_feature(feature_id, values)
            genome.score_set.set(REDUNDANCY, redundancy, higher_is_better=False)
        elif genome.output_type == CATEGORICAL:
            # Record distinct-value count so the model factory can constrain one-hot
            # encoding at build time rather than only as a training-time fallback.
            obj = np.asarray(values, dtype=object)
            present = obj[~missing_mask(obj)]
            genome.cardinality = int(np.unique(present.astype(str)).size)

    def score_tree_importance(
        self,
        *,
        context: MaterializationContext,
        y: np.ndarray,
        task: str = "classification",
        rng: np.random.Generator | None = None,
    ) -> None:
        """Fit one small tree ensemble over active numeric features; store importances."""
        active_numeric = [
            g for g in self.get_active_features() if g.output_type in (NUMERIC, "boolean")
        ]
        named = {g.feature_id: self.materialize(g.feature_id, context) for g in active_numeric}
        importances = self.scorer.tree_importances(named, y, task=task, rng=rng)
        for feature_id, imp in importances.items():
            self.get_feature(feature_id).score_set.set(TREE_IMPORTANCE, imp)

    def recompute_utilities(self) -> None:
        """Normalise scores across the active pool and recompute each utility."""
        features = self.get_active_features()
        if not features:
            return
        for name in _NORMALIZED_SCORES:
            present = [f for f in features if f.score_set.has(name)]
            if not present:
                continue
            normalized = rank_normalize(np.array([f.score_set.value(name) for f in present]))
            for f, nv in zip(present, normalized, strict=True):
                score = f.score_set.get(name)
                if score is not None:
                    score.normalized_value = float(nv)
        for f in features:
            self.utility.combine(f)

    def rescore_active(
        self,
        *,
        context: MaterializationContext,
        y: np.ndarray,
        task: str = "classification",
        rng: np.random.Generator | None = None,
        with_tree_importance: bool = True,
    ) -> None:
        """Full per-batch scoring pass over the active pool."""
        for genome in self.get_active_features():
            self.score_feature(genome.feature_id, context=context, y=y, task=task)
        if with_tree_importance:
            self.score_tree_importance(context=context, y=y, task=task, rng=rng)
        self.recompute_utilities()

    # --- selection ----------------------------------------------------------
    def compute_selection_probabilities(
        self, features: list[FeatureGenome] | None = None
    ) -> dict[str, float]:
        """Softmax-with-exploration probabilities over ``features`` (default: active)."""
        features = features if features is not None else self.get_active_features()
        if not features:
            return {}
        utilities = np.array([f.utility for f in features])
        probs = softmax_with_exploration(
            utilities,
            temperature=self.settings.feature_selection_temperature,
            exploration_rate=self.settings.feature_selection_exploration_rate,
        )
        return {f.feature_id: float(p) for f, p in zip(features, probs, strict=True)}

    def sample_feature(
        self,
        rng: np.random.Generator | None = None,
        *,
        output_type: str | None = None,
        exclude: set[str] | None = None,
    ) -> str | None:
        """Sample one active feature id by utility, optionally filtered by type."""
        ids = self.sample_features(1, rng, output_type=output_type, exclude=exclude)
        return ids[0] if ids else None

    def sample_features(
        self,
        n: int,
        rng: np.random.Generator | None = None,
        *,
        output_type: str | None = None,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """Sample up to ``n`` distinct active feature ids by selection probability."""
        rng = spawn_rng(rng)
        exclude = exclude or set()
        candidates = [
            g
            for g in self.get_active_features()
            if g.feature_id not in exclude and (output_type is None or g.output_type == output_type)
        ]
        if not candidates:
            return []
        probs = self.compute_selection_probabilities(candidates)
        return weighted_choice(rng, probs, n)

    def get_candidate_parents(self, *, output_type: str | None = None) -> list[FeatureGenome]:
        """Features eligible as parents for generation under the current depth rules."""
        if self.settings.allow_generated_feature_parents:
            pool = [
                g for g in self.get_active_features() if g.depth < self.settings.max_feature_depth
            ]
        else:
            pool = [g for g in self.get_original_features() if g.active]
        if output_type is not None:
            pool = [g for g in pool if g.output_type == output_type]
        return pool

    # --- similarity ---------------------------------------------------------
    def get_similarity(self, feature_id_a: str, feature_id_b: str) -> float:
        return self.similarity.get(feature_id_a, feature_id_b)

    def update_similarity_for_feature(self, feature_id: str, values: np.ndarray) -> float:
        return self.similarity.update_for_feature(feature_id, values)

    # --- deletion / insertion ----------------------------------------------
    def find_weakest_removable_feature(self) -> FeatureGenome | None:
        return self.deletion.weakest(self.get_removable_features(), self.current_batch)

    def maybe_insert_generated_feature(self, genome: FeatureGenome) -> InsertionResult:
        """Insert a *scored* generated candidate, evicting the weakest if the pool is full.

        Assumes ``genome.score_set.utility`` has been computed by the generator.
        Duplicate recipes return the existing feature untouched. When the pool is
        full, the candidate replaces the weakest removable feature only if it
        scores higher; otherwise it is stored inactive (still reproducible).
        """
        if self.has_recipe_hash(genome.recipe_hash):
            existing = self.get_feature(self._by_recipe_hash[genome.recipe_hash])
            return InsertionResult(existing, DUPLICATE)

        self._store(genome)
        genome.active = False

        if len(self.get_active_features()) < self.effective_max_active_features:
            genome.active = True
            logger.debug("activated generated feature %s", genome.feature_id)
            return InsertionResult(genome, INSERTED)

        weakest = self.find_weakest_removable_feature()
        if weakest is not None and self.deletion.score(
            genome, self.current_batch
        ) > self.deletion.score(weakest, self.current_batch):
            weakest.active = False
            self.similarity.remove(weakest.feature_id)
            genome.active = True
            logger.info(
                "feature %s evicted %s (full active pool)", genome.feature_id, weakest.feature_id
            )
            return InsertionResult(genome, REPLACED, evicted=weakest)

        return InsertionResult(genome, STORED_INACTIVE)

    # --- batch bookkeeping --------------------------------------------------
    def advance_batch(self) -> int:
        self.current_batch += 1
        return self.current_batch
