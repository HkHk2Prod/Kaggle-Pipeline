"""The :class:`EvolutionSettings` object -- every knob the evolutionary layer needs.

Mirrors the role of :class:`kaggle_pipeline.config.Config` for the v1 pipeline:
one explicit, serialisable place for the metaparameters of feature generation,
feature scoring/selection, feature deletion, model generation, model mutation,
credit assignment and the controller loop. Defaults follow the design contract in
the README. Group weights live in small nested dataclasses so new scores can be
added without reshaping the top-level object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kaggle_pipeline.preprocessing.encoders import ONEHOT_MAX_CARDINALITY


@dataclass
class FeatureScoringWeights:
    """Weights combining intrinsic feature scores into ``feature_intrinsic_score``.

    All weights are positive; the *sign* with which each score enters the utility
    is fixed by the formula (redundancy/complexity/cost subtract). See
    :mod:`kaggle_pipeline.evolution.features.scoring`.
    """

    target_correlation: float = 1.0
    tree_importance: float = 0.75
    redundancy: float = 1.0
    complexity: float = 0.25
    cost: float = 0.10


@dataclass
class DownstreamWeights:
    """Weights combining downstream model credit into ``feature_downstream_score``."""

    add_mutation_delta: float = 1.0
    remove_mutation_delta: float = 1.0
    model_usage_credit: float = 0.5
    elite_usage_rate: float = 0.5


@dataclass
class EvolutionSettings:
    """All tunable parameters for an evolutionary run.

    Grouped by concern: feature pool, feature generation, feature scoring/selection,
    feature deletion, model mutation, model scoring/utility, credit assignment,
    controller policy and reproducibility.
    """

    # --- Feature pool -------------------------------------------------------
    # Cap on *logical* active features (originals excepted -- see the registry).
    max_active_features: int = 300
    # Planned cap on *physical* materialized columns (one logical feature can
    # expand to many columns, e.g. one-hot). Not yet enforced; reserved.
    max_materialized_columns: int | None = None
    # Extra utility granted to original features so they are not crowded out
    # early before downstream evidence accumulates.
    original_feature_bonus: float = 0.25

    # --- Feature generation -------------------------------------------------
    # Candidates generated per batch = ceil(max_active_features * ratio).
    feature_generation_ratio: float = 0.10
    # Depth/parentage rules. Generated features may compose on top of other
    # generated features up to ``max_feature_depth``; deeper features are not
    # forbidden but pay a cost -- their added depth/complexity feeds the complexity
    # penalty in the feature-utility formula, so they must earn their keep.
    max_feature_depth: int = 2
    allow_generated_feature_parents: bool = True
    # A categorical with more distinct levels than this is never one-hot encoded
    # (it falls back to frequency encoding) so a single feature cannot explode into
    # a huge materialized width. Shares the v1 default so the two stay in lockstep.
    onehot_max_cardinality: int = ONEHOT_MAX_CARDINALITY

    # --- Feature scoring & selection ---------------------------------------
    feature_scoring_weights: FeatureScoringWeights = field(default_factory=FeatureScoringWeights)
    downstream_weights: DownstreamWeights = field(default_factory=DownstreamWeights)
    # Softmax temperature for turning utility into selection probability.
    feature_selection_temperature: float = 1.0
    # Floor of uniform exploration mixed into the selection probability.
    feature_selection_exploration_rate: float = 0.05
    # Rank-/robust-normalize scores before combining them.
    use_rank_normalization_for_scores: bool = True
    # Confidence pivot k in beta = n_obs / (n_obs + k): how many downstream
    # observations before downstream credit outweighs the intrinsic score.
    downstream_confidence_k: float = 10.0

    # --- Feature deletion ---------------------------------------------------
    # A generated feature is protected from eviction for this many batches after
    # creation, so a fresh feature is not deleted before it can be used.
    feature_deletion_cooldown_batches: int = 3

    # --- Model mutation -----------------------------------------------------
    # Per-gene mutation probability is stored but is NOT the primary mechanism:
    # prefer the count distribution below, which mutates a small number of genes.
    model_gene_mutation_probability: float = 0.05
    # P(number of genes mutated in a child). Keys are gene counts.
    preferred_num_mutated_genes_distribution: dict[int, float] = field(
        default_factory=lambda: {1: 0.70, 2: 0.20, 3: 0.07, 4: 0.03}
    )
    # signed_amount ~ Uniform(drift - scale, drift + scale).
    mutation_scale: float = 0.20
    mutation_drift: float = 0.00

    # --- Model scoring & utility -------------------------------------------
    # adj_score = score - score_std_penalty * score_std.
    score_std_penalty: float = 1.75
    # Divide the utility numerator by 1 + log(1 + time / t_ref) when enabled.
    compute_penalty_enabled: bool = True

    # --- Controller policy --------------------------------------------------
    # Parent selection: sample this many candidates, keep the best by utility.
    tournament_size: int = 5
    # Exploration floors -- neither action's probability drops below these, so the
    # search never permanently stops generating or mutating.
    p_generate_new_model_floor: float = 0.10
    p_mutate_existing_model_floor: float = 0.10

    # --- Reproducibility ----------------------------------------------------
    # Master seed for every random process in the evolutionary layer. ``None``
    # leaves it unseeded (non-reproducible), matching v1's Config.seed default.
    default_random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_active_features < 1:
            raise ValueError(f"max_active_features must be >= 1, got {self.max_active_features}.")
        if not 0.0 < self.feature_generation_ratio <= 1.0:
            raise ValueError(
                f"feature_generation_ratio must be in (0, 1], got {self.feature_generation_ratio}."
            )
        if self.max_feature_depth < 1:
            raise ValueError(f"max_feature_depth must be >= 1, got {self.max_feature_depth}.")
        if not 0.0 <= self.feature_selection_exploration_rate <= 1.0:
            raise ValueError(
                "feature_selection_exploration_rate must be in [0, 1], "
                f"got {self.feature_selection_exploration_rate}."
            )
        if self.feature_selection_temperature <= 0.0:
            raise ValueError(
                f"feature_selection_temperature must be > 0, "
                f"got {self.feature_selection_temperature}."
            )
        dist = self.preferred_num_mutated_genes_distribution
        if not dist or any(p < 0 for p in dist.values()):
            raise ValueError(
                "preferred_num_mutated_genes_distribution must hold non-negative probs."
            )
        total = sum(dist.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"preferred_num_mutated_genes_distribution must sum to 1.0, got {total}."
            )
        if self.tournament_size < 1:
            raise ValueError(f"tournament_size must be >= 1, got {self.tournament_size}.")

    @property
    def num_feature_candidates_per_batch(self) -> int:
        """How many feature candidates to propose each batch (>= 1)."""
        from math import ceil

        return max(1, ceil(self.max_active_features * self.feature_generation_ratio))

    def effective_max_active_features(self, n_original: int) -> int:
        """Active limit, raised to the original count when there are more originals.

        Original features are protected from deletion, so the active pool can
        never be smaller than the number of originals.
        """
        return max(self.max_active_features, n_original)
