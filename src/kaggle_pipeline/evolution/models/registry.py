"""The model population: active set, elite archive, rankings and parent selection.

Holds every genome (deduplicated by genome hash), the *active* set eligible for
mutation, and an *elite archive* of the best ever found. Utilities are recomputed
against comparable sets (same fidelity level) so a low-fidelity trial is never
measured against a full-fidelity one. Parent selection is tournament-based with a
small diversity penalty so the search does not collapse onto the single best
model or one model family.

Three rankings are maintained: efficient-search (utility), absolute-score, and
ensemble-candidate (stability-penalised score).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus
from kaggle_pipeline.evolution.models.mutation import MutationRecord
from kaggle_pipeline.evolution.models.parameter_spaces import (
    DEFAULT_MIN_MODELS,
    FamilyDefinition,
    resolve_min_count,
)
from kaggle_pipeline.evolution.models.scoring import ModelUtility, comparable_stats
from kaggle_pipeline.evolution.utils.logging import get_logger
from kaggle_pipeline.evolution.utils.random import spawn_rng

logger = get_logger(__name__)


class ModelPopulation:
    """Stores model genomes and manages the active set, elites and rankings."""

    def __init__(
        self,
        settings: EvolutionSettings,
        *,
        max_active: int = 200,
        elite_size: int = 25,
        families: dict[str, FamilyDefinition] | None = None,
    ):
        self.settings = settings
        self.utility = ModelUtility(settings)
        self.max_active = max_active
        self.elite_size = elite_size
        # Per-family floor specs only (not the full catalog -- ``FamilyDefinition``
        # carries un-picklable estimator/availability lambdas and the population
        # is checkpointed). Families absent here fall back to ``DEFAULT_MIN_MODELS``
        # so every family keeps at least its best model with no catalog wired in.
        self.family_min_models: dict[str, int | float] = {}
        self._by_id: dict[str, ModelGenome] = {}
        self._by_hash: dict[str, str] = {}
        self.active: list[str] = []
        self.elite: list[str] = []
        self.mutation_records: list[MutationRecord] = []
        # Optional store whose big OOF arrays are freed when a model is pruned;
        # wired by the controller. The genome + scores are always kept.
        self.oof_store: Any = None
        # Target vector used to recompute residual-error correlation penalties
        # on demand. Set by the pipeline once the search subsample is built and
        # again on resume; ``None`` short-circuits the correlation recompute to
        # zero (the warning still fires once).
        self._search_y: np.ndarray | None = None
        if families:
            self.set_family_minimums(families)

    def set_family_minimums(self, families: dict[str, FamilyDefinition]) -> None:
        """Cache each family's ``min_models`` spec from the catalog (picklable)."""
        self.family_min_models = {name: fam.min_models for name, fam in families.items()}

    def set_search_target(self, y: np.ndarray | None) -> None:
        """Bind the y vector that recomputers use to derive residual penalties.

        Also cascades into the OOF store so its correlation cache pivots to the
        same target (rebuilding once if the bound ``y`` is new). The two
        signatures of "what y are we comparing residuals against" then can't
        drift out of sync.
        """
        self._search_y = y
        if self.oof_store is not None:
            self.oof_store.set_residual_target(y)

    # --- registration -------------------------------------------------------
    def has_genome_hash(self, genome_hash: str) -> bool:
        return genome_hash in self._by_hash

    def get(self, model_id: str) -> ModelGenome:
        return self._by_id[model_id]

    def get_by_hash(self, genome_hash: str) -> ModelGenome | None:
        model_id = self._by_hash.get(genome_hash)
        return self._by_id.get(model_id) if model_id else None

    def register(self, genome: ModelGenome) -> ModelGenome:
        """Store a genome; if its hash already exists, return the existing one."""
        existing = self.get_by_hash(genome.genome_hash)
        if existing is not None:
            return existing
        self._by_id[genome.model_id] = genome
        self._by_hash[genome.genome_hash] = genome.model_id
        self._wire_score_recomputers(genome)
        return genome

    # --- score-recomputer wiring -------------------------------------------
    # The lazy-recompute mechanism on the genome is generic: each individual
    # score (``correlation_penalty``, ``utility``, future scores) is read via
    # ``genome.get_score(name)``, which calls back into a closure registered
    # here. Adding a new score type therefore means: declare the field on
    # ``ModelGenome``, add a recompute method below, and register it in
    # ``_wire_score_recomputers``. Every existing leaderboard that consumes
    # ``effective_*`` automatically picks the new value up.
    def wire_all_score_recomputers(self) -> None:
        """Re-bind recomputers on every stored genome, used after pickle resume."""
        for g in self._by_id.values():
            self._wire_score_recomputers(g)

    def _wire_score_recomputers(self, g: ModelGenome) -> None:
        def _recompute_correlation() -> None:
            self._recompute_correlation_for(g)

        g.register_score_recomputer("correlation_penalty", _recompute_correlation)
        g.register_score_recomputer("utility", self.update_utilities)

    def _recompute_correlation_for(self, g: ModelGenome) -> None:
        if self._search_y is None or self.oof_store is None:
            # Nothing to compare against; record the absence so we don't keep
            # firing the missing-score warning on every subsequent access.
            g.correlation_penalty = 0.0
            return
        self.compute_correlation_penalties(
            self._search_y,
            threshold=self.settings.correlation_penalty_threshold,
            scale=self.settings.correlation_penalty_scale,
        )

    def add_mutation_record(self, record: MutationRecord) -> None:
        self.mutation_records.append(record)

    # --- results ------------------------------------------------------------
    def record_result(self, genome: ModelGenome) -> None:
        """Note a finished genome; admit completed ones to the active set."""
        if genome.status == ModelStatus.COMPLETED and genome.model_id not in self.active:
            self.active.append(genome.model_id)
        self.update_utilities()
        self.update_elite()
        self.prune_active()

    def completed(self) -> list[ModelGenome]:
        return [g for g in self._by_id.values() if g.status == ModelStatus.COMPLETED]

    def update_utilities(self) -> None:
        """Recompute every completed genome's utility against its fidelity peers."""
        completed = self.completed()
        by_fidelity: dict[int, list[ModelGenome]] = {}
        for g in completed:
            by_fidelity.setdefault(g.fidelity_level, []).append(g)
        std_penalty = self.settings.score_std_penalty
        for genomes in by_fidelity.values():
            stats = comparable_stats(
                [g.score_set for g in genomes if g.score_set], std_penalty=std_penalty
            )
            for g in genomes:
                if g.score_set is not None:
                    g.utility = self.utility.utility(g.score_set, stats)

    # --- elites & pruning ---------------------------------------------------
    def update_elite(self) -> None:
        penalty = self.settings.score_std_penalty
        completed = [g for g in self.completed() if g.score_set is not None]
        completed.sort(key=lambda g: self.effective_adj_score(g, penalty), reverse=True)
        self.elite = [g.model_id for g in completed[: self.elite_size]]
        # Sticky tag: once a genome makes the leaderboard we mark it so the
        # end-of-cycle compute summary can distinguish "evicted later" from
        # "never made it". Older pickles may lack the attribute; ``setattr`` is
        # safe either way.
        for mid in self.elite:
            self._by_id[mid].was_elite = True

    def family_min_count(self, family: str) -> int:
        """Per-family survival floor, resolved against the population cap."""
        # ``getattr`` guards pickles from before the field existed.
        specs = getattr(self, "family_min_models", None) or {}
        spec = specs.get(family, DEFAULT_MIN_MODELS)
        return resolve_min_count(spec, self.max_active)

    def _family_protected(self, ranked: list[str]) -> set[str]:
        """The highest-utility ``min_models`` ids of each family in ``ranked``.

        ``ranked`` is already sorted best-first, so walking it and taking the
        first ``min_count`` ids per family yields each family's strongest
        survivors -- the ones a per-family floor must shield from pruning.
        """
        kept: Counter = Counter()
        protected: set[str] = set()
        for mid in ranked:
            family = self._by_id[mid].family
            if kept[family] < self.family_min_count(family):
                protected.add(mid)
                kept[family] += 1
        return protected

    def prune_active(self) -> None:
        """Cap the active set, never evicting elites or a family's floor.

        Lowest-utility models are dropped first, but elites and each family's
        top ``min_models`` are protected -- so a whole family is never culled to
        extinction purely on utility. Protected members can push the kept set
        past ``max_active``, exactly as elites already do.
        """
        if len(self.active) <= self.max_active:
            return
        ranked = sorted(
            self.active,
            key=lambda mid: self.effective_utility(self._by_id[mid]),
            reverse=True,
        )
        protected = set(self.elite) | self._family_protected(ranked)
        keep: list[str] = []
        for mid in ranked:
            if len(keep) < self.max_active or mid in protected:
                keep.append(mid)
            else:
                self._by_id[mid].status = ModelStatus.PRUNED
                # Free the pruned model's big data (OOF); keep genome + scores.
                if self.oof_store is not None:
                    self.oof_store.remove(mid)
        self.active = keep

    # --- correlation penalty ------------------------------------------------
    def compute_correlation_penalties(
        self,
        y: np.ndarray,
        *,
        threshold: float,
        scale: float,
    ) -> int:
        """Set each active model's ``correlation_penalty`` from cached residual correlations.

        For every active model, look up its largest ``|Olkin-Pratt r|`` against
        any active model with a *strictly higher* raw ``score_set.score`` and
        apply ``penalty = scale * max(0, max_r - threshold)``. The top scorer
        is never penalised (no higher peers). Reads from the OOF store's
        incremental correlation cache, so no standardization or dot products
        run here -- the matrix was built once as each OOF was stored.

        Returns the count of models with a non-zero penalty.
        """
        # Reset every active model's penalty before recomputation so a stale
        # value from a previous batch never lingers on a now-uncontested model.
        for mid in self.active:
            self._by_id[mid].correlation_penalty = 0.0

        if self.oof_store is None or scale <= 0.0 or len(self.active) < 2:
            return 0

        # Make sure the cache is bound to this ``y``. If the bound target is
        # already this exact array, ``set_target`` is a no-op; otherwise the
        # cache rebuilds from the stored OOFs.
        self.oof_store.set_residual_target(y)
        cache = self.oof_store.correlation_cache

        actives = [self._by_id[m] for m in self.active if self._by_id[m].score_set is not None]
        actives.sort(key=lambda g: g.score_set.score if g.score_set else 0.0, reverse=True)

        affected = 0
        for i, g in enumerate(actives):
            if g.score_set is None or not cache.has(g.model_id):
                continue
            max_r = -np.inf
            self_score = g.score_set.score
            for j in range(i):
                other = actives[j]
                if other.score_set is None or other.score_set.score <= self_score:
                    # ``actives`` is sorted, so once a peer is not strictly
                    # higher, no later peer will be either.
                    break
                r = cache.correlation(g.model_id, other.model_id)
                if r is None:
                    continue
                if r > max_r:
                    max_r = r
            if not np.isfinite(max_r):
                continue
            penalty = scale * max(0.0, max_r - threshold)
            g.correlation_penalty = penalty
            if penalty > 0.0:
                affected += 1
        return affected

    # --- parent selection ---------------------------------------------------
    def family_counts(self) -> Counter:
        return Counter(self._by_id[mid].family for mid in self.active)

    def select_parent(
        self, rng: np.random.Generator | None = None, *, k: int | None = None
    ) -> ModelGenome | None:
        """Tournament selection: best of ``k`` sampled actives, minus a diversity penalty."""
        if not self.active:
            return None
        rng = spawn_rng(rng)
        k = k or self.settings.tournament_size
        ids = list(self.active)
        idx = rng.choice(len(ids), size=min(k, len(ids)), replace=False)
        counts = self.family_counts()

        def score(model_id: str) -> float:
            g = self._by_id[model_id]
            diversity_penalty = 0.02 * counts.get(g.family, 0)
            return self.effective_utility(g) - diversity_penalty

        best_id = max((ids[int(i)] for i in np.atleast_1d(idx)), key=score)
        return self._by_id[best_id]

    # --- rankings -----------------------------------------------------------
    def efficient_search_ranking(self) -> list[ModelGenome]:
        return sorted(self.completed(), key=self.effective_utility, reverse=True)

    def absolute_score_ranking(self) -> list[ModelGenome]:
        return sorted(
            (g for g in self.completed() if g.score_set),
            key=self.effective_raw_score,
            reverse=True,
        )

    def ensemble_candidate_ranking(self) -> list[ModelGenome]:
        penalty = self.settings.score_std_penalty
        return sorted(
            (g for g in self.completed() if g.score_set),
            key=lambda g: self.effective_adj_score(g, penalty),
            reverse=True,
        )

    # --- combined scores ----------------------------------------------------
    # The "final" leaderboard score is never stored. Every selection path goes
    # through these functions, which stack individual scores read via the
    # genome's ``get_score`` (lazy-recomputed on miss with a warning). Adding a
    # new score: declare a field on ``ModelGenome``, wire its recomputer in
    # ``_wire_score_recomputers``, then include it in the formulas below.
    def effective_raw_score(self, g: ModelGenome) -> float:
        base = g.score_set.score if g.score_set else 0.0
        return base - _coerce_float(g.get_score("correlation_penalty"))

    def effective_adj_score(self, g: ModelGenome, std_penalty: float) -> float:
        base = g.score_set.adj_score(std_penalty) if g.score_set else 0.0
        return base - _coerce_float(g.get_score("correlation_penalty"))

    def effective_utility(self, g: ModelGenome) -> float:
        utility = _coerce_float(g.get_score("utility"))
        penalty = _coerce_float(g.get_score("correlation_penalty"))
        return utility - penalty

    def elite_genomes(self) -> list[ModelGenome]:
        return [self._by_id[mid] for mid in self.elite]

    def all_genomes(self) -> list[ModelGenome]:
        return list(self._by_id.values())

    def status_counts(self) -> dict[str, int]:
        return dict(Counter(g.status for g in self._by_id.values()))


def _coerce_float(value: Any) -> float:
    """``get_score`` may return ``None`` if no recomputer is registered; treat as 0."""
    return float(value) if value is not None else 0.0
