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

import numpy as np

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus
from kaggle_pipeline.evolution.models.mutation import MutationRecord
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
    ):
        self.settings = settings
        self.utility = ModelUtility(settings)
        self.max_active = max_active
        self.elite_size = elite_size
        self._by_id: dict[str, ModelGenome] = {}
        self._by_hash: dict[str, str] = {}
        self.active: list[str] = []
        self.elite: list[str] = []
        self.mutation_records: list[MutationRecord] = []

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
        return genome

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
        completed.sort(
            key=lambda g: g.score_set.adj_score(penalty) if g.score_set else 0.0, reverse=True
        )
        self.elite = [g.model_id for g in completed[: self.elite_size]]

    def prune_active(self) -> None:
        """Cap the active set, never evicting elites; drop the lowest-utility rest."""
        if len(self.active) <= self.max_active:
            return
        elite = set(self.elite)
        ranked = sorted(self.active, key=lambda mid: self._by_id[mid].utility or 0.0, reverse=True)
        keep: list[str] = []
        for mid in ranked:
            if len(keep) < self.max_active or mid in elite:
                keep.append(mid)
            else:
                self._by_id[mid].status = ModelStatus.PRUNED
        self.active = keep

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
            return (g.utility or 0.0) - diversity_penalty

        best_id = max((ids[int(i)] for i in np.atleast_1d(idx)), key=score)
        return self._by_id[best_id]

    # --- rankings -----------------------------------------------------------
    def efficient_search_ranking(self) -> list[ModelGenome]:
        return sorted(self.completed(), key=lambda g: g.utility or 0.0, reverse=True)

    def absolute_score_ranking(self) -> list[ModelGenome]:
        return sorted(
            (g for g in self.completed() if g.score_set),
            key=lambda g: g.score_set.score if g.score_set else 0.0,
            reverse=True,
        )

    def ensemble_candidate_ranking(self) -> list[ModelGenome]:
        penalty = self.settings.score_std_penalty
        return sorted(
            (g for g in self.completed() if g.score_set),
            key=lambda g: g.score_set.adj_score(penalty) if g.score_set else 0.0,
            reverse=True,
        )

    def elite_genomes(self) -> list[ModelGenome]:
        return [self._by_id[mid] for mid in self.elite]

    def all_genomes(self) -> list[ModelGenome]:
        return list(self._by_id.values())

    def status_counts(self) -> dict[str, int]:
        return dict(Counter(g.status for g in self._by_id.values()))
