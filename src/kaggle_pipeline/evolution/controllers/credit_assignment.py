"""Credit assignment: from a trained child back to its genes and features.

Two signals, per the README:

* **Gene credit** -- after a child trains, each *mutated* gene's stats are updated
  by ``delta_utility * max(eps, behavior_delta)`` (falling back to ``|delta_score|``
  when OOF behaviour is unavailable), with a separate compute-credit of
  ``-delta_time``. Counts/means/variances are kept so one lucky mutation cannot
  dominate.
* **Feature credit** -- the cleanest signal is add/remove mutation credit (a child
  that *added* feature D and improved gives D positive credit; one where *removing*
  D improved gives D remove-credit). Noisier model-usage credit (optionally
  importance-weighted) is also distributed, and elite usage is recorded.

Intrinsic vs. downstream feature scores are blended with confidence weighting in
the feature utility combiner; this module only feeds it the downstream evidence.
"""

from __future__ import annotations

from kaggle_pipeline.evolution.config import EvolutionSettings
from kaggle_pipeline.evolution.evaluation.oof_store import OOFStore
from kaggle_pipeline.evolution.features.registry import FeatureRegistry
from kaggle_pipeline.evolution.genes.base import Gene
from kaggle_pipeline.evolution.models.genome import ModelGenome
from kaggle_pipeline.evolution.models.mutation import MutationRecord
from kaggle_pipeline.evolution.utils.logging import get_logger

logger = get_logger(__name__)

_EPS = 1e-3


class CreditAssigner:
    """Distributes a trained model's outcome to its genes and features."""

    def __init__(
        self,
        registry: FeatureRegistry,
        settings: EvolutionSettings,
        *,
        oof_store: OOFStore | None = None,
    ):
        self.registry = registry
        self.settings = settings
        self.oof_store = oof_store

    # --- selection ----------------------------------------------------------
    def assign_selection(self, genome: ModelGenome) -> None:
        """Record that a model selected each of its features (a weak signal)."""
        for fid in genome.feature_ids():
            try:
                self.registry.get_feature(fid).usage_stats.record_selected()
            except KeyError:
                continue

    # --- gene credit --------------------------------------------------------
    def assign_gene_credit(self, record: MutationRecord, child: ModelGenome) -> None:
        """Update the mutated genes' stats from the child's measured outcome."""
        delta_utility = record.delta_utility or 0.0
        behavior = record.behavior_delta
        if behavior is None:
            behavior = abs(record.delta_score) if record.delta_score is not None else 1.0
        credit = delta_utility * max(_EPS, behavior)
        compute_credit = -(record.delta_compute_time or 0.0)

        genes_by_id = {g.gene_id: g for g in self._all_genes(child)}
        for gid, amount in zip(record.mutated_gene_ids, record.signed_amounts, strict=False):
            gene = genes_by_id.get(gid)
            if gene is not None:
                gene.mutation_stats.record(
                    amount, credit, behavior_delta=behavior, compute_delta=compute_credit
                )

    # --- feature credit -----------------------------------------------------
    def assign_feature_mutation_credit(self, record: MutationRecord) -> None:
        """Give add/remove credit to the features a mutation added or removed."""
        delta_utility = record.delta_utility or 0.0
        for fid in record.added_feature_ids:
            self._with_feature(fid, lambda u: u.record_added(delta_utility))
        for fid in record.removed_feature_ids:
            self._with_feature(fid, lambda u: u.record_removed(delta_utility))

    def assign_usage_credit(
        self,
        genome: ModelGenome,
        *,
        is_elite: bool = False,
        importances: dict[str, float] | None = None,
    ) -> None:
        """Distribute the model's utility across its selected features.

        Importance-weighted when available, else split evenly (labelled weak/noisy).
        """
        utility = genome.utility or 0.0
        fids = genome.feature_ids()
        if not fids:
            return
        even = 1.0 / len(fids)
        for fid in fids:
            importance = importances.get(fid) if importances else None
            credit = utility * (importance if importance is not None else even)

            def update(u, credit=credit, importance=importance, is_elite=is_elite):
                u.record_completed(credit, family=genome.family, importance=importance)
                if is_elite:
                    u.record_elite()

            self._with_feature(fid, update)

    # --- helpers ------------------------------------------------------------
    def _with_feature(self, feature_id: str, fn) -> None:
        try:
            feature = self.registry.get_feature(feature_id)
        except KeyError:
            return
        fn(feature.usage_stats)

    @staticmethod
    def _all_genes(genome: ModelGenome) -> list[Gene]:
        genes: list[Gene] = [
            genome.base_model_gene,
            *genome.parameter_genes,
            *genome.resource_genes,
        ]
        for fr in genome.feature_reference_genes:
            genes.append(fr)
            genes.extend(fr.children)
        return genes
