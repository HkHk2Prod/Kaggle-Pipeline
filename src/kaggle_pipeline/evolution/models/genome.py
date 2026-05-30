"""The :class:`ModelGenome` -- a model defined as a set of dependent genes.

Immutable once created/trained: mutation produces a *child* genome (the mutator
clones the genes, mutates a few, and builds a new genome). A genome references
global features by id via :class:`FeatureReferenceGene`s and owns model-local
encoding/parameter/resource genes; the :class:`BaseModelGene` (model family) is
immutable within the genome -- changing family is a new genome, not a mutation.

The :attr:`genome_hash` covers the base model, the *set* of feature ids + their
encodings, the parameter values, and the resource/fidelity settings (order
independent), plus any validation-scheme / target-transform tags in
``metadata``. Identical genomes hash identically, which is what lets the trainer
skip retraining duplicates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kaggle_pipeline.evolution.genes.base import BaseModelGene, Gene
from kaggle_pipeline.evolution.genes.feature_reference_gene import FeatureReferenceGene
from kaggle_pipeline.evolution.genes.parameter_gene import ParameterGene
from kaggle_pipeline.evolution.genes.resource_gene import ResourceGene
from kaggle_pipeline.evolution.models.lifecycle import ModelStatus
from kaggle_pipeline.evolution.models.scoring import ModelScoreSet
from kaggle_pipeline.evolution.storage.hashing import short_hash, stable_hash
from kaggle_pipeline.evolution.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelGenome:
    """A model genome: base model + feature references + parameters + resources."""

    base_model_gene: BaseModelGene
    feature_reference_genes: list[FeatureReferenceGene] = field(default_factory=list)
    parameter_genes: list[ParameterGene] = field(default_factory=list)
    resource_genes: list[ResourceGene] = field(default_factory=list)
    parent_model_id: str | None = None
    created_at_batch: int = 0
    fidelity_level: int = 1
    mutation_history: list[str] = field(default_factory=list)
    status: str = ModelStatus.CREATED
    metadata: dict[str, Any] = field(default_factory=dict)
    # Individual scores. Each one is read via :meth:`get_score` so that
    # ``None`` (or a wholly missing attribute on a pickled-from-older-version
    # genome) triggers a lazy recompute via a recomputer that the surrounding
    # ``ModelPopulation`` registers at admission time. The combined leaderboard
    # score is never stored -- it's always derived in
    # ``ModelPopulation.effective_*`` from these individual fields.
    score_set: ModelScoreSet | None = None
    utility: float | None = None
    correlation_penalty: float | None = None
    # Sticky flag: ``True`` from the first time this genome appears in the
    # elite list and never reset, so the end-of-cycle compute-waste summary
    # can tell "made it onto the leaderboard but got evicted" apart from
    # "never made it at all". Set by ``ModelPopulation.update_elite``.
    was_elite: bool = False
    # Derived; set in __post_init__.
    genome_hash: str = field(default="", compare=False)
    model_id: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        self.genome_hash = self.compute_hash()
        if not self.model_id:
            self.model_id = f"m_{short_hash(self.genome_hash, 16)}"
        # Per-instance score plumbing. Not part of identity / hash. Excluded
        # from pickle (see ``__getstate__``) because the recompute callbacks
        # close over the live ``ModelPopulation``; on resume the population
        # rewires them via ``wire_all_score_recomputers``.
        self._score_recomputers: dict[str, Callable[[], None]] = {}
        self._warned_score_names: set[str] = set()

    # --- hashing ------------------------------------------------------------
    def compute_hash(self) -> str:
        """Order-independent hash over the genome's identity-defining genes."""
        payload = {
            "base_model": self.base_model_gene.hash_component(),
            "features": sorted(
                (g.hash_component() for g in self.feature_reference_genes),
                key=lambda c: c.get("feature_id", ""),
            ),
            "parameters": sorted(
                (g.hash_component() for g in self.parameter_genes),
                key=lambda c: c.get("name", ""),
            ),
            "resources": sorted(
                (g.hash_component() for g in self.resource_genes),
                key=lambda c: c.get("resource_name", ""),
            ),
            "fidelity_level": self.fidelity_level,
            "validation_scheme": self.metadata.get("validation_scheme"),
            "target_transform": self.metadata.get("target_transform"),
            "config_version": self.metadata.get("config_version"),
        }
        return stable_hash(payload)

    # --- individual scores (lazy recompute on miss) -------------------------
    # Each individual score (``utility``, ``correlation_penalty``, future ones)
    # is read through :meth:`get_score`. When the stored value is ``None`` or
    # the attribute is wholly missing (an older pickle predating the field),
    # we warn once per genome-name pair and run the registered recomputer; the
    # recomputer is expected to set the attribute as a side effect. Combined
    # scores (the leaderboard, ensemble-candidate ranking) are never stored:
    # they live in ``ModelPopulation.effective_*`` and stack individual scores
    # on the fly, so a new score type only has to wire its recomputer to be
    # consumed by all existing leaderboards.
    def register_score_recomputer(self, name: str, recompute: Callable[[], None]) -> None:
        """Tell this genome how to recompute the individual score ``name``."""
        recs = self._score_recomputer_map()
        recs[name] = recompute

    def get_score(self, name: str) -> Any:
        """Return individual score ``name``; on miss, warn once and recompute.

        ``None`` or a missing attribute counts as miss. The recomputer is
        invoked and is expected to populate the attribute; we return whatever
        it set (or ``None`` if it produced nothing).
        """
        value = getattr(self, name, None)
        if value is not None:
            return value
        warned = self._warned_score_names_set()
        if name not in warned:
            logger.warning(
                "score '%s' missing for model %s; recomputing on demand",
                name,
                self.model_id,
            )
            warned.add(name)
        recompute = self._score_recomputer_map().get(name)
        if recompute is None:
            return None
        recompute()
        return getattr(self, name, None)

    def _score_recomputer_map(self) -> dict[str, Callable[[], None]]:
        # Lazy-init handles pickles from before the attribute existed.
        recs = getattr(self, "_score_recomputers", None)
        if recs is None:
            self._score_recomputers = {}
            recs = self._score_recomputers
        return recs

    def _warned_score_names_set(self) -> set[str]:
        warned = getattr(self, "_warned_score_names", None)
        if warned is None:
            self._warned_score_names = set()
            warned = self._warned_score_names
        return warned

    def __getstate__(self) -> dict[str, Any]:
        # Drop the live recomputer closures / warn-set from the pickle; the
        # population re-wires recomputers via ``wire_all_score_recomputers``.
        state = self.__dict__.copy()
        state.pop("_score_recomputers", None)
        state.pop("_warned_score_names", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._score_recomputers = {}
        self._warned_score_names = set()

    # --- views --------------------------------------------------------------
    @property
    def family(self) -> str:
        return self.base_model_gene.family

    def feature_ids(self) -> list[str]:
        return [g.feature_id for g in self.feature_reference_genes]

    def all_genes(self) -> list[Gene]:
        genes: list[Gene] = [self.base_model_gene]
        genes.extend(self.feature_reference_genes)
        genes.extend(self.parameter_genes)
        genes.extend(self.resource_genes)
        return genes

    def mutable_genes(self) -> list[Gene]:
        """Genes eligible for ordinary mutation (feature refs, encodings, params).

        Resources and the base model are excluded (resources change by promotion,
        the base model defines the genome). Encoding children of feature references
        are included so encodings can be mutated too.
        """
        out: list[Gene] = []
        for fr in self.feature_reference_genes:
            if fr.mutable:
                out.append(fr)
            enc = fr.encoding
            if enc is not None and enc.mutable:
                out.append(enc)
        out.extend(g for g in self.parameter_genes if g.mutable)
        return out

    def get_parameter(self, name: str) -> ParameterGene | None:
        for g in self.parameter_genes:
            if g.parameter_name == name:
                return g
        return None

    def get_resource(self, name: str) -> ResourceGene | None:
        for g in self.resource_genes:
            if g.resource_name == name:
                return g
        return None

    def gene_summary(self) -> list[str]:
        """A readable list of this genome's *structural* genes, for printing.

        Lists the base model, resource genes, and each feature reference (with its
        encoding) -- i.e. what the model is made of. It deliberately omits the
        hyperparameter (``ParameterGene``) values, which are tuning detail rather
        than structure.
        """

        def fmt(value: Any) -> str:
            return f"{value:.4g}" if isinstance(value, float) else str(value)

        genes = [f"base={self.family}"]
        genes.extend(f"{r.resource_name}={fmt(r.value)}" for r in self.resource_genes)
        for fr in self.feature_reference_genes:
            encoding = fr.encoding.value if fr.encoding is not None else None
            genes.append(f"feat:{fr.feature_id}" + (f"/{encoding}" if encoding else ""))
        return genes

    def validate(self) -> None:
        if not self.feature_reference_genes:
            raise ValueError(f"genome {self.model_id} has no feature references")
        for gene in self.parameter_genes:
            gene.validate()

    def to_serializable(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "genome_hash": self.genome_hash,
            "parent_model_id": self.parent_model_id,
            "family": self.family,
            "status": self.status,
            "fidelity_level": self.fidelity_level,
            "created_at_batch": self.created_at_batch,
            "feature_ids": self.feature_ids(),
            "base_model_gene": self.base_model_gene.to_serializable(),
            "feature_reference_genes": [g.to_serializable() for g in self.feature_reference_genes],
            "parameter_genes": [g.to_serializable() for g in self.parameter_genes],
            "resource_genes": [g.to_serializable() for g in self.resource_genes],
            "mutation_history": list(self.mutation_history),
            "metadata": dict(self.metadata),
            "score_set": self.score_set.to_serializable() if self.score_set else None,
            "utility": self.utility,
        }
