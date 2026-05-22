"""The leaderboard: per-model-class buckets of the best estimators found.

The leaderboard holds, for each model class, a score-sorted list of
:class:`ModelEntry` records (each pointing at a pickled model on disk). It
enforces per-class ``lower``/``upper`` capacity bounds and a global cap, evicts
the weakest evictable entry when full, and adapts each class's ``complexity``
based on score-per-log-compute-time. It also picks which class to try next and
selects the final ensemble members.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from kaggle_pipeline.models.base import Model

if TYPE_CHECKING:
    from kaggle_pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

LEADERBOARD_FILENAME = "LeaderBoard"


@dataclass
class ModelEntry:
    """A scored, on-disk model on the leaderboard."""

    score: float
    name: str
    file_path: str
    compute_time: int

    def load_model(self, ctx: PipelineContext) -> Model:
        return Model.load(self.file_path, ctx)

    def delete_file(self) -> None:
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def __lt__(self, other: ModelEntry) -> bool:
        return self.score < other.score


@dataclass
class ModelClass:
    """A capacity-bounded, score-descending bucket of entries for one model type."""

    lower: int
    upper: int
    entries: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def is_full(self) -> bool:
        return len(self) >= self.upper

    def is_satisfied(self, at_lower_bound_check: bool = False) -> bool:
        # Whether an entry can be evicted without breaking len >= lower.
        if at_lower_bound_check:
            return len(self) > self.lower
        return len(self) >= self.lower

    def pop(self) -> None:
        worst = self.entries.pop(-1)
        worst.delete_file()

    def mean_score(self, top: int = 10):
        if len(self) == 0:
            return None
        score, count = 0, 0
        for entry in self.entries[:top]:
            score += entry.score
            count += 1
        return score / count

    def insert(self, entry: ModelEntry) -> None:
        if self.is_full():
            if entry < self.entries[-1]:
                entry.delete_file()
                return
            self.pop()
        # Binary search insertion to keep the list sorted descending by score.
        lo, hi = 0, len(self.entries)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.entries[mid].score > entry.score:
                lo = mid + 1
            else:
                hi = mid
        self.entries.insert(lo, entry)


class LeaderBoard:
    """Collection of model classes with capacity control and class selection."""

    def __init__(self, num_models: int, storage_dir: Path, seed_seq: np.random.SeedSequence):
        self.classes: dict[str, ModelClass] = {}
        self._complexities: dict[str, float] = {}
        self.num_models = num_models
        # Runtime-coupled; not restored from a loaded board (see ``load``).
        self.storage_dir = storage_dir
        self.seed_seq = seed_seq
        os.makedirs(self.storage_dir, exist_ok=True)

    def add_class(self, name: str, lower: int | float, upper: int | float) -> None:
        self.classes[name] = ModelClass(
            lower=self._resolve_bound(lower), upper=self._resolve_bound(upper)
        )
        self._complexities[name] = 1.0

    def _resolve_bound(self, value: int | float) -> int:
        """Resolve a capacity bound to an absolute count.

        An ``int`` is taken literally; a ``float`` is a fraction of the
        leaderboard size (``num_models``), rounded *up* to the next integer. This
        lets per-class bounds scale with ``num_models`` -- e.g. ``0.05`` means
        "5% of the board" whether the board holds 100 or 300 models.
        """
        if isinstance(value, float):
            return math.ceil(value * self.num_models)
        return int(value)

    def increase_complexity(self, name: str | None = None, val: float = 0.5) -> None:
        names = self._complexities.keys() if name is None else [name]
        for name in names:
            self._complexities[name] += val
            self._complexities[name] = max(self._complexities[name], 1.0)

    def complexity(self, name: str) -> float:
        return self._complexities[name]

    def evaluate_models(self) -> None:
        """Adjust each class's complexity from score-per-log-compute-time.

        Scores are shifted to be positive, divided by ``log1p(compute_time)``,
        then standardised across all entries; each class's mean of that quantity
        (scaled down by 10) becomes its complexity increment. Complexities across
        classes are not directly comparable -- this is a heuristic that nudges
        cheaper, better-scoring classes toward more capacity.
        """
        cls_scores = {
            name: np.array([e.score for e in cl.entries]) for name, cl in self.classes.items()
        }
        times = {
            name: np.array([e.compute_time for e in cl.entries])
            for name, cl in self.classes.items()
        }

        min_score = min(s for scores in cls_scores.values() for s in scores)
        adjusted = {
            name: (cls_scores[name] - min_score) / np.log1p(times[name]) for name in self.classes
        }

        all_adjusted = np.concatenate(list(adjusted.values()))
        mean_as, std_as = all_adjusted.mean(), all_adjusted.std()

        for name, adj in adjusted.items():
            final_score = ((adj - mean_as + 0.25 * std_as) / std_as).mean()
            final_score /= 10
            self.increase_complexity(name=name, val=final_score)

    def __len__(self) -> int:
        return sum(len(c) for c in self.classes.values())

    def _pop(self, new_score: float) -> bool:
        worst_score, candidate = new_score, None
        for _name, cl in self.classes.items():
            if not cl.is_satisfied(at_lower_bound_check=True):
                continue
            worst_in_class = cl.entries[-1]
            if worst_in_class.score < worst_score:
                worst_score, candidate = worst_in_class.score, cl

        if candidate is None:
            return False
        candidate.pop()
        return True

    def add(self, class_name: str, model_entry: ModelEntry) -> None:
        score = model_entry.score
        if len(self) >= self.num_models and not self._pop(score):
            model_entry.delete_file()
            return
        self.classes[class_name].insert(model_entry)

    def generate_model_entry(
        self, model: Model, score: float, compute_time: int, class_name: str
    ) -> tuple[str, ModelEntry]:
        now = datetime.now()
        # A short random suffix keeps names unique even when several models of the
        # same class are saved within the same millisecond (parallel workers),
        # which would otherwise collide and overwrite each other's pickle.
        model_name = class_name + now.strftime("_%Y%m%d_%H%M%S%f")[:-3] + "_" + uuid.uuid4().hex[:8]
        path = self.storage_dir / model_name
        model.save(path)
        return class_name, ModelEntry(
            score=score, name=model_name, file_path=str(path), compute_time=compute_time
        )

    def __str__(self) -> str:
        table = []
        for name, cl in self.classes.items():
            for model in cl.entries:
                table.append((name, model.score))
        table.sort(key=lambda x: -x[1])
        output = f"{'Model':<20} | {'Score':>10}\n"
        output += "-" * 18 + "\n"
        for model, score in table:
            output += f"{model:<30} | {score:>10.4f}" + "\n"
        output += f"Complexities of the models are {self._complexities}\n"
        return output

    def get(self) -> str:
        """Pick the next model class to try.

        Prioritises classes that have not yet reached their lower bound; among
        saturated classes it samples proportionally to a softmax of mean scores.
        The class lookup is shuffled so repeated calls don't always return the
        first unsatisfied class. All randomness draws from a child of the run's
        seed sequence, so with a fixed ``seed`` the search is reproducible.
        """
        models: list[str] = []
        scores: list[float] = []
        rng = np.random.default_rng(self.seed_seq.spawn(1)[0])
        items = list(self.classes.items())
        rng.shuffle(items)
        for name, cl in items:
            if not cl.is_satisfied():
                return name
            models.append(name)
            # ``mean_score`` is None for an empty class; carried through as NaN
            # and handled below.
            scores.append(cl.mean_score())
        prob = np.array(scores, dtype=float)
        finite = np.isfinite(prob)
        # No saturated classes, or none with a usable mean score: pick uniformly.
        if not finite.any():
            return str(rng.choice(list(self.classes.keys())))
        # Treat any missing class mean as the lowest score so it is least likely.
        prob = np.where(finite, prob, prob[finite].min())
        spread = prob.std()
        if spread == 0:  # all means equal -> softmax is uniform anyway.
            return str(rng.choice(models))
        prob = np.exp((prob - prob.mean()) / (spread / 2))  # normalise to std = 2
        prob = prob / prob.sum()
        return str(rng.choice(models, p=prob))

    def get_best(self, length: int = 20, min_repr: int = 0) -> list[tuple[str, str]]:
        """Select ensemble members: a minimum per class, then top scorers."""
        length = min(length, len(self))
        files: set[tuple[str, str]] = set()
        # Per-member listing of the chosen ensemble -- verbose only; the final
        # ensemble's score is reported at the default level in ``Judge.predict``.
        logger.debug("Picked models (minimal requirement):")
        if min_repr:
            for cl in self.classes.values():
                for entry in cl.entries[:min_repr]:
                    files.add((entry.name, entry.file_path))
                    logger.debug("Score: %s. Name: %s", entry.score, entry.name)
        logger.debug("Picked models (best score):")
        table = []
        for _name, cl in self.classes.items():
            for entry in cl.entries:
                table.append((entry.score, (entry.name, entry.file_path)))

        table.sort(key=lambda x: -x[0])
        for score, data in table:
            if len(files) >= length:
                break
            logger.debug("Score: %s. Name: %s", score, data[0])
            files.add(data)
        return list(files)

    def save(self) -> None:
        path = self.storage_dir / LEADERBOARD_FILENAME
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> bool:
        """Restore a saved board, keeping the *current* storage dir and seed.

        Everything else (classes, entries, complexities, num_models) comes from
        the pickled board. ``storage_dir`` and ``seed_seq`` are runtime-coupled
        and must reflect the current environment, so they are preserved.
        """
        path = self.storage_dir / LEADERBOARD_FILENAME
        if not path.exists():
            return False
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, LeaderBoard):
            current_storage_dir, current_seed_seq = self.storage_dir, self.seed_seq
            self.__dict__.update(loaded.__dict__)
            self.storage_dir, self.seed_seq = current_storage_dir, current_seed_seq
            self._rebase_entry_paths()
            return True
        logger.warning(
            "Loaded leaderboard was corrupted. Class was %s instead of %s",
            type(loaded),
            type(self),
        )
        return False

    def _rebase_entry_paths(self) -> None:
        """Point every entry at its model file in the *current* storage dir.

        Each entry pickles an absolute ``file_path`` from the run that saved it
        (a previous Kaggle kernel's ``/kaggle/working/Models``, say). On resume
        the model files are warm-started into *this* run's storage dir, so the
        baked-in path may no longer exist. The on-disk basename always equals the
        entry name (see :meth:`generate_model_entry`), so rebuild each path as
        ``storage_dir / name``. Within a single run this is a no-op; across runs
        it is what lets the ensemble actually reload the resumed models.
        """
        for model_class in self.classes.values():
            for entry in model_class.entries:
                entry.file_path = str(self.storage_dir / entry.name)
