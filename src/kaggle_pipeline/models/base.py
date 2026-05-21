"""The abstract :class:`Model` and the random-parameter sampler.

A concrete model is the *whole* estimator pipeline (preprocessing + estimator)
for one algorithm. Each subclass declares a hyperparameter *distribution* keyed
by a ``complexity`` knob and knows how to build its pipeline from a sampled
parameter set. The search machinery instantiates these, cross-validates them,
stores out-of-fold predictions and pickles the winners.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from kaggle_pipeline.models.registry import registry

if TYPE_CHECKING:
    from kaggle_pipeline.context import PipelineContext


def sample_parameters(dist: dict[str, Any], rng: np.random.Generator | None = None) -> dict:
    """Draw one concrete parameter set from a distribution spec.

    Each value may be a scipy distribution (anything with ``rvs``), a list/tuple
    to choose uniformly from, or a constant. Sorting the items first makes the
    draw order deterministic given ``rng``. Pass distinct fixed ``rng`` objects
    to different threads for reproducibility; ``None`` means non-reproducible.
    """
    if rng is None:
        rng = np.random.default_rng(None)

    params: dict[str, Any] = {}
    for k, v in sorted(dist.items()):
        if hasattr(v, "rvs"):
            params[k] = v.rvs(random_state=rng)
        elif isinstance(v, (list, tuple)):  # must be False for str
            params[k] = v[rng.integers(len(v))]
        else:
            params[k] = v
    return params


class Model(ABC):
    """Base class for a registered model. Subclasses implement the two hooks."""

    # Attributes persisted by ``save`` (overridable per subclass).
    vars_to_save: list[str] = ["_param", "_oof"]
    # Extra kwargs forwarded to the underlying pipeline's ``fit``.
    _fit_params: dict[str, Any] = {}
    # Set by the @register_model decorator.
    _model_name: str
    # Whether the estimator consumes raw categorical columns natively. When True
    # the model is handed the raw columns and ``Config.categorical_encoding`` is
    # ignored for it (capability wins). When False the categoricals are encoded
    # per that config (default: frequency) before reaching the model.
    handles_categoricals: bool = False
    # For native handlers with a hard cap on categorical cardinality (sklearn's
    # HistGradientBoosting caps at ``max_bins`` = 255): columns above this many
    # levels are encoded instead of passed natively. ``None`` means no cap.
    native_cardinality_cap: int | None = None

    def __init__(
        self,
        ctx: PipelineContext,
        complexity: float | None = None,
        param_dist: dict | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.ctx = ctx
        param_dist = param_dist or self.generate_distribution(complexity)
        self._param = sample_parameters(param_dist, rng=rng)
        self._oof: np.ndarray | None = None
        self._model = self.build_pipeline(self._param)

    @abstractmethod
    def generate_distribution(self, complexity: float | None) -> dict:
        """Return the hyperparameter distribution, scaled by ``complexity``."""

    @abstractmethod
    def build_pipeline(self, param: dict):
        """Build and parametrise the full sklearn pipeline for this model."""

    @property
    def name(self) -> str:
        return type(self)._model_name

    @property
    def params(self) -> dict:
        return self._param

    def fit(self, X, y) -> None:
        self._model.fit(X, y, **type(self)._fit_params)

    def predict(self, X) -> np.ndarray:
        if self.ctx.target_is_num:
            return self._model.predict(X)
        return self._model.predict_proba(X)

    @property
    def oof(self) -> np.ndarray | None:
        return self._oof

    def set_oof(self, oof: np.ndarray) -> None:
        self._oof = oof

    def save(self, filepath: str | Path) -> None:
        name = (self.name,)
        values = tuple(getattr(self, var) for var in self.vars_to_save)
        with open(filepath, "wb") as f:
            pickle.dump(name + values, f)

    @classmethod
    def load(cls, filepath: str | Path, ctx: PipelineContext) -> Model:
        """Reconstruct a saved model and rebuild its pipeline from saved params.

        Note: the notebook rebuilt the pipeline with random (complexity=1.0)
        params and never re-applied the *saved* hyperparameters, so reloaded
        ensemble members were refit with the wrong params. We rebuild the
        pipeline from ``_param`` after loading so the tuned model is actually
        the one that gets refit. (Correctness fix over the original.)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise ValueError("Loading a model with a missing data file.")
        with open(filepath, "rb") as f:
            name, *loaded_vars = pickle.load(f)

        model_cls = registry[name]
        # complexity is irrelevant; the params are overwritten just below.
        model = model_cls(ctx, complexity=1.0)
        for var_name, value in zip(model_cls.vars_to_save, loaded_vars, strict=False):
            setattr(model, var_name, value)
        model._model = model.build_pipeline(model._param)
        return model
