"""The model registry and the ``@register_model`` decorator.

Models register themselves at import time. The registry both maps a name to its
class and groups model names by *purpose* (the kind of task they suit), with the
``lower``/``upper`` bounds on how many of each may sit on the leaderboard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaggle_pipeline.context import PipelineContext
    from kaggle_pipeline.models.base import Model

# Purpose for binary/multiclass single-column probability prediction.
SINGLE_TARGET_PROB_PRED = "single_target_prob_pred"


class ModelRegistry:
    """Maps model names to classes and lists them by purpose."""

    def __init__(self) -> None:
        self._models: dict[str, type[Model]] = {}
        self._model_lists: dict[str, list[tuple[str, int | float, int | float]]] = {}

    def add(self, name, model_class, purposes, lower, upper) -> None:
        if not isinstance(purposes, (list, tuple)):
            purposes = [purposes]
        self._models[name] = model_class
        for purpose in purposes:
            self._model_lists.setdefault(purpose, []).append((name, lower, upper))

    def get_purpose(self, ctx: PipelineContext) -> str:
        """Infer the purpose for the task described by ``ctx``."""
        if not ctx.target_is_num and len(ctx.target) == 1:
            return SINGLE_TARGET_PROB_PRED
        raise ValueError("Purpose generator cannot generate a purpose for the task at hand.")

    def get_list(
        self, ctx: PipelineContext, purpose: str | None = None
    ) -> list[tuple[str, int | float, int | float]]:
        """Return ``(name, lower, upper)`` triples for the task's purpose."""
        purpose = purpose or self.get_purpose(ctx)
        return self._model_lists[purpose]

    def __getitem__(self, name: str) -> type[Model]:
        if name not in self._models:
            raise ValueError(f"Model {name} not found in registry.")
        return self._models[name]


# Module-level singleton populated by the @register_model decorators.
registry = ModelRegistry()


def register_model(name: str, purposes, lower: int | float = 0.05, upper: int | float = 0.30):
    """Register a :class:`~kaggle_pipeline.models.base.Model` subclass.

    A single model may suit several tasks, so ``purposes`` may be a list or a
    single string. ``lower``/``upper`` bound how many instances of this model
    type the leaderboard keeps. Each bound is either an ``int`` (an absolute
    count) or a ``float`` (a fraction of the leaderboard size ``num_models``,
    rounded up). The defaults -- 5% and 30% -- are the proportions that the
    original 100-model board used (5 and 30), now expressed as fractions so they
    scale with ``num_models``.

    Current purposes:
        ``'single_target_prob_pred'`` -- single-column probability prediction.
    """

    def decorator(cls):
        registry.add(name=name, model_class=cls, purposes=purposes, lower=lower, upper=upper)
        cls._model_name = name
        return cls

    return decorator
