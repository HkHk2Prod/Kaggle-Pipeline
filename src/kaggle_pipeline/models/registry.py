"""The model registry and the ``@register_model`` decorator.

Models register themselves at import time. The registry both maps a name to its
class and groups model names by *purpose* (the kind of task they suit), with the
``lower``/``upper`` bounds on how many of each may sit on the leaderboard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaggle_pipeline.context import PipelineContext

# Purpose for binary/multiclass single-column probability prediction.
SINGLE_TARGET_PROB_PRED = "single_target_prob_pred"


class ModelRegistry:
    """Maps model names to classes and lists them by purpose."""

    def __init__(self) -> None:
        self._models: dict[str, type] = {}
        self._model_lists: dict[str, list[tuple[str, int, int]]] = {}

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
    ) -> list[tuple[str, int, int]]:
        """Return ``(name, lower, upper)`` triples for the task's purpose."""
        purpose = purpose or self.get_purpose(ctx)
        return self._model_lists[purpose]

    def __getitem__(self, name: str) -> type:
        if name not in self._models:
            raise ValueError(f"Model {name} not found in registry.")
        return self._models[name]


# Module-level singleton populated by the @register_model decorators.
registry = ModelRegistry()


def register_model(name: str, purposes, lower: int = 5, upper: int = 30):
    """Register a :class:`~kaggle_pipeline.models.base.Model` subclass.

    A single model may suit several tasks, so ``purposes`` may be a list or a
    single string. ``lower``/``upper`` bound how many instances of this model
    type the leaderboard keeps.

    Current purposes:
        ``'single_target_prob_pred'`` -- single-column probability prediction.
    """

    def decorator(cls):
        registry.add(name=name, model_class=cls, purposes=purposes, lower=lower, upper=upper)
        cls._model_name = name
        return cls

    return decorator
