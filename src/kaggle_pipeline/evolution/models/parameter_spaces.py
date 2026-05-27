"""Per-family parameter spaces, estimator builders and availability checks.

Each model family is a :class:`FamilyDefinition`: its mutable behaviour parameters
(as :class:`ParameterSpec`s with bounds, log-scale and complexity direction), the
number of estimators per fidelity level (a *resource*, not a behaviour gene), a
``build_estimator`` callable, and an availability probe. Optional libraries
(LightGBM/XGBoost/CatBoost) are imported lazily and degrade gracefully -- only
*available* families are offered to the factory.

Parameter names match the estimators' kwargs so a sampled value passes straight
through. New families are added by registering another ``FamilyDefinition``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from kaggle_pipeline.evolution.genes.parameter_gene import (
    CATEGORICAL,
    FLOAT,
    INT,
    NEGATIVE,
    POSITIVE,
    ParameterSpec,
)

# Default estimator counts per fidelity level (a resource gene's values).
DEFAULT_FIDELITY_N_ESTIMATORS = {1: 120, 2: 300, 3: 600, 4: 900}


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@dataclass
class FamilyDefinition:
    """Everything the factory/trainer need to handle one model family."""

    name: str
    handles_categoricals: bool
    needs_scaling: bool
    parameter_specs: list[ParameterSpec]
    build_estimator: Callable[..., Any]
    available: Callable[[], bool] = lambda: True
    fidelity_n_estimators: dict[int, int] = field(
        default_factory=lambda: dict(DEFAULT_FIDELITY_N_ESTIMATORS)
    )

    def n_estimators_for(self, fidelity_level: int) -> int:
        levels = self.fidelity_n_estimators
        return levels.get(fidelity_level, max(levels.values()))

    def is_available(self) -> bool:
        return self.available()


# --- estimator builders ------------------------------------------------------


def _build_logistic(params: dict, n_estimators: int, random_state: int | None) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=params.get("C", 1.0),
        max_iter=int(params.get("max_iter", 400)),
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )


def _build_random_forest(params: dict, n_estimators: int, random_state: int | None) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params.get("max_features", "sqrt"),
        n_jobs=1,
        class_weight="balanced_subsample",
        random_state=random_state,
    )


def _build_extra_trees(params: dict, n_estimators: int, random_state: int | None) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params.get("max_features", "sqrt"),
        n_jobs=1,
        class_weight="balanced_subsample",
        random_state=random_state,
    )


def _build_hist_gb(params: dict, n_estimators: int, random_state: int | None) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        learning_rate=params.get("learning_rate", 0.1),
        max_iter=n_estimators,
        max_depth=int(params["max_depth"]),
        l2_regularization=params.get("l2_regularization", 0.0),
        max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
        random_state=random_state,
    )


def _build_lightgbm(params: dict, n_estimators: int, random_state: int | None) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=int(params.get("num_leaves", 31)),
        max_depth=int(params.get("max_depth", -1)),
        min_child_samples=int(params.get("min_child_samples", 20)),
        reg_lambda=params.get("reg_lambda", 0.0),
        subsample=params.get("subsample", 1.0),
        subsample_freq=1,
        colsample_bytree=params.get("colsample_bytree", 1.0),
        class_weight="balanced",
        n_jobs=1,
        verbose=-1,
        random_state=random_state,
    )


def _build_xgboost(params: dict, n_estimators: int, random_state: int | None) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=int(params.get("max_depth", 6)),
        min_child_weight=params.get("min_child_weight", 1.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        subsample=params.get("subsample", 1.0),
        colsample_bytree=params.get("colsample_bytree", 1.0),
        n_jobs=1,
        verbosity=0,
        random_state=random_state,
    )


def _build_catboost(params: dict, n_estimators: int, random_state: int | None) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=n_estimators,
        learning_rate=params.get("learning_rate", 0.05),
        depth=int(params.get("depth", 6)),
        l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
        verbose=False,
        allow_writing_files=False,
        random_seed=random_state,
    )


# --- family definitions ------------------------------------------------------


def _spec(name: str, **kw: Any) -> ParameterSpec:
    return ParameterSpec(name=name, **kw)


def build_default_families() -> dict[str, FamilyDefinition]:
    """Return all *available* families keyed by name."""
    definitions = [
        FamilyDefinition(
            name="logistic",
            handles_categoricals=False,
            needs_scaling=True,
            parameter_specs=[
                _spec(
                    "C",
                    kind=FLOAT,
                    low=1e-3,
                    high=100.0,
                    log_scale=True,
                    complexity_direction=POSITIVE,
                    model_family="logistic",
                ),
            ],
            build_estimator=_build_logistic,
        ),
        FamilyDefinition(
            name="random_forest",
            handles_categoricals=False,
            needs_scaling=False,
            parameter_specs=[
                _spec("max_depth", kind=INT, low=3, high=24, complexity_direction=POSITIVE),
                _spec("min_samples_leaf", kind=INT, low=1, high=40, complexity_direction=NEGATIVE),
                _spec("max_features", kind=CATEGORICAL, choices=("sqrt", "log2")),
            ],
            build_estimator=_build_random_forest,
        ),
        FamilyDefinition(
            name="extra_trees",
            handles_categoricals=False,
            needs_scaling=False,
            parameter_specs=[
                _spec("max_depth", kind=INT, low=3, high=24, complexity_direction=POSITIVE),
                _spec("min_samples_leaf", kind=INT, low=1, high=40, complexity_direction=NEGATIVE),
                _spec("max_features", kind=CATEGORICAL, choices=("sqrt", "log2")),
            ],
            build_estimator=_build_extra_trees,
        ),
        FamilyDefinition(
            name="hist_gb",
            handles_categoricals=True,
            needs_scaling=False,
            parameter_specs=[
                _spec("learning_rate", kind=FLOAT, low=0.01, high=0.3, log_scale=True),
                _spec("max_depth", kind=INT, low=2, high=16, complexity_direction=POSITIVE),
                _spec(
                    "max_leaf_nodes",
                    kind=INT,
                    low=15,
                    high=255,
                    log_scale=True,
                    complexity_direction=POSITIVE,
                ),
                _spec(
                    "l2_regularization",
                    kind=FLOAT,
                    low=0.0,
                    high=10.0,
                    complexity_direction=NEGATIVE,
                ),
            ],
            build_estimator=_build_hist_gb,
        ),
        FamilyDefinition(
            name="lightgbm",
            handles_categoricals=True,
            needs_scaling=False,
            parameter_specs=[
                _spec("learning_rate", kind=FLOAT, low=0.01, high=0.3, log_scale=True),
                _spec(
                    "num_leaves",
                    kind=INT,
                    low=8,
                    high=256,
                    log_scale=True,
                    complexity_direction=POSITIVE,
                ),
                _spec("max_depth", kind=INT, low=3, high=16, complexity_direction=POSITIVE),
                _spec(
                    "min_child_samples", kind=INT, low=5, high=200, complexity_direction=NEGATIVE
                ),
                _spec("reg_lambda", kind=FLOAT, low=0.0, high=50.0, complexity_direction=NEGATIVE),
                _spec("subsample", kind=FLOAT, low=0.5, high=1.0, complexity_direction=POSITIVE),
                _spec(
                    "colsample_bytree", kind=FLOAT, low=0.4, high=1.0, complexity_direction=POSITIVE
                ),
            ],
            build_estimator=_build_lightgbm,
            available=lambda: _have("lightgbm"),
        ),
        FamilyDefinition(
            name="xgboost",
            handles_categoricals=False,
            needs_scaling=False,
            parameter_specs=[
                _spec("learning_rate", kind=FLOAT, low=0.01, high=0.3, log_scale=True),
                _spec("max_depth", kind=INT, low=3, high=12, complexity_direction=POSITIVE),
                _spec(
                    "min_child_weight",
                    kind=FLOAT,
                    low=0.5,
                    high=20.0,
                    complexity_direction=NEGATIVE,
                ),
                _spec("reg_lambda", kind=FLOAT, low=0.1, high=50.0, complexity_direction=NEGATIVE),
                _spec("subsample", kind=FLOAT, low=0.5, high=1.0, complexity_direction=POSITIVE),
                _spec(
                    "colsample_bytree", kind=FLOAT, low=0.4, high=1.0, complexity_direction=POSITIVE
                ),
            ],
            build_estimator=_build_xgboost,
            available=lambda: _have("xgboost"),
        ),
        FamilyDefinition(
            name="catboost",
            handles_categoricals=True,
            needs_scaling=False,
            parameter_specs=[
                _spec("learning_rate", kind=FLOAT, low=0.01, high=0.3, log_scale=True),
                _spec("depth", kind=INT, low=3, high=10, complexity_direction=POSITIVE),
                _spec("l2_leaf_reg", kind=FLOAT, low=1.0, high=30.0, complexity_direction=NEGATIVE),
            ],
            build_estimator=_build_catboost,
            available=lambda: _have("catboost"),
        ),
    ]
    return {d.name: d for d in definitions if d.is_available()}
