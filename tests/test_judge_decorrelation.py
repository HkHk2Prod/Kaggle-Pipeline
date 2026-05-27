"""Leaderboard-level wiring for correlated-model pruning.

Builds a tiny real context and a handful of real (unfitted) models with
hand-set OOF predictions, then checks ``Judge.prune_correlated_models`` removes
the redundant entries *and* deletes their pickles from disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from kaggle_pipeline import Config, build_pipeline
from kaggle_pipeline.models import registry
from kaggle_pipeline.search import Judge


def _judge(config: Config) -> Judge:
    ctx, _ = build_pipeline(config)
    cv = StratifiedKFold(n_splits=config.cv_splits, shuffle=True, random_state=config.seed)
    return Judge(ctx, cv)


def _add_model(judge: Judge, cls_name: str, oof: np.ndarray, score: float):
    """Save a real model of ``cls_name`` with a given OOF and add it to the board."""
    model = registry[cls_name](judge.ctx, complexity=1.0)
    model.set_oof(oof)
    entry = judge.board.generate_model_entry(
        model=model, score=score, compute_time=1, class_name=cls_name
    )
    judge.board.classes[cls_name].insert(entry)
    return entry


def test_prune_removes_redundant_models_and_their_files(smoke_config):
    judge = _judge(smoke_config)
    cls_name = next(iter(judge.board.classes))
    # Make room for our hand-made entries regardless of the class's default cap.
    judge.board.classes[cls_name].upper = 10

    n = len(judge.y)
    rng = np.random.default_rng(0)
    oof_best = rng.uniform(0.05, 0.95, size=(n, 1))
    e_best = _add_model(judge, cls_name, oof_best, score=0.90)
    e_dupe = _add_model(judge, cls_name, oof_best.copy(), score=0.80)  # same errors, worse
    e_diverse = _add_model(judge, cls_name, rng.uniform(0.05, 0.95, size=(n, 1)), score=0.85)

    pruned = judge.prune_correlated_models()

    assert pruned == 1
    survivors = {e.name for e in judge.board.classes[cls_name].entries}
    assert survivors == {e_best.name, e_diverse.name}  # kept the best + the diverse one
    assert not Path(e_dupe.file_path).exists()  # redundant model's pickle deleted
    assert Path(e_best.file_path).exists()


def test_prune_is_a_noop_when_disabled(smoke_config):
    cfg = Config(**{**smoke_config.__dict__, "prune_correlated_models": False})
    judge = _judge(cfg)
    cls_name = next(iter(judge.board.classes))
    judge.board.classes[cls_name].upper = 10

    n = len(judge.y)
    oof = np.random.default_rng(1).uniform(0.05, 0.95, size=(n, 1))
    _add_model(judge, cls_name, oof, score=0.90)
    _add_model(judge, cls_name, oof.copy(), score=0.80)

    assert judge.prune_correlated_models() == 0
    assert len(judge.board.classes[cls_name].entries) == 2  # nothing removed
