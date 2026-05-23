"""The :class:`Judge` -- orchestrates the model search and final ensembling.

Each ``step`` draws a batch of model classes from the leaderboard, samples and
cross-validates a model from each (in parallel threads), records the scores and
adapts complexities. After the search, ``predict`` stacks the selected models'
out-of-fold predictions with a logistic-regression meta-model and returns the
decoded test predictions.
"""

from __future__ import annotations

import logging
import time
from typing import cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import loguniform

from kaggle_pipeline.context import PipelineContext
from kaggle_pipeline.models import Model, registry
from kaggle_pipeline.search.cv import CrossValScore
from kaggle_pipeline.search.leaderboard import LeaderBoard, ModelEntry

logger = logging.getLogger(__name__)


class Judge:
    """Runs the search loop and builds the stacked ensemble."""

    def __init__(self, ctx: PipelineContext, cv, model_list=None):
        self.ctx = ctx
        self.X = ctx.train_df[ctx.predictor_columns]
        self.X_test = ctx.test_df[ctx.predictor_columns]
        self.y = ctx.target_transforms.forward(ctx.train_df[ctx.target])
        self.splits = list(cv.split(self.X, self.y))

        if model_list is None:
            model_list = registry.get_list(ctx)
        self.board = LeaderBoard(
            num_models=ctx.config.num_models,
            storage_dir=ctx.storage_dir,
            seed_seq=ctx.seed_seq,
        )
        for cls_name, lower, upper in model_list:
            self.board.add_class(cls_name, lower, upper)

    def _evaluate_one(self, cls_name, X, y, splits, rng=None):
        timer = time.perf_counter()
        model_cls = registry[cls_name]
        model_complexity = self.board.complexity(cls_name)
        model = model_cls(self.ctx, rng=rng, complexity=model_complexity)
        cv_results = CrossValScore(model, X, y, splits=splits, ctx=self.ctx)
        score, std = cv_results.score
        timer = time.perf_counter() - timer
        entry = self.board.generate_model_entry(
            model=model, score=score, compute_time=timer, class_name=cls_name
        )
        timer = time.strftime("%H:%M:%S", time.gmtime(timer))
        # Two separate lines so they can be logged at different levels: the
        # name/score/time summary at the normal level, the sampled
        # parameters only at the verbose level.
        summary = (
            f"Tested new model in the class {cls_name}. Score = {score:.4f} ± {std:.4f}. "
            f"It took {timer}."
        )
        params_msg = f"  Parameters are {model.params}"
        return cls_name, entry, summary, params_msg

    def step(self, n_workers: int | None = None) -> float:
        n_workers = self.ctx.config.n_workers if n_workers is None else n_workers
        timer = time.perf_counter()
        batch_size = self.ctx.config.step_batch_size

        cls_names = [self.board.get() for _ in range(batch_size)]
        rngs = [np.random.default_rng(s) for s in self.ctx.seed_seq.spawn(batch_size)]

        # joblib's ``Parallel`` ships no type hints, so the checker can't tell
        # that calling it returns the list of per-model results (it infers None);
        # cast to the known item type produced by ``_evaluate_one``.
        results = cast(
            "list[tuple[str, ModelEntry, str, str]]",
            Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(self._evaluate_one)(cls_name, self.X, self.y, self.splits, rng=rng)
                for cls_name, rng in zip(cls_names, rngs, strict=False)
            ),
        )

        for cls_name, entry, summary, params_msg in results:
            # Per-model summary (name + CV score + time) at the normal level;
            # the sampled parameters only at the verbose level.
            logger.info("%s", summary)
            logger.debug("%s", params_msg)
            self.board.add(cls_name, entry)
        self.board.evaluate_models()

        timer = time.perf_counter() - timer
        time_spent = time.strftime("%H:%M:%S", time.gmtime(timer))
        # The full leaderboard table each step is verbose; the one-line batch
        # summary below stays at the normal level.
        logger.debug("%s", self)
        logger.info("Tested a batch of %d model(s). It took %s.", batch_size, time_spent)
        return timer

    def construct_df(self, ensemble_length: int | None = None, min_repr: int | None = None):
        """Build the meta-feature frames: OOF preds (train) and test preds."""
        ensemble_length = (
            self.ctx.config.ensemble_length if ensemble_length is None else ensemble_length
        )
        min_repr = self.ctx.config.ensemble_min_repr if min_repr is None else min_repr

        ens_train_df = pd.DataFrame()
        ens_test_df = pd.DataFrame()
        file_paths = self.board.get_best(ensemble_length=ensemble_length, min_repr=min_repr)
        for name, path in file_paths:
            model = Model.load(path, self.ctx)
            model.fit(self.X, self.y)
            pred = model.predict(self.X_test)
            oof = model.oof
            assert oof is not None, "loaded model is missing its out-of-fold predictions"
            if oof.ndim > 1:
                for i in range(oof.shape[1]):
                    ens_train_df[name + f"{i}"] = oof[:, i]
                    ens_test_df[name + f"{i}"] = pred[:, i]
            else:
                ens_train_df[name] = oof
                ens_test_df[name] = pred
        return ens_train_df, ens_test_df

    def predict(self, model=None) -> np.ndarray:
        """Stack the selected models and return decoded test predictions."""
        if self.ctx.target_is_num:
            raise ValueError("Numerical target prediction is not realised.")

        if model is None:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression()
            param_dist = {
                "random_state": [self.ctx.config.seed],
                "max_iter": [500],
                "solver": ["lbfgs"],
                "C": loguniform(1e-3, 1.0),
                "class_weight": ["balanced"],
            }
        else:
            raise ValueError("Unknown prediction model")

        from sklearn.model_selection import RandomizedSearchCV

        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=30,
            n_jobs=-1,
            cv=self.splits,
            scoring=self.ctx.config.scoring,
            random_state=self.ctx.config.seed,
        )

        X_ens_train, X_ens_test = self.construct_df()
        search.fit(X_ens_train, self.y)

        best_score = search.best_score_
        best_index = search.best_index_
        best_std = search.cv_results_["std_test_score"][best_index]
        logger.info(
            "Ensembling is done. Best score: %.6f ± %.6f, Best params: %s",
            best_score,
            best_std,
            search.best_params_,
        )
        p_pred = search.best_estimator_.predict_proba(X_ens_test)
        return self.ctx.target_transforms.inverse(p_pred)

    def save(self) -> None:
        self.board.save()

    def load(self) -> None:
        if self.board.load():
            logger.info("Loaded existing leaderboard (%d model(s)).", len(self.board))
            logger.debug("The leaderboard is: \n %s \n\n", self)
        else:
            logger.info("No existing leaderboard found; starting a new one.")

    def format_complexities(self) -> str:
        """One-line summary of the leaderboard's per-class complexities."""
        return self.board.format_complexities()

    def __str__(self) -> str:
        return self.board.__str__()
