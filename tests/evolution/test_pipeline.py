"""End-to-end KagglePipeline: batched run, threading, ensemble, submission, resume."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from kaggle_pipeline.evolution import KagglePipeline, KagglePipelineSettings
from kaggle_pipeline.evolution.ecosystem.summary import format_summary
from kaggle_pipeline.evolution.logging_utils import Verbosity
from kaggle_pipeline.evolution.models.parameter_spaces import build_default_families


def _data(n: int = 240):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "id": range(n),
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "cat1": rng.choice(list("abc"), n),
        }
    )
    logit = df["num1"] + 0.5 * df["num2"] + (df["cat1"] == "a") * 1.0
    df["target"] = (logit + rng.normal(scale=0.5, size=n) > 0).astype(int)
    test = df.drop(columns=["target"]).iloc[:50].copy()
    test["id"] = range(5000, 5050)
    return df, test


def _fast_pipeline(tmp_path, **overrides):
    settings = KagglePipelineSettings(
        max_runtime_seconds=overrides.pop("max_runtime_seconds", 20),
        safety_margin_seconds=1,
        checkpoint_time_reserve_seconds=1,
        ensemble_time_reserve_seconds=2,
        finalization_time_reserve_seconds=1,
        verbosity=overrides.pop("verbosity", 1),
        models_per_batch=3,
        cv_splits=3,
        max_active_features=20,
        num_workers=2,
        seed=1,
        ensemble_min_models=overrides.pop("ensemble_min_models", 2),
        ensemble_max_models=overrides.pop("ensemble_max_models", 8),
        state_dir=str(tmp_path / "state"),
        # Tests opt out of auto-submission by default so they don't write
        # `submission.csv` into the working directory; the auto-submit test
        # below overrides this back to True explicitly.
        make_submission_on_run=overrides.pop("make_submission_on_run", False),
        **overrides,
    )
    pipeline = KagglePipeline(settings)
    families = build_default_families()
    pipeline.families = {
        name: families[name] for name in ("logistic", "random_forest") if name in families
    }
    return pipeline


def test_pipeline_runs_and_writes_submission(tmp_path):
    warnings.simplefilter("ignore")
    train, test = _data()
    pipeline = _fast_pipeline(tmp_path)
    try:
        pipeline.fit(train, target="target", test_df=test, scoring="roc_auc", id_col="id")
        summary = pipeline.summarize_state()
        assert summary["models"]["completed"] > 0
        assert summary["batch_index"] >= 1
        out = pipeline.make_submission(tmp_path / "submission.csv")
        written = pd.read_csv(out)
        assert list(written.columns) == ["id", "target"]
        assert len(written) == len(test)
        assert written["target"].between(0.0, 1.0).all()
    finally:
        pipeline.shutdown()


def test_pipeline_state_saves_and_reloads(tmp_path):
    warnings.simplefilter("ignore")
    train, _ = _data()
    pipeline = _fast_pipeline(tmp_path)
    try:
        pipeline.fit(train, target="target", scoring="roc_auc", id_col="id")
        n_models = len(pipeline.controller.population.all_genomes())
        assert n_models > 0
    finally:
        pipeline.shutdown()

    resumed = _fast_pipeline(tmp_path)
    try:
        state = resumed.load_state()
        assert len(state.population.all_genomes()) == n_models
        assert resumed.serializer.read_manifest()["model_count"] == n_models
    finally:
        resumed.shutdown()


def test_autodetect_feature_expressions_and_subsample(tmp_path):
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(1)
    n = 500
    train = pd.DataFrame(
        {
            "id": range(n),
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )
    pipeline = _fast_pipeline(tmp_path, max_runtime_seconds=15, search_sample_fraction=0.2)
    try:
        # target/task/scoring all autodetected; feature_expressions add a column.
        pipeline.fit(train, feature_expressions=["ratio = num1 - num2"], id_col="id")
        assert pipeline._task == "classification"  # autodetected from 0/1 target
        assert "ratio" in list(pipeline._train_features.columns)  # engineered, no encodings
        assert len(pipeline._train_features) == n  # full data retained
        assert len(pipeline._search_y) == round(n * 0.2)  # search on 20% subsample
        assert pipeline.summarize_state()["models"]["completed"] > 0
    finally:
        pipeline.shutdown()


def test_submission_matches_sample_columns(tmp_path):
    warnings.simplefilter("ignore")
    train, test = _data()
    # A competition-style sample_submission with a non-"target" target column name.
    sample = pd.DataFrame({"id": test["id"].to_numpy(), "Survived": 0})
    pipeline = _fast_pipeline(tmp_path)
    try:
        pipeline.fit(train, target="target", test_df=test, sample_df=sample, id_col="id")
        out = pipeline.make_submission(tmp_path / "submission.csv")
        written = pd.read_csv(out)
        assert list(written.columns) == ["id", "Survived"]  # matches the sample, not "target"
        assert len(written) == len(test)
        assert written["Survived"].between(0.0, 1.0).all()  # probability aim
    finally:
        pipeline.shutdown()


def test_submission_category_aim_writes_labels(tmp_path):
    warnings.simplefilter("ignore")
    train, test = _data()
    sample = pd.DataFrame({"id": test["id"].to_numpy(), "label": 0})
    pipeline = _fast_pipeline(tmp_path)
    try:
        pipeline.fit(
            train,
            target="target",
            test_df=test,
            sample_df=sample,
            prediction_aim="category",
            id_col="id",
        )
        written = pd.read_csv(pipeline.make_submission(tmp_path / "submission.csv"))
        assert set(written["label"].unique()).issubset({0, 1})  # decoded class labels
    finally:
        pipeline.shutdown()


def test_pipeline_tolerates_nulls(tmp_path):
    warnings.simplefilter("ignore")
    train, test = _data(n=260)
    rng = np.random.default_rng(0)
    for frame in (train, test):
        frame.loc[rng.choice(len(frame), 20, replace=False), "num1"] = np.nan
        frame.loc[rng.choice(len(frame), 20, replace=False), "cat1"] = None
    pipeline = _fast_pipeline(tmp_path)
    try:
        # No load_datasets gate here; the pipeline imputes numerics and treats the
        # missing category as its own level, so training completes despite nulls.
        pipeline.fit(train, target="target", test_df=test, id_col="id")
        assert pipeline.summarize_state()["models"]["completed"] > 0
        preds = pipeline.predict()
        assert np.all(np.isfinite(preds))
    finally:
        pipeline.shutdown()


def test_silent_verbosity_prints_nothing(tmp_path):
    # format_summary at level 0 yields nothing; print_state must be a no-op.
    assert (
        format_summary(
            {"batch_index": 0, "features": {}, "models": {}, "mutations": {}, "ensemble": {}},
            Verbosity.SILENT,
        )
        == ""
    )


def test_make_submission_on_run_writes_csv_inside_fit(tmp_path):
    # With the flag set, fit() -> run() -> make_submission all happen in one
    # call -- no separate make_submission() needed afterwards. The budget is
    # large enough that the dynamic submission estimate (per-model time *
    # 1/sample_fraction * 1.3 * ensemble_size) fits within the run.
    warnings.simplefilter("ignore")
    train, test = _data()
    out_path = tmp_path / "auto_submission.csv"
    pipeline = _fast_pipeline(
        tmp_path,
        max_runtime_seconds=60,
        make_submission_on_run=True,
        submission_path=str(out_path),
        ensemble_max_models=2,
        search_sample_fraction=1.0,
        # Small bootstrap so the run isn't gated by the 30-min default before
        # the dynamic estimator overwrites it after the first batch.
        submission_time_reserve_seconds=5,
    )
    try:
        pipeline.fit(train, target="target", test_df=test, id_col="id")
        assert out_path.exists(), "auto-submission should have written the CSV"
        written = pd.read_csv(out_path)
        assert len(written) == len(test)
    finally:
        pipeline.shutdown()


def test_make_submission_on_run_skips_when_no_test_data(tmp_path):
    # If fit() got no test_df, auto-submission has nothing to predict on -- it
    # must skip silently rather than raising.
    warnings.simplefilter("ignore")
    train, _ = _data()
    out_path = tmp_path / "should_not_exist.csv"
    pipeline = _fast_pipeline(
        tmp_path,
        max_runtime_seconds=60,
        make_submission_on_run=True,
        submission_path=str(out_path),
        ensemble_max_models=2,
        search_sample_fraction=1.0,
        submission_time_reserve_seconds=5,
    )
    try:
        pipeline.fit(train, target="target", id_col="id")  # no test_df
        assert not out_path.exists()
    finally:
        pipeline.shutdown()


def test_ensembling_can_be_disabled(tmp_path):
    warnings.simplefilter("ignore")
    train, _ = _data()
    pipeline = _fast_pipeline(tmp_path, enable_ensembling=False)
    try:
        pipeline.fit(train, target="target", scoring="roc_auc", id_col="id")
        # With ensembling off, no ensemble is built; the best single model stands.
        assert pipeline.ensemble_result is None
        assert pipeline.best_genome() is not None
    finally:
        pipeline.shutdown()
