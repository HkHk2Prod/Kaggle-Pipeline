"""The three notebook modes via flags: parallel-train (no submit) and blend-only."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from kaggle_pipeline.evolution import KagglePipeline, KagglePipelineSettings
from kaggle_pipeline.evolution.ecosystem.serialization import EcosystemSerializer
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
    sample = pd.DataFrame({"id": test["id"], "target": 0})
    return df, test, sample


def _settings(tmp_path, **overrides):
    base = dict(
        max_runtime_seconds=20,
        safety_margin_seconds=1,
        checkpoint_time_reserve_seconds=1,
        ensemble_time_reserve_seconds=2,
        finalization_time_reserve_seconds=1,
        verbosity=0,
        models_per_batch=4,
        cv_splits=3,
        max_active_features=20,
        num_workers=2,
        ensemble_min_models=2,
    )
    base.update(overrides)
    return KagglePipelineSettings(**base)


def _pipeline(settings):
    pipeline = KagglePipeline(settings)
    families = build_default_families()
    pipeline.families = {
        name: families[name] for name in ("logistic", "random_forest") if name in families
    }
    return pipeline


def _run_worker(tmp_path, *, seed):
    settings = _settings(
        tmp_path, seed=seed, state_dir=str(tmp_path / f"state_{seed}"), make_submission_on_run=False
    )
    _pipeline(settings).fit(_data()[0], target="target", scoring="roc_auc", id_col="id")
    return tmp_path / f"state_{seed}"


def test_train_mode_writes_no_submission(tmp_path):
    """Parallel-train: train_models=True, make_submission_on_run=False -> no CSV."""
    warnings.simplefilter("ignore")
    train, _, _ = _data()
    out = tmp_path / "submission.csv"
    settings = _settings(
        tmp_path,
        seed=1,
        state_dir=str(tmp_path / "state"),
        make_submission_on_run=False,
        submission_path=str(out),
    )
    pipeline = _pipeline(settings)
    pipeline.fit(train, target="target", scoring="roc_auc", id_col="id")
    assert not out.exists()
    # ...but an ecosystem checkpoint is written for the blend step to consume.
    assert EcosystemSerializer(settings.state_dir).latest_path() is not None


def test_blend_mode_merges_inputs_and_submits_without_training(tmp_path, monkeypatch):
    """Blend-only: train_models=False merges two ecosystems and writes a submission."""
    warnings.simplefilter("ignore")
    dir_a = _run_worker(tmp_path, seed=1)
    dir_b = _run_worker(tmp_path, seed=2)

    # The blend run discovers its inputs via find_all_state_dirs; point it at the
    # two worker outputs (stands in for two attached /kaggle/input datasets).
    monkeypatch.setattr(
        "kaggle_pipeline.evolution.state_io.find_all_state_dirs",
        lambda **_kw: [dir_a, dir_b],
    )

    n_a = len(EcosystemSerializer(dir_a).load().population.all_genomes())
    n_b = len(EcosystemSerializer(dir_b).load().population.all_genomes())

    train, test, sample = _data()
    out = tmp_path / "blend_submission.csv"
    settings = _settings(
        tmp_path,
        seed=3,
        state_dir=str(tmp_path / "blend_state"),  # empty -> falls through to inputs
        train_models=False,
        make_submission_on_run=True,
        submission_path=str(out),
    )
    pipeline = _pipeline(settings)
    pipeline.fit(
        train,
        target="target",
        test_df=test,
        sample_df=sample,
        scoring="roc_auc",
        id_col="id",
        resume=True,
    )

    assert out.exists()
    merged = pipeline.controller.population.all_genomes()
    # The merged leaderboard holds the union of both workers (>= the larger one).
    assert len(merged) >= max(n_a, n_b)
    # No new batches were trained: the batch counter never advanced past the
    # merged checkpoint's index (training was skipped entirely).
    assert pipeline._last_batch is None
