"""Tests for the standalone EDA flow and its decoupling from training."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import
import warnings  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pandas.errors import PerformanceWarning  # noqa: E402

from kaggle_pipeline import Config, analyze  # noqa: E402
from kaggle_pipeline.eda.plots import (  # noqa: E402
    EdaContext,
    _cap_categories,
    plot_cat_vs_any,
    plot_num_vs_cat,
)
from kaggle_pipeline.preprocessing.association import (  # noqa: E402
    association_matrix,
    correlation_ratio,
    cramers_v,
)


def test_analyze_runs_without_error(smoke_config: Config):
    smoke_config.run_eda = True  # opt in -- EDA is off by default
    analyze(smoke_config)
    plt.close("all")


def test_analyze_skipped_when_run_eda_false(monkeypatch):
    """With the default flag, analyze returns early without rendering or loading data."""
    import kaggle_pipeline.eda as eda_mod

    def _should_not_run(*args, **kwargs):  # pragma: no cover - asserts it isn't called
        raise AssertionError("run_eda must not be called when config.run_eda is False")

    monkeypatch.setattr(eda_mod, "run_eda", _should_not_run)
    assert Config().run_eda is False  # default
    analyze(Config())  # no data_dir needed: it returns before loading anything


def test_analyze_with_low_cardinality_numeric(tmp_path: Path):
    """A low-cardinality numeric column (e.g. Year) is plotted as categorical.

    Such columns are not in the typer's ``ordered_cats`` (which orders only true
    categoricals), so the plot dispatch used to ``KeyError`` on them. EDA must
    render them with a sorted-unique hue order instead.
    """
    rng = np.random.default_rng(0)
    n = 120
    train = pd.DataFrame(
        {
            "id": range(n),
            "Year": rng.choice([2019, 2020, 2021], size=n),  # low-cardinality numeric
            "num1": rng.normal(size=n),
            "cat1": rng.choice(["low", "high"], size=n),
            "y": rng.choice(["no", "yes"], size=n),
        }
    )
    train.to_csv(tmp_path / "train.csv", index=False)
    train.drop(columns=["y"]).head(40).to_csv(tmp_path / "test.csv", index=False)
    pd.DataFrame({"id": range(40), "y": ["no"] * 40}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )

    analyze(Config(competition="synthetic", data_dir=tmp_path, run_eda=True))
    plt.close("all")


def test_cramers_v_detects_perfect_and_zero_association():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.choice(["x", "y", "z"], size=300))
    perfectly_dependent = a.map({"x": "p", "y": "q", "z": "r"})
    independent = pd.Series(rng.choice(["p", "q", "r"], size=300))
    assert cramers_v(a, perfectly_dependent) > 0.95
    assert cramers_v(a, independent) < 0.2


def test_correlation_ratio_perfect_and_zero():
    cats = pd.Series(["a"] * 50 + ["b"] * 50 + ["c"] * 50)
    fully_determined = pd.Series([0.0] * 50 + [10.0] * 50 + [20.0] * 50)
    assert correlation_ratio(cats, fully_determined) > 0.99
    assert correlation_ratio(cats, pd.Series([5.0] * 150)) == 0.0  # constant -> 0


def test_correlation_ratio_bias_corrected_for_high_cardinality():
    """An independent high-cardinality category must not show a spurious η.

    Raw η² = SS_between / SS_total inflates with the group count: under
    independence η ≈ √((k-1)/(n-1)), which for a ``driver``-like column on a
    subsample is ~0.5 against *every* numeric (a noise floor, not a relationship).
    The ε² correction collapses it toward 0, keeping the cat–num block consistent
    with the Bergsma-corrected cramers_v (which already reads ~0 for such columns).
    """
    rng = np.random.default_rng(0)
    n = 1000
    hi_card = pd.Series(rng.integers(0, 350, size=n).astype(str))  # ~r/n = 0.35
    independent_num = pd.Series(rng.normal(size=n))
    assert correlation_ratio(hi_card, independent_num) < 0.1  # raw η would be ~0.58


def test_association_matrix_collapses_highcardinality_to_one_cell():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "driver": rng.choice([f"d{i}" for i in range(40)], size=n),
        }
    )
    matrix, ordered_num, ordered_cat = association_matrix(df, ["num1", "num2"], ["driver"])
    # One row/column per ORIGINAL column: driver is a single cell, not 40 dummies.
    assert list(matrix.index) == ["num1", "num2", "driver"]
    assert ordered_num == ["num1", "num2"] and ordered_cat == ["driver"]
    arr = matrix.to_numpy()
    assert np.allclose(np.diag(arr), 1.0)  # self-association is 1
    assert np.allclose(arr, arr.T)  # symmetric
    assert ((arr >= 0) & (arr <= 1)).all()  # unsigned, in [0, 1]


def test_cap_categories_folds_high_cardinality_and_preserves_nan():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.integers(0, 200, size=1000).astype(str))
    s[s.sample(20, random_state=1).index] = np.nan
    capped, order = _cap_categories(s, ordered=None, max_levels=12)
    assert capped.nunique() == 13  # top 12 + "Other"
    assert order[-1] == "Other" and len(order) == 13
    assert capped.isna().sum() == 20  # NaNs are preserved, never bucketed
    # A column already within the cap is returned untouched, in the given order.
    low = pd.Series(["a", "b", "c", "a"])
    same, low_order = _cap_categories(low, ordered=["a", "b", "c"], max_levels=12)
    assert same.equals(low) and low_order == ["a", "b", "c"]


def test_high_cardinality_plots_emit_no_performance_warning():
    """A high-cardinality categorical must not flood seaborn with fragmentation warnings.

    Plotting hundreds of hue levels / boxplot groups makes seaborn build one
    column per level, raising pandas' "highly fragmented DataFrame"
    PerformanceWarning (and an unusable plot). The top-N + "Other" cap prevents it.
    """
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame(
        {
            "lap": rng.normal(size=n),
            "driver": pd.Series(rng.integers(0, 400, size=n).astype(str)).astype("category"),
        }
    )
    ctx = EdaContext(
        df=df,
        ordered_cats={},
        columns=[],
        columns_x=[],
        num_cols=["lap"],
        cat_cols=["driver"],
        target=["lap"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        _, ax = plt.subplots()
        plot_cat_vs_any(ctx, ax, "lap", "driver")  # histplot hue path
        _, ax = plt.subplots()
        plot_num_vs_cat(ctx, ax, "driver", "lap")  # boxplot group path
    plt.close("all")


def test_training_import_does_not_pull_matplotlib():
    # Importing the package (the training entry point) must not import plotting
    # libraries -- EDA deps are loaded lazily only inside analyze().
    code = (
        "import kaggle_pipeline, sys; "
        "pulled = [m for m in sys.modules if m.split('.')[0] in {'matplotlib', 'seaborn'}]; "
        "assert not pulled, pulled"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
