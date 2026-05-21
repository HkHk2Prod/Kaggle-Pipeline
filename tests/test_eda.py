"""Tests for the standalone EDA flow and its decoupling from training."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kaggle_pipeline import Config, analyze  # noqa: E402
from kaggle_pipeline.eda.association import (  # noqa: E402
    association_matrix,
    correlation_ratio,
    cramers_v,
)


def test_analyze_runs_without_error(smoke_config: Config):
    analyze(smoke_config)
    plt.close("all")


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

    analyze(Config(competition="synthetic", data_dir=tmp_path))
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


def test_training_import_does_not_pull_matplotlib():
    # Importing the package (the training entry point) must not import plotting
    # libraries -- EDA deps are loaded lazily only inside analyze().
    code = (
        "import kaggle_pipeline, sys; "
        "pulled = [m for m in sys.modules if m.split('.')[0] in {'matplotlib', 'seaborn'}]; "
        "assert not pulled, pulled"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
