"""Tests for the standalone EDA flow and its decoupling from training."""

from __future__ import annotations

import subprocess
import sys

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from kaggle_pipeline import Config, analyze  # noqa: E402


def test_analyze_runs_without_error(smoke_config: Config):
    analyze(smoke_config)
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
