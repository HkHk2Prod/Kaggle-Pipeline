#!/usr/bin/env python3
"""Publish this package as a Kaggle Dataset for offline (internet-off) use.

Code competitions disable internet, so the ``notebooks/pipeline_*.ipynb``
runners cannot ``pip install`` the package and instead import it from an attached
Kaggle Dataset (see each notebook's "Offline use" section). This script automates
building / refreshing that dataset from the repo's ``src/`` directory.

It must run somewhere with internet *and* your Kaggle credentials -- your own
machine or CI -- not inside the offline competition kernel.

Prerequisites
-------------
* ``pip install kaggle``
* A Kaggle API token at ``~/.kaggle/kaggle.json`` (Kaggle: Account -> Settings ->
  Create New API Token), or ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` env vars.

Usage
-----
    python scripts/publish_dataset.py                  # owner from token/env, slug "kaggle-pipeline"
    python scripts/publish_dataset.py --owner alice    # explicit owner
    python scripts/publish_dataset.py -m "v0.1.0"      # version note when updating
    python scripts/publish_dataset.py --create         # force first-time create

The dataset is created on the first run and uploaded as a new version after
that. It mounts at ``/kaggle/input/<slug>/src/kaggle_pipeline``, which the
notebook's setup cell discovers automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DEFAULT_SLUG = "kaggle-pipeline"
DEFAULT_TITLE = "Kaggle Pipeline (package)"


def _fail(msg: str) -> NoReturn:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _resolve_owner(explicit: str | None) -> str:
    """Kaggle username from --owner, then KAGGLE_USERNAME, then the API token."""
    if explicit:
        return explicit
    if env := os.environ.get("KAGGLE_USERNAME"):
        return env
    token = Path.home() / ".kaggle" / "kaggle.json"
    if token.is_file():
        try:
            return json.loads(token.read_text())["username"]
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    _fail(
        "could not determine the Kaggle owner. Pass --owner, set KAGGLE_USERNAME, "
        "or create ~/.kaggle/kaggle.json (Kaggle Account -> Create New API Token)."
    )


def _kaggle(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["kaggle", *args], check=check, text=True, capture_output=capture)


def _dataset_exists(dataset_id: str) -> bool:
    """True if ``owner/slug`` already exists (so we version rather than create)."""
    proc = _kaggle("datasets", "files", dataset_id, check=False, capture=True)
    return proc.returncode == 0


def stage_package(stage: Path, dataset_id: str, title: str) -> None:
    """Lay out ``src/`` plus a dataset-metadata.json inside ``stage``."""
    shutil.copytree(
        SRC_DIR,
        stage / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.egg-info"),
    )
    (stage / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": title, "id": dataset_id, "licenses": [{"name": "MIT"}]},
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--owner", help="Kaggle username (default: from token / KAGGLE_USERNAME)")
    parser.add_argument(
        "--slug", default=DEFAULT_SLUG, help=f"dataset slug (default: {DEFAULT_SLUG})"
    )
    parser.add_argument("--title", default=DEFAULT_TITLE, help="dataset title")
    parser.add_argument("-m", "--message", default="update", help="version note when updating")
    parser.add_argument(
        "--create", action="store_true", help="force first-time create instead of versioning"
    )
    args = parser.parse_args()

    if shutil.which("kaggle") is None:
        _fail("the Kaggle CLI is not installed. Run `pip install kaggle` first.")
    if not SRC_DIR.is_dir():
        _fail(f"expected the package source at {SRC_DIR}, but it does not exist.")

    dataset_id = f"{_resolve_owner(args.owner)}/{args.slug}"

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        stage_package(stage, dataset_id, args.title)

        # `--dir-mode zip` archives the src/ subtree; Kaggle extracts it on upload,
        # so the mounted dataset contains src/kaggle_pipeline/... as a real tree.
        if args.create or not _dataset_exists(dataset_id):
            print(f"[publish] creating new dataset {dataset_id}")
            _kaggle("datasets", "create", "-p", str(stage), "--dir-mode", "zip")
        else:
            print(f"[publish] uploading a new version of {dataset_id}")
            _kaggle(
                "datasets", "version", "-p", str(stage), "-m", args.message, "--dir-mode", "zip"
            )

    print(
        f"[publish] done. Attach it to the notebook via Add Input -> '{dataset_id}'. "
        "The setup cell imports it automatically when there is no internet."
    )


if __name__ == "__main__":
    main()
