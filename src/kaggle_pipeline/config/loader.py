"""Load a :class:`Config` from a YAML file."""

from __future__ import annotations

from pathlib import Path

import yaml

from kaggle_pipeline.config.config import Config


def load_config(path: str | Path) -> Config:
    """Read a YAML file and return a validated :class:`Config`.

    The YAML keys map one-to-one onto Config fields. Unknown keys raise, so a
    typo fails loudly instead of being silently ignored.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return Config.from_dict(data)
