"""Command-line entry point: ``kaggle-pipeline run --config path/to/config.yaml``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from kaggle_pipeline.config import Config
from kaggle_pipeline.logconfig import configure_logging
from kaggle_pipeline.pipeline import run

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle-pipeline",
        description="Config-driven AutoML pipeline for tabular Kaggle competitions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    config_help = "Path to a YAML config file (see configs/ for an example)."

    run_parser = sub.add_parser("run", help="Train and write a submission from a YAML config.")
    run_parser.add_argument("--config", "-c", required=True, help=config_help)

    analyze_parser = sub.add_parser(
        "analyze", help="Run exploratory data analysis (plots + reports) from a YAML config."
    )
    analyze_parser.add_argument("--config", "-c", required=True, help=config_help)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    if args.command == "run":
        out_path = run(Config.from_yaml(args.config))
        logger.info("Done. Submission at: %s", out_path)
        return 0
    if args.command == "analyze":
        from kaggle_pipeline.analysis import analyze

        analyze(Config.from_yaml(args.config))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
