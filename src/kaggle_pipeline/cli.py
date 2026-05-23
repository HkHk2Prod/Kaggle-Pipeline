"""Command-line entry point: ``kaggle-pipeline run --config path/to/config.yaml``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from kaggle_pipeline.config import Config
from kaggle_pipeline.pipeline import run

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle-pipeline",
        description="Config-driven AutoML pipeline for tabular Kaggle competitions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    config_help = (
        "Path to a YAML config file whose keys map onto Config fields. Omit to use "
        "defaults that autodetect everything from the data (notebook style)."
    )

    # Shared --verbose/--quiet flags so both subcommands accept them after the
    # command name (e.g. ``kaggle-pipeline run -c cfg.yaml -v``). When given they
    # override the config's ``verbosity``; otherwise the config value is used.
    verbosity = argparse.ArgumentParser(add_help=False)
    group = verbosity.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output: per-model scores, the full leaderboard each step, the encoding plan.",
    )
    group.add_argument(
        "-q", "--quiet", action="store_true", help="Quiet output: only warnings and errors."
    )

    run_parser = sub.add_parser(
        "run", parents=[verbosity], help="Train and write a submission from a YAML config."
    )
    run_parser.add_argument("--config", "-c", help=config_help)

    analyze_parser = sub.add_parser(
        "analyze",
        parents=[verbosity],
        help="Run exploratory data analysis (plots + reports) from a YAML config.",
    )
    analyze_parser.add_argument("--config", "-c", help=config_help)
    return parser


def _apply_cli_verbosity(config: Config, args: argparse.Namespace) -> None:
    """Let ``--verbose`` / ``--quiet`` override the config's ``verbosity``."""
    if getattr(args, "quiet", False):
        config.verbosity = "quiet"
    elif getattr(args, "verbose", False):
        config.verbosity = "verbose"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # No --config: fall back to a bare Config() that autodetects every field from
    # the data, exactly like a default Config() in the runner notebook.
    config = Config.from_yaml(args.config) if args.config else Config()
    _apply_cli_verbosity(config, args)
    # The entry points (run/analyze) configure the package logger from
    # ``config.verbosity``, so the level reflects the flag override above.
    if args.command == "run":
        out_path = run(config)
        logger.info("Done. Submission at: %s", out_path)
        return 0
    if args.command == "analyze":
        from kaggle_pipeline.analysis import analyze

        analyze(config)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
