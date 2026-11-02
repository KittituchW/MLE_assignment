"""Command-line interface for independently runnable pipeline stages."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from reco.logging import configure_logging


def _not_implemented(_: argparse.Namespace) -> int:
    raise RuntimeError("Pipeline stage is not implemented yet")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(prog="python -m reco", description="Netflix recommender pipeline")
    parser.add_argument("--config", default="configs/base.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "validate", "prepare", "simulate", "train", "evaluate", "promote", "run-all", "serve"):
        command = sub.add_parser(name)
        command.add_argument("--source")
        command.add_argument("--version")
        command.set_defaults(func=_not_implemented)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the selected pipeline stage."""
    configure_logging()
    args = build_parser().parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    raise SystemExit(func(args))

