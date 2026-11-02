"""Command-line interface for independently runnable pipeline stages."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from reco.config import load_config, runtime_paths
from reco.logging import configure_logging


def _not_implemented(_: argparse.Namespace) -> int:
    raise RuntimeError("Pipeline stage is not implemented yet")


def _ingest(args: argparse.Namespace) -> int:
    from reco.data.ingest import ingest

    ingest(load_config(args.config), source=args.source)
    return 0


def _validate(args: argparse.Namespace) -> int:
    from reco.data.validate import validate

    validate(load_config(args.config))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    from reco.data.prepare import prepare

    prepare(load_config(args.config))
    return 0


def _simulate(args: argparse.Namespace) -> int:
    import pandas as pd

    from reco.interactions.simulator import SimulatedInteractionSource, write_simulated

    cfg = load_config(args.config)
    paths = runtime_paths()
    catalog_path = paths["processed"] / "catalog.parquet"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Prepared catalogue missing; run prepare first: {catalog_path}")
    source = SimulatedInteractionSource(pd.read_parquet(catalog_path), cfg)
    write_simulated(source, paths["interactions"] / "interactions.parquet")
    return 0


def _common(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--config", default="configs/base.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(prog="python -m reco", description="Netflix recommender pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser("ingest", help="download or copy and verify raw catalogues")
    _common(ingest_parser)
    ingest_parser.add_argument("--source", type=Path)
    ingest_parser.set_defaults(func=_ingest)
    for name, func, help_text in (
        ("validate", _validate, "run the raw data-quality gate"),
        ("prepare", _prepare, "create the canonical catalogue"),
        ("simulate", _simulate, "generate deterministic interactions"),
    ):
        subparser = commands.add_parser(name, help=help_text)
        _common(subparser)
        subparser.set_defaults(func=func)
    for name in ("train", "evaluate", "promote", "serve", "run-all"):
        subparser = commands.add_parser(name)
        _common(subparser)
        subparser.set_defaults(func=_not_implemented)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the selected pipeline stage."""
    configure_logging()
    args = build_parser().parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    raise SystemExit(func(args))
