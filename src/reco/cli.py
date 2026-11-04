"""Command-line interface for independently runnable pipeline stages."""

from __future__ import annotations

import argparse
import os
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


def _train(args: argparse.Namespace) -> int:
    from reco.bundle.pipeline import train_bundle

    train_bundle(load_config(args.config), version=args.version)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from reco.bundle.pipeline import evaluate_bundle

    evaluate_bundle(load_config(args.config), version=args.version)
    return 0


def _promote(args: argparse.Namespace) -> int:
    from reco.bundle.registry import promote_bundle

    promote_bundle(version=args.version, force=args.force)
    return 0


def _serve(args: argparse.Namespace) -> int:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import uvicorn

    from reco.serving.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)
    return 0


def _run_all(args: argparse.Namespace) -> int:
    """Run the ordered, fail-fast training pipeline."""
    cfg = load_config(args.config)
    from reco.bundle.pipeline import evaluate_bundle, train_bundle
    from reco.bundle.registry import promote_bundle
    from reco.data.ingest import ingest
    from reco.data.prepare import prepare
    from reco.data.validate import validate
    from reco.interactions.simulator import SimulatedInteractionSource, write_simulated

    ingest(cfg, source=args.source)
    validate(cfg)
    catalog_path = prepare(cfg)
    import pandas as pd

    paths = runtime_paths()
    write_simulated(
        SimulatedInteractionSource(pd.read_parquet(catalog_path), cfg),
        paths["interactions"] / "interactions.parquet",
    )
    bundle = train_bundle(cfg, version=args.version)
    evaluate_bundle(cfg, version=bundle.name)
    promote_bundle(version=bundle.name)
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
        ("train", _train, "fit and write a portable candidate bundle"),
        ("evaluate", _evaluate, "evaluate the candidate on the final temporal holdout"),
        ("promote", _promote, "gate and promote a candidate; override only with explicit --force"),
    ):
        subparser = commands.add_parser(name, help=help_text)
        _common(subparser)
        if name in {"train", "evaluate", "promote"}:
            subparser.add_argument("--version")
        if name == "promote":
            subparser.add_argument(
                "--force",
                action="store_true",
                help="Explicitly bypass the quality gate for a trusted rollback or test fixture",
            )
        subparser.set_defaults(func=func)
    serve_parser = commands.add_parser("serve", help="serve the promoted bundle")
    _common(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=8000, type=int)
    serve_parser.set_defaults(func=_serve)
    run_all = commands.add_parser("run-all", help="run ingest through promotion")
    _common(run_all)
    run_all.add_argument("--source", type=Path)
    run_all.add_argument("--version")
    run_all.set_defaults(func=_run_all)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the selected pipeline stage."""
    configure_logging()
    args = build_parser().parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    raise SystemExit(func(args))
