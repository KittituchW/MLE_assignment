"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .catalog import known_items, prepare_catalog
from .evaluate import temporal_evaluate
from .model import train


def _events(path: str | None, catalog: pd.DataFrame) -> pd.DataFrame:
    columns = ["user_id", "item_id", "event_type", "timestamp"]
    if not path:
        return pd.DataFrame(columns=columns).assign(timestamp=pd.to_datetime([]))
    return known_items(pd.read_csv(path), catalog.item_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--catalog", required=True)
    build.add_argument("--events")
    build.add_argument("--output", default="artifacts/model")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--catalog", required=True)
    evaluate.add_argument("--events", required=True)
    args = parser.parse_args()
    catalog = prepare_catalog(args.catalog)
    events = _events(args.events, catalog)
    if args.command == "build":
        print(train(catalog, events).save(Path(args.output)))
    else:
        print(json.dumps(temporal_evaluate(catalog, events), indent=2))


if __name__ == "__main__":
    main()

