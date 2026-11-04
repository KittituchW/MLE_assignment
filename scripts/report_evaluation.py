"""Print reproducible full-data evaluation results and a live-bundle API example.

Run after `python -m reco run-all` from a development installation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from reco.serving.app import create_app


def main() -> None:
    home = Path(os.environ.get("RECO_HOME", "var")).expanduser().resolve()
    version = (home / "artifacts" / "CHAMPION").read_text(encoding="utf-8").strip()
    report_path = home / "artifacts" / "models" / version / "eval_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report["promotion_gate"]["passed"]:
        raise RuntimeError("Cannot report an unapproved model as champion")

    print("=== FULL-DATA EVALUATION TABLE ===", flush=True)
    print(f"Model version: {version}", flush=True)
    print("| Model | Recall@10 | NDCG@10 | Coverage | Cold-item share |", flush=True)
    print("|---|---:|---:|---:|---:|", flush=True)
    for name in ("popularity", "item_knn", "content", "hybrid"):
        values = report["models"][name]
        print(
            f"| {name} | {values['recall_at_10']:.4f} | "
            f"{values['ndcg_at_10']:.4f} | {values['catalogue_coverage']:.2%} | "
            f"{values['cold_item_share']:.2%} |",
            flush=True,
        )
    print(f"Test events: {report['split']['test_events']}", flush=True)
    print(f"Evaluated users: {int(report['models']['hybrid']['evaluated_users'])}", flush=True)
    print(f"Gate passed: {report['promotion_gate']['passed']}", flush=True)
    print("=== FULL-DATA API EXAMPLE ===", flush=True)
    with TestClient(create_app(home)) as client:
        response = client.post("/v1/recommendations", json={"user_id": "A", "limit": 3})
        response.raise_for_status()
        example = response.json()
        if len(example["items"]) != 3:
            raise RuntimeError("Expected exactly three recommendations in the full-data example")
        print(json.dumps(example, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
