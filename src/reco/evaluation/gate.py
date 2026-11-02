"""Candidate promotion criteria."""

from __future__ import annotations


def evaluate_gate(
    candidate: dict[str, float],
    baseline: dict[str, float],
    champion: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    """Require 1.5x baseline quality and no >5% regression from champion."""
    reasons: list[str] = []
    for metric in ("recall_at_10", "ndcg_at_10"):
        if candidate.get(metric, 0.0) < 1.5 * baseline.get(metric, 0.0):
            reasons.append(f"{metric} below 1.5x popularity")
        if champion and candidate.get(metric, 0.0) < 0.95 * champion.get(metric, 0.0):
            reasons.append(f"{metric} regressed more than 5% from champion")
    return not reasons, reasons
