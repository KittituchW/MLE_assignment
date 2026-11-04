"""Candidate promotion criteria."""

from __future__ import annotations


def evaluate_gate(
    candidate: dict[str, float],
    baseline: dict[str, float],
    champion: dict[str, float] | None = None,
    *,
    baseline_ratio: float = 1.5,
    champion_tolerance: float = 0.05,
) -> tuple[bool, list[str]]:
    """Require configured baseline improvement and avoid champion regression."""
    reasons: list[str] = []
    for metric in ("recall_at_10", "ndcg_at_10"):
        if candidate.get(metric, 0.0) < baseline_ratio * baseline.get(metric, 0.0):
            reasons.append(f"{metric} below {baseline_ratio:g}x popularity")
        if champion and candidate.get(metric, 0.0) < (1 - champion_tolerance) * champion.get(metric, 0.0):
            reasons.append(f"{metric} regressed more than {champion_tolerance:.0%} from champion")
    return not reasons, reasons
