import math

import pandas as pd

from reco.evaluation.gate import evaluate_gate
from reco.evaluation.metrics import catalogue_coverage, ndcg_at_k, recall_at_k
from reco.evaluation.split import temporal_split


def test_metrics_against_hand_computed_examples() -> None:
    recommended = ["a", "x", "b"]
    relevant = {"a", "b"}
    assert recall_at_k(recommended, relevant, 3) == 1.0
    expected = (1.0 + 1.0 / math.log2(4)) / (1.0 + 1.0 / math.log2(3))
    assert ndcg_at_k(recommended, relevant, 3) == expected
    assert catalogue_coverage([["a", "b"], ["b", "c"]], 6) == 0.5


def test_temporal_split_has_no_future_training_events() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u"] * 4,
            "item_id": ["old", "validation", "test", "after"],
            "event_ts": pd.to_datetime(["2026-01-01", "2026-01-10", "2026-01-25", "2026-02-01"], utc=True),
            "event_type": ["play"] * 4,
            "watch_ratio": [0.5] * 4,
        }
    )
    split = temporal_split(events, "2026-01-31", 7)
    assert (split.train.event_ts < split.validation_start).all()
    assert (split.validation.event_ts < split.test_start).all()
    assert (split.test.event_ts >= split.test_start).all()
    assert "after" not in set(split.test.item_id)


def test_temporal_split_removes_seen_targets() -> None:
    events = pd.DataFrame(
        {
            "user_id": ["u", "u", "u"],
            "item_id": ["same", "same", "new"],
            "event_ts": pd.to_datetime(["2025-12-01", "2026-01-20", "2026-01-25"], utc=True),
            "event_type": ["play"] * 3,
            "watch_ratio": [0.5] * 3,
        }
    )
    split = temporal_split(events, "2026-01-31", 14)
    assert set(split.test.item_id) == {"new"}


def test_gate_checks_baseline_and_champion() -> None:
    baseline = {"recall_at_10": 0.01, "ndcg_at_10": 0.01}
    champion = {"recall_at_10": 0.02, "ndcg_at_10": 0.02}
    assert evaluate_gate({"recall_at_10": 0.03, "ndcg_at_10": 0.03}, baseline, champion)[0]
    passed, reasons = evaluate_gate({"recall_at_10": 0.014, "ndcg_at_10": 0.03}, baseline, champion)
    assert not passed
    assert any("recall" in reason for reason in reasons)
