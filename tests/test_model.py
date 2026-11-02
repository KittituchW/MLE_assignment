from discovery_reco.evaluate import temporal_evaluate
from discovery_reco.model import DiscoveryModel, train


def test_personalized_and_cold_start_paths(catalog, events):
    model = train(catalog, events)
    strategy, known = model.recommend("A", 5)
    cold_strategy, cold = model.recommend("unknown", 5, "tv")
    assert strategy == "behavior_content_profile"
    assert cold_strategy == "cold_start_discovery"
    assert len(known) == len(cold) == 5
    assert all(item.content_type == "tv" for item in cold)
    watched = set(events.loc[events.user_id == "A", "item_id"])
    assert not watched.intersection(item.item_id for item in known)


def test_artifact_round_trip(tmp_path, catalog, events):
    expected = train(catalog, events)
    expected.save(tmp_path)
    actual = DiscoveryModel.load(tmp_path)
    assert [x.item_id for x in expected.recommend("A", 5)[1]] == [
        x.item_id for x in actual.recommend("A", 5)[1]
    ]


def test_temporal_evaluation_is_finite(catalog, events):
    result = temporal_evaluate(catalog, events, 10)
    assert result["users"] == 2
    assert 0 <= result["hit_rate@10"] <= 1
    assert 0 <= result["catalog_coverage"] <= 1
