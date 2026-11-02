import numpy as np

from discovery_reco.catalog import known_items


def test_catalog_is_namespaced_unique_and_zero_ratings_are_shrunk(catalog):
    assert catalog.item_id.is_unique
    assert catalog.item_id.str.match(r"^(movie|tv)_\d+$").all()
    assert np.isfinite(catalog.quality_score).all()
    assert catalog.discovery_score.between(0, 1).all()


def test_catalog_does_not_model_untrustworthy_fields(catalog):
    assert "date_added" not in catalog
    assert "duration" not in catalog
    assert "rating" not in catalog


def test_events_require_contract_and_drop_unknown(catalog, events):
    dirty = events.copy()
    dirty.loc[len(dirty)] = ["A", "movie_999999999", "play", events.timestamp.max()]
    clean = known_items(dirty, catalog.item_id)
    assert "movie_999999999" not in set(clean.item_id)

