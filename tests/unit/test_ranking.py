import numpy as np

from reco.serving.ranking import rank_items


def _item(index: int, genres: list[str], title: str | None = None, kind: str = "movie") -> dict:
    name = title or f"Title {index}"
    return {
        "item_id": f"{kind}_{index}",
        "title": name,
        "normalized_title": name.lower(),
        "content_type": kind,
        "genres": genres,
        "release_year": 2020,
        "recommendable": True,
    }


def test_ranking_all_rules_in_order() -> None:
    catalog = [
        _item(0, ["Action"]),
        _item(1, ["Action"]),
        _item(2, ["Action"]),
        _item(3, ["Action"]),
        _item(4, ["Action"]),
        _item(5, ["Drama"], title="Duplicate"),
        _item(6, ["Comedy"], title="Duplicate"),
        _item(7, ["Comedy"], kind="tv"),
    ]
    catalog[0]["recommendable"] = False
    scores = np.arange(8, 0, -1, dtype=np.float32)
    ranked = rank_items(scores, catalog, watched={1}, limit=5, content_type="movie")
    assert 0 not in ranked
    assert 1 not in ranked
    assert 7 not in ranked
    assert not ({5, 6} <= set(ranked))
    assert sum(catalog[index]["genres"] == ["Action"] for index in ranked) <= 2
