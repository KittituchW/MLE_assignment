from pathlib import Path

import pandas as pd

from reco.data.policy import apply_content_policy
from reco.data.prepare import harmonize_genres, namespace_item_id


def test_id_namespacing_handles_real_18823_collision() -> None:
    assert namespace_item_id("movie", 18823) == "movie_18823"
    assert namespace_item_id("tv", 18823) == "tv_18823"


def test_genre_mapping() -> None:
    mapping = {"Action & Adventure": ["Action", "Adventure"], "TV Movie": []}
    assert harmonize_genres("Drama, Action & Adventure, TV Movie", mapping) == ["Action", "Adventure", "Drama"]


def _policy_frame() -> pd.DataFrame:
    titles = [
        "Sex Game 6969",
        "Beauty Salon: Special Service",
        "Purpose of Reunion 3",
        "Busty Girlfriend",
        "Bosomy Mom",
        "Obscene House: Slave Wife",
        "Youthful Older Sister",
        "Mom's Friend 2",
        "Love Lesson",
        "Mickey 17",
        "Flight Risk",
        "Bridgerton",
        "Angels of Sex",
    ]
    rows = []
    for index, title in enumerate(titles):
        rows.append(
            {
                "item_id": "movie_104634" if title == "Angels of Sex" else f"movie_{index}",
                "content_type": "tv" if title == "Bridgerton" else "movie",
                "title": title,
                "description": title
                if title not in {"Beauty Salon: Special Service", "Purpose of Reunion 3", "Love Lesson"}
                else "",
                "genres": ["Science Fiction"] if title == "Mickey 17" else ["Romance"],
                "vote_count": 15 if title == "Mickey 17" else (100 if title in {"Flight Risk", "Bridgerton"} else 1),
                "budget": 80000000 if title == "Mickey 17" else (1000000 if title == "Flight Risk" else 0),
            }
        )
    return pd.DataFrame(rows)


def test_policy_regressions() -> None:
    result = apply_content_policy(_policy_frame())
    blocked = set(result.loc[result.policy_blocked, "title"])
    expected_blocked = {
        "Sex Game 6969",
        "Beauty Salon: Special Service",
        "Purpose of Reunion 3",
        "Busty Girlfriend",
        "Bosomy Mom",
        "Obscene House: Slave Wife",
        "Youthful Older Sister",
        "Mom's Friend 2",
        "Love Lesson",
        "Angels of Sex",
    }
    assert expected_blocked <= blocked
    assert not ({"Mickey 17", "Flight Risk", "Bridgerton"} & blocked)


def test_policy_each_layer(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "item_id": "movie_1",
                "content_type": "movie",
                "title": "Erotic Story",
                "description": "",
                "genres": ["Comedy"],
                "vote_count": 500,
                "budget": 2,
            },
            {
                "item_id": "movie_2",
                "content_type": "movie",
                "title": "Quiet Love",
                "description": "",
                "genres": ["Drama"],
                "vote_count": 2,
                "budget": 0,
            },
            {
                "item_id": "tv_3",
                "content_type": "tv",
                "title": "Ordinary",
                "description": "",
                "genres": ["Comedy"],
                "vote_count": 500,
                "budget": 0,
            },
        ]
    )
    deny = tmp_path / "deny.txt"
    deny.write_text("tv_3 # curated\n", encoding="utf-8")
    result = apply_content_policy(frame, denylist_path=deny)
    assert result.policy_reason.tolist() == ["keyword", "evidence_floor", "denylist"]
