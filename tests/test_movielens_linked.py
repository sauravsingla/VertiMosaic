from __future__ import annotations

import pandas as pd

from vertimosaic.experiments.linked_movielens import build_movielens_user_parties


def test_movielens_builder_uses_exact_users_and_temporal_target() -> None:
    users = pd.DataFrame(
        {
            "UserID": range(1, 121),
            "Gender": ["M", "F"] * 60,
            "Age": [25, 35] * 60,
            "Occupation": [12, 15] * 60,
            "ZipCode": ["560001", "110001"] * 60,
        }
    )
    movies = pd.DataFrame(
        {
            "MovieID": [1, 2, 3, 4],
            "Title": ["A", "B", "C", "D"],
            "Genres": ["Action|Comedy", "Drama", "Sci-Fi|Thriller", "Romance|Crime"],
        }
    )
    rows: list[dict[str, int]] = []
    for user_id in range(1, 121):
        positive = user_id % 2 == 0
        for position in range(20):
            is_late = position >= 16
            rating = 5 if is_late and positive else (2 if is_late else 3)
            rows.append(
                {
                    "UserID": user_id,
                    "MovieID": position % 4 + 1,
                    "Rating": rating,
                    "Timestamp": 1_000_000 + position,
                }
            )
    ratings = pd.DataFrame(rows)

    active, passive, label, entity_ids = build_movielens_user_parties(users, ratings, movies)
    assert len(active) == len(passive) == len(label) == len(entity_ids) == 120
    assert list(active.index) == list(passive.index) == list(label.index)
    assert set(label.unique()) == {0, 1}
    assert "rating_count" in passive.columns
    assert "gender_male" in active.columns
    assert active.index.name == "UserID"
