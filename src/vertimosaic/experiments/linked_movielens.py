# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from vertimosaic.evaluation import (
    binary_metrics,
    bootstrap_confidence_intervals,
    communication_totals,
    entity_level_split,
    select_f1_threshold,
)
from vertimosaic.experiments.linked_uci_credit import _fit_transform_train_only
from vertimosaic.experiments.resources import PeakRSSSampler
from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.reproducibility import environment_snapshot

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_1M_README = "https://files.grouplens.org/datasets/movielens/ml-1m-README.txt"


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def download_movielens_1m(*, timeout_seconds: float = 60.0) -> bytes:
    request = Request(
        MOVIELENS_1M_URL,
        headers={"User-Agent": "VertiMosaic research benchmark"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310 - fixed HTTPS URL
        payload = response.read()
    if len(payload) < 1_000_000:
        raise RuntimeError("MovieLens 1M download was unexpectedly small")
    return payload


def parse_movielens_1m(archive: bytes) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        expected = {
            "ml-1m/users.dat",
            "ml-1m/ratings.dat",
            "ml-1m/movies.dat",
        }
        missing = expected - set(bundle.namelist())
        if missing:
            raise RuntimeError(f"MovieLens archive is missing required files: {sorted(missing)}")
        users = pd.read_csv(
            bundle.open("ml-1m/users.dat"),
            sep="::",
            engine="python",
            names=["UserID", "Gender", "Age", "Occupation", "ZipCode"],
            encoding="latin-1",
        )
        ratings = pd.read_csv(
            bundle.open("ml-1m/ratings.dat"),
            sep="::",
            engine="python",
            names=["UserID", "MovieID", "Rating", "Timestamp"],
            encoding="latin-1",
        )
        movies = pd.read_csv(
            bundle.open("ml-1m/movies.dat"),
            sep="::",
            engine="python",
            names=["MovieID", "Title", "Genres"],
            encoding="latin-1",
        )
    return users, ratings, movies


def _genre_flags(frame: pd.DataFrame, genres: tuple[str, ...]) -> pd.DataFrame:
    values = frame["Genres"].fillna("").astype(str)
    return pd.DataFrame(
        {
            f"genre_{genre.lower().replace('-', '_')}": values.str.contains(
                genre,
                regex=False,
            ).astype(float)
            for genre in genres
        },
        index=frame.index,
    )


def build_movielens_user_parties(
    users: pd.DataFrame,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create exact UserID-linked demographic and historical-behaviour parties.

    The passive-party features use the earliest 80% of each user's observed ratings.
    The active-party label is derived only from the latest 20%: whether the mean late
    rating is at least 4 stars. This prevents the exact target events from appearing
    in the passive feature block while retaining an entirely observed-data target.
    """

    required_users = {"UserID", "Gender", "Age", "Occupation", "ZipCode"}
    required_ratings = {"UserID", "MovieID", "Rating", "Timestamp"}
    required_movies = {"MovieID", "Genres"}
    if not required_users.issubset(users.columns):
        raise ValueError("users frame does not match MovieLens 1M schema")
    if not required_ratings.issubset(ratings.columns):
        raise ValueError("ratings frame does not match MovieLens 1M schema")
    if not required_movies.issubset(movies.columns):
        raise ValueError("movies frame does not match MovieLens 1M schema")

    ordered = ratings.sort_values(["UserID", "Timestamp", "MovieID"]).copy()
    ordered["row_number"] = ordered.groupby("UserID").cumcount()
    ordered["user_count"] = ordered.groupby("UserID")["MovieID"].transform("size")
    ordered["early_cutoff"] = np.floor(0.8 * ordered["user_count"]).astype(int)
    early = ordered.loc[ordered["row_number"] < ordered["early_cutoff"]].copy()
    late = ordered.loc[ordered["row_number"] >= ordered["early_cutoff"]].copy()
    if early.empty or late.empty:
        raise ValueError("temporal split produced an empty MovieLens partition")

    late_summary = late.groupby("UserID").agg(
        late_mean_rating=("Rating", "mean"),
        late_rating_count=("Rating", "size"),
    )
    label = (late_summary["late_mean_rating"] >= 4.0).astype(int).rename("target")

    early = early.merge(
        movies[["MovieID", "Genres"]],
        on="MovieID",
        how="left",
        validate="many_to_one",
    )
    genres = (
        "Action",
        "Comedy",
        "Drama",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "Crime",
        "Adventure",
    )
    flags = _genre_flags(early, genres)
    early = pd.concat([early.reset_index(drop=True), flags.reset_index(drop=True)], axis=1)
    early["high_rating"] = (
        pd.to_numeric(early["Rating"], errors="coerce") >= 4.0
    ).astype(float)
    aggregate_spec: dict[str, tuple[str, str]] = {
        "rating_count": ("Rating", "size"),
        "mean_rating": ("Rating", "mean"),
        "rating_std": ("Rating", "std"),
        "high_rating_fraction": ("high_rating", "mean"),
        "unique_movies": ("MovieID", "nunique"),
    }
    for genre in genres:
        column = f"genre_{genre.lower().replace('-', '_')}"
        aggregate_spec[f"{column}_fraction"] = (column, "mean")
    passive = early.groupby("UserID").agg(**aggregate_spec).fillna(0.0)

    active = users.copy()
    active["gender_male"] = (active["Gender"].astype(str) == "M").astype(float)
    active["age_code"] = pd.to_numeric(active["Age"], errors="coerce")
    active["occupation_code"] = pd.to_numeric(active["Occupation"], errors="coerce")
    active["zip_prefix"] = pd.to_numeric(
        active["ZipCode"].astype(str).str.extract(r"(\d{3})", expand=False),
        errors="coerce",
    )
    active = active.set_index("UserID")[
        ["gender_male", "age_code", "occupation_code", "zip_prefix"]
    ]

    common_ids = active.index.intersection(passive.index).intersection(label.index).sort_values()
    if len(common_ids) < 100:
        raise ValueError("MovieLens exact-link benchmark requires at least 100 linked users")
    active = active.loc[common_ids]
    passive = passive.loc[common_ids]
    label = label.loc[common_ids]
    if label.nunique() != 2:
        raise ValueError("MovieLens temporal target must contain both classes")
    return (
        active,
        passive,
        label,
        pd.Series(common_ids.to_numpy(), index=common_ids, name="UserID"),
    )


def run_movielens_linked_experiment(
    *,
    seed: int = 42,
    bootstrap_replicates: int = 1000,
    output: Path = Path("reports/movielens_1m_exact_linked.json"),
) -> dict[str, Any]:
    archive = download_movielens_1m()
    users, ratings, movies = parse_movielens_1m(archive)
    active_frame, passive_frame, label, entity_ids = build_movielens_user_parties(
        users,
        ratings,
        movies,
    )
    split = entity_level_split(label.to_numpy(dtype=float), seed=seed)

    active_train, active_validation, active_test, active_preprocess = _fit_transform_train_only(
        active_frame.iloc[split.train],
        active_frame.iloc[split.validation],
        active_frame.iloc[split.test],
    )
    passive_train, passive_validation, passive_test, passive_preprocess = _fit_transform_train_only(
        passive_frame.iloc[split.train],
        passive_frame.iloc[split.validation],
        passive_frame.iloc[split.test],
    )
    y = label.to_numpy(dtype=float)
    train_active = ActiveParty("demographics", active_train, y[split.train])
    train_passive = [PassiveParty("rating_history", passive_train)]
    validation_active = ActiveParty("demographics", active_validation, y[split.validation])
    validation_passive = [PassiveParty("rating_history", passive_validation)]
    test_active = ActiveParty("demographics", active_test, y[split.test])
    test_passive = [PassiveParty("rating_history", passive_test)]

    model = VFLLogisticRegression(
        learning_rate=0.08,
        max_iter=500,
        l2=1e-3,
        early_stopping_rounds=5,
        seed=seed,
    )
    started = time.perf_counter()
    with PeakRSSSampler(interval_seconds=0.005) as memory:
        model.fit(train_active, train_passive, validation_active, validation_passive)
    training_seconds = time.perf_counter() - started
    validation_probability = model.predict_proba([validation_active, *validation_passive])[:, 1]
    threshold = select_f1_threshold(validation_active.labels, validation_probability)
    inference_started = time.perf_counter()
    probability = model.predict_proba([test_active, *test_passive])[:, 1]
    inference_seconds = time.perf_counter() - inference_started
    metrics = binary_metrics(test_active.labels, probability, threshold=threshold)
    intervals = bootstrap_confidence_intervals(
        test_active.labels,
        probability,
        threshold=threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    communication = communication_totals(model.transport.audit_log)

    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output.with_name(output.stem + "_predictions.csv")
    test_ids = entity_ids.iloc[split.test].to_numpy()
    pseudonyms = [
        sha256(f"movielens1m:{int(user_id)}".encode()).hexdigest()[:20]
        for user_id in test_ids
    ]
    pd.DataFrame(
        {
            "entity_id": pseudonyms,
            "target": test_active.labels.astype(int),
            "probability": probability,
        }
    ).to_csv(predictions_path, index=False)

    payload: dict[str, Any] = {
        "benchmark": "movielens_1m_exact_multitable_linked",
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "source": {
            "provider": "GroupLens Research, University of Minnesota",
            "dataset": "MovieLens 1M",
            "download_url": MOVIELENS_1M_URL,
            "readme_url": MOVIELENS_1M_README,
            "archive_sha256": _sha256_bytes(archive),
            "license_summary": (
                "research use; acknowledgement required; source data may not be redistributed "
                "without separate permission; commercial/revenue-bearing use requires permission"
            ),
            "redistributed_by_vertimosaic": False,
            "users_rows": len(users),
            "ratings_rows": len(ratings),
            "movies_rows": len(movies),
        },
        "linkage": {
            "kind": "exact_public_multitable_user_id_and_movie_id_linkage",
            "same_real_service_entities_across_parties": True,
            "single_service_dataset": True,
            "cross_organization_claim": False,
            "semi_synthetic_linkage": False,
            "entity_key": "UserID",
            "secondary_key": "MovieID",
            "linked_users": len(active_frame),
        },
        "target": {
            "kind": "derived_observed_temporal_target",
            "definition": "mean rating in latest 20% of each user's ratings is >= 4 stars",
            "target_events_excluded_from_passive_features": True,
        },
        "partitions": {
            "active_party": "users.dat demographics",
            "passive_party": "early ratings.dat history joined to movies.dat genres",
            "active_columns": list(active_frame.columns),
            "passive_columns": list(passive_frame.columns),
            "preprocessing": {
                "active": active_preprocess,
                "passive": passive_preprocess,
            },
        },
        "metrics": metrics,
        "confidence_intervals": intervals,
        "threshold_selected_on_validation": threshold,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "peak_rss_bytes": memory.peak_rss_bytes,
        "peak_rss_method": "sampled_process_rss_5ms",
        "estimated_communication_bytes": int(model.transport.estimated_payload_bytes),
        "communication": communication,
        "environment": environment_snapshot(seed),
        "predictions_file": predictions_path.name,
        "claim_boundary": (
            "This is genuine exact linkage across multiple public tables from one service. "
            "It is not evidence of cross-organization record linkage or cryptographic PSI."
        ),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the public exact-linked MovieLens 1M VFL benchmark."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/movielens_1m_exact_linked.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_movielens_linked_experiment(
                seed=args.seed,
                bootstrap_replicates=args.bootstrap_replicates,
                output=args.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
