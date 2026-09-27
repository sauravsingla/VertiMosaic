from __future__ import annotations

from pathlib import Path

import pandas as pd

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import entity_level_split
from vertimosaic.experiments.pipeline import slice_parties
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.reporting import gbdt_local_feature_importance, logistic_local_feature_importance

_PARTIES = ("bank", "telecom", "insurance", "retail")


def run_feature_importance_study(
    *,
    rows: int = 1200,
    seed: int = 42,
    directory: Path = Path("results"),
) -> dict[str, pd.DataFrame]:
    """Write measured party-local logistic and GBDT importance summaries."""
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    parties = [train_active, *train_passive]

    logistic = VFLLogisticRegression(
        learning_rate=0.08,
        max_iter=350,
        l2=1e-3,
        seed=seed,
    )
    logistic.fit(train_active, train_passive)
    logistic_frames = logistic_local_feature_importance(logistic, parties)

    gbdt = VFLHistGBDT(
        n_estimators=12,
        max_depth=3,
        min_samples_leaf=15,
        seed=seed,
    )
    gbdt.fit(train_active, train_passive)
    gbdt_frames = gbdt_local_feature_importance(gbdt)

    output: dict[str, pd.DataFrame] = {}
    directory.mkdir(parents=True, exist_ok=True)
    for party in _PARTIES:
        logistic_frame = logistic_frames[party].copy()
        logistic_frame.insert(0, "model", "logistic")
        logistic_frame.insert(2, "method", "standardized_coefficient_magnitude")
        logistic_frame = logistic_frame.rename(
            columns={"standardized_coefficient_importance": "importance"}
        )
        gbdt_frame = gbdt_frames.get(
            party,
            pd.DataFrame(columns=["party", "feature", "split_count", "gain_sum", "gain_mean"]),
        ).copy()
        if not gbdt_frame.empty:
            gbdt_frame.insert(0, "model", "vfl-hist-gbdt")
            gbdt_frame.insert(2, "method", "local_split_gain")
            gbdt_frame["importance"] = gbdt_frame["gain_sum"]
        else:
            gbdt_frame = pd.DataFrame(
                columns=[
                    "model",
                    "party",
                    "method",
                    "feature",
                    "split_count",
                    "gain_sum",
                    "gain_mean",
                    "importance",
                ]
            )
        combined = pd.concat([logistic_frame, gbdt_frame], ignore_index=True, sort=False)
        path = directory / f"{party}_feature_importance.csv"
        combined.to_csv(path, index=False)
        output[party] = combined
    return output
