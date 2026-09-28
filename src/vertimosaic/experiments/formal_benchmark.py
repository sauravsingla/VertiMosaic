# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import (
    binary_metrics,
    communication_totals,
    entity_level_split,
    select_f1_threshold,
)
from vertimosaic.experiments.overlap_study import run_overlap_study
from vertimosaic.experiments.pipeline import slice_parties
from vertimosaic.experiments.resources import PeakRSSSampler
from vertimosaic.experiments.studies import run_dropout_study
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.privacy import confidence_membership_inference


def _matrix(parties: list[PassiveParty]) -> np.ndarray:
    return np.column_stack([party._x for party in parties])


def _markdown_table(frame: pd.DataFrame) -> str:
    preferred = [
        "protocol",
        "roc_auc",
        "pr_auc",
        "f1",
        "training_seconds",
        "peak_rss_bytes",
        "estimated_communication_bytes",
        "overlap_50_roc_auc",
        "worst_inference_dropout_roc_auc",
        "membership_attack_auc",
        "formal_privacy_guarantee",
    ]
    columns = [column for column in preferred if column in frame.columns]
    view = frame[columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    values = [[str(item) if not pd.isna(item) else "" for item in row] for row in view.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in values]
    return "\n".join([header, rule, *body]) + "\n"


def _membership(train_probability: np.ndarray, test_probability: np.ndarray) -> tuple[float, float]:
    result = confidence_membership_inference(train_probability, test_probability)
    return float(result.roc_auc), float(result.attack_advantage)


def _non_federated_row(
    *,
    protocol: str,
    train_active: ActiveParty,
    train_passive: list[PassiveParty],
    validation_active: ActiveParty,
    validation_passive: list[PassiveParty],
    test_active: ActiveParty,
    test_passive: list[PassiveParty],
    seed: int,
    all_features: bool,
) -> dict[str, Any]:
    train_parties: list[PassiveParty] = (
        [train_active, *train_passive] if all_features else [train_active]
    )
    validation_parties: list[PassiveParty] = (
        [validation_active, *validation_passive] if all_features else [validation_active]
    )
    test_parties: list[PassiveParty] = (
        [test_active, *test_passive] if all_features else [test_active]
    )
    x_train = _matrix(train_parties)
    x_validation = _matrix(validation_parties)
    x_test = _matrix(test_parties)
    estimator = LogisticRegression(max_iter=1000, random_state=seed)
    start = time.perf_counter()
    with PeakRSSSampler(interval_seconds=0.005) as memory:
        estimator.fit(x_train, train_active.labels)
    training_seconds = time.perf_counter() - start
    validation_probability = estimator.predict_proba(x_validation)[:, 1]
    threshold = select_f1_threshold(validation_active.labels, validation_probability)
    inference_start = time.perf_counter()
    test_probability = estimator.predict_proba(x_test)[:, 1]
    inference_seconds = time.perf_counter() - inference_start
    metrics = binary_metrics(test_active.labels, test_probability, threshold=threshold)
    train_probability = estimator.predict_proba(x_train)[:, 1]
    membership_auc, membership_advantage = _membership(train_probability, test_probability)
    return {
        "protocol": protocol,
        "reference_model": "sklearn-logistic",
        "federated": False,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "f1": metrics["f1"],
        "brier": metrics["brier"],
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "peak_rss_bytes": memory.peak_rss_bytes,
        "peak_rss_method": "sampled_process_rss_5ms",
        "estimated_communication_bytes": 0,
        "communication_message_count": 0,
        "overlap_50_roc_auc": np.nan,
        "overlap_status": "not applicable to pooled baseline",
        "worst_inference_dropout_roc_auc": np.nan,
        "dropout_status": "not applicable to pooled baseline",
        "membership_attack_auc": membership_auc,
        "membership_attack_advantage": membership_advantage,
        "privacy_surface": (
            "all selected feature columns are pooled into one process"
            if all_features
            else "active-party features only; no cross-party messages"
        ),
        "formal_privacy_guarantee": "none",
    }


def _vfl_row(
    *,
    model_name: str,
    train_active: ActiveParty,
    train_passive: list[PassiveParty],
    validation_active: ActiveParty,
    validation_passive: list[PassiveParty],
    test_active: ActiveParty,
    test_passive: list[PassiveParty],
    seed: int,
    logistic_max_iter: int,
    gbdt_estimators: int,
) -> dict[str, Any]:
    if model_name == "logistic":
        model: VFLLogisticRegression | VFLHistGBDT = VFLLogisticRegression(
            learning_rate=0.08,
            max_iter=logistic_max_iter,
            l2=1e-3,
            early_stopping_rounds=5 if logistic_max_iter >= 10 else None,
            seed=seed,
        )
    elif model_name == "vfl-hist-gbdt":
        model = VFLHistGBDT(
            n_estimators=gbdt_estimators,
            max_depth=3,
            min_samples_leaf=10,
            early_stopping_rounds=3 if gbdt_estimators >= 6 else None,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown VFL model: {model_name}")

    start = time.perf_counter()
    with PeakRSSSampler(interval_seconds=0.005) as memory:
        if getattr(model, "early_stopping_rounds", None) is None:
            model.fit(train_active, train_passive)
        else:
            model.fit(train_active, train_passive, validation_active, validation_passive)
    training_seconds = time.perf_counter() - start
    communication = communication_totals(model.transport.audit_log)

    if isinstance(model, VFLHistGBDT):
        training_parties: list[PassiveParty] = [train_active, *train_passive]
        for evaluation_parties in (
            [validation_active, *validation_passive],
            [test_active, *test_passive],
        ):
            for source_party, target_party in zip(
                training_parties,
                evaluation_parties,
                strict=True,
            ):
                source_party.share_histogram_routing_state_with(target_party)

    validation_probability = model.predict_proba([validation_active, *validation_passive])[:, 1]
    threshold = select_f1_threshold(validation_active.labels, validation_probability)
    inference_start = time.perf_counter()
    test_probability = model.predict_proba([test_active, *test_passive])[:, 1]
    inference_seconds = time.perf_counter() - inference_start
    train_probability = model.predict_proba([train_active, *train_passive])[:, 1]
    metrics = binary_metrics(test_active.labels, test_probability, threshold=threshold)
    membership_auc, membership_advantage = _membership(train_probability, test_probability)
    if model_name == "logistic":
        privacy_surface = (
            "raw features stay party-local; passive parties receive residual-derived "
            "signals and send local logits"
        )
    else:
        privacy_surface = (
            "raw features and numeric split thresholds stay party-local; parties receive "
            "target-derived gradient/Hessian signals and routing information is exposed"
        )
    return {
        "protocol": "vfl_logistic" if model_name == "logistic" else "vfl_hist_gbdt",
        "reference_model": model_name,
        "federated": True,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "f1": metrics["f1"],
        "brier": metrics["brier"],
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "peak_rss_bytes": memory.peak_rss_bytes,
        "peak_rss_method": "sampled_process_rss_5ms",
        "estimated_communication_bytes": int(model.transport.estimated_payload_bytes),
        "communication_message_count": int(communication["message_count"]),
        "overlap_50_roc_auc": np.nan,
        "overlap_status": "not measured in core run",
        "worst_inference_dropout_roc_auc": np.nan,
        "dropout_status": "not measured in core run",
        "membership_attack_auc": membership_auc,
        "membership_attack_advantage": membership_advantage,
        "privacy_surface": privacy_surface,
        "formal_privacy_guarantee": "none by default",
    }


def run_formal_benchmark(
    *,
    rows: int = 1200,
    seed: int = 42,
    include_robustness: bool = True,
    logistic_max_iter: int = 350,
    gbdt_estimators: int = 12,
    output: Path = Path("benchmarks/formal_comparison.csv"),
) -> pd.DataFrame:
    """Create a directly comparable four-protocol research benchmark table."""
    if rows < 200:
        raise ValueError("formal benchmark requires at least 200 rows")
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    validation_active, validation_passive = slice_parties(active, passive, split.validation)
    test_active, test_passive = slice_parties(active, passive, split.test)

    common = dict(
        train_active=train_active,
        train_passive=train_passive,
        validation_active=validation_active,
        validation_passive=validation_passive,
        test_active=test_active,
        test_passive=test_passive,
        seed=seed,
    )
    records = [
        _non_federated_row(
            protocol="centralized_all_features",
            all_features=True,
            **common,
        ),
        _non_federated_row(
            protocol="single_party_bank",
            all_features=False,
            **common,
        ),
        _vfl_row(
            model_name="logistic",
            logistic_max_iter=logistic_max_iter,
            gbdt_estimators=gbdt_estimators,
            **common,
        ),
        _vfl_row(
            model_name="vfl-hist-gbdt",
            logistic_max_iter=logistic_max_iter,
            gbdt_estimators=gbdt_estimators,
            **common,
        ),
    ]
    frame = pd.DataFrame(records)
    frame.insert(1, "rows", rows)
    frame.insert(2, "seed", seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    if include_robustness:
        for model_name, protocol in (
            ("logistic", "vfl_logistic"),
            ("vfl-hist-gbdt", "vfl_hist_gbdt"),
        ):
            overlap = run_overlap_study(
                rows=max(rows, 400),
                seed=seed,
                model_name=model_name,
                fractions=(0.5,),
                methods=("availability_indicator",),
                output=output.parent / f"robustness_overlap_{model_name}.csv",
                write_run=False,
            )
            mask = frame["protocol"] == protocol
            frame.loc[mask, "overlap_50_roc_auc"] = float(overlap.iloc[0]["roc_auc"])
            frame.loc[mask, "overlap_status"] = "measured: availability_indicator at 50% overlap"
        dropout = run_dropout_study(
            rows=max(rows, 400),
            seed=seed,
            max_iter=logistic_max_iter,
            output=output.parent / "robustness_dropout_logistic.csv",
            write_run=False,
        )
        inference = dropout.loc[dropout["phase"] == "inference_time_dropout", "roc_auc"]
        if not inference.empty:
            mask = frame["protocol"] == "vfl_logistic"
            frame.loc[mask, "worst_inference_dropout_roc_auc"] = float(inference.min())
            frame.loc[mask, "dropout_status"] = "measured over documented inference scenarios"
        frame.loc[frame["protocol"] == "vfl_hist_gbdt", "dropout_status"] = (
            "trained-party omission is not supported; retrain/evaluate availability separately"
        )

    frame.to_csv(output, index=False)
    output.with_suffix(".md").write_text(
        "# VertiMosaic formal benchmark matrix\n\n"
        "Centralized rows are explicitly non-federated research baselines. "
        "Communication values are protocol payload accounting. Peak RSS is sampled "
        "throughout training rather than inferred from before/after snapshots. "
        "Membership leakage is a simple empirical attack baseline.\n\n" + _markdown_table(frame),
        encoding="utf-8",
    )
    metadata = {
        "rows": rows,
        "seed": seed,
        "include_robustness": include_robustness,
        "logistic_max_iter": logistic_max_iter,
        "gbdt_estimators": gbdt_estimators,
        "memory_measurement": "process RSS sampled every 5ms during model fitting",
        "claims": {
            "centralized_rows_are_federated": False,
            "communication_is_packet_capture": False,
            "privacy_measurement_is_formal_proof": False,
        },
        "records": frame.replace({np.nan: None}).to_dict(orient="records"),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the VertiMosaic formal comparison matrix.")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/formal_comparison.csv"))
    parser.add_argument("--no-robustness", action="store_true")
    parser.add_argument("--logistic-max-iter", type=int, default=350)
    parser.add_argument("--gbdt-estimators", type=int, default=12)
    args = parser.parse_args()
    frame = run_formal_benchmark(
        rows=args.rows,
        seed=args.seed,
        include_robustness=not args.no_robustness,
        logistic_max_iter=args.logistic_max_iter,
        gbdt_estimators=args.gbdt_estimators,
        output=args.output,
    )
    print(frame.to_csv(index=False))


if __name__ == "__main__":
    main()
