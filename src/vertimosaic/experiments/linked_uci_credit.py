# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from ucimlrepo import fetch_ucirepo

from vertimosaic.datasets import DatasetRegistry
from vertimosaic.evaluation import (
    binary_metrics,
    bootstrap_confidence_intervals,
    communication_totals,
    entity_level_split,
    select_f1_threshold,
)
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.reproducibility import environment_snapshot


def _frame_sha256(frame: pd.DataFrame) -> str:
    digest = sha256()
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    digest.update(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    hashed = pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy(
        dtype=np.uint64
    )
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_transform_train_only(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_numeric = train.apply(pd.to_numeric, errors="coerce")
    validation_numeric = validation.apply(pd.to_numeric, errors="coerce")
    test_numeric = test.apply(pd.to_numeric, errors="coerce")
    medians = train_numeric.median(axis=0).fillna(0.0)
    train_filled = train_numeric.fillna(medians)
    means = train_filled.mean(axis=0)
    std = train_filled.std(axis=0).replace(0.0, 1.0).fillna(1.0)

    def transform(frame: pd.DataFrame) -> np.ndarray:
        values = frame.fillna(medians)
        values = (values - means) / std
        return values.to_numpy(dtype=float)

    metadata = {
        "columns": [str(column) for column in train.columns],
        "median_fit_scope": "training rows only",
        "scaling_fit_scope": "training rows only",
    }
    return (
        transform(train_numeric),
        transform(validation_numeric),
        transform(test_numeric),
        metadata,
    )


def prepare_exact_row_vertical_partitions(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Create two disjoint source-column parties over the exact same source rows."""
    if len(features) != len(target):
        raise ValueError("features and target must contain the same source rows")
    if features.shape[1] < 2:
        raise ValueError("at least two source feature columns are required")
    target_numeric = pd.to_numeric(target, errors="coerce")
    valid = target_numeric.notna()
    features = features.loc[valid].reset_index(drop=True)
    target_numeric = target_numeric.loc[valid].astype(float).reset_index(drop=True)
    if not set(np.unique(target_numeric)).issubset({0.0, 1.0}):
        raise ValueError("linked benchmark requires a binary 0/1 target")

    midpoint = max(1, features.shape[1] // 2)
    active_columns = list(features.columns[:midpoint])
    passive_columns = list(features.columns[midpoint:])
    if not passive_columns:
        raise ValueError("column partition produced an empty passive party")
    if set(active_columns) & set(passive_columns):
        raise RuntimeError("source-column partitions must be disjoint")

    split = entity_level_split(target_numeric.to_numpy(), seed=seed)
    active_train, active_validation, active_test, active_preprocess = _fit_transform_train_only(
        features.loc[split.train, active_columns],
        features.loc[split.validation, active_columns],
        features.loc[split.test, active_columns],
    )
    passive_train, passive_validation, passive_test, passive_preprocess = (
        _fit_transform_train_only(
            features.loc[split.train, passive_columns],
            features.loc[split.validation, passive_columns],
            features.loc[split.test, passive_columns],
        )
    )
    y = target_numeric.to_numpy(dtype=float)
    return {
        "train_active": ActiveParty("credit_active", active_train, y[split.train]),
        "train_passive": [PassiveParty("credit_passive", passive_train)],
        "validation_active": ActiveParty(
            "credit_active", active_validation, y[split.validation]
        ),
        "validation_passive": [PassiveParty("credit_passive", passive_validation)],
        "test_active": ActiveParty("credit_active", active_test, y[split.test]),
        "test_passive": [PassiveParty("credit_passive", passive_test)],
        "test_source_rows": split.test,
        "active_columns": [str(column) for column in active_columns],
        "passive_columns": [str(column) for column in passive_columns],
        "preprocessing": {
            "active": active_preprocess,
            "passive": passive_preprocess,
        },
    }


def run_uci_credit_linked_experiment(
    *,
    model_name: str = "logistic",
    seed: int = 42,
    bootstrap_replicates: int = 1000,
    output: Path = Path("reports/uci_credit_exact_linked.json"),
) -> dict[str, Any]:
    """Run a public exact-row linked VFL sanity benchmark on UCI dataset 350.

    This is a single-source exact vertical partition. It proves same-entity row
    alignment without claiming that the two partitions came from independent
    organizations.
    """
    data = fetch_ucirepo(id=350)
    features = data.data.features.copy()
    targets = data.data.targets
    if targets is None or targets.empty:
        raise RuntimeError("UCI dataset 350 did not provide its binary target")
    target = targets.iloc[:, 0]
    original = getattr(data.data, "original", None)
    source_frame = original.copy() if original is not None else pd.concat(
        [features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1
    )
    prepared = prepare_exact_row_vertical_partitions(features, target, seed=seed)
    train_active: ActiveParty = prepared["train_active"]
    train_passive: list[PassiveParty] = prepared["train_passive"]
    validation_active: ActiveParty = prepared["validation_active"]
    validation_passive: list[PassiveParty] = prepared["validation_passive"]
    test_active: ActiveParty = prepared["test_active"]
    test_passive: list[PassiveParty] = prepared["test_passive"]

    if model_name == "logistic":
        model: VFLLogisticRegression | VFLHistGBDT = VFLLogisticRegression(
            learning_rate=0.08,
            max_iter=500,
            l2=1e-3,
            early_stopping_rounds=5,
            seed=seed,
        )
    elif model_name == "vfl-hist-gbdt":
        model = VFLHistGBDT(
            n_estimators=20,
            max_depth=3,
            min_samples_leaf=20,
            early_stopping_rounds=3,
            seed=seed,
        )
    else:
        raise ValueError("model_name must be logistic or vfl-hist-gbdt")

    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = time.perf_counter()
    model.fit(train_active, train_passive, validation_active, validation_passive)
    training_seconds = time.perf_counter() - start
    peak_rss_bytes = int(max(rss_before, process.memory_info().rss))
    communication = communication_totals(model.transport.audit_log)

    if isinstance(model, VFLHistGBDT):
        for evaluation_parties in (
            [validation_active, *validation_passive],
            [test_active, *test_passive],
        ):
            for source_party, target_party in zip(
                [train_active, *train_passive], evaluation_parties, strict=True
            ):
                source_party.share_histogram_routing_state_with(target_party)

    validation_probability = model.predict_proba([validation_active, *validation_passive])[:, 1]
    threshold = select_f1_threshold(validation_active.labels, validation_probability)
    inference_start = time.perf_counter()
    probability = model.predict_proba([test_active, *test_passive])[:, 1]
    inference_seconds = time.perf_counter() - inference_start
    metrics = binary_metrics(test_active.labels, probability, threshold=threshold)
    intervals = bootstrap_confidence_intervals(
        test_active.labels,
        probability,
        threshold=threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )

    registry = DatasetRegistry().get("bank")
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = output.with_name(output.stem + "_predictions.csv")
    pseudonyms = [
        sha256(f"uci350:{int(index)}".encode()).hexdigest()[:20]
        for index in prepared["test_source_rows"]
    ]
    pd.DataFrame(
        {
            "entity_id": pseudonyms,
            "target": test_active.labels.astype(int),
            "probability": probability,
        }
    ).to_csv(predictions_path, index=False)

    payload = {
        "benchmark": "uci_credit_exact_row_linked",
        "model": model_name,
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "source": {
            "provider": registry.provider,
            "dataset_name": registry.dataset_name,
            "dataset_id": registry.dataset_id,
            "doi": registry.doi,
            "license": registry.license,
            "provider_url": registry.provider_url,
            "source_sha256": _frame_sha256(source_frame),
            "raw_rows": len(source_frame),
        },
        "linkage": {
            "kind": "exact_source_row_vertical_partition",
            "same_real_entities_across_parties": True,
            "single_source_dataset": True,
            "cross_organization_claim": False,
            "semi_synthetic": False,
            "source_column_overlap": False,
            "entity_alignment": (
                "exact row identity before deterministic train/validation/test split"
            ),
        },
        "partitions": {
            "active_columns": prepared["active_columns"],
            "passive_columns": prepared["passive_columns"],
            "preprocessing": prepared["preprocessing"],
        },
        "metrics": metrics,
        "confidence_intervals": intervals,
        "threshold_selected_on_validation": threshold,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "estimated_communication_bytes": int(model.transport.estimated_payload_bytes),
        "communication": communication,
        "environment": environment_snapshot(seed),
        "predictions_file": predictions_path.name,
        "claim_boundary": (
            "This benchmark demonstrates exact same-entity vertical partitioning on a public "
            "single-source dataset. It does not demonstrate cross-organization record linkage."
        ),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = output.with_suffix(output.suffix + ".sha256")
    manifest_path.write_text(
        f"{_file_sha256(output)}  {output.name}\n"
        f"{_file_sha256(predictions_path)}  {predictions_path.name}\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the exact-row linked public UCI credit VFL sanity benchmark."
    )
    parser.add_argument("--model", choices=["logistic", "vfl-hist-gbdt"], default="logistic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/uci_credit_exact_linked.json")
    )
    args = parser.parse_args()
    payload = run_uci_credit_linked_experiment(
        model_name=args.model,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        output=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
