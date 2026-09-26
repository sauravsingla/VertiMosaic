from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from vertimosaic.baselines import fit_centralized_baseline
from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import (
    binary_metrics,
    bootstrap_confidence_intervals,
    communication_event_frame,
    communication_totals,
    confusion_at_threshold,
    entity_level_split,
    paired_bootstrap_difference,
    select_f1_threshold,
)
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.provenance import FeatureProvenance
from vertimosaic.reproducibility import RunArtifacts


def slice_parties(
    active: ActiveParty, passive: list[PassiveParty], indices: np.ndarray
) -> tuple[ActiveParty, list[PassiveParty]]:
    return (
        ActiveParty(active.name, active._x[indices], active.labels[indices]),
        [PassiveParty(item.name, item._x[indices]) for item in passive],
    )


def _make_model(model_name: str, seed: int) -> VFLLogisticRegression | VFLHistGBDT:
    if model_name == "logistic":
        return VFLLogisticRegression(
            learning_rate=0.08,
            max_iter=500,
            l2=1e-3,
            seed=seed,
        )
    if model_name == "vfl-hist-gbdt":
        return VFLHistGBDT(
            n_estimators=20,
            max_depth=3,
            min_samples_leaf=20,
            early_stopping_rounds=3,
            seed=seed,
        )
    raise ValueError(f"unknown model: {model_name}")


def _communication_frame(model: VFLLogisticRegression | VFLHistGBDT) -> pd.DataFrame:
    return communication_event_frame(model.transport.audit_log)


def _training_frame(model: VFLLogisticRegression | VFLHistGBDT) -> pd.DataFrame:
    if isinstance(model, VFLLogisticRegression):
        return pd.DataFrame(
            {"iteration": np.arange(len(model.loss_history_)), "loss": model.loss_history_}
        )
    frame = pd.DataFrame(
        {
            "tree": np.arange(len(model.training_loss_history_)),
            "training_loss": model.training_loss_history_,
        }
    )
    if model.validation_loss_history_:
        frame["validation_loss"] = model.validation_loss_history_[: len(frame)]
    return frame


def _synthetic_provenance(active: ActiveParty, passive: list[PassiveParty]) -> pd.DataFrame:
    records: list[FeatureProvenance] = []
    for party in [active, *passive]:
        for index in range(party.n_features):
            records.append(
                FeatureProvenance(
                    party=party.name,
                    feature=f"feature_{index}",
                    external_dataset="synthetic_scale",
                    source_column=f"synthetic_column_{index}",
                    transformation="controlled synthetic generator",
                    source_type="fully_synthetic",
                    observed_or_derived="derived",
                    semi_synthetic=False,
                )
            )
    return pd.DataFrame([asdict(item) for item in records])


def run_synthetic_experiment(
    *,
    rows: int = 2000,
    seed: int = 42,
    model_name: str = "logistic",
    bootstrap_replicates: int = 100,
    write_run: bool = False,
    runs_root: Path = Path("runs"),
) -> dict[str, Any]:
    preparation_start = time.perf_counter()
    active, passive = make_vertical_synthetic(rows, seed)
    data_preparation_seconds = time.perf_counter() - preparation_start
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    val_active, val_passive = slice_parties(active, passive, split.validation)
    test_active, test_passive = slice_parties(active, passive, split.test)
    model = _make_model(model_name, seed)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = time.perf_counter()
    if isinstance(model, VFLHistGBDT):
        model.fit(train_active, train_passive, val_active, val_passive)
    else:
        model.fit(train_active, train_passive)
    training_seconds = time.perf_counter() - start
    peak_rss_bytes = max(rss_before, process.memory_info().rss)
    val_p = model.predict_proba([val_active, *val_passive])[:, 1]
    threshold = select_f1_threshold(val_active.labels, val_p)
    inference_start = time.perf_counter()
    test_p = model.predict_proba([test_active, *test_passive])[:, 1]
    inference_seconds = time.perf_counter() - inference_start
    metrics = binary_metrics(test_active.labels, test_p, threshold=threshold)
    intervals = bootstrap_confidence_intervals(
        test_active.labels,
        test_p,
        threshold=threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    baseline_name = "logistic" if model_name == "logistic" else "hist-gbdt"
    bank_baseline = fit_centralized_baseline(
        [train_active._x],
        train_active.labels,
        [test_active._x],
        model=baseline_name,
        seed=seed,
    )
    all_party_baseline = fit_centralized_baseline(
        [train_active._x, *[party._x for party in train_passive]],
        train_active.labels,
        [test_active._x, *[party._x for party in test_passive]],
        model=baseline_name,
        seed=seed,
    )
    bank_comparison = paired_bootstrap_difference(
        test_active.labels,
        test_p,
        bank_baseline.probabilities,
        metric="roc_auc",
        threshold=threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    bank_comparison["baseline"] = "bank_only_non_federated"
    all_party_comparison = paired_bootstrap_difference(
        test_active.labels,
        test_p,
        all_party_baseline.probabilities,
        metric="roc_auc",
        threshold=threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    all_party_comparison["baseline"] = "centralized_all_party_non_federated"
    communication = communication_totals(model.transport.audit_log)
    training_steps = (
        model.n_iter_ if isinstance(model, VFLLogisticRegression) else len(model.trees_)
    )
    payload: dict[str, Any] = {
        "mode": "synthetic_scale",
        "model": model_name,
        "rows": rows,
        "seed": seed,
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": confusion_at_threshold(test_active.labels, test_p, threshold),
        "confidence_intervals": intervals,
        "comparisons": [bank_comparison, all_party_comparison],
        "data_preparation_seconds": data_preparation_seconds,
        "entity_alignment_seconds": None,
        "preprocessing_seconds": None,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "training_steps": training_steps,
        "peak_rss_bytes": int(peak_rss_bytes),
        "estimated_communication_bytes": int(model.transport.estimated_payload_bytes),
        "communication": communication,
    }
    if write_run:
        run = RunArtifacts.create(root=runs_root)
        predictions = pd.DataFrame(
            {
                "row_index": split.test,
                "target": test_active.labels,
                "probability": test_p,
            }
        )
        directory = run.finalize(
            config={
                "mode": "synthetic_scale",
                "model": model_name,
                "rows": rows,
                "seed": seed,
                "bootstrap_replicates": bootstrap_replicates,
            },
            seed=seed,
            dataset_provenance={"source_type": "fully_synthetic", "rows": rows},
            linkage_manifest={"method": "shared synthetic latent entities", "target_blind": True},
            metrics=payload,
            predictions=predictions,
            training_history=_training_frame(model),
            communication=_communication_frame(model),
            feature_provenance=_synthetic_provenance(active, passive),
        )
        payload["run_directory"] = str(directory)
    return payload
