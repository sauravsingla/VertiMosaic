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


def _make_model(model_name: str) -> VFLLogisticRegression | VFLHistGBDT:
    if model_name == "logistic":
        return VFLLogisticRegression(learning_rate=0.08, max_iter=500, l2=1e-3)
    if model_name == "vfl-hist-gbdt":
        return VFLHistGBDT(n_estimators=20, max_depth=3, min_samples_leaf=20)
    raise ValueError(f"unknown model: {model_name}")


def _communication_frame(model: VFLLogisticRegression | VFLHistGBDT) -> pd.DataFrame:
    return pd.DataFrame([asdict(event) for event in model.transport.audit_log])


def _training_frame(model: VFLLogisticRegression | VFLHistGBDT) -> pd.DataFrame:
    if isinstance(model, VFLLogisticRegression):
        return pd.DataFrame(
            {"iteration": np.arange(len(model.loss_history_)), "loss": model.loss_history_}
        )
    return pd.DataFrame({"tree": np.arange(len(model.trees_))})


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
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    val_active, val_passive = slice_parties(active, passive, split.validation)
    test_active, test_passive = slice_parties(active, passive, split.test)
    model = _make_model(model_name)
    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = time.perf_counter()
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
    baseline = fit_centralized_baseline(
        [train_active._x],
        train_active.labels,
        [test_active._x],
        model=baseline_name,
        seed=seed,
    )
    comparison = paired_bootstrap_difference(
        test_active.labels,
        test_p,
        baseline.probabilities,
        metric="roc_auc",
        threshold=threshold,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    payload: dict[str, Any] = {
        "mode": "synthetic_scale",
        "model": model_name,
        "rows": rows,
        "seed": seed,
        "threshold": threshold,
        "metrics": metrics,
        "confidence_intervals": intervals,
        "comparisons": [comparison],
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "peak_rss_bytes": int(peak_rss_bytes),
        "estimated_communication_bytes": int(model.transport.estimated_payload_bytes),
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
