from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vertimosaic.baselines import fit_centralized_baseline
from vertimosaic.evaluation import (
    binary_metrics,
    bootstrap_confidence_intervals,
    confusion_at_threshold,
    entity_level_split,
    paired_bootstrap_difference,
    select_f1_threshold,
)
from vertimosaic.experiments.external import linkage_manifest_dict, prepare_external_benchmark
from vertimosaic.experiments.external_preprocessing import prepare_external_splits_locally
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.reporting import select_sanitized_case_study


def _external_model(
    model_name: str,
    seed: int,
    *,
    early_stopping: bool,
) -> VFLLogisticRegression | VFLHistGBDT:
    if model_name == "logistic":
        return VFLLogisticRegression(
            learning_rate=0.08,
            max_iter=500,
            l2=1e-3,
            early_stopping_rounds=5 if early_stopping else None,
            seed=seed,
        )
    if model_name == "vfl-hist-gbdt":
        return VFLHistGBDT(
            n_estimators=20,
            max_depth=3,
            min_samples_leaf=20,
            early_stopping_rounds=3 if early_stopping else None,
            seed=seed,
        )
    raise ValueError(f"unknown model: {model_name}")


def _fit_external_model(
    model: VFLLogisticRegression | VFLHistGBDT,
    train_active: ActiveParty,
    train_passive: list[PassiveParty],
    validation_active: ActiveParty,
    validation_passive: list[PassiveParty],
) -> None:
    model.fit(train_active, train_passive, validation_active, validation_passive)


def _party_permuted_probabilities(
    model: VFLLogisticRegression | VFLHistGBDT,
    active: ActiveParty,
    passive: list[PassiveParty],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    all_names = [active.name, *[party.name for party in passive]]
    for party_index, party_name in enumerate(all_names):
        rng = np.random.default_rng(seed + 500 + party_index)
        permutation = rng.permutation(active.n_rows)
        permuted_active = ActiveParty(active.name, active._x.copy(), active.labels)
        permuted_passive = [PassiveParty(item.name, item._x.copy()) for item in passive]
        if party_name == active.name:
            permuted_active = ActiveParty(active.name, active._x[permutation], active.labels)
        else:
            permuted_passive = [
                PassiveParty(
                    item.name,
                    item._x[permutation] if item.name == party_name else item._x,
                )
                for item in passive
            ]
        output[party_name] = model.predict_proba([permuted_active, *permuted_passive])[:, 1]
    return output


def run_external_experiment(
    *,
    mode: str = "observed_target_external",
    model_name: str = "vfl-hist-gbdt",
    seed: int = 42,
    cross_party_correlation: float = 0.25,
    insurance_sample_size: int | None = None,
    bootstrap_replicates: int = 100,
    output: Path | None = None,
) -> dict[str, Any]:
    preparation_start = time.perf_counter()
    benchmark = prepare_external_benchmark(
        mode=mode,
        cross_party_correlation=cross_party_correlation,
        seed=seed,
        insurance_sample_size=insurance_sample_size,
    )
    benchmark_preparation_seconds = time.perf_counter() - preparation_start
    split = entity_level_split(benchmark.active.labels, seed=seed)
    prepared = prepare_external_splits_locally(benchmark, split)
    train_active = prepared.train_active
    train_passive = prepared.train_passive
    val_active = prepared.validation_active
    val_passive = prepared.validation_passive
    test_active = prepared.test_active
    test_passive = prepared.test_passive

    model = _external_model(model_name, seed, early_stopping=True)
    start = time.perf_counter()
    _fit_external_model(model, train_active, train_passive, val_active, val_passive)
    training_seconds = time.perf_counter() - start
    validation_p = model.predict_proba([val_active, *val_passive])[:, 1]
    threshold = select_f1_threshold(val_active.labels, validation_p)
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
    centralized_name = "logistic" if model_name == "logistic" else "hist-gbdt"
    bank_baseline = fit_centralized_baseline(
        [train_active._x],
        train_active.labels,
        [test_active._x],
        model=centralized_name,
        seed=seed,
    )
    all_party_baseline = fit_centralized_baseline(
        [train_active._x, *[item._x for item in train_passive]],
        train_active.labels,
        [test_active._x, *[item._x for item in test_passive]],
        model=centralized_name,
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
    payload: dict[str, Any] = {
        "benchmark_description": "externally grounded semi-synthetic cross-industry VFL benchmark",
        "mode": mode,
        "target_description": "observed Bank default target"
        if mode == "observed_target_external"
        else "semi_synthetic_cross_industry_risk",
        "model": model_name,
        "seed": seed,
        "metrics": metrics,
        "confidence_intervals": intervals,
        "comparisons": [bank_comparison, all_party_comparison],
        "confusion_matrix": confusion_at_threshold(test_active.labels, test_p, threshold),
        "threshold_selected_on_validation": threshold,
        "benchmark_preparation_seconds": benchmark_preparation_seconds,
        "preprocessing_seconds": prepared.preprocessing_seconds,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "estimated_communication_bytes": model.transport.estimated_payload_bytes,
        "preprocessor_artifacts": prepared.preprocessor_paths,
        "preprocessing_fit_scope": "TRAIN only, independently per party",
        "linkage": linkage_manifest_dict(benchmark),
        "four_sources_same_real_people": False,
    }
    if mode == "distributed_signal_external":
        bank_model = _external_model(model_name, seed + 1, early_stopping=True)
        _fit_external_model(bank_model, train_active, [], val_active, [])
        bank_only_p = bank_model.predict_proba([test_active])[:, 1]
        party_permuted = _party_permuted_probabilities(
            model,
            test_active,
            test_passive,
            seed=seed,
        )
        payload["case_study"] = select_sanitized_case_study(
            split.test,
            bank_only_p,
            test_p,
            party_permuted,
            threshold=threshold,
        )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
