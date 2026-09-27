from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import binary_metrics, entity_level_split
from vertimosaic.experiments.pipeline import slice_parties
from vertimosaic.experiments.robustness import (
    apply_categorical_frequency_drift,
    apply_numeric_drift,
    dropout_scenarios,
)
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty

_PASSIVE_NAMES = ("telecom", "insurance", "retail")


def _model(name: str, seed: int) -> VFLLogisticRegression | VFLHistGBDT:
    if name == "logistic":
        return VFLLogisticRegression(learning_rate=0.08, max_iter=350, l2=1e-3, seed=seed)
    if name == "vfl-hist-gbdt":
        return VFLHistGBDT(
            n_estimators=12,
            max_depth=2,
            min_samples_leaf=15,
            seed=seed,
        )
    raise ValueError(f"unknown model: {name}")


def _passive_map(items: list[PassiveParty]) -> dict[str, PassiveParty]:
    return {item.name: item for item in items}


def run_ablation_study(
    *,
    rows: int = 1200,
    seed: int = 42,
    model_name: str = "logistic",
    output: Path = Path("results/party_ablation.csv"),
) -> pd.DataFrame:
    """Run all scientifically valid Bank-plus-passive party subsets under VFL training."""
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    test_active, test_passive = slice_parties(active, passive, split.test)
    train_map = _passive_map(train_passive)
    test_map = _passive_map(test_passive)
    records: list[dict[str, Any]] = []
    for subset_size in range(len(_PASSIVE_NAMES) + 1):
        for subset in combinations(_PASSIVE_NAMES, subset_size):
            model = _model(model_name, seed)
            selected_train = [train_map[name] for name in subset]
            selected_test = [test_map[name] for name in subset]
            start = time.perf_counter()
            model.fit(train_active, selected_train)
            training_seconds = time.perf_counter() - start
            probability = model.predict_proba([test_active, *selected_test])[:, 1]
            metrics = binary_metrics(test_active.labels, probability)
            records.append(
                {
                    "model": model_name,
                    "parties": "+".join(("bank", *subset)),
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "f1": metrics["f1"],
                    "log_loss": metrics["log_loss"],
                    "brier": metrics["brier"],
                    "ece": metrics["ece"],
                    "training_seconds": training_seconds,
                    "estimated_communication_bytes": model.transport.estimated_payload_bytes,
                }
            )
    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def run_overlap_study(
    *,
    rows: int = 2000,
    seed: int = 42,
    model_name: str = "logistic",
    fractions: tuple[float, ...] = (1.0, 0.9, 0.75, 0.5, 0.25),
    output: Path = Path("results/partial_overlap.csv"),
) -> pd.DataFrame:
    """Evaluate explicit intersection-only performance over controlled entity overlap."""
    active, passive = make_vertical_synthetic(rows, seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(rows)
    records: list[dict[str, Any]] = []
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("overlap fractions must be in (0, 1]")
        count = max(40, int(round(rows * fraction)))
        common = np.sort(order[: min(count, rows)])
        subset_active = ActiveParty("bank", active._x[common], active.labels[common])
        subset_passive = [PassiveParty(item.name, item._x[common]) for item in passive]
        split = entity_level_split(subset_active.labels, seed=seed)
        train_active, train_passive = slice_parties(subset_active, subset_passive, split.train)
        test_active, test_passive = slice_parties(subset_active, subset_passive, split.test)
        model = _model(model_name, seed)
        start = time.perf_counter()
        model.fit(train_active, train_passive)
        training_seconds = time.perf_counter() - start
        probability = model.predict_proba([test_active, *test_passive])[:, 1]
        metrics = binary_metrics(test_active.labels, probability)
        records.append(
            {
                "overlap_fraction": fraction,
                "coverage": len(common) / rows,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1": metrics["f1"],
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "training_seconds": training_seconds,
                "estimated_communication_bytes": model.transport.estimated_payload_bytes,
            }
        )
    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def _dropout_metrics_record(
    *,
    phase: str,
    scenario: str,
    dropped: tuple[str, ...],
    model: VFLLogisticRegression,
    test_active: ActiveParty,
    test_parties: list[PassiveParty],
    training_seconds: float,
) -> dict[str, Any]:
    probability = model.predict_proba([test_active, *test_parties])[:, 1]
    metrics = binary_metrics(test_active.labels, probability)
    return {
        "phase": phase,
        "scenario": scenario,
        "dropped_parties": "+".join(dropped),
        "available_parties": "+".join(("bank", *[party.name for party in test_parties])),
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "f1": metrics["f1"],
        "log_loss": metrics["log_loss"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
        "training_seconds": training_seconds,
        "estimated_communication_bytes": model.transport.estimated_payload_bytes,
    }


def run_dropout_study(
    *,
    rows: int = 1600,
    seed: int = 42,
    max_iter: int = 350,
    output: Path = Path("results/party_dropout.csv"),
) -> pd.DataFrame:
    """Evaluate inference-time dropout and training-time party availability differences."""
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    test_active, test_passive = slice_parties(active, passive, split.test)
    train_map = _passive_map(train_passive)
    test_map = _passive_map(test_passive)

    full_model = VFLLogisticRegression(
        learning_rate=0.08,
        max_iter=max_iter,
        l2=1e-3,
        seed=seed,
    )
    full_start = time.perf_counter()
    full_model.fit(train_active, train_passive)
    full_training_seconds = time.perf_counter() - full_start

    records: list[dict[str, Any]] = []
    for scenario, dropped in dropout_scenarios().items():
        available_names = [name for name in _PASSIVE_NAMES if name not in dropped]
        inference_parties = [test_map[name] for name in available_names]
        records.append(
            _dropout_metrics_record(
                phase="inference_time_dropout",
                scenario=scenario,
                dropped=dropped,
                model=full_model,
                test_active=test_active,
                test_parties=inference_parties,
                training_seconds=full_training_seconds,
            )
        )

        availability_model = VFLLogisticRegression(
            learning_rate=0.08,
            max_iter=max_iter,
            l2=1e-3,
            seed=seed,
        )
        available_train = [train_map[name] for name in available_names]
        training_start = time.perf_counter()
        availability_model.fit(train_active, available_train)
        training_seconds = time.perf_counter() - training_start
        records.append(
            _dropout_metrics_record(
                phase="training_and_inference_availability",
                scenario=scenario,
                dropped=dropped,
                model=availability_model,
                test_active=test_active,
                test_parties=inference_parties,
                training_seconds=training_seconds,
            )
        )

    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def _categorical_shifted_matrix(values: np.ndarray, *, seed: int) -> np.ndarray:
    """Discretize one synthetic feature, shift its frequencies, and retain numeric codes."""
    out = values.copy()
    feature = out[:, 0]
    quantiles = np.quantile(feature, [1.0 / 3.0, 2.0 / 3.0])
    categories = np.digitize(feature, quantiles).astype(object)
    shifted = apply_categorical_frequency_drift(categories, strength=0.55, seed=seed)
    out[:, 0] = np.asarray(shifted, dtype=float)
    return out


def _apply_drift_scenario(
    values: np.ndarray,
    scenario: dict[str, Any],
    *,
    seed: int,
) -> np.ndarray:
    if bool(scenario.get("categorical_frequency_shift", False)):
        return _categorical_shifted_matrix(values, seed=seed)
    shifted = apply_numeric_drift(
        values,
        mean_shift=float(scenario.get("mean_shift", 0.0)),
        variance_scale=float(scenario.get("variance_scale", 1.0)),
        missingness_increase=float(scenario.get("missingness_increase", 0.0)),
        seed=seed,
    )
    return np.nan_to_num(shifted, nan=0.0)


def run_drift_study(
    *,
    rows: int = 1600,
    seed: int = 42,
    output: Path = Path("results/feature_drift.csv"),
) -> pd.DataFrame:
    """Evaluate mean, variance, missingness, and categorical-frequency shifts."""
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    test_active, test_passive = slice_parties(active, passive, split.test)
    model = VFLLogisticRegression(learning_rate=0.08, max_iter=350, l2=1e-3, seed=seed)
    model.fit(train_active, train_passive)
    scenarios: list[dict[str, Any]] = [
        {"scenario": "baseline", "party": None},
        {"scenario": "telecom_mean_shift", "party": "telecom", "mean_shift": 0.75},
        {
            "scenario": "insurance_variance_shift",
            "party": "insurance",
            "variance_scale": 1.5,
        },
        {
            "scenario": "retail_missingness_shift",
            "party": "retail",
            "missingness_increase": 0.20,
        },
        {
            "scenario": "retail_categorical_frequency_shift",
            "party": "retail",
            "categorical_frequency_shift": True,
        },
        {
            "scenario": "bank_repayment_proxy_shift",
            "party": "bank",
            "mean_shift": 0.5,
            "variance_scale": 1.2,
        },
    ]
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        party_name = scenario.get("party")
        active_eval = ActiveParty(
            test_active.name,
            test_active._x.copy(),
            test_active.labels.copy(),
        )
        passive_eval = [PassiveParty(item.name, item._x.copy()) for item in test_passive]
        if party_name == "bank":
            shifted = _apply_drift_scenario(active_eval._x, scenario, seed=seed)
            active_eval = ActiveParty("bank", shifted, active_eval.labels)
        elif isinstance(party_name, str):
            passive_eval = [
                PassiveParty(
                    item.name,
                    _apply_drift_scenario(item._x, scenario, seed=seed)
                    if item.name == party_name
                    else item._x,
                )
                for item in passive_eval
            ]
        probability = model.predict_proba([active_eval, *passive_eval])[:, 1]
        metrics = binary_metrics(active_eval.labels, probability)
        records.append(
            {
                "scenario": scenario["scenario"],
                "party": party_name or "none",
                "mean_shift": float(scenario.get("mean_shift", 0.0)),
                "variance_scale": float(scenario.get("variance_scale", 1.0)),
                "missingness_increase": float(scenario.get("missingness_increase", 0.0)),
                "categorical_frequency_shift": bool(
                    scenario.get("categorical_frequency_shift", False)
                ),
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1": metrics["f1"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
            }
        )
    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
