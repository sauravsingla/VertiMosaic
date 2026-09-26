from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import binary_metrics, entity_level_split
from vertimosaic.experiments.contribution import exact_shapley_party_utility
from vertimosaic.experiments.pipeline import slice_parties
from vertimosaic.experiments.robustness import apply_numeric_drift, dropout_scenarios
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty

_PASSIVE_NAMES = ("telecom", "insurance", "retail")
_ALL_NAMES = ("bank", *_PASSIVE_NAMES)


def _model(name: str) -> VFLLogisticRegression | VFLHistGBDT:
    if name == "logistic":
        return VFLLogisticRegression(learning_rate=0.08, max_iter=350, l2=1e-3)
    if name == "vfl-hist-gbdt":
        return VFLHistGBDT(n_estimators=12, max_depth=2, min_samples_leaf=15)
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
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    test_active, test_passive = slice_parties(active, passive, split.test)
    train_map = _passive_map(train_passive)
    test_map = _passive_map(test_passive)
    records: list[dict[str, Any]] = []
    for r in range(len(_PASSIVE_NAMES) + 1):
        for subset in combinations(_PASSIVE_NAMES, r):
            model = _model(model_name)
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
        model = _model(model_name)
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


def run_dropout_study(
    *,
    rows: int = 1600,
    seed: int = 42,
    output: Path = Path("results/party_dropout.csv"),
) -> pd.DataFrame:
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    test_active, test_passive = slice_parties(active, passive, split.test)
    model = VFLLogisticRegression(learning_rate=0.08, max_iter=350, l2=1e-3)
    model.fit(train_active, train_passive)
    test_map = _passive_map(test_passive)
    records: list[dict[str, Any]] = []
    for scenario, dropped in dropout_scenarios().items():
        remaining = [test_map[name] for name in _PASSIVE_NAMES if name not in dropped]
        probability = model.predict_proba([test_active, *remaining])[:, 1]
        metrics = binary_metrics(test_active.labels, probability)
        records.append(
            {
                "scenario": scenario,
                "dropped_parties": "+".join(dropped),
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


def run_drift_study(
    *,
    rows: int = 1600,
    seed: int = 42,
    output: Path = Path("results/feature_drift.csv"),
) -> pd.DataFrame:
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    test_active, test_passive = slice_parties(active, passive, split.test)
    model = VFLLogisticRegression(learning_rate=0.08, max_iter=350, l2=1e-3)
    model.fit(train_active, train_passive)
    scenarios: list[tuple[str, str | None, float, float]] = [
        ("baseline", None, 0.0, 1.0),
        ("telecom_mean_shift", "telecom", 0.75, 1.0),
        ("insurance_variance_shift", "insurance", 0.0, 1.5),
        ("retail_mean_variance_shift", "retail", 0.5, 1.3),
        ("bank_repayment_proxy_shift", "bank", 0.5, 1.2),
    ]
    records: list[dict[str, Any]] = []
    for name, party_name, mean_shift, variance_scale in scenarios:
        active_eval = ActiveParty(
            test_active.name, test_active._x.copy(), test_active.labels.copy()
        )
        passive_eval = [PassiveParty(item.name, item._x.copy()) for item in test_passive]
        if party_name == "bank":
            active_eval = ActiveParty(
                "bank",
                apply_numeric_drift(
                    active_eval._x,
                    mean_shift=mean_shift,
                    variance_scale=variance_scale,
                    seed=seed,
                ),
                active_eval.labels,
            )
        elif party_name is not None:
            passive_eval = [
                PassiveParty(
                    item.name,
                    apply_numeric_drift(
                        item._x,
                        mean_shift=mean_shift,
                        variance_scale=variance_scale,
                        seed=seed,
                    )
                    if item.name == party_name
                    else item._x,
                )
                for item in passive_eval
            ]
        probability = model.predict_proba([active_eval, *passive_eval])[:, 1]
        metrics = binary_metrics(active_eval.labels, probability)
        records.append(
            {
                "scenario": name,
                "party": party_name or "none",
                "mean_shift": mean_shift,
                "variance_scale": variance_scale,
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


def run_contribution_study(
    *,
    rows: int = 800,
    seed: int = 42,
    output: Path = Path("results/party_contribution.csv"),
) -> pd.DataFrame:
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    test_active, test_passive = slice_parties(active, passive, split.test)
    train_map = _passive_map(train_passive)
    test_map = _passive_map(test_passive)
    cache: dict[tuple[str, ...], float] = {}

    def score(subset: tuple[str, ...]) -> float:
        key = tuple(sorted(subset))
        if key in cache:
            return cache[key]
        bank_included = "bank" in key
        train_x = train_active._x if bank_included else np.zeros((train_active.n_rows, 0))
        test_x = test_active._x if bank_included else np.zeros((test_active.n_rows, 0))
        local_train_active = ActiveParty("bank", train_x, train_active.labels)
        local_test_active = ActiveParty("bank", test_x, test_active.labels)
        selected = [name for name in _PASSIVE_NAMES if name in key]
        model = VFLLogisticRegression(learning_rate=0.08, max_iter=300, l2=1e-3)
        model.fit(local_train_active, [train_map[name] for name in selected])
        selected_test = [test_map[name] for name in selected]
        probability = model.predict_proba([local_test_active, *selected_test])[:, 1]
        value = float(roc_auc_score(local_test_active.labels, probability))
        cache[key] = value
        return value

    full = score(_ALL_NAMES)
    shapley = exact_shapley_party_utility(_ALL_NAMES, score)
    records: list[dict[str, Any]] = []
    for party in _ALL_NAMES:
        without = tuple(name for name in _ALL_NAMES if name != party)
        ablated = score(without)
        records.append(
            {
                "model": "logistic",
                "party": party,
                "method": "leave_one_party_out",
                "metric": "roc_auc",
                "full_score": full,
                "ablated_score": ablated,
                "delta": full - ablated,
                "run_id": f"synthetic-{seed}",
            }
        )
        records.append(
            {
                "model": "logistic",
                "party": party,
                "method": "exact_shapley_predictive_utility",
                "metric": "roc_auc",
                "full_score": full,
                "ablated_score": np.nan,
                "delta": shapley[party],
                "run_id": f"synthetic-{seed}",
            }
        )
    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
