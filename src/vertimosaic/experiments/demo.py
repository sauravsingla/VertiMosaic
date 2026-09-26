from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import binary_metrics
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty


def _slice(active: ActiveParty, passive: list[PassiveParty], idx: np.ndarray) -> tuple[ActiveParty, list[PassiveParty]]:
    a = ActiveParty(active.name, active._x[idx], active.labels[idx])
    p = [PassiveParty(item.name, item._x[idx]) for item in passive]
    return a, p


def run_demo(rows: int = 2000, seed: int = 42, model_name: str = "logistic") -> dict[str, float]:
    active, passive = make_vertical_synthetic(rows, seed)
    indices = np.arange(rows)
    train_idx, test_idx = train_test_split(indices, test_size=0.30, random_state=seed, stratify=active.labels)
    train_active, train_passive = _slice(active, passive, train_idx)
    test_active, test_passive = _slice(active, passive, test_idx)
    if model_name == "logistic":
        model = VFLLogisticRegression(learning_rate=0.08, max_iter=500, l2=1e-3)
    elif model_name == "vfl-hist-gbdt":
        model = VFLHistGBDT(n_estimators=12, max_depth=2, min_samples_leaf=20)
    else:
        raise ValueError(f"unknown model: {model_name}")
    model.fit(train_active, train_passive)
    p = model.predict_proba([test_active, *test_passive])[:, 1]
    metrics = binary_metrics(test_active.labels, p)
    metrics["estimated_communication_bytes"] = float(model.transport.estimated_payload_bytes)
    return metrics


def write_demo_report(metrics: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
