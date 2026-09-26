from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import binary_metrics, entity_level_split, select_f1_threshold
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty


def _slice(
    active: ActiveParty, passive: list[PassiveParty], idx: np.ndarray
) -> tuple[ActiveParty, list[PassiveParty]]:
    a = ActiveParty(active.name, active._x[idx], active.labels[idx])
    p = [PassiveParty(item.name, item._x[idx]) for item in passive]
    return a, p


def run_demo(rows: int = 2000, seed: int = 42, model_name: str = "logistic") -> dict[str, float]:
    """CPU-friendly smoke demo using one entity split shared across all parties."""
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = _slice(active, passive, split.train)
    validation_active, validation_passive = _slice(active, passive, split.validation)
    test_active, test_passive = _slice(active, passive, split.test)
    model: VFLLogisticRegression | VFLHistGBDT
    if model_name == "logistic":
        model = VFLLogisticRegression(learning_rate=0.08, max_iter=500, l2=1e-3)
    elif model_name == "vfl-hist-gbdt":
        model = VFLHistGBDT(n_estimators=12, max_depth=2, min_samples_leaf=20)
    else:
        raise ValueError(f"unknown model: {model_name}")
    model.fit(train_active, train_passive)
    validation_p = model.predict_proba([validation_active, *validation_passive])[:, 1]
    threshold = select_f1_threshold(validation_active.labels, validation_p)
    test_p = model.predict_proba([test_active, *test_passive])[:, 1]
    metrics = binary_metrics(test_active.labels, test_p, threshold=threshold)
    metrics["threshold_selected_on_validation"] = threshold
    metrics["estimated_communication_bytes"] = float(model.transport.estimated_payload_bytes)
    return metrics


def write_demo_report(metrics: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
