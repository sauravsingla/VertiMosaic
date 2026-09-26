from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from vertimosaic.evaluation import binary_metrics, entity_level_split, select_f1_threshold
from vertimosaic.experiments.external import linkage_manifest_dict, prepare_external_benchmark
from vertimosaic.experiments.pipeline import slice_parties
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression


def run_external_experiment(
    *,
    mode: str = "observed_target_external",
    model_name: str = "vfl-hist-gbdt",
    seed: int = 42,
    cross_party_correlation: float = 0.25,
    insurance_sample_size: int | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    benchmark = prepare_external_benchmark(
        mode=mode,
        cross_party_correlation=cross_party_correlation,
        seed=seed,
        insurance_sample_size=insurance_sample_size,
    )
    split = entity_level_split(benchmark.active.labels, seed=seed)
    train_active, train_passive = slice_parties(benchmark.active, benchmark.passive, split.train)
    val_active, val_passive = slice_parties(benchmark.active, benchmark.passive, split.validation)
    test_active, test_passive = slice_parties(benchmark.active, benchmark.passive, split.test)
    if model_name == "logistic":
        model: VFLLogisticRegression | VFLHistGBDT = VFLLogisticRegression(
            learning_rate=0.08, max_iter=500, l2=1e-3
        )
    elif model_name == "vfl-hist-gbdt":
        model = VFLHistGBDT(n_estimators=20, max_depth=3, min_samples_leaf=20)
    else:
        raise ValueError(f"unknown model: {model_name}")
    start = time.perf_counter()
    model.fit(train_active, train_passive)
    training_seconds = time.perf_counter() - start
    validation_p = model.predict_proba([val_active, *val_passive])[:, 1]
    threshold = select_f1_threshold(val_active.labels, validation_p)
    test_p = model.predict_proba([test_active, *test_passive])[:, 1]
    payload: dict[str, Any] = {
        "benchmark_description": "externally grounded semi-synthetic cross-industry VFL benchmark",
        "mode": mode,
        "target_description": "observed Bank default target"
        if mode == "observed_target_external"
        else "semi_synthetic_cross_industry_risk",
        "model": model_name,
        "seed": seed,
        "metrics": binary_metrics(test_active.labels, test_p, threshold=threshold),
        "threshold_selected_on_validation": threshold,
        "training_seconds": training_seconds,
        "estimated_communication_bytes": model.transport.estimated_payload_bytes,
        "linkage": linkage_manifest_dict(benchmark),
        "four_sources_same_real_people": False,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
