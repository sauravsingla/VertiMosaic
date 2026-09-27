from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import binary_metrics, entity_level_split, select_f1_threshold
from vertimosaic.experiments.missing_parties import prepare_missing_party_method
from vertimosaic.experiments.pipeline import slice_parties
from vertimosaic.experiments.robustness import AvailabilityMasks
from vertimosaic.experiments.study_artifacts import (
    model_communication_frame,
    model_training_frame,
    prediction_frame,
    write_synthetic_study_run,
)
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression

_OVERLAP_METHODS = ("intersection_only", "availability_indicator")


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


def _shared_overlap_masks(rows: int, fraction: float, *, seed: int) -> AvailabilityMasks:
    """Create an exact Bank-to-passive overlap fraction shared by all passive parties."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("overlap fractions must be in (0, 1]")
    rng = np.random.default_rng(seed)
    count = max(1, int(round(rows * fraction)))
    chosen = rng.permutation(rows)[:count]
    passive_mask = np.zeros(rows, dtype=bool)
    passive_mask[chosen] = True
    return AvailabilityMasks(
        bank=np.ones(rows, dtype=bool),
        telecom=passive_mask.copy(),
        insurance=passive_mask.copy(),
        retail=passive_mask.copy(),
    )


def _positions(retained_entities: np.ndarray, split_entities: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(retained_entities, split_entities))


def run_overlap_study(
    *,
    rows: int = 2000,
    seed: int = 42,
    model_name: str = "logistic",
    fractions: tuple[float, ...] = (1.0, 0.9, 0.75, 0.5, 0.25),
    methods: tuple[str, ...] = _OVERLAP_METHODS,
    output: Path = Path("results/partial_overlap.csv"),
    write_run: bool = True,
    runs_root: Path = Path("runs"),
) -> pd.DataFrame:
    """Compare intersection-only and missing-party-aware training over overlap levels.

    Each overlap level uses one deterministic availability realization shared by the
    two methods. The global entity split is fixed before applying availability masks,
    preventing entities from moving between train/validation/test across methods.
    """
    unsupported = set(methods) - set(_OVERLAP_METHODS)
    if unsupported:
        raise ValueError(f"unsupported overlap study methods: {sorted(unsupported)}")
    active, passive = make_vertical_synthetic(rows, seed)
    global_split = entity_level_split(active.labels, seed=seed)
    records: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    communications: list[pd.DataFrame] = []

    for fraction_index, fraction in enumerate(fractions):
        masks = _shared_overlap_masks(rows, fraction, seed=seed + 1000 + fraction_index)
        intersection_coverage = float(masks.intersection().mean())
        for method in methods:
            prepared = prepare_missing_party_method(active, passive, masks, method)
            train_idx = _positions(prepared.retained_entity_indices, global_split.train)
            validation_idx = _positions(prepared.retained_entity_indices, global_split.validation)
            test_idx = _positions(prepared.retained_entity_indices, global_split.test)
            if min(len(train_idx), len(validation_idx), len(test_idx)) == 0:
                raise ValueError(
                    f"overlap={fraction:.2f} and method={method} left an empty entity split"
                )
            train_active, train_passive = slice_parties(
                prepared.active,
                prepared.passive,
                train_idx,
            )
            validation_active, validation_passive = slice_parties(
                prepared.active,
                prepared.passive,
                validation_idx,
            )
            test_active, test_passive = slice_parties(
                prepared.active,
                prepared.passive,
                test_idx,
            )
            model = _model(model_name, seed)
            start = time.perf_counter()
            model.fit(train_active, train_passive)
            training_seconds = time.perf_counter() - start
            validation_probability = model.predict_proba([validation_active, *validation_passive])[
                :, 1
            ]
            threshold = select_f1_threshold(validation_active.labels, validation_probability)
            probability = model.predict_proba([test_active, *test_passive])[:, 1]
            metrics = binary_metrics(test_active.labels, probability, threshold=threshold)
            condition = f"{method}:overlap={fraction:.2f}"
            records.append(
                {
                    "model": model_name,
                    "method": method,
                    "overlap_fraction": fraction,
                    "coverage": prepared.active.n_rows / rows,
                    "intersection_coverage": intersection_coverage,
                    "retained_entities": prepared.active.n_rows,
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "f1": metrics["f1"],
                    "log_loss": metrics["log_loss"],
                    "brier": metrics["brier"],
                    "ece": metrics["ece"],
                    "threshold_selected_on_validation": threshold,
                    "training_seconds": training_seconds,
                    "estimated_communication_bytes": model.transport.estimated_payload_bytes,
                }
            )
            predictions.append(
                prediction_frame(
                    prepared.retained_entity_indices[test_idx],
                    test_active.labels,
                    probability,
                    seed=seed,
                    condition=condition,
                )
            )
            histories.append(model_training_frame(model, condition=condition))
            communications.append(model_communication_frame(model, condition=condition))

    frame = pd.DataFrame(records)
    if write_run:
        run_id, directory = write_synthetic_study_run(
            study_name="partial_overlap",
            seed=seed,
            active=active,
            passive=passive,
            config={
                "rows": rows,
                "model": model_name,
                "fractions": list(fractions),
                "methods": list(methods),
                "missing_party_aware_method": "availability_indicator",
                "entity_split": {"train": 0.70, "validation": 0.15, "test": 0.15},
            },
            results=frame,
            predictions=pd.concat(predictions, ignore_index=True),
            training_history=pd.concat(histories, ignore_index=True),
            communication=pd.concat(communications, ignore_index=True),
            runs_root=runs_root,
        )
        frame["run_id"] = run_id
        frame["run_directory"] = str(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
