from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import binary_metrics, entity_level_split
from vertimosaic.experiments.pipeline import slice_parties
from vertimosaic.experiments.robustness import AvailabilityMasks, make_availability_masks
from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty

_METHODS = (
    "intersection_only",
    "zero_contribution",
    "availability_indicator",
    "learned_party_bias",
)


@dataclass(frozen=True)
class MissingPartyPrepared:
    active: ActiveParty
    passive: list[PassiveParty]
    method: str
    retained_entity_indices: np.ndarray


def _passive_mask(masks: AvailabilityMasks, name: str) -> np.ndarray:
    value = getattr(masks, name, None)
    if value is None:
        raise ValueError(f"availability mask missing party: {name}")
    return np.asarray(value, dtype=bool)


def prepare_missing_party_method(
    active: ActiveParty,
    passive: list[PassiveParty],
    masks: AvailabilityMasks,
    method: str,
) -> MissingPartyPrepared:
    """Apply one documented missing-party strategy without target-dependent masking."""
    if method not in _METHODS:
        raise ValueError(f"unknown missing-party method: {method}")
    if active.n_rows != len(masks.bank):
        raise ValueError("availability masks must match entity count")
    if method == "intersection_only":
        retained = np.flatnonzero(masks.intersection())
        return MissingPartyPrepared(
            ActiveParty(active.name, active._x[retained], active.labels[retained]),
            [PassiveParty(party.name, party._x[retained]) for party in passive],
            method,
            retained,
        )

    retained = np.flatnonzero(masks.bank)
    active_x = active._x[retained].copy()
    labels = active.labels[retained]
    availability_columns: list[np.ndarray] = []
    prepared_passive: list[PassiveParty] = []
    for party in passive:
        available = _passive_mask(masks, party.name)[retained]
        values = party._x[retained].copy()
        values[~available] = 0.0
        if method == "availability_indicator":
            availability_columns.append(available.astype(float)[:, None])
        elif method == "learned_party_bias":
            missing_indicator = (~available).astype(float)[:, None]
            values = np.column_stack([values, missing_indicator])
        prepared_passive.append(PassiveParty(party.name, values))
    if availability_columns:
        active_x = np.column_stack([active_x, *availability_columns])
    return MissingPartyPrepared(
        ActiveParty(active.name, active_x, labels),
        prepared_passive,
        method,
        retained,
    )


def run_missing_party_methods_study(
    *,
    rows: int = 1800,
    seed: int = 42,
    telecom: float = 0.75,
    insurance: float = 0.65,
    retail: float = 0.80,
    output: Path = Path("results/missing_party_methods.csv"),
) -> pd.DataFrame:
    """Compare all four missing-party methods on one fixed availability realization."""
    active, passive = make_vertical_synthetic(rows, seed)
    masks = make_availability_masks(
        rows,
        seed=seed,
        bank=1.0,
        telecom=telecom,
        insurance=insurance,
        retail=retail,
    )
    records: list[dict[str, Any]] = []
    for method in _METHODS:
        prepared = prepare_missing_party_method(active, passive, masks, method)
        split = entity_level_split(prepared.active.labels, seed=seed)
        train_active, train_passive = slice_parties(prepared.active, prepared.passive, split.train)
        test_active, test_passive = slice_parties(prepared.active, prepared.passive, split.test)
        model = VFLLogisticRegression(learning_rate=0.08, max_iter=350, l2=1e-3, seed=seed)
        model.fit(train_active, train_passive)
        probability = model.predict_proba([test_active, *test_passive])[:, 1]
        metrics = binary_metrics(test_active.labels, probability)
        records.append(
            {
                "method": method,
                "retained_entities": prepared.active.n_rows,
                "coverage": prepared.active.n_rows / rows,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "f1": metrics["f1"],
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "estimated_communication_bytes": model.transport.estimated_payload_bytes,
            }
        )
    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
