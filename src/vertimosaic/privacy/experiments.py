# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import numpy as np

from vertimosaic.datasets.synthetic import make_vertical_synthetic
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.models.vfl_hist_gbdt import TreeNode
from vertimosaic.privacy.attacks import (
    confidence_membership_inference,
    residual_label_inference,
    routing_membership_exposure,
)


def _passive_routing_messages(node: TreeNode, active_name: str) -> list[np.ndarray]:
    messages: list[np.ndarray] = []
    if not node.is_leaf and node.party is not None and node.party != active_name:
        messages.append(node.indices.copy())
    if node.left is not None:
        messages.extend(_passive_routing_messages(node.left, active_name))
    if node.right is not None:
        messages.extend(_passive_routing_messages(node.right, active_name))
    return messages


def run_privacy_audit(
    *,
    rows: int = 1200,
    seed: int = 42,
    logistic_epochs: int = 80,
    gbdt_estimators: int = 5,
) -> dict[str, Any]:
    """Run deterministic empirical leakage baselines on synthetic VFL data.

    The measurements are attack baselines, not formal privacy guarantees. They are
    intended to make documented leakage surfaces observable and regression-testable.
    """

    if rows < 200:
        raise ValueError("privacy audit requires at least 200 rows")
    train_rows = rows // 2
    holdout_rows = rows - train_rows
    train_active, train_passive = make_vertical_synthetic(train_rows, seed=seed)
    holdout_active, holdout_passive = make_vertical_synthetic(holdout_rows, seed=seed + 1)

    logistic = VFLLogisticRegression(
        learning_rate=0.05,
        max_iter=logistic_epochs,
        l2=1e-3,
        tolerance=0.0,
        seed=seed,
    )
    logistic.fit(train_active, train_passive)
    train_probability = logistic.predict_proba([train_active, *train_passive])[:, 1]
    holdout_probability = logistic.predict_proba([holdout_active, *holdout_passive])[:, 1]
    membership = confidence_membership_inference(train_probability, holdout_probability)
    residual = train_probability - train_active.labels
    label_inference = residual_label_inference(residual, train_active.labels)

    gbdt = VFLHistGBDT(
        n_estimators=gbdt_estimators,
        max_depth=2,
        min_samples_leaf=max(10, train_rows // 50),
        max_bins=12,
        seed=seed,
    )
    gbdt.fit(train_active, train_passive)
    routing_messages: list[np.ndarray] = []
    for tree in gbdt.trees_:
        routing_messages.extend(_passive_routing_messages(tree, active_name=train_active.name))
    routing = routing_membership_exposure(train_rows, routing_messages)

    return {
        "schema_version": 1,
        "seed": seed,
        "rows": rows,
        "train_rows": train_rows,
        "holdout_rows": holdout_rows,
        "attacks": {
            "confidence_membership_inference": membership.to_dict(),
            "residual_label_inference": label_inference.to_dict(),
            "passive_gbdt_routing_exposure": routing.to_dict(),
        },
        "interpretation": {
            "membership": "Simple confidence-threshold attack; AUC 0.5 is chance.",
            "residual": "Residual sign can reveal the binary target in plain logistic VFL.",
            "routing": "Counts entity-index exposure at passive-party-owned tree splits.",
        },
        "formal_privacy_guarantee": False,
    }
