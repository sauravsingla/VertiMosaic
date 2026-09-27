# SPDX-License-Identifier: Apache-2.0
import numpy as np

from vertimosaic.privacy import (
    confidence_membership_inference,
    residual_label_inference,
    routing_membership_exposure,
    run_privacy_audit,
)


def test_confidence_membership_attack_detects_separation() -> None:
    train = np.array([0.99, 0.98, 0.02, 0.01, 0.95, 0.05])
    holdout = np.array([0.55, 0.45, 0.60, 0.40, 0.52, 0.48])
    result = confidence_membership_inference(train, holdout)
    assert result.roc_auc > 0.95
    assert result.attack_advantage > 0.8


def test_residual_sign_exposes_binary_label() -> None:
    probability = np.array([0.2, 0.9, 0.4, 0.8])
    labels = np.array([0, 1, 1, 0])
    result = residual_label_inference(probability - labels, labels)
    assert result.accuracy == 1.0
    assert result.exposed_count == 4
    assert result.ambiguous_count == 0


def test_routing_exposure_counts_unique_entities_and_events() -> None:
    result = routing_membership_exposure(
        6,
        [np.array([0, 1, 2, 3]), np.array([1, 3, 5])],
    )
    assert result.routed_entity_events == 7
    assert result.unique_entities_exposed == 5
    assert np.isclose(result.unique_exposure_fraction, 5 / 6)


def test_privacy_audit_is_structured_and_explicitly_non_formal() -> None:
    result = run_privacy_audit(rows=240, seed=11, logistic_epochs=3, gbdt_estimators=1)
    assert result["formal_privacy_guarantee"] is False
    assert set(result["attacks"]) == {
        "confidence_membership_inference",
        "residual_label_inference",
        "passive_gbdt_routing_exposure",
    }
    residual = result["attacks"]["residual_label_inference"]
    assert residual["accuracy"] == 1.0
