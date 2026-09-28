from __future__ import annotations

import numpy as np

from vertimosaic.privacy import (
    gradient_label_inference,
    message_feature_reconstruction,
    privacy_attack_scenarios,
)


def test_message_feature_reconstruction_detects_linear_exposure() -> None:
    rng = np.random.default_rng(7)
    calibration_message = rng.normal(size=(200, 2))
    evaluation_message = rng.normal(size=(100, 2))
    calibration_features = np.column_stack(
        [2.0 * calibration_message[:, 0], -3.0 * calibration_message[:, 1]]
    )
    evaluation_features = np.column_stack(
        [2.0 * evaluation_message[:, 0], -3.0 * evaluation_message[:, 1]]
    )
    result = message_feature_reconstruction(
        calibration_message,
        calibration_features,
        evaluation_message,
        evaluation_features,
    )
    assert result.mean_r2 > 0.99
    assert result.max_r2 > 0.99
    assert result.reconstructed_features == 2


def test_gradient_label_inference_exposes_binary_gradient_sign() -> None:
    labels = np.array([0, 1, 0, 1])
    probabilities = np.array([0.2, 0.8, 0.4, 0.6])
    gradient = probabilities - labels
    result = gradient_label_inference(gradient, labels)
    assert result.accuracy == 1.0
    assert result.exposed_count == 4


def test_attack_matrix_names_both_active_and_passive_observers() -> None:
    scenarios = privacy_attack_scenarios()
    observers = {row["observer"] for row in scenarios}
    messages = {row["observed_message"] for row in scenarios}
    assert "passive party" in observers
    assert "active party" in observers
    assert any("gradient" in message for message in messages)
    assert any("local logits" in message for message in messages)
