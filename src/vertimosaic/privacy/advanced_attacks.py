# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from vertimosaic.privacy.attacks import ResidualLabelInferenceResult, residual_label_inference


@dataclass(frozen=True)
class MessageFeatureReconstructionResult:
    """Linear reconstruction baseline for party features from exposed messages."""

    mean_r2: float
    max_r2: float
    mean_mae: float
    exposed_dimensions: int
    reconstructed_features: int
    calibration_count: int
    evaluation_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def message_feature_reconstruction(
    calibration_messages: np.ndarray,
    calibration_features: np.ndarray,
    evaluation_messages: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    alpha: float = 1e-3,
) -> MessageFeatureReconstructionResult:
    """Fit a reproducible ridge attacker from exposed messages to private features.

    A high score demonstrates linear recoverability from the supplied protocol
    messages. A low score is not proof that a stronger nonlinear attack will fail.
    """

    x_cal = np.asarray(calibration_messages, dtype=float)
    y_cal = np.asarray(calibration_features, dtype=float)
    x_eval = np.asarray(evaluation_messages, dtype=float)
    y_eval = np.asarray(evaluation_features, dtype=float)
    if x_cal.ndim == 1:
        x_cal = x_cal[:, None]
    if x_eval.ndim == 1:
        x_eval = x_eval[:, None]
    if y_cal.ndim == 1:
        y_cal = y_cal[:, None]
    if y_eval.ndim == 1:
        y_eval = y_eval[:, None]
    if x_cal.shape[0] != y_cal.shape[0] or x_eval.shape[0] != y_eval.shape[0]:
        raise ValueError("message and feature rows must align")
    if x_cal.shape[1] != x_eval.shape[1] or y_cal.shape[1] != y_eval.shape[1]:
        raise ValueError("calibration and evaluation dimensions must match")
    if x_cal.shape[0] < 2 or x_eval.shape[0] < 2:
        raise ValueError("reconstruction requires at least two rows in each split")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if not all(np.isfinite(array).all() for array in (x_cal, y_cal, x_eval, y_eval)):
        raise ValueError("reconstruction inputs must be finite")

    attacker = Ridge(alpha=alpha)
    attacker.fit(x_cal, y_cal)
    prediction = np.asarray(attacker.predict(x_eval), dtype=float)
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    r2_values = [
        float(r2_score(y_eval[:, column], prediction[:, column]))
        for column in range(y_eval.shape[1])
    ]
    mae_values = [
        float(mean_absolute_error(y_eval[:, column], prediction[:, column]))
        for column in range(y_eval.shape[1])
    ]
    return MessageFeatureReconstructionResult(
        mean_r2=float(np.mean(r2_values)),
        max_r2=float(np.max(r2_values)),
        mean_mae=float(np.mean(mae_values)),
        exposed_dimensions=int(x_cal.shape[1]),
        reconstructed_features=int(y_cal.shape[1]),
        calibration_count=int(x_cal.shape[0]),
        evaluation_count=int(x_eval.shape[0]),
    )


def gradient_label_inference(
    gradient_signal: np.ndarray,
    labels: np.ndarray,
    *,
    ambiguity_tolerance: float = 1e-12,
) -> ResidualLabelInferenceResult:
    """Measure binary-label leakage from first-order logistic/GBDT gradient signals."""

    return residual_label_inference(
        gradient_signal,
        labels,
        ambiguity_tolerance=ambiguity_tolerance,
    )


def privacy_attack_scenarios() -> tuple[dict[str, str], ...]:
    """Document concrete observer/message/target combinations in reference protocols."""

    return (
        {
            "scenario": "passive_party_label_inference_logistic",
            "observer": "passive party",
            "observed_message": "per-entity residual signal",
            "private_target": "active-party binary label",
            "baseline_attack": "residual sign label inference",
        },
        {
            "scenario": "passive_party_label_inference_gbdt",
            "observer": "split-owning party",
            "observed_message": "per-entity first-order gradient/Hessian signal",
            "private_target": "active-party binary label",
            "baseline_attack": "gradient sign label inference",
        },
        {
            "scenario": "active_party_feature_reconstruction",
            "observer": "active party",
            "observed_message": "passive-party local logits",
            "private_target": "passive-party local feature block",
            "baseline_attack": "ridge message-to-feature reconstruction",
        },
        {
            "scenario": "active_party_membership_routing",
            "observer": "active/coordinator role",
            "observed_message": "tree routing and aggregate split statistics",
            "private_target": "party/entity membership and local split structure",
            "baseline_attack": "routing membership exposure",
        },
    )
