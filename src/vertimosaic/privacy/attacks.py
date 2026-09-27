# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class MembershipInferenceResult:
    """Measured confidence-threshold membership-inference result."""

    roc_auc: float
    attack_advantage: float
    threshold: float
    train_count: int
    holdout_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ResidualLabelInferenceResult:
    """Measured label leakage from logistic residual signs."""

    accuracy: float
    exposed_count: int
    ambiguous_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class RoutingExposureResult:
    """Summary of entity-membership exposure through tree-routing messages."""

    total_entities: int
    routed_entity_events: int
    unique_entities_exposed: int
    unique_exposure_fraction: float
    mean_messages_per_exposed_entity: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _confidence(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=float).reshape(-1)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return np.abs(values - 0.5) * 2.0


def confidence_membership_inference(
    train_probability: np.ndarray,
    holdout_probability: np.ndarray,
) -> MembershipInferenceResult:
    """Run a simple confidence-based membership-inference attack.

    Higher predictive confidence is treated as evidence of membership. This is a
    deliberately simple, reproducible baseline rather than a claim that it is the
    strongest available attack.
    """

    train_score = _confidence(train_probability)
    holdout_score = _confidence(holdout_probability)
    if train_score.size == 0 or holdout_score.size == 0:
        raise ValueError("membership inference requires non-empty train and holdout samples")

    labels = np.concatenate(
        [np.ones(train_score.size, dtype=int), np.zeros(holdout_score.size, dtype=int)]
    )
    scores = np.concatenate([train_score, holdout_score])
    auc = float(roc_auc_score(labels, scores))

    thresholds = np.unique(np.concatenate([[0.0], scores, [1.0 + np.finfo(float).eps]]))
    best_advantage = -np.inf
    best_threshold = 0.5
    train_n = float(train_score.size)
    holdout_n = float(holdout_score.size)
    for threshold in thresholds:
        true_positive_rate = float(np.sum(train_score >= threshold) / train_n)
        false_positive_rate = float(np.sum(holdout_score >= threshold) / holdout_n)
        advantage = true_positive_rate - false_positive_rate
        if advantage > best_advantage:
            best_advantage = advantage
            best_threshold = float(threshold)

    return MembershipInferenceResult(
        roc_auc=auc,
        attack_advantage=float(best_advantage),
        threshold=best_threshold,
        train_count=train_score.size,
        holdout_count=holdout_score.size,
    )


def residual_label_inference(
    residual: np.ndarray,
    labels: np.ndarray,
    *,
    ambiguity_tolerance: float = 1e-12,
) -> ResidualLabelInferenceResult:
    """Measure label inference from a binary-logistic residual signal.

    For residual = p - y with p in (0, 1), the sign exposes the binary label:
    negative implies y=1 and positive implies y=0. Values close to zero are marked
    ambiguous to make the metric numerically explicit.
    """

    values = np.asarray(residual, dtype=float).reshape(-1)
    target = np.asarray(labels, dtype=int).reshape(-1)
    if values.shape != target.shape:
        raise ValueError("residual and labels must have equal shape")
    if not np.all(np.isin(target, [0, 1])):
        raise ValueError("labels must contain only 0/1")
    if ambiguity_tolerance < 0:
        raise ValueError("ambiguity_tolerance must be non-negative")

    ambiguous = np.abs(values) <= ambiguity_tolerance
    exposed = ~ambiguous
    prediction = (values < 0.0).astype(int)
    exposed_count = int(exposed.sum())
    accuracy = (
        float(np.mean(prediction[exposed] == target[exposed]))
        if exposed_count
        else float("nan")
    )
    return ResidualLabelInferenceResult(
        accuracy=accuracy,
        exposed_count=exposed_count,
        ambiguous_count=int(ambiguous.sum()),
    )


def routing_membership_exposure(
    total_entities: int,
    routed_index_messages: list[np.ndarray],
) -> RoutingExposureResult:
    """Measure entity-membership exposure from routed index payloads."""

    if total_entities <= 0:
        raise ValueError("total_entities must be positive")
    seen: set[int] = set()
    events = 0
    for message in routed_index_messages:
        indices = np.asarray(message, dtype=int).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= total_entities):
            raise ValueError("routed index message contains an out-of-range entity index")
        events += int(indices.size)
        seen.update(int(index) for index in indices)
    unique = len(seen)
    return RoutingExposureResult(
        total_entities=total_entities,
        routed_entity_events=events,
        unique_entities_exposed=unique,
        unique_exposure_fraction=float(unique / total_entities),
        mean_messages_per_exposed_entity=float(events / unique) if unique else 0.0,
    )
