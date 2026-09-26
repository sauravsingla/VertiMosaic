from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.transport import InMemoryTransport


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class VFLLogisticRegression:
    """Reference first-principles vertical logistic regression protocol.

    Setting both ``l1`` and ``l2`` to non-zero values gives an elastic-net
    objective. Mini-batches are entity-aligned across every party and are
    shuffled deterministically from ``seed``.
    """

    learning_rate: float = 0.1
    max_iter: int = 500
    l2: float = 0.0
    l1: float = 0.0
    tolerance: float = 1e-7
    gradient_clip: float | None = 10.0
    class_weight: str | dict[int, float] | None = None
    batch_size: int | None = None
    learning_rate_schedule: str = "constant"
    warm_start: bool = False
    seed: int = 42
    transport: InMemoryTransport = field(default_factory=InMemoryTransport)
    weights_: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    intercept_: float = field(default=0.0, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)
    n_iter_: int = field(default=0, init=False)
    converged_: bool = field(default=False, init=False)

    def _validate_hyperparameters(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.l1 < 0 or self.l2 < 0:
            raise ValueError("l1 and l2 must be non-negative")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive when supplied")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive when supplied")
        if self.learning_rate_schedule not in {"constant", "inverse_sqrt", "linear_decay"}:
            raise ValueError(
                "learning_rate_schedule must be constant, inverse_sqrt or linear_decay"
            )
        if isinstance(self.class_weight, str) and self.class_weight != "balanced":
            raise ValueError("class_weight string must be 'balanced'")

    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        if self.class_weight == "balanced":
            pos = max(float(y.sum()), 1.0)
            neg = max(float((1.0 - y).sum()), 1.0)
            return np.where(y == 1.0, n / (2.0 * pos), n / (2.0 * neg))
        if isinstance(self.class_weight, dict):
            return np.where(
                y == 1.0,
                float(self.class_weight.get(1, 1.0)),
                float(self.class_weight.get(0, 1.0)),
            )
        return np.ones(n, dtype=float)

    def _epoch_learning_rate(self, epoch: int) -> float:
        if self.learning_rate_schedule == "inverse_sqrt":
            return self.learning_rate / np.sqrt(epoch + 1.0)
        if self.learning_rate_schedule == "linear_decay":
            fraction = max(0.05, 1.0 - epoch / max(self.max_iter, 1))
            return self.learning_rate * fraction
        return self.learning_rate

    def _initialize(self, parties: list[PassiveParty]) -> None:
        expected = {party.name: party.n_features for party in parties}
        can_reuse = self.warm_start and set(self.weights_) == set(expected)
        if can_reuse:
            can_reuse = all(len(self.weights_[name]) == width for name, width in expected.items())
        if not can_reuse:
            self.weights_ = {
                party.name: np.zeros(party.n_features, dtype=float) for party in parties
            }
            self.intercept_ = 0.0

    def _logits(self, parties: list[PassiveParty], indices: np.ndarray | None = None) -> np.ndarray:
        n = parties[0].n_rows if indices is None else len(indices)
        logits = np.full(n, self.intercept_, dtype=float)
        for party in parties:
            local = party.local_logits(self.weights_[party.name], indices)
            logits += self.transport.send(
                local,
                message_type="local_logits",
                sender_role=party.name,
                receiver_role="active",
            )
        return logits

    def fit(self, active: ActiveParty, passive: list[PassiveParty]) -> VFLLogisticRegression:
        self._validate_hyperparameters()
        parties: list[PassiveParty] = [active, *passive]
        n = active.n_rows
        if any(party.n_rows != n for party in parties):
            raise ValueError("all VFL parties must align to the same row count")
        self._initialize(parties)
        self.loss_history_.clear()
        self.n_iter_ = 0
        self.converged_ = False
        y = active.labels
        sample_weight = self._sample_weights(y)
        rng = np.random.default_rng(self.seed)
        batch_size = min(self.batch_size or n, n)

        previous = np.inf
        for epoch in range(self.max_iter):
            logits = self._logits(parties)
            probs = _sigmoid(logits)
            eps = 1e-12
            data_loss = -np.average(
                y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps),
                weights=sample_weight,
            )
            penalty = sum(
                0.5 * self.l2 * float(weights @ weights)
                + self.l1 * float(np.abs(weights).sum())
                for weights in self.weights_.values()
            )
            loss = float(data_loss + penalty)
            self.loss_history_.append(loss)
            self.n_iter_ = epoch + 1
            if abs(previous - loss) < self.tolerance:
                self.converged_ = True
                break
            previous = loss

            if batch_size == n:
                order = np.arange(n, dtype=int)
            else:
                order = rng.permutation(n)
            rate = self._epoch_learning_rate(epoch)
            for start in range(0, n, batch_size):
                batch = order[start : start + batch_size]
                batch_logits = self._logits(parties, batch)
                batch_probs = _sigmoid(batch_logits)
                residual = (batch_probs - y[batch]) * sample_weight[batch]
                self.transport.send(
                    residual,
                    message_type="residual_signal",
                    sender_role="active",
                    receiver_role="parties",
                )
                for party in parties:
                    grad = party.local_gradient(residual, batch)
                    grad += self.l2 * self.weights_[party.name]
                    if self.gradient_clip is not None:
                        norm = float(np.linalg.norm(grad))
                        if norm > self.gradient_clip:
                            grad *= self.gradient_clip / max(norm, 1e-12)
                    weights = self.weights_[party.name] - rate * grad
                    if self.l1 > 0:
                        shrink = rate * self.l1
                        weights = np.sign(weights) * np.maximum(np.abs(weights) - shrink, 0.0)
                    self.weights_[party.name] = weights
                self.intercept_ -= rate * float(residual.mean())
        return self

    def decision_function(self, parties: list[PassiveParty]) -> np.ndarray:
        if not self.weights_:
            raise RuntimeError("model is not fitted")
        if not parties:
            raise ValueError("at least one party is required")
        n = parties[0].n_rows
        if any(party.n_rows != n for party in parties):
            raise ValueError("inference parties must have equal row counts")
        logits = np.full(n, self.intercept_, dtype=float)
        for party in parties:
            if party.name not in self.weights_:
                raise ValueError(f"unknown inference party: {party.name}")
            logits += party.local_logits(self.weights_[party.name])
        return logits

    def predict_proba(self, parties: list[PassiveParty]) -> np.ndarray:
        probability = _sigmoid(self.decision_function(parties))
        return np.column_stack([1.0 - probability, probability])

    def predict(self, parties: list[PassiveParty], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(parties)[:, 1] >= threshold).astype(int)
