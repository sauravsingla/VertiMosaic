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
    learning_rate: float = 0.1
    max_iter: int = 500
    l2: float = 0.0
    l1: float = 0.0
    tolerance: float = 1e-7
    gradient_clip: float | None = 10.0
    class_weight: str | None = None
    transport: InMemoryTransport = field(default_factory=InMemoryTransport)
    weights_: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    intercept_: float = field(default=0.0, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)

    def fit(self, active: ActiveParty, passive: list[PassiveParty]) -> "VFLLogisticRegression":
        parties: list[PassiveParty] = [active, *passive]
        n = active.n_rows
        if any(p.n_rows != n for p in parties):
            raise ValueError("all VFL parties must align to the same row count")
        self.weights_ = {p.name: np.zeros(p.n_features, dtype=float) for p in parties}
        self.intercept_ = 0.0
        self.loss_history_.clear()
        y = active.labels
        if self.class_weight == "balanced":
            pos = max(float(y.sum()), 1.0)
            neg = max(float((1.0 - y).sum()), 1.0)
            sample_weight = np.where(y == 1.0, n / (2.0 * pos), n / (2.0 * neg))
        else:
            sample_weight = np.ones(n, dtype=float)
        previous = np.inf
        for _ in range(self.max_iter):
            logits = np.full(n, self.intercept_, dtype=float)
            for party in parties:
                local = party.local_logits(self.weights_[party.name])
                logits += self.transport.send(local,message_type="local_logits",sender_role=party.name,receiver_role="active")
            probs = _sigmoid(logits)
            eps = 1e-12
            data_loss = -np.average(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps),weights=sample_weight)
            penalty = sum(0.5 * self.l2 * float(w @ w) + self.l1 * float(np.abs(w).sum()) for w in self.weights_.values())
            loss = float(data_loss + penalty)
            self.loss_history_.append(loss)
            if abs(previous - loss) < self.tolerance:
                break
            previous = loss
            residual = (probs - y) * sample_weight
            self.transport.send(residual,message_type="residual_signal",sender_role="active",receiver_role="parties")
            for party in parties:
                grad = party.local_gradient(residual)
                grad += self.l2 * self.weights_[party.name]
                if self.gradient_clip is not None:
                    norm = float(np.linalg.norm(grad))
                    if norm > self.gradient_clip:
                        grad *= self.gradient_clip / max(norm, 1e-12)
                w = self.weights_[party.name] - self.learning_rate * grad
                if self.l1 > 0:
                    shrink = self.learning_rate * self.l1
                    w = np.sign(w) * np.maximum(np.abs(w) - shrink, 0.0)
                self.weights_[party.name] = w
            self.intercept_ -= self.learning_rate * float(residual.mean())
        return self

    def decision_function(self, parties: list[PassiveParty]) -> np.ndarray:
        if not self.weights_:
            raise RuntimeError("model is not fitted")
        if not parties:
            raise ValueError("at least one party is required")
        n = parties[0].n_rows
        logits = np.full(n, self.intercept_, dtype=float)
        for party in parties:
            logits += party.local_logits(self.weights_[party.name])
        return logits

    def predict_proba(self, parties: list[PassiveParty]) -> np.ndarray:
        p = _sigmoid(self.decision_function(parties))
        return np.column_stack([1.0 - p, p])

    def predict(self, parties: list[PassiveParty], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(parties)[:, 1] >= threshold).astype(int)
