"""Reference vertical federated logistic regression implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vertimosaic.parties import ActiveParty, Party
from vertimosaic.transport import InMemoryTransport, Message


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(slots=True)
class VFLLogisticRegression:
    """Column-partitioned logistic regression.

    Each party computes its local logit and local gradient. The coordinator never stores
    a passive party's raw feature matrix.
    """

    learning_rate: float = 0.1
    epochs: int = 100
    batch_size: int = 512
    l2: float = 1e-4
    tol: float = 1e-6
    patience: int = 10
    seed: int = 42
    transport: InMemoryTransport = field(default_factory=InMemoryTransport)
    weights_: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    intercept_: float = field(default=0.0, init=False)
    history_: list[dict[str, float]] = field(default_factory=list, init=False)
    party_order_: tuple[str, ...] = field(default_factory=tuple, init=False)

    def fit(self, parties: list[Party], active_party: ActiveParty) -> "VFLLogisticRegression":
        if not parties:
            raise ValueError("at least one party is required")
        n = active_party.n_samples
        if any(p.n_samples != n for p in parties):
            raise ValueError("all parties must be aligned to the same number of rows")
        if active_party.name not in {p.name for p in parties}:
            raise ValueError("active_party must be included in parties")
        self.party_order_ = tuple(p.name for p in parties)
        self.weights_ = {p.name: np.zeros(p.n_features, dtype=float) for p in parties}
        self.intercept_ = 0.0
        self.history_.clear()
        rng = np.random.default_rng(self.seed)
        best_loss = float("inf")
        stale = 0

        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                local_logits: dict[str, np.ndarray] = {}
                for party in parties:
                    z = party.X[idx] @ self.weights_[party.name]
                    local_logits[party.name] = self.transport.send(
                        Message("local_logit", party.name, "coordinator", {"values": z})
                    ).payload["values"]
                logits = self.intercept_ + sum(local_logits.values())
                prob = _sigmoid(logits)
                residual = prob - active_party.y[idx]
                self.intercept_ -= self.learning_rate * float(np.mean(residual))
                for party in parties:
                    signal = self.transport.send(
                        Message("residual_signal", "active", party.name, {"values": residual})
                    ).payload["values"]
                    grad = party.X[idx].T @ signal / len(idx)
                    grad += self.l2 * self.weights_[party.name]
                    self.weights_[party.name] -= self.learning_rate * grad

            loss = self._loss(parties, active_party)
            self.history_.append({"epoch": float(epoch + 1), "loss": loss})
            if best_loss - loss > self.tol:
                best_loss = loss
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        return self

    def decision_function(self, parties: list[Party]) -> np.ndarray:
        if not self.weights_:
            raise RuntimeError("model is not fitted")
        self._validate_prediction_parties(parties)
        return self.intercept_ + sum(p.X @ self.weights_[p.name] for p in parties)

    def predict_proba(self, parties: list[Party]) -> np.ndarray:
        p = _sigmoid(self.decision_function(parties))
        return np.column_stack([1.0 - p, p])

    def predict(self, parties: list[Party], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(parties)[:, 1] >= threshold).astype(int)

    def _loss(self, parties: list[Party], active_party: ActiveParty) -> float:
        p = np.clip(self.predict_proba(parties)[:, 1], 1e-12, 1.0 - 1e-12)
        y = active_party.y
        data_loss = -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        penalty = 0.5 * self.l2 * sum(float(w @ w) for w in self.weights_.values())
        return data_loss + penalty

    def _validate_prediction_parties(self, parties: list[Party]) -> None:
        names = tuple(p.name for p in parties)
        if names != self.party_order_:
            raise ValueError(f"party order mismatch: expected {self.party_order_}, got {names}")
        if len({p.n_samples for p in parties}) != 1:
            raise ValueError("prediction parties must have aligned row counts")
