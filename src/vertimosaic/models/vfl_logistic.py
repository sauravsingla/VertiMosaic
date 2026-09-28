from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vertimosaic.alignment import validate_exact_entity_alignment
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.privacy.backends import ClippedGaussianDPBackend
from vertimosaic.transport import InMemoryTransport


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class VFLLogisticRegression:
    """Reference first-principles vertical logistic regression protocol.

    Setting both ``l1`` and ``l2`` to non-zero values gives an elastic-net
    objective. Mini-batches are entity-aligned across every party and are
    shuffled deterministically from ``seed``. Passive-party logit contributions
    and residual signals cross the simulated ``Message`` transport boundary;
    raw party feature matrices remain local.

    ``require_entity_ids`` enables protocol-boundary verification of exact ordered
    entity alignment. Official VertiMosaic research/reproduction entry points enable
    it. The low-level class keeps ``False`` as a backwards-compatibility bridge for
    callers that have not yet bound identifiers with ``bind_entity_ids``.

    ``missing_party_policy='error'`` is the default and rejects incomplete inference.
    ``zero_contribution`` is an explicit research fallback that treats absent passive
    parties as contributing zero logits; the active party may never be omitted.

    ``residual_dp_backend`` can apply the executable clipped-Gaussian message-level
    release mechanism to residual messages sent to passive parties. This protects that
    release only and is not an end-to-end VFL privacy claim.
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
    early_stopping_rounds: int | None = None
    warm_start: bool = False
    residual_noise_std: float = 0.0
    residual_dp_backend: ClippedGaussianDPBackend | None = None
    require_entity_ids: bool = False
    missing_party_policy: str = "error"
    seed: int = 42
    transport: InMemoryTransport = field(default_factory=InMemoryTransport)
    weights_: dict[str, np.ndarray] = field(default_factory=dict, init=False)
    intercept_: float = field(default=0.0, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)
    validation_loss_history_: list[float] = field(default_factory=list, init=False)
    n_iter_: int = field(default=0, init=False)
    converged_: bool = field(default=False, init=False)
    best_iteration_: int | None = field(default=None, init=False)
    trained_party_names_: tuple[str, ...] = field(default=(), init=False)
    active_party_name_: str | None = field(default=None, init=False)

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
        if self.early_stopping_rounds is not None and self.early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be positive when supplied")
        if not np.isfinite(self.residual_noise_std) or self.residual_noise_std < 0:
            raise ValueError("residual_noise_std must be finite and non-negative")
        if self.residual_noise_std > 0.0 and self.residual_dp_backend is not None:
            raise ValueError("residual_noise_std and residual_dp_backend are mutually exclusive")
        if self.missing_party_policy not in {"error", "zero_contribution"}:
            raise ValueError("missing_party_policy must be error or zero_contribution")
        if isinstance(self.class_weight, str) and self.class_weight != "balanced":
            raise ValueError("class_weight string must be 'balanced'")
        if isinstance(self.class_weight, dict) and any(
            float(value) < 0.0 for value in self.class_weight.values()
        ):
            raise ValueError("class weights must be non-negative")

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

    def _validate_party_collection(
        self,
        parties: list[PassiveParty],
        *,
        context: str,
        require_entity_ids: bool | None = None,
    ) -> None:
        if not parties:
            raise ValueError(f"{context}: at least one party is required")
        names = [party.name for party in parties]
        if len(set(names)) != len(names):
            raise ValueError(f"{context}: party names must be unique")
        n = parties[0].n_rows
        if any(party.n_rows != n for party in parties):
            raise ValueError(f"{context}: all VFL parties must align to the same row count")
        validate_exact_entity_alignment(
            parties,
            context=context,
            require_bound_ids=self.require_entity_ids
            if require_entity_ids is None
            else require_entity_ids,
        )

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

    def _penalty(self) -> float:
        return float(
            sum(
                0.5 * self.l2 * float(weights @ weights) + self.l1 * float(np.abs(weights).sum())
                for weights in self.weights_.values()
            )
        )

    def _loss(
        self,
        y: np.ndarray,
        probabilities: np.ndarray,
        sample_weight: np.ndarray,
    ) -> float:
        eps = 1e-12
        data_loss = -np.average(
            y * np.log(probabilities + eps) + (1.0 - y) * np.log(1.0 - probabilities + eps),
            weights=sample_weight,
        )
        return float(data_loss + self._penalty())

    def _logits(
        self,
        parties: list[PassiveParty],
        indices: np.ndarray | None = None,
        *,
        stage: str = "epoch",
        step: int | None = None,
    ) -> np.ndarray:
        n = parties[0].n_rows if indices is None else len(indices)
        logits = np.full(n, self.intercept_, dtype=float)
        active_name = parties[0].name
        for party in parties:
            local = party.local_logits(self.weights_[party.name], indices)
            if party.name == active_name:
                logits += local
            else:
                delivered = self.transport.send(
                    local,
                    message_type="local_logits",
                    sender_role=party.name,
                    receiver_role=active_name,
                    direction="forward",
                    stage=stage,
                    step=step,
                )
                logits += np.asarray(delivered, dtype=float)
        return logits

    def fit(
        self,
        active: ActiveParty,
        passive: list[PassiveParty],
        validation_active: ActiveParty | None = None,
        validation_passive: list[PassiveParty] | None = None,
    ) -> VFLLogisticRegression:
        self._validate_hyperparameters()
        parties: list[PassiveParty] = [active, *passive]
        self._validate_party_collection(parties, context="training")
        self.active_party_name_ = active.name
        self.trained_party_names_ = tuple(party.name for party in parties)
        n = active.n_rows
        if self.early_stopping_rounds is not None and validation_active is None:
            raise ValueError("validation data are required when early stopping is enabled")
        self._initialize(parties)
        self.loss_history_.clear()
        self.validation_loss_history_.clear()
        self.n_iter_ = 0
        self.converged_ = False
        self.best_iteration_ = None
        y = active.labels
        sample_weight = self._sample_weights(y)
        if not np.isfinite(sample_weight).all() or float(sample_weight.sum()) <= 0.0:
            raise ValueError("sample weights must be finite with positive total weight")
        rng = np.random.default_rng(self.seed)
        noise_rng = np.random.default_rng(self.seed + 104729)
        batch_size = min(self.batch_size or n, n)

        validation_parties: list[PassiveParty] | None = None
        validation_labels: np.ndarray | None = None
        validation_weights: np.ndarray | None = None
        if validation_active is not None:
            validation_passive = validation_passive or []
            validation_parties = [validation_active, *validation_passive]
            self._validate_party_collection(validation_parties, context="validation")
            if tuple(party.name for party in validation_parties) != self.trained_party_names_:
                raise ValueError(
                    "validation data must provide the same ordered VFL parties as training"
                )
            validation_labels = validation_active.labels
            validation_weights = self._sample_weights(validation_labels)
            if not np.isfinite(validation_weights).all() or float(validation_weights.sum()) <= 0.0:
                raise ValueError("validation weights must be finite with positive total weight")

        best_validation_loss = np.inf
        best_weights: dict[str, np.ndarray] | None = None
        best_intercept = self.intercept_
        rounds_without_improvement = 0
        previous = np.inf
        for epoch in range(self.max_iter):
            logits = self._logits(parties, stage="epoch", step=epoch)
            probs = _sigmoid(logits)
            loss = self._loss(y, probs, sample_weight)
            self.loss_history_.append(loss)
            self.n_iter_ = epoch + 1
            if abs(previous - loss) < self.tolerance:
                self.converged_ = True
                break
            previous = loss

            order = np.arange(n, dtype=int) if batch_size == n else rng.permutation(n)
            rate = self._epoch_learning_rate(epoch)
            for start in range(0, n, batch_size):
                batch = order[start : start + batch_size]
                batch_logits = self._logits(parties, batch, stage="epoch", step=epoch)
                batch_probs = _sigmoid(batch_logits)
                batch_weights = sample_weight[batch]
                batch_weight_sum = float(batch_weights.sum())
                if batch_weight_sum <= 0.0:
                    raise ValueError("every optimization batch must have positive total weight")
                residual = (batch_probs - y[batch]) * batch_weights
                residual *= len(batch) / batch_weight_sum
                delivered_residuals: dict[str, np.ndarray] = {active.name: residual}
                for party in passive:
                    party_residual = residual
                    if self.residual_dp_backend is not None:
                        party_residual = self.residual_dp_backend.release(residual)
                    elif self.residual_noise_std > 0.0:
                        party_residual = residual + noise_rng.normal(
                            scale=self.residual_noise_std,
                            size=residual.shape,
                        )
                    delivered_residuals[party.name] = np.asarray(
                        self.transport.send(
                            party_residual,
                            message_type="residual_signal",
                            sender_role=active.name,
                            receiver_role=party.name,
                            direction="backward",
                            stage="epoch",
                            step=epoch,
                        ),
                        dtype=float,
                    )
                for party in parties:
                    grad = party.local_gradient(delivered_residuals[party.name], batch)
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

            if (
                validation_parties is not None
                and validation_labels is not None
                and validation_weights is not None
            ):
                validation_probability = _sigmoid(
                    self._logits(validation_parties, stage="validation_epoch", step=epoch)
                )
                validation_loss = self._loss(
                    validation_labels,
                    validation_probability,
                    validation_weights,
                )
                self.validation_loss_history_.append(validation_loss)
                if validation_loss < best_validation_loss - 1e-12:
                    best_validation_loss = validation_loss
                    best_weights = {name: weights.copy() for name, weights in self.weights_.items()}
                    best_intercept = self.intercept_
                    self.best_iteration_ = epoch
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                if (
                    self.early_stopping_rounds is not None
                    and rounds_without_improvement >= self.early_stopping_rounds
                ):
                    self.converged_ = True
                    break

        if self.early_stopping_rounds is not None and best_weights is not None:
            self.weights_ = best_weights
            self.intercept_ = best_intercept
        elif self.best_iteration_ is None and self.n_iter_:
            self.best_iteration_ = self.n_iter_ - 1

        final_probability = _sigmoid(self._logits(parties, stage="final_training"))
        final_loss = self._loss(y, final_probability, sample_weight)
        if self.loss_history_:
            self.loss_history_[-1] = final_loss
        else:
            self.loss_history_.append(final_loss)
        return self

    def _inference_parties(self, parties: list[PassiveParty]) -> list[PassiveParty]:
        if not self.weights_ or not self.trained_party_names_ or self.active_party_name_ is None:
            raise RuntimeError("model is not fitted")
        if not parties:
            raise ValueError("at least one party is required")
        by_name = {party.name: party for party in parties}
        if len(by_name) != len(parties):
            raise ValueError("inference party names must be unique")
        unknown = set(by_name) - set(self.trained_party_names_)
        if unknown:
            raise ValueError(f"unknown inference parties: {sorted(unknown)}")
        if self.active_party_name_ not in by_name:
            raise ValueError("the active party cannot be omitted at inference")
        missing = set(self.trained_party_names_) - set(by_name)
        if missing and self.missing_party_policy == "error":
            raise ValueError(
                "missing inference parties: " + ", ".join(sorted(missing)) + "; "
                "set missing_party_policy='zero_contribution' only for an explicitly "
                "evaluated fallback configuration"
            )
        ordered = [by_name[name] for name in self.trained_party_names_ if name in by_name]
        self._validate_party_collection(ordered, context="inference")
        return ordered

    def decision_function(self, parties: list[PassiveParty]) -> np.ndarray:
        ordered = self._inference_parties(parties)
        return self._logits(ordered, stage="inference")

    def predict_proba(self, parties: list[PassiveParty]) -> np.ndarray:
        probability = _sigmoid(self.decision_function(parties))
        return np.column_stack([1.0 - probability, probability])

    def predict(self, parties: list[PassiveParty], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(parties)[:, 1] >= threshold).astype(int)

    def privacy_report(self, *, delta: float) -> dict[str, float | int | str] | None:
        """Return accounting for the optional residual message mechanism, if enabled."""
        if self.residual_dp_backend is None:
            return None
        report = self.residual_dp_backend.privacy_report(delta=delta)
        report["protected_message"] = "active-to-passive residual_signal"
        report["non_guarantee"] = (
            "other VFL messages are outside this mechanism; this is not end-to-end VFL DP"
        )
        return report
