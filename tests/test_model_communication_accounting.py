import numpy as np

from vertimosaic.evaluation import communication_breakdown
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.transport import InMemoryTransport


def _parties() -> tuple[ActiveParty, list[PassiveParty]]:
    rng = np.random.default_rng(5)
    rows = 80
    active = ActiveParty("bank", rng.normal(size=(rows, 2)), np.tile([0.0, 1.0], rows // 2))
    passive = [
        PassiveParty("telecom", rng.normal(size=(rows, 2))),
        PassiveParty("insurance", rng.normal(size=(rows, 2))),
        PassiveParty("retail", rng.normal(size=(rows, 2))),
    ]
    return active, passive


def test_logistic_counts_cross_party_messages_by_passive_party() -> None:
    active, passive = _parties()
    model = VFLLogisticRegression(max_iter=2, learning_rate=0.05, seed=5)
    model.fit(active, passive)
    frame = communication_breakdown(model.transport.audit_log)
    assert set(frame["party"]) == {"telecom", "insurance", "retail"}
    assert set(frame["direction"]) == {"forward", "backward"}
    assert not any(event.sender_role == event.receiver_role for event in model.transport.audit_log)


def test_logistic_inference_uses_message_transport_for_passive_logits() -> None:
    active, passive = _parties()
    model = VFLLogisticRegression(max_iter=2, learning_rate=0.05, seed=5)
    model.fit(active, passive)
    before = len(model.transport.audit_log)
    model.predict_proba([active, *passive])
    inference_events = model.transport.audit_log[before:]
    assert len(inference_events) == len(passive)
    assert all(event.message_type == "local_logits" for event in inference_events)
    assert all(event.stage == "inference" for event in inference_events)


class ZeroResidualTransport(InMemoryTransport):
    """Test transport proving passive gradients consume the delivered payload."""

    def send(self, payload: object, **kwargs: object) -> object:
        delivered = super().send(payload, **kwargs)
        if kwargs.get("message_type") == "residual_signal":
            return np.zeros_like(np.asarray(delivered, dtype=float))
        return delivered


def test_logistic_passive_gradient_uses_message_delivered_residual() -> None:
    active, passive = _parties()
    model = VFLLogisticRegression(
        max_iter=1,
        learning_rate=0.05,
        seed=5,
        transport=ZeroResidualTransport(),
    )
    model.fit(active, [passive[0]])
    assert np.allclose(model.weights_["telecom"], 0.0)
    assert not np.allclose(model.weights_["bank"], 0.0)


def test_gbdt_counts_gradient_and_histogram_messages_per_passive_party() -> None:
    active, passive = _parties()
    model = VFLHistGBDT(n_estimators=2, max_depth=1, min_samples_leaf=5, seed=5)
    model.fit(active, passive)
    frame = communication_breakdown(model.transport.audit_log)
    assert set(frame["party"]) == {"telecom", "insurance", "retail"}
    assert set(frame["direction"]) == {"forward", "backward"}
