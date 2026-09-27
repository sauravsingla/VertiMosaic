import numpy as np

from vertimosaic.evaluation import communication_breakdown
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty


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


def test_gbdt_counts_gradient_and_histogram_messages_per_passive_party() -> None:
    active, passive = _parties()
    model = VFLHistGBDT(n_estimators=2, max_depth=1, min_samples_leaf=5, seed=5)
    model.fit(active, passive)
    frame = communication_breakdown(model.transport.audit_log)
    assert set(frame["party"]) == {"telecom", "insurance", "retail"}
    assert set(frame["direction"]) == {"forward", "backward"}
