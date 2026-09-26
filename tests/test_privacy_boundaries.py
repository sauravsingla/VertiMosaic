import numpy as np

from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty


def test_transport_audit_contains_no_raw_feature_matrix() -> None:
    rng = np.random.default_rng(2)
    x1 = rng.normal(size=(120, 2))
    x2 = rng.normal(size=(120, 3))
    y = (x1[:, 0] + x2[:, 1] > 0).astype(float)
    bank = ActiveParty("bank", x1, ("b1", "b2"), y)
    telco = PassiveParty("telecom", x2, ("t1", "t2", "t3"))
    model = VFLLogisticRegression(epochs=2, batch_size=50, seed=2).fit([bank, telco], bank)
    assert model.transport.events
    assert all(not hasattr(event, "payload") for event in model.transport.events)
    assert all(event.kind in {"local_logit", "residual_signal"} for event in model.transport.events)
