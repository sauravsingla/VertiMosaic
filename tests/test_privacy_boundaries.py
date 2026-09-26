import numpy as np

from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty


def test_transport_and_audit_do_not_retain_passive_raw_matrix() -> None:
    rng = np.random.default_rng(3)
    bank_x = rng.normal(size=(100, 2)); telco_x = rng.normal(size=(100, 3)); y = rng.integers(0, 2, size=100).astype(float)
    active = ActiveParty("bank", bank_x, y); telecom = PassiveParty("telecom", telco_x.copy())
    model = VFLLogisticRegression(max_iter=3).fit(active, [telecom])
    assert not hasattr(model.transport, "payloads")
    for event in model.transport.audit_log:
        assert not hasattr(event, "payload")
        assert vars(event).get("_x") is None


def test_raw_source_ids_are_not_message_fields() -> None:
    from vertimosaic.transport import Message
    assert "source_id" not in Message.__dataclass_fields__
