import numpy as np

from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, Coordinator, PassiveParty
from vertimosaic.transport import Message


def _fixture() -> tuple[
    ActiveParty,
    PassiveParty,
    PassiveParty,
    PassiveParty,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(3)
    bank_x = rng.normal(size=(100, 2))
    telecom_x = rng.normal(size=(100, 3))
    insurance_x = rng.normal(size=(100, 2))
    retail_x = rng.normal(size=(100, 4))
    y = rng.integers(0, 2, size=100).astype(float)
    return (
        ActiveParty("bank", bank_x, y),
        PassiveParty("telecom", telecom_x),
        PassiveParty("insurance", insurance_x),
        PassiveParty("retail", retail_x),
        telecom_x,
        insurance_x,
        retail_x,
    )


def _active_does_not_retain(active: ActiveParty, raw: np.ndarray) -> bool:
    return all(value is not raw for value in vars(active).values())


def test_bank_never_receives_telecom_raw_matrix() -> None:
    active, telecom, insurance, retail, raw, _, _ = _fixture()
    VFLLogisticRegression(max_iter=2).fit(active, [telecom, insurance, retail])
    assert _active_does_not_retain(active, raw)


def test_bank_never_receives_insurance_raw_matrix() -> None:
    active, telecom, insurance, retail, _, raw, _ = _fixture()
    VFLLogisticRegression(max_iter=2).fit(active, [telecom, insurance, retail])
    assert _active_does_not_retain(active, raw)


def test_bank_never_receives_retail_raw_matrix() -> None:
    active, telecom, insurance, retail, _, _, raw = _fixture()
    VFLLogisticRegression(max_iter=2).fit(active, [telecom, insurance, retail])
    assert _active_does_not_retain(active, raw)


def test_coordinator_contains_no_passive_raw_features() -> None:
    coordinator = Coordinator(("bank", "telecom", "insurance", "retail"))
    assert set(vars(coordinator)) == {"party_names", "transport"}
    assert not hasattr(coordinator, "_x")
    assert not hasattr(coordinator, "features")


def test_transport_contains_no_raw_feature_values() -> None:
    active, telecom, insurance, retail, _, _, _ = _fixture()
    model = VFLLogisticRegression(max_iter=3)
    model.fit(active, [telecom, insurance, retail])
    assert set(vars(model.transport)) == {"audit_log"}
    assert not hasattr(model.transport, "payloads")


def test_audit_log_contains_no_raw_rows() -> None:
    active, telecom, insurance, retail, _, _, _ = _fixture()
    model = VFLLogisticRegression(max_iter=3)
    model.fit(active, [telecom, insurance, retail])
    for event in model.transport.audit_log:
        assert not hasattr(event, "payload")
        assert not any(isinstance(value, np.ndarray) for value in vars(event).values())


def test_passive_party_never_receives_target_vector_in_logistic_unnecessarily() -> None:
    active, telecom, insurance, retail, _, _, _ = _fixture()
    model = VFLLogisticRegression(max_iter=2)
    model.fit(active, [telecom, insurance, retail])
    for party in (telecom, insurance, retail):
        assert not hasattr(party, "_y")
        assert not hasattr(party, "labels")
    assert all(event.message_type != "target" for event in model.transport.audit_log)


def test_raw_source_ids_not_in_messages() -> None:
    assert "source_id" not in Message.__dataclass_fields__
    assert "raw_source_id" not in Message.__dataclass_fields__


def test_transport_and_audit_do_not_retain_passive_raw_matrix() -> None:
    active, telecom, _, _, _, _, _ = _fixture()
    model = VFLLogisticRegression(max_iter=3)
    model.fit(active, [telecom])
    assert not hasattr(model.transport, "payloads")
    for event in model.transport.audit_log:
        assert not hasattr(event, "payload")
        assert vars(event).get("_x") is None
