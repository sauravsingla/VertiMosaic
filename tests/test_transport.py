import numpy as np

from vertimosaic.transport import InMemoryTransport


def test_transport_metadata_for_array_scalar_list_and_object() -> None:
    t = InMemoryTransport()
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert t.send(arr, message_type="a", sender_role="p", receiver_role="q") is arr
    t.send(3.0, message_type="b", sender_role="p", receiver_role="q")
    t.send([1, 2, 3], message_type="c", sender_role="p", receiver_role="q")
    t.send({"opaque": "metadata"}, message_type="d", sender_role="p", receiver_role="q")
    assert t.audit_log[0].shape == (2, 3)
    assert t.audit_log[0].scalar_count == 6
    assert t.audit_log[-1].shape is None
    assert t.audit_log[-1].scalar_count == 0
    assert t.estimated_payload_bytes >= arr.nbytes
