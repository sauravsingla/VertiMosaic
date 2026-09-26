import numpy as np

from vertimosaic.evaluation import communication_breakdown, communication_event_frame, communication_totals
from vertimosaic.transport import InMemoryTransport


def test_communication_benchmarking_tracks_metadata_without_payload_values() -> None:
    transport = InMemoryTransport()
    payload = np.arange(8, dtype=float)
    transport.send(
        payload,
        message_type="local_logits",
        sender_role="telecom",
        receiver_role="active",
        direction="forward",
        stage="epoch",
        step=2,
    )
    transport.send(
        np.ones(4),
        message_type="residual_signal",
        sender_role="active",
        receiver_role="parties",
        direction="backward",
        stage="epoch",
        step=2,
    )

    frame = communication_event_frame(transport.audit_log)
    totals = communication_totals(transport.audit_log)
    breakdown = communication_breakdown(transport.audit_log)

    assert list(frame["direction"]) == ["forward", "backward"]
    assert list(frame["step"]) == [2, 2]
    assert frame["cumulative_bytes"].is_monotonic_increasing
    assert totals["forward_message_count"] == 1
    assert totals["backward_message_count"] == 1
    assert totals["traffic_type"] == "SIMULATED PAYLOAD SIZE"
    assert breakdown["estimated_bytes"].sum() == totals["estimated_bytes"]
    assert not any(isinstance(value, np.ndarray) for event in transport.audit_log for value in vars(event).values())
