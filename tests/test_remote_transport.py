# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading

import numpy as np

from vertimosaic.transport import ReferenceRelayServer, RemoteHTTPTransport, StructuredPayload


def test_remote_transport_round_trip_and_audit_metadata() -> None:
    server = ReferenceRelayServer(("127.0.0.1", 0), receiver_role="telecom", bearer_token="secret")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    transport = RemoteHTTPTransport(
        endpoints={"telecom": f"http://{host}:{port}/v1/messages"},
        bearer_tokens={"telecom": "secret"},
        allow_insecure_http=True,
        timeout_seconds=2.0,
    )
    payload = np.arange(12, dtype=np.float64).reshape(4, 3)
    try:
        delivered = transport.send(
            payload,
            message_type="local_logits",
            sender_role="bank",
            receiver_role="telecom",
            direction="forward",
            stage="test",
            step=1,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    assert np.array_equal(delivered, payload)
    assert len(transport.audit_log) == 1
    assert transport.audit_log[0].shape == (4, 3)
    assert transport.audit_log[0].estimated_bytes == payload.nbytes


def test_remote_transport_supports_structured_payload() -> None:
    server = ReferenceRelayServer(("127.0.0.1", 0), receiver_role="bank")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    transport = RemoteHTTPTransport(
        endpoints={"bank": f"http://{host}:{port}/v1/messages"},
        allow_insecure_http=True,
    )
    value = (np.array([1, 2]), np.array([3, 4]))
    payload = StructuredPayload(value=value, shape=(4,), scalar_count=4, estimated_bytes=32)
    try:
        delivered = transport.send(
            payload,
            message_type="partition_routing_indices",
            sender_role="telecom",
            receiver_role="bank",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    assert isinstance(delivered, tuple)
    assert np.array_equal(delivered[0], value[0])
    assert np.array_equal(delivered[1], value[1])
    assert transport.estimated_payload_bytes == 32


def test_remote_transport_rejects_plain_http_by_default() -> None:
    try:
        RemoteHTTPTransport(endpoints={"bank": "http://127.0.0.1:9999/v1/messages"})
    except ValueError as exc:
        assert "HTTPS" in str(exc)
    else:
        raise AssertionError("plain HTTP must require explicit test-only opt-in")
