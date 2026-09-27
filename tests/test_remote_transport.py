# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading

import numpy as np

from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty
from vertimosaic.transport import (
    InMemoryTransport,
    ReferenceRelayServer,
    RemoteHTTPTransport,
    StructuredPayload,
    Transport,
)


def _start_server(
    role: str,
    token: str | None = None,
) -> tuple[ReferenceRelayServer, threading.Thread, str]:
    server = ReferenceRelayServer(("127.0.0.1", 0), receiver_role=role, bearer_token=token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}/v1/messages"


def _stop_server(server: ReferenceRelayServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def test_transport_protocol_is_structural() -> None:
    assert isinstance(InMemoryTransport(), Transport)


def test_remote_transport_round_trip_and_audit_metadata() -> None:
    server, thread, endpoint = _start_server("telecom", "secret")
    transport = RemoteHTTPTransport(
        endpoints={"telecom": endpoint},
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
        _stop_server(server, thread)
    assert np.array_equal(delivered, payload)
    assert len(transport.audit_log) == 1
    assert transport.audit_log[0].shape == (4, 3)
    assert transport.audit_log[0].estimated_bytes == payload.nbytes


def test_remote_transport_supports_structured_payload() -> None:
    server, thread, endpoint = _start_server("bank")
    transport = RemoteHTTPTransport(
        endpoints={"bank": endpoint},
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
        _stop_server(server, thread)
    assert isinstance(delivered, tuple)
    assert np.array_equal(delivered[0], value[0])
    assert np.array_equal(delivered[1], value[1])
    assert transport.estimated_payload_bytes == 32


def test_logistic_vfl_can_use_remote_network_transport() -> None:
    bank_server, bank_thread, bank_endpoint = _start_server("bank", "bank-token")
    telco_server, telco_thread, telco_endpoint = _start_server("telecom", "telco-token")
    transport = RemoteHTTPTransport(
        endpoints={"bank": bank_endpoint, "telecom": telco_endpoint},
        bearer_tokens={"bank": "bank-token", "telecom": "telco-token"},
        allow_insecure_http=True,
        timeout_seconds=2.0,
        max_retries=1,
    )
    rng = np.random.default_rng(4)
    x_bank = rng.normal(size=(60, 2))
    x_telco = rng.normal(size=(60, 2))
    y = (x_bank[:, 0] + x_telco[:, 0] > 0).astype(float)
    active = ActiveParty("bank", x_bank, y)
    passive = PassiveParty("telecom", x_telco)
    model = VFLLogisticRegression(
        learning_rate=0.05,
        max_iter=3,
        tolerance=0.0,
        transport=transport,
    )
    try:
        model.fit(active, [passive])
        probability = model.predict_proba([active, passive])[:, 1]
    finally:
        _stop_server(bank_server, bank_thread)
        _stop_server(telco_server, telco_thread)
    assert probability.shape == (60,)
    assert np.isfinite(probability).all()
    assert transport.audit_log


def test_remote_transport_rejects_plain_http_by_default() -> None:
    try:
        RemoteHTTPTransport(endpoints={"bank": "http://127.0.0.1:9999/v1/messages"})
    except ValueError as exc:
        assert "HTTPS" in str(exc)
    else:
        raise AssertionError("plain HTTP must require explicit test-only opt-in")
