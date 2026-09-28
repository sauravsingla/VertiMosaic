from __future__ import annotations

import hashlib
import threading

import numpy as np
import pytest

from vertimosaic.transport import ReferenceRelayServer, RemoteHTTPTransport


def _start_server(**kwargs: object) -> tuple[ReferenceRelayServer, threading.Thread, str]:
    server = ReferenceRelayServer(("127.0.0.1", 0), receiver_role="telecom", **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}/v1/messages"


def _stop(server: ReferenceRelayServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def test_signed_request_response_and_wire_accounting() -> None:
    secret = b"signed-transport-secret-32-bytes!!"
    server, thread, endpoint = _start_server(
        sender_signing_keys={"bank": {"current": secret, "previous": b"p" * 32}},
        require_signed_messages=True,
    )
    transport = RemoteHTTPTransport(
        endpoints={"telecom": endpoint},
        signing_keys={"telecom": ("current", secret)},
        allow_insecure_http=True,
        array_codec="npy-zlib-base64",
    )
    try:
        value = np.arange(64, dtype=float)
        delivered = transport.send(
            value,
            message_type="local_logits",
            sender_role="bank",
            receiver_role="telecom",
        )
    finally:
        _stop(server, thread)
    assert np.array_equal(delivered, value)
    totals = transport.network_totals()
    assert totals["message_count"] == 1
    assert int(totals["request_body_bytes"]) > 0
    assert int(totals["response_body_bytes"]) > 0
    assert int(totals["measured_application_bytes"]) > int(totals["request_body_bytes"])
    assert totals["tls_record_bytes_measured"] is False
    assert float(totals["round_trip_p95_seconds"]) >= 0.0


def test_signed_server_rejects_unsigned_client() -> None:
    secret = b"signed-transport-secret-32-bytes!!"
    server, thread, endpoint = _start_server(
        sender_signing_keys={"bank": {"current": secret}},
        require_signed_messages=True,
    )
    transport = RemoteHTTPTransport(
        endpoints={"telecom": endpoint},
        allow_insecure_http=True,
        max_retries=0,
    )
    try:
        with pytest.raises(RuntimeError):
            transport.send(
                np.array([1.0]),
                message_type="local_logits",
                sender_role="bank",
                receiver_role="telecom",
            )
    finally:
        _stop(server, thread)


def test_idempotency_key_is_bound_to_request_digest() -> None:
    server = ReferenceRelayServer(("127.0.0.1", 0), receiver_role="telecom")
    try:
        first = {"payload": 1}
        digest = hashlib.sha256(b"first").hexdigest()
        server.put_cached("same-id", digest, first)
        assert server.get_cached("same-id", digest) == first
        with pytest.raises(ValueError, match="different request content"):
            server.get_cached("same-id", hashlib.sha256(b"second").hexdigest())
    finally:
        server.server_close()


def test_rotating_bearer_tokens_and_external_rate_limiter() -> None:
    calls: list[str] = []

    def external_limit(sender: str) -> bool:
        calls.append(sender)
        return len(calls) <= 1

    server = ReferenceRelayServer(
        ("127.0.0.1", 0),
        receiver_role="telecom",
        sender_tokens={"bank": ("old-token", "new-token")},
        external_rate_limiter=external_limit,
    )
    try:
        assert server.authorize("bank", "Bearer old-token")
        assert server.authorize("bank", "Bearer new-token")
        assert not server.authorize("bank", "Bearer invalid")
        assert server.allow_request("bank")
        assert not server.allow_request("bank")
        assert calls == ["bank", "bank"]
    finally:
        server.server_close()
