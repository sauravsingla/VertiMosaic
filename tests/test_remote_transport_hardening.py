from __future__ import annotations

import threading

import numpy as np
import pytest

from vertimosaic.transport import ReferenceRelayServer, RemoteHTTPTransport


def _start_server(
    *,
    role: str = "telecom",
    **kwargs: object,
) -> tuple[ReferenceRelayServer, threading.Thread, str]:
    server = ReferenceRelayServer(("127.0.0.1", 0), receiver_role=role, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}/v1/messages"


def _stop(server: ReferenceRelayServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def test_remote_transport_supports_compressed_numpy_codec() -> None:
    server, thread, endpoint = _start_server()
    transport = RemoteHTTPTransport(
        endpoints={"telecom": endpoint},
        allow_insecure_http=True,
        array_codec="npy-zlib-base64",
    )
    values = np.arange(200, dtype=np.float64).reshape(20, 10)
    try:
        delivered = transport.send(
            values,
            message_type="local_logits",
            sender_role="bank",
            receiver_role="telecom",
        )
    finally:
        _stop(server, thread)
    assert np.array_equal(delivered, values)


def test_remote_server_enforces_sender_and_message_authorization() -> None:
    server, thread, endpoint = _start_server(
        allowed_senders={"bank"},
        allowed_message_types={"bank": {"local_logits"}},
        sender_tokens={"bank": "sender-secret"},
    )
    authorized = RemoteHTTPTransport(
        endpoints={"telecom": endpoint},
        bearer_tokens={"telecom": "sender-secret"},
        allow_insecure_http=True,
        max_retries=0,
    )
    wrong_sender = RemoteHTTPTransport(
        endpoints={"telecom": endpoint},
        bearer_tokens={"telecom": "sender-secret"},
        allow_insecure_http=True,
        max_retries=0,
    )
    try:
        delivered = authorized.send(
            np.array([1.0]),
            message_type="local_logits",
            sender_role="bank",
            receiver_role="telecom",
        )
        assert np.array_equal(delivered, np.array([1.0]))
        with pytest.raises(RuntimeError):
            wrong_sender.send(
                np.array([1.0]),
                message_type="local_logits",
                sender_role="retail",
                receiver_role="telecom",
            )
        with pytest.raises(RuntimeError):
            authorized.send(
                np.array([1.0]),
                message_type="gradient_hessian",
                sender_role="bank",
                receiver_role="telecom",
            )
    finally:
        _stop(server, thread)


def test_idempotency_cache_is_bounded_lru() -> None:
    server = ReferenceRelayServer(
        ("127.0.0.1", 0),
        receiver_role="bank",
        max_idempotency_entries=2,
    )
    try:
        server.put_cached("one", {"payload": 1})
        server.put_cached("two", {"payload": 2})
        assert server.get_cached("one") is not None
        server.put_cached("three", {"payload": 3})
        assert len(server.idempotency_cache) == 2
        assert server.get_cached("one") is not None
        assert server.get_cached("two") is None
        assert server.get_cached("three") is not None
    finally:
        server.server_close()
