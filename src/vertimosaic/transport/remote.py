# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import ssl
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np

from vertimosaic.transport.core import AuditEvent, InMemoryTransport, Message, StructuredPayload

SCHEMA_VERSION = 1


def _encode_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode_value(item) for key, item in value.items()}
    if value.__class__.__name__ == "OpaqueSplitReference" and hasattr(value, "feature_ref"):
        return {
            "__type__": "opaque_split_reference",
            "feature_ref": int(value.feature_ref),
            "bin_ref": int(value.bin_ref),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"remote transport cannot serialize {type(value).__name__}")


def _decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    marker = value.get("__type__")
    if marker == "ndarray":
        array = np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
        return array.reshape(tuple(int(item) for item in value["shape"]))
    if marker == "tuple":
        return tuple(_decode_value(item) for item in value["items"])
    if marker == "opaque_split_reference":
        from vertimosaic.parties.core import OpaqueSplitReference

        return OpaqueSplitReference(
            feature_ref=int(value["feature_ref"]),
            bin_ref=int(value["bin_ref"]),
        )
    return {key: _decode_value(item) for key, item in value.items()}


@dataclass
class RemoteHTTPTransport(InMemoryTransport):
    """Reference network transport for VertiMosaic protocol messages.

    The transport posts versioned JSON envelopes to receiver-specific HTTP(S)
    endpoints. HTTPS is required by default. Client certificates can be supplied
    for mutual TLS, while receiver-specific bearer tokens provide application-level
    authentication. A stable message ID is reused across retries so a receiver can
    deduplicate repeated delivery attempts.

    This class transports protocol messages only. It does not by itself make the
    VertiMosaic threat model cryptographically private; payloads such as residuals
    and gradients retain their documented leakage surface.
    """

    endpoints: dict[str, str] = field(default_factory=dict)
    bearer_tokens: dict[str, str] = field(default_factory=dict, repr=False)
    ca_file: str | None = None
    client_cert: str | None = None
    client_key: str | None = None
    timeout_seconds: float = 10.0
    max_retries: int = 2
    backoff_seconds: float = 0.25
    allow_insecure_http: bool = False

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")
        for role, endpoint in self.endpoints.items():
            if endpoint.startswith("https://"):
                continue
            if endpoint.startswith("http://") and self.allow_insecure_http:
                continue
            raise ValueError(
                f"remote endpoint for {role} must use HTTPS; "
                "allow_insecure_http is intended only for local tests"
            )

    def _ssl_context(self) -> ssl.SSLContext | None:
        if not any(endpoint.startswith("https://") for endpoint in self.endpoints.values()):
            return None
        context = ssl.create_default_context(cafile=self.ca_file)
        if self.client_cert is not None:
            context.load_cert_chain(self.client_cert, keyfile=self.client_key)
        return context

    def send(
        self,
        payload: Any,
        *,
        message_type: str,
        sender_role: str,
        receiver_role: str,
        direction: str | None = None,
        stage: str | None = None,
        step: int | None = None,
    ) -> Any:
        endpoint = self.endpoints.get(receiver_role)
        if endpoint is None:
            raise ValueError(f"no remote endpoint configured for receiver role {receiver_role}")

        if isinstance(payload, StructuredPayload):
            wire_value = payload.value
            shape = payload.shape
            count = payload.scalar_count
            size = payload.estimated_bytes
        else:
            wire_value = payload
            shape, count, size = self._metadata(payload)

        message_id = uuid4().hex
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "message_id": message_id,
            "message_type": message_type,
            "sender_role": sender_role,
            "receiver_role": receiver_role,
            "direction": direction,
            "stage": stage,
            "step": step,
            "payload": _encode_value(wire_value),
        }
        body = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-VertiMosaic-Schema": str(SCHEMA_VERSION),
            "X-VertiMosaic-Message-ID": message_id,
        }
        token = self.bearer_tokens.get(receiver_role)
        if token:
            headers["Authorization"] = f"Bearer {token}"

        context = self._ssl_context()
        last_error: Exception | None = None
        response_payload: Any = None
        for attempt in range(self.max_retries + 1):
            request = Request(endpoint, data=body, headers=headers, method="POST")
            try:
                # Endpoint schemes are restricted above to HTTPS, or explicitly opted-in
                # HTTP for loopback/local tests, so urllib cannot reach file/custom schemes.
                with urlopen(  # nosec B310
                    request,
                    timeout=self.timeout_seconds,
                    context=context,
                ) as response:
                    response_body = response.read()
                    decoded = json.loads(response_body.decode("utf-8"))
                    if decoded.get("schema_version") != SCHEMA_VERSION:
                        raise RuntimeError("remote response schema version mismatch")
                    if decoded.get("message_id") != message_id:
                        raise RuntimeError("remote response message ID mismatch")
                    if "payload" not in decoded:
                        raise RuntimeError("remote response is missing payload")
                    response_payload = _decode_value(decoded["payload"])
                    break
            except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"remote delivery to {receiver_role} failed after {attempt + 1} attempts"
                    ) from exc
                time.sleep(self.backoff_seconds * (2**attempt))
        else:
            raise RuntimeError(
                "remote transport retry loop terminated unexpectedly"
            ) from last_error

        message = Message(
            message_type=message_type,
            sender_role=sender_role,
            receiver_role=receiver_role,
            shape=shape,
            scalar_count=count,
            estimated_bytes=size,
            direction=direction,
            stage=stage,
            step=step,
            payload=None,
        )
        self.audit_log.append(AuditEvent.from_message(message))
        return response_payload


class _RelayHandler(BaseHTTPRequestHandler):
    server_version = "VertiMosaicReferenceRelay/1"

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        server = self.server
        if not isinstance(server, ReferenceRelayServer):
            self.send_error(500)
            return
        if self.path != server.path:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > server.max_body_bytes:
            self.send_error(413)
            return
        expected_token = server.bearer_token
        if expected_token is not None:
            supplied = self.headers.get("Authorization")
            if supplied != f"Bearer {expected_token}":
                self.send_error(401)
                return
        try:
            envelope = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(400)
            return
        if envelope.get("schema_version") != SCHEMA_VERSION:
            self.send_error(409, "schema version mismatch")
            return
        message_id = envelope.get("message_id")
        if not isinstance(message_id, str) or not message_id:
            self.send_error(400, "missing message ID")
            return
        if not isinstance(envelope.get("message_type"), str):
            self.send_error(400, "missing message type")
            return
        if envelope.get("receiver_role") != server.receiver_role:
            self.send_error(403, "receiver role mismatch")
            return
        cached = server.idempotency_cache.get(message_id)
        if cached is None:
            cached = {
                "schema_version": SCHEMA_VERSION,
                "message_id": message_id,
                "payload": envelope.get("payload"),
            }
            server.idempotency_cache[message_id] = cached
        response = json.dumps(cached, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:
        return


class ReferenceRelayServer(ThreadingHTTPServer):
    """Small reference receiver used for process/network transport demonstrations."""

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        receiver_role: str,
        bearer_token: str | None = None,
        path: str = "/v1/messages",
        max_body_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        super().__init__(server_address, _RelayHandler)
        self.receiver_role = receiver_role
        self.bearer_token = bearer_token
        self.path = path
        self.max_body_bytes = max_body_bytes
        self.idempotency_cache: dict[str, dict[str, Any]] = {}

    def enable_mtls(
        self,
        *,
        server_cert: str,
        server_key: str,
        client_ca: str,
    ) -> None:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(server_cert, keyfile=server_key)
        context.load_verify_locations(cafile=client_ca)
        context.verify_mode = ssl.CERT_REQUIRED
        self.socket = context.wrap_socket(self.socket, server_side=True)
