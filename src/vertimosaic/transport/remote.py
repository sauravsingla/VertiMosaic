# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import hmac
import io
import json
import ssl
import threading
import time
import zlib
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np

from vertimosaic.transport.core import AuditEvent, InMemoryTransport, Message, StructuredPayload

SCHEMA_VERSION = 2
_ARRAY_CODECS = {"json", "npy-zlib-base64"}
_MAX_DECOMPRESSED_ARRAY_BYTES = 256 * 1024 * 1024


def _encode_value(value: Any, *, array_codec: str = "json") -> Any:
    if array_codec not in _ARRAY_CODECS:
        raise ValueError(f"unsupported array codec: {array_codec}")
    if isinstance(value, np.ndarray):
        if array_codec == "json":
            return {
                "__type__": "ndarray",
                "codec": "json",
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "data": value.tolist(),
            }
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(value), allow_pickle=False)
        packed = zlib.compress(buffer.getvalue())
        return {
            "__type__": "ndarray",
            "codec": "npy-zlib-base64",
            "nbytes": int(np.asarray(value).nbytes),
            "data": base64.b64encode(packed).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {
            "__type__": "tuple",
            "items": [_encode_value(item, array_codec=array_codec) for item in value],
        }
    if isinstance(value, list):
        return [_encode_value(item, array_codec=array_codec) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _encode_value(item, array_codec=array_codec) for key, item in value.items()
        }
    if value.__class__.__name__ == "OpaqueSplitReference" and hasattr(value, "feature_ref"):
        return {
            "__type__": "opaque_split_reference",
            "feature_ref": int(value.feature_ref),
            "bin_ref": int(value.bin_ref),
        }
    if value.__class__.__name__ == "HistogramRoutingState" and hasattr(value, "state_ref"):
        return {
            "__type__": "histogram_routing_state",
            "party_name": str(value.party_name),
            "state_ref": str(value.state_ref),
            "n_features": int(value.n_features),
            "max_bins": int(value.max_bins),
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
        codec = value.get("codec", "json")
        if codec == "json":
            array = np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
            return array.reshape(tuple(int(item) for item in value["shape"]))
        if codec == "npy-zlib-base64":
            claimed_nbytes = int(value.get("nbytes", -1))
            if claimed_nbytes < 0 or claimed_nbytes > _MAX_DECOMPRESSED_ARRAY_BYTES:
                raise ValueError("compressed ndarray exceeds configured decoded-size limit")
            compressed = base64.b64decode(value["data"], validate=True)
            decompressor = zlib.decompressobj()
            raw = decompressor.decompress(
                compressed,
                _MAX_DECOMPRESSED_ARRAY_BYTES + 64 * 1024 + 1,
            )
            if decompressor.unconsumed_tail or len(raw) > _MAX_DECOMPRESSED_ARRAY_BYTES + 64 * 1024:
                raise ValueError("compressed ndarray expands beyond configured limit")
            raw += decompressor.flush()
            with io.BytesIO(raw) as buffer:
                array = np.load(buffer, allow_pickle=False)
            if int(array.nbytes) != claimed_nbytes:
                raise ValueError("compressed ndarray byte count mismatch")
            return array
        raise ValueError(f"unsupported ndarray codec: {codec}")
    if marker == "tuple":
        return tuple(_decode_value(item) for item in value["items"])
    if marker == "opaque_split_reference":
        from vertimosaic.parties.core import OpaqueSplitReference

        return OpaqueSplitReference(
            feature_ref=int(value["feature_ref"]),
            bin_ref=int(value["bin_ref"]),
        )
    if marker == "histogram_routing_state":
        from vertimosaic.parties.core import HistogramRoutingState

        return HistogramRoutingState(
            party_name=str(value["party_name"]),
            state_ref=str(value["state_ref"]),
            n_features=int(value["n_features"]),
            max_bins=int(value["max_bins"]),
        )
    return {key: _decode_value(item) for key, item in value.items()}


def _validate_envelope(envelope: Any) -> str | None:
    if not isinstance(envelope, dict):
        return "request body must be a JSON object"
    required = {
        "schema_version": int,
        "message_id": str,
        "message_type": str,
        "sender_role": str,
        "receiver_role": str,
        "sent_at_unix": (int, float),
        "nonce": str,
    }
    for key, expected in required.items():
        value = envelope.get(key)
        if not isinstance(value, expected) or (isinstance(value, str) and not value):
            return f"invalid or missing {key}"
    if "payload" not in envelope:
        return "missing payload"
    return None


@dataclass
class RemoteHTTPTransport(InMemoryTransport):
    """Reference network transport with bounded replay protection and optional compression.

    HTTPS remains required by default. Bearer tokens are configured per receiver and
    can be paired with per-sender authorization on ``ReferenceRelayServer``. The
    default JSON ndarray codec is retained for compatibility; ``npy-zlib-base64`` is
    available for substantially smaller numeric payloads while keeping a JSON envelope.

    These controls harden transport integrity/availability. They do not make residuals,
    gradients, Hessians, or routing messages cryptographically private.
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
    array_codec: str = "json"

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")
        if self.array_codec not in _ARRAY_CODECS:
            raise ValueError(f"array_codec must be one of {sorted(_ARRAY_CODECS)}")
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
            "sent_at_unix": time.time(),
            "nonce": uuid4().hex,
            "array_codec": self.array_codec,
            "payload": _encode_value(wire_value, array_codec=self.array_codec),
        }
        body = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-VertiMosaic-Schema": str(SCHEMA_VERSION),
            "X-VertiMosaic-Message-ID": message_id,
            "X-VertiMosaic-Sender": sender_role,
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
                with urlopen(  # nosec B310 - endpoint schemes are restricted above
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
    server_version = "VertiMosaicReferenceRelay/2"

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        server = self.server
        if not isinstance(server, ReferenceRelayServer):
            self.send_error(500)
            return
        if self.path != server.path:
            self.send_error(404)
            return

        length_header = self.headers.get("Content-Length", "0")
        try:
            length = int(length_header)
        except ValueError:
            self.send_error(400, "invalid Content-Length")
            return
        if length <= 0 or length > server.max_body_bytes:
            self.send_error(413)
            return

        try:
            envelope = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_error(400)
            return

        schema_error = _validate_envelope(envelope)
        if schema_error is not None:
            self.send_error(400, schema_error)
            return
        if envelope["schema_version"] != SCHEMA_VERSION:
            self.send_error(409, "schema version mismatch")
            return
        if envelope["receiver_role"] != server.receiver_role:
            self.send_error(403, "receiver role mismatch")
            return

        sender_role = str(envelope["sender_role"])
        message_type = str(envelope["message_type"])
        if server.allowed_senders is not None and sender_role not in server.allowed_senders:
            self.send_error(403, "sender role not authorized")
            return
        allowed_types = server.allowed_message_types.get(sender_role)
        if allowed_types is not None and message_type not in allowed_types:
            self.send_error(403, "message type not authorized for sender")
            return

        if not server.authorize(sender_role, self.headers.get("Authorization")):
            self.send_error(401)
            return

        sent_at = float(envelope["sent_at_unix"])
        if abs(time.time() - sent_at) > server.max_clock_skew_seconds:
            self.send_error(408, "message timestamp outside replay window")
            return
        if not server.allow_request(sender_role):
            self.send_error(429, "rate limit exceeded")
            return

        message_id = str(envelope["message_id"])
        cached = server.get_cached(message_id)
        if cached is None:
            encoded_payload = envelope["payload"]
            if server.request_handler is not None:
                try:
                    result = server.request_handler(
                        message_type,
                        sender_role,
                        _decode_value(encoded_payload),
                    )
                    encoded_payload = _encode_value(
                        result,
                        array_codec=str(envelope.get("array_codec", "json")),
                    )
                except (TypeError, ValueError, RuntimeError) as exc:
                    self.send_error(422, str(exc))
                    return
            cached = {
                "schema_version": SCHEMA_VERSION,
                "message_id": message_id,
                "payload": encoded_payload,
            }
            server.put_cached(message_id, cached)

        response = json.dumps(cached, separators=(",", ":")).encode("utf-8")
        if len(response) > server.max_response_bytes:
            self.send_error(413, "response exceeds configured maximum")
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:
        return


class ReferenceRelayServer(ThreadingHTTPServer):
    """Reference receiver with bounded idempotency, authorization and rate controls."""

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        receiver_role: str,
        bearer_token: str | None = None,
        sender_tokens: dict[str, str] | None = None,
        allowed_senders: set[str] | None = None,
        allowed_message_types: dict[str, set[str]] | None = None,
        path: str = "/v1/messages",
        max_body_bytes: int = 64 * 1024 * 1024,
        max_response_bytes: int = 64 * 1024 * 1024,
        idempotency_ttl_seconds: float = 300.0,
        max_idempotency_entries: int = 10_000,
        max_clock_skew_seconds: float = 120.0,
        rate_limit_per_minute: int = 600,
        request_handler: Callable[[str, str, Any], Any] | None = None,
    ) -> None:
        super().__init__(server_address, _RelayHandler)
        if max_body_bytes <= 0 or max_response_bytes <= 0:
            raise ValueError("body and response limits must be positive")
        if idempotency_ttl_seconds <= 0 or max_idempotency_entries <= 0:
            raise ValueError("idempotency limits must be positive")
        if max_clock_skew_seconds <= 0 or rate_limit_per_minute <= 0:
            raise ValueError("replay window and rate limit must be positive")
        self.receiver_role = receiver_role
        self.bearer_token = bearer_token
        self.sender_tokens = dict(sender_tokens or {})
        self.allowed_senders = set(allowed_senders) if allowed_senders is not None else None
        self.allowed_message_types = {
            key: set(value) for key, value in (allowed_message_types or {}).items()
        }
        self.path = path
        self.max_body_bytes = max_body_bytes
        self.max_response_bytes = max_response_bytes
        self.idempotency_ttl_seconds = idempotency_ttl_seconds
        self.max_idempotency_entries = max_idempotency_entries
        self.max_clock_skew_seconds = max_clock_skew_seconds
        self.rate_limit_per_minute = rate_limit_per_minute
        self.request_handler = request_handler
        self.idempotency_cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._request_times: dict[str, deque[float]] = {}
        self._security_lock = threading.Lock()

    def authorize(self, sender_role: str, supplied_header: str | None) -> bool:
        expected = self.sender_tokens.get(sender_role, self.bearer_token)
        if expected is None:
            return True
        if supplied_header is None or not supplied_header.startswith("Bearer "):
            return False
        supplied = supplied_header.removeprefix("Bearer ")
        return hmac.compare_digest(supplied, expected)

    def _prune_cache(self, now: float) -> None:
        cutoff = now - self.idempotency_ttl_seconds
        while self.idempotency_cache:
            _, (created, _) = next(iter(self.idempotency_cache.items()))
            if created >= cutoff:
                break
            self.idempotency_cache.popitem(last=False)
        while len(self.idempotency_cache) > self.max_idempotency_entries:
            self.idempotency_cache.popitem(last=False)

    def get_cached(self, message_id: str) -> dict[str, Any] | None:
        with self._security_lock:
            now = time.time()
            self._prune_cache(now)
            entry = self.idempotency_cache.get(message_id)
            if entry is None:
                return None
            created, payload = entry
            if created < now - self.idempotency_ttl_seconds:
                self.idempotency_cache.pop(message_id, None)
                return None
            self.idempotency_cache.move_to_end(message_id)
            return payload

    def put_cached(self, message_id: str, payload: dict[str, Any]) -> None:
        with self._security_lock:
            now = time.time()
            self._prune_cache(now)
            self.idempotency_cache[message_id] = (now, payload)
            self.idempotency_cache.move_to_end(message_id)
            while len(self.idempotency_cache) > self.max_idempotency_entries:
                self.idempotency_cache.popitem(last=False)

    def allow_request(self, sender_role: str) -> bool:
        with self._security_lock:
            now = time.time()
            cutoff = now - 60.0
            timestamps = self._request_times.setdefault(sender_role, deque())
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.rate_limit_per_minute:
                return False
            timestamps.append(now)
            return True

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
