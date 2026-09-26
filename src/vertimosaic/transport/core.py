from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Message:
    message_type: str
    sender_role: str
    receiver_role: str
    shape: tuple[int, ...] | None
    scalar_count: int
    estimated_bytes: int


@dataclass(frozen=True)
class AuditEvent:
    message_type: str
    sender_role: str
    receiver_role: str
    shape: tuple[int, ...] | None
    scalar_count: int
    estimated_bytes: int


@dataclass
class InMemoryTransport:
    """Metadata-only communication audit.

    Payload values are returned to the caller but never retained by the transport or audit log.
    """

    audit_log: list[AuditEvent] = field(default_factory=list)

    def send(self, payload: Any, *, message_type: str, sender_role: str, receiver_role: str) -> Any:
        shape, count, size = self._metadata(payload)
        self.audit_log.append(
            AuditEvent(message_type, sender_role, receiver_role, shape, count, size)
        )
        return payload

    @staticmethod
    def _metadata(payload: Any) -> tuple[tuple[int, ...] | None, int, int]:
        if isinstance(payload, np.ndarray):
            return tuple(payload.shape), int(payload.size), int(payload.nbytes)
        if np.isscalar(payload):
            arr = np.asarray(payload)
            return tuple(arr.shape), 1, int(arr.nbytes)
        if isinstance(payload, (tuple, list)) and all(np.isscalar(x) for x in payload):
            arr = np.asarray(payload)
            return tuple(arr.shape), int(arr.size), int(arr.nbytes)
        return None, 0, 0

    @property
    def estimated_payload_bytes(self) -> int:
        return sum(event.estimated_bytes for event in self.audit_log)
