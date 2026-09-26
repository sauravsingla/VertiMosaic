from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Message:
    """Ephemeral metadata envelope for one simulated federated communication."""

    message_type: str
    sender_role: str
    receiver_role: str
    shape: tuple[int, ...] | None
    scalar_count: int
    estimated_bytes: int
    direction: str | None = None
    stage: str | None = None
    step: int | None = None


@dataclass(frozen=True)
class AuditEvent:
    """Persisted communication metadata; private payload values are never retained."""

    message_type: str
    sender_role: str
    receiver_role: str
    shape: tuple[int, ...] | None
    scalar_count: int
    estimated_bytes: int
    direction: str | None = None
    stage: str | None = None
    step: int | None = None

    @classmethod
    def from_message(cls, message: Message) -> AuditEvent:
        return cls(
            message.message_type,
            message.sender_role,
            message.receiver_role,
            message.shape,
            message.scalar_count,
            message.estimated_bytes,
            message.direction,
            message.stage,
            message.step,
        )


@dataclass
class InMemoryTransport:
    """Metadata-only communication audit.

    Each send creates an ephemeral :class:`Message`. Payload values are returned
    to the protocol caller but never retained by the transport or audit log.
    """

    audit_log: list[AuditEvent] = field(default_factory=list)

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
        shape, count, size = self._metadata(payload)
        message = Message(
            message_type,
            sender_role,
            receiver_role,
            shape,
            count,
            size,
            direction,
            stage,
            step,
        )
        self.audit_log.append(AuditEvent.from_message(message))
        return payload

    @staticmethod
    def _metadata(payload: Any) -> tuple[tuple[int, ...] | None, int, int]:
        if isinstance(payload, np.ndarray):
            return tuple(payload.shape), int(payload.size), int(payload.nbytes)
        if np.isscalar(payload):
            array = np.asarray(payload)
            return tuple(array.shape), 1, int(array.nbytes)
        if isinstance(payload, (tuple, list)) and all(np.isscalar(value) for value in payload):
            array = np.asarray(payload)
            return tuple(array.shape), int(array.size), int(array.nbytes)
        return None, 0, 0

    @property
    def estimated_payload_bytes(self) -> int:
        return sum(event.estimated_bytes for event in self.audit_log)
