from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class StructuredPayload:
    """Explicit metadata for a structured simulated protocol payload.

    The wrapped value is ephemeral. Only shape/scalar-count/estimated-byte metadata
    is copied into the audit trail.
    """

    value: Any = field(repr=False, compare=False)
    shape: tuple[int, ...] | None
    scalar_count: int
    estimated_bytes: int

    def __post_init__(self) -> None:
        if self.scalar_count < 0 or self.estimated_bytes < 0:
            raise ValueError("structured payload metadata must be non-negative")


@dataclass(frozen=True)
class Message:
    """Ephemeral envelope for one federated communication."""

    message_type: str
    sender_role: str
    receiver_role: str
    shape: tuple[int, ...] | None
    scalar_count: int
    estimated_bytes: int
    direction: str | None = None
    stage: str | None = None
    step: int | None = None
    payload: Any = field(default=None, repr=False, compare=False)


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


@runtime_checkable
class Transport(Protocol):
    """Structural interface used by VertiMosaic protocol message transports."""

    audit_log: list[AuditEvent]

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
    ) -> Any: ...

    @property
    def estimated_payload_bytes(self) -> int: ...


@dataclass
class InMemoryTransport:
    """Ephemeral Message transport with metadata-only persistent auditing.

    Every protocol ``send`` creates a :class:`Message` carrying the ephemeral payload.
    The transport returns that message payload to the protocol caller but persists only
    an :class:`AuditEvent`, so row-level values are never retained in the audit log.
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
        if isinstance(payload, StructuredPayload):
            wire_value = payload.value
            shape = payload.shape
            count = payload.scalar_count
            size = payload.estimated_bytes
        else:
            wire_value = payload
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
            wire_value,
        )
        self.audit_log.append(AuditEvent.from_message(message))
        return message.payload

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
