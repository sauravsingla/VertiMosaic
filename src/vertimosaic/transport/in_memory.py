"""Single-process transport for research simulations."""

from __future__ import annotations

from dataclasses import dataclass, field

from .messages import Message


@dataclass(slots=True)
class AuditEvent:
    kind: str
    sender: str
    receiver: str
    scalar_count: int
    estimated_bytes: int


@dataclass(slots=True)
class InMemoryTransport:
    """Records protocol metadata without retaining private payload values."""

    events: list[AuditEvent] = field(default_factory=list)

    def send(self, message: Message) -> Message:
        self.events.append(
            AuditEvent(
                kind=message.kind,
                sender=message.sender,
                receiver=message.receiver,
                scalar_count=message.scalar_count,
                estimated_bytes=message.estimated_bytes,
            )
        )
        return message

    def summary(self) -> dict[str, int]:
        return {
            "messages": len(self.events),
            "scalar_values": sum(event.scalar_count for event in self.events),
            "estimated_bytes": sum(event.estimated_bytes for event in self.events),
        }
