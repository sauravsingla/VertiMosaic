"""Protocol privacy-boundary validation helpers."""

from __future__ import annotations

from vertimosaic.transport import InMemoryTransport


def audit_is_metadata_only(transport: InMemoryTransport) -> bool:
    """Return true when persisted audit events expose metadata fields only."""
    allowed = {"kind", "sender", "receiver", "scalar_count", "estimated_bytes"}
    return all(set(event.__dataclass_fields__) == allowed for event in transport.events)
