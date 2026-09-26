"""Sanitized audit event helpers."""

from __future__ import annotations

from dataclasses import asdict

from vertimosaic.transport import InMemoryTransport


def export_audit_summary(transport: InMemoryTransport) -> list[dict[str, int | str]]:
    """Return metadata-only audit events."""
    return [asdict(event) for event in transport.events]
