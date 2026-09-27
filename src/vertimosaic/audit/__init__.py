"""Metadata-only protocol audit, structured logging and communication telemetry API."""

from vertimosaic.audit.logging import UnsafeLogFieldError, log_event
from vertimosaic.evaluation.communication import (
    communication_breakdown,
    communication_event_frame,
    communication_totals,
)
from vertimosaic.transport import AuditEvent

__all__ = [
    "AuditEvent",
    "UnsafeLogFieldError",
    "communication_breakdown",
    "communication_event_frame",
    "communication_totals",
    "log_event",
]
