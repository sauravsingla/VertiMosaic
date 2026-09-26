from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from vertimosaic.transport import AuditEvent


def communication_event_frame(events: Iterable[AuditEvent]) -> pd.DataFrame:
    """Return metadata-only per-message communication telemetry with cumulative payload."""
    rows: list[dict[str, Any]] = []
    cumulative_bytes = 0
    for sequence, event in enumerate(events):
        cumulative_bytes += event.estimated_bytes
        rows.append(
            {
                "sequence": sequence,
                "direction": event.direction or "unspecified",
                "stage": event.stage or "unspecified",
                "step": event.step,
                "message_type": event.message_type,
                "sender_role": event.sender_role,
                "receiver_role": event.receiver_role,
                "shape": event.shape,
                "scalar_count": event.scalar_count,
                "estimated_bytes": event.estimated_bytes,
                "cumulative_bytes": cumulative_bytes,
                "traffic_type": "SIMULATED PAYLOAD SIZE",
            }
        )
    return pd.DataFrame(rows)


def communication_totals(events: Iterable[AuditEvent]) -> dict[str, int | str]:
    """Summarize simulated payload without implying measured network traffic."""
    items = list(events)
    forward = [event for event in items if event.direction == "forward"]
    backward = [event for event in items if event.direction == "backward"]
    return {
        "traffic_type": "SIMULATED PAYLOAD SIZE",
        "message_count": len(items),
        "scalar_count": sum(event.scalar_count for event in items),
        "estimated_bytes": sum(event.estimated_bytes for event in items),
        "forward_message_count": len(forward),
        "forward_estimated_bytes": sum(event.estimated_bytes for event in forward),
        "backward_message_count": len(backward),
        "backward_estimated_bytes": sum(event.estimated_bytes for event in backward),
    }


def communication_breakdown(events: Iterable[AuditEvent]) -> pd.DataFrame:
    """Aggregate simulated payload by direction, party, and epoch/tree step."""
    frame = communication_event_frame(events)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "direction",
                "sender_role",
                "stage",
                "step",
                "message_count",
                "scalar_count",
                "estimated_bytes",
            ]
        )
    grouped = (
        frame.groupby(
            ["direction", "sender_role", "stage", "step"],
            dropna=False,
            sort=True,
        )
        .agg(
            message_count=("sequence", "count"),
            scalar_count=("scalar_count", "sum"),
            estimated_bytes=("estimated_bytes", "sum"),
        )
        .reset_index()
    )
    return grouped
