from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any


class UnsafeLogFieldError(ValueError):
    """Raised when structured logging is asked to persist non-scalar/private data."""


_ALLOWED_SCALARS = (str, int, float, bool, type(None))


def _validate_fields(fields: Mapping[str, Any]) -> dict[str, str | int | float | bool | None]:
    safe: dict[str, str | int | float | bool | None] = {}
    for key, value in fields.items():
        if not isinstance(key, str) or not key:
            raise UnsafeLogFieldError("structured log field names must be non-empty strings")
        if not isinstance(value, _ALLOWED_SCALARS):
            raise UnsafeLogFieldError(
                f"structured log field {key!r} must be scalar metadata; "
                "row-level values are forbidden"
            )
        safe[key] = value
    return safe


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit one deterministic JSON metadata event without row-level payloads.

    Callers may record scalar operational metadata such as mode, party, seed,
    counts, durations, or status. Arrays, frames, mappings and other structured
    objects are rejected so raw rows/features cannot accidentally enter logs.
    """
    if not event:
        raise UnsafeLogFieldError("structured log event name must be non-empty")
    payload: dict[str, str | int | float | bool | None] = {"event": event}
    payload.update(_validate_fields(fields))
    logger.log(level, json.dumps(payload, sort_keys=True, separators=(",", ":")))
