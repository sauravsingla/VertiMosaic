"""Federated protocol message types.

Messages intentionally contain summaries or protocol signals, never raw feature matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Message:
    """A typed in-memory protocol message."""

    kind: str
    sender: str
    receiver: str
    payload: dict[str, Any]

    @property
    def scalar_count(self) -> int:
        """Estimate the number of scalar values represented by the payload."""
        total = 0
        for value in self.payload.values():
            if isinstance(value, np.ndarray):
                total += int(value.size)
            elif np.isscalar(value):
                total += 1
            elif isinstance(value, (list, tuple)):
                total += len(value)
        return total

    @property
    def estimated_bytes(self) -> int:
        """Approximate payload bytes for numeric arrays/scalars."""
        total = 0
        for value in self.payload.values():
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
            elif isinstance(value, (float, int, np.floating, np.integer)):
                total += 8
            elif isinstance(value, (list, tuple)):
                total += 8 * len(value)
        return total
