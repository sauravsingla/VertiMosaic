"""Active party that owns labels."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import Party


@dataclass(slots=True)
class ActiveParty(Party):
    y: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    def __post_init__(self) -> None:
        Party.__post_init__(self)
        self.y = np.asarray(self.y, dtype=float).reshape(-1)
        if len(self.y) != self.n_samples:
            raise ValueError("y length must match X rows")
