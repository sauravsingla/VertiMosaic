"""Partial entity-overlap simulation."""

from __future__ import annotations

import numpy as np


def availability_mask(rows: int, availability: float, seed: int) -> np.ndarray:
    if not 0 <= availability <= 1:
        raise ValueError("availability must be between 0 and 1")
    rng = np.random.default_rng(seed)
    count = int(round(rows * availability))
    selected = rng.choice(rows, size=count, replace=False)
    mask = np.zeros(rows, dtype=bool)
    mask[selected] = True
    return mask
