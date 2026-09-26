"""Controlled local feature drift utilities."""

from __future__ import annotations

import numpy as np


def apply_mean_shift(
    x: np.ndarray, amount: float, columns: tuple[int, ...] | None = None
) -> np.ndarray:
    out = np.asarray(x, dtype=float).copy()
    cols = columns if columns is not None else tuple(range(out.shape[1]))
    out[:, cols] += amount
    return out


def apply_missingness(x: np.ndarray, rate: float, seed: int = 42) -> np.ndarray:
    if not 0 <= rate <= 1:
        raise ValueError("rate must be between 0 and 1")
    out = np.asarray(x, dtype=float).copy()
    rng = np.random.default_rng(seed)
    out[rng.random(out.shape) < rate] = np.nan
    return out
