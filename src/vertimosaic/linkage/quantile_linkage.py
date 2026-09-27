from __future__ import annotations

import numpy as np


def target_blind_rank_linkage(anchor: np.ndarray, donor: np.ndarray, seed: int = 42) -> np.ndarray:
    """Return donor row indices using target-blind rank proximity on unsupervised row summaries."""
    if anchor.ndim != 2 or donor.ndim != 2:
        raise ValueError("anchor and donor must be 2D")
    rng = np.random.default_rng(seed)
    a_score = np.nanmean(anchor, axis=1)
    d_score = np.nanmean(donor, axis=1)
    a_rank = np.argsort(np.argsort(a_score)) / max(len(a_score) - 1, 1)
    d_order = np.argsort(d_score)
    desired = np.clip(np.rint(a_rank * (len(donor) - 1)).astype(int), 0, len(donor) - 1)
    jitter = rng.integers(-1, 2, size=len(anchor))
    desired = np.clip(desired + jitter, 0, len(donor) - 1)
    return d_order[desired]
