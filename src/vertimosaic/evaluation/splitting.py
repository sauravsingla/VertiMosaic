from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def entity_level_split(
    labels: np.ndarray,
    *,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> SplitIndices:
    """Create one entity-level split reused by every vertical party."""
    y = np.asarray(labels).reshape(-1)
    total = train_fraction + validation_fraction + test_fraction
    if abs(total - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    indices = np.arange(len(y), dtype=int)
    holdout_fraction = validation_fraction + test_fraction
    train, holdout = train_test_split(
        indices, test_size=holdout_fraction, random_state=seed, stratify=y
    )
    relative_test = test_fraction / holdout_fraction
    validation, test = train_test_split(
        holdout,
        test_size=relative_test,
        random_state=seed + 1,
        stratify=y[holdout],
    )
    return SplitIndices(
        train=np.sort(train), validation=np.sort(validation), test=np.sort(test)
    )
