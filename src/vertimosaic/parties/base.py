"""Party abstractions for vertically partitioned feature ownership."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Party:
    """A party that exclusively owns its local feature matrix."""

    name: str
    X: np.ndarray = field(repr=False)
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        if self.X.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix")
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError("feature_names length must match X columns")

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])
