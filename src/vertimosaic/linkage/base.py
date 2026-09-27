from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class LinkageResult:
    donor_indices: np.ndarray
    donor_reuse_fraction: float
    unique_donors: int
    anchor_rows: int
    donor_rows: int


class Linker(Protocol):
    def link(self, anchor: np.ndarray, donor: np.ndarray) -> LinkageResult: ...
