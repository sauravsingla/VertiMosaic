from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, rankdata

from vertimosaic.linkage.base import LinkageResult


def _row_factor(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("linkage inputs must be 2D")
    if len(arr) == 0:
        raise ValueError("linkage inputs must be non-empty")
    medians = np.nanmedian(arr, axis=0)
    safe = np.where(np.isnan(arr), medians, arr)
    center = np.median(safe, axis=0)
    scale = np.median(np.abs(safe - center), axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardized = (safe - center) / scale
    return np.mean(np.tanh(standardized), axis=1)


def _gaussian_rank(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average")
    quantiles = (ranks - 0.5) / len(values)
    return norm.ppf(np.clip(quantiles, 1e-6, 1.0 - 1e-6))


@dataclass(frozen=True)
class GaussianCopulaLinker:
    """Target-blind rank/Gaussian-copula donor linker for external profiles."""

    cross_party_correlation: float = 0.25
    seed: int = 42
    stochasticity: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.cross_party_correlation <= 1.0:
            raise ValueError("cross_party_correlation must be in [0, 1]")
        if self.stochasticity < 0.0:
            raise ValueError("stochasticity must be non-negative")

    def link(self, anchor: np.ndarray, donor: np.ndarray) -> LinkageResult:
        anchor_factor = _gaussian_rank(_row_factor(anchor))
        donor_factor = _gaussian_rank(_row_factor(donor))
        rng = np.random.default_rng(self.seed)
        rho = self.cross_party_correlation
        independent = rng.normal(size=len(anchor_factor))
        desired = rho * anchor_factor + np.sqrt(max(1.0 - rho * rho, 0.0)) * independent
        if self.stochasticity:
            desired += rng.normal(scale=self.stochasticity, size=len(desired))
        donor_order = np.argsort(donor_factor)
        sorted_factor = donor_factor[donor_order]
        positions = np.searchsorted(sorted_factor, desired, side="left")
        positions = np.clip(positions, 0, len(donor_order) - 1)
        left = np.maximum(positions - 1, 0)
        choose_left = np.abs(sorted_factor[left] - desired) < np.abs(
            sorted_factor[positions] - desired
        )
        positions = np.where(choose_left, left, positions)
        donor_indices = donor_order[positions].astype(int)
        unique = int(np.unique(donor_indices).size)
        reuse_fraction = 1.0 - unique / max(len(donor_indices), 1)
        return LinkageResult(
            donor_indices=donor_indices,
            donor_reuse_fraction=float(reuse_fraction),
            unique_donors=unique,
            anchor_rows=len(anchor_factor),
            donor_rows=len(donor_factor),
        )
