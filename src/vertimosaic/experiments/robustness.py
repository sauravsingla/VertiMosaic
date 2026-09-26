from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_PARTIES = ("bank", "telecom", "insurance", "retail")


@dataclass(frozen=True)
class AvailabilityMasks:
    bank: np.ndarray
    telecom: np.ndarray
    insurance: np.ndarray
    retail: np.ndarray

    def intersection(self) -> np.ndarray:
        return self.bank & self.telecom & self.insurance & self.retail


def make_availability_masks(
    n_rows: int,
    *,
    seed: int = 42,
    bank: float = 1.0,
    telecom: float = 1.0,
    insurance: float = 1.0,
    retail: float = 1.0,
) -> AvailabilityMasks:
    """Create deterministic party-availability masks without silently dropping overlap."""
    rng = np.random.default_rng(seed)
    fractions = {"bank": bank, "telecom": telecom, "insurance": insurance, "retail": retail}
    masks: dict[str, np.ndarray] = {}
    for name in _PARTIES:
        fraction = fractions[name]
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("availability fractions must be in [0, 1]")
        count = int(round(n_rows * fraction))
        chosen = rng.permutation(n_rows)[:count]
        mask = np.zeros(n_rows, dtype=bool)
        mask[chosen] = True
        masks[name] = mask
    return AvailabilityMasks(**masks)


def apply_numeric_drift(
    values: np.ndarray,
    *,
    mean_shift: float = 0.0,
    variance_scale: float = 1.0,
    missingness_increase: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Apply controlled numeric feature drift without mutating the source array."""
    if variance_scale <= 0:
        raise ValueError("variance_scale must be positive")
    if not 0.0 <= missingness_increase <= 1.0:
        raise ValueError("missingness_increase must be in [0, 1]")
    out = np.asarray(values, dtype=float).copy()
    center = np.nanmean(out, axis=0)
    out = center + variance_scale * (out - center) + mean_shift
    if missingness_increase:
        rng = np.random.default_rng(seed)
        mask = rng.random(out.shape) < missingness_increase
        out[mask] = np.nan
    return out


def dropout_scenarios() -> dict[str, tuple[str, ...]]:
    return {
        "none": (),
        "telecom_absent": ("telecom",),
        "insurance_absent": ("insurance",),
        "retail_absent": ("retail",),
        "telecom_retail_absent": ("telecom", "retail"),
    }
