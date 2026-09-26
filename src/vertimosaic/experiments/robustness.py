from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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


def apply_categorical_frequency_drift(
    values: np.ndarray,
    *,
    strength: float = 0.25,
    seed: int = 42,
) -> np.ndarray:
    """Shift category frequencies toward each column's dominant observed category.

    ``strength=0`` preserves the empirical sampling distribution and
    ``strength=1`` puts all sampled non-missing values on the dominant category.
    The operation is deterministic for a fixed seed and does not mutate the input.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("categorical drift strength must be in [0, 1]")
    original = np.asarray(values, dtype=object)
    squeeze = original.ndim == 1
    if original.ndim not in {1, 2}:
        raise ValueError("categorical drift accepts one- or two-dimensional arrays")
    matrix = original.reshape(-1, 1).copy() if squeeze else original.copy()
    rng = np.random.default_rng(seed)
    for column_index in range(matrix.shape[1]):
        column = pd.Series(matrix[:, column_index], dtype="object")
        observed = column.dropna()
        if observed.empty:
            continue
        counts = observed.value_counts(sort=False)
        categories = counts.index.to_numpy(dtype=object)
        empirical = counts.to_numpy(dtype=float)
        empirical /= empirical.sum()
        dominant = int(np.argmax(empirical))
        shifted = (1.0 - strength) * empirical
        shifted[dominant] += strength
        non_missing = column.notna().to_numpy()
        matrix[non_missing, column_index] = rng.choice(
            categories,
            size=int(non_missing.sum()),
            p=shifted,
        )
    return matrix[:, 0] if squeeze else matrix


def dropout_scenarios() -> dict[str, tuple[str, ...]]:
    return {
        "none": (),
        "telecom_absent": ("telecom",),
        "insurance_absent": ("insurance",),
        "retail_absent": ("retail",),
        "telecom_retail_absent": ("telecom", "retail"),
    }
