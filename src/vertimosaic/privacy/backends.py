# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt

import numpy as np


@dataclass
class GaussianZCDPAccountant:
    """Account repeated Gaussian-mechanism releases using zCDP composition.

    The accounting is formal for Gaussian releases whose L2 sensitivity is bounded
    by the caller-supplied value. It does not establish end-to-end DP for a VFL
    training protocol unless every sensitive release satisfies the stated bound
    and is routed through the mechanism.
    """

    noise_multiplier: float
    releases: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.noise_multiplier) or self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be finite and positive")
        if self.releases < 0:
            raise ValueError("releases must be non-negative")

    @property
    def rho(self) -> float:
        return self.releases / (2.0 * self.noise_multiplier**2)

    def step(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count must be positive")
        self.releases += int(count)

    def epsilon(self, *, delta: float) -> float:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.releases == 0:
            return 0.0
        rho = self.rho
        return float(rho + 2.0 * sqrt(rho * log(1.0 / delta)))

    def summary(self, *, delta: float) -> dict[str, float | int | str]:
        return {
            "mechanism": "Gaussian",
            "accounting": "zCDP composition converted to (epsilon, delta)-DP",
            "releases": self.releases,
            "noise_multiplier": float(self.noise_multiplier),
            "rho": float(self.rho),
            "epsilon": self.epsilon(delta=delta),
            "delta": float(delta),
        }


@dataclass
class GaussianDPBackend:
    """Optional Gaussian release mechanism for sensitivity-bounded research messages.

    ``l2_sensitivity`` is a protocol contract, not something this class can infer.
    The caller must prove or enforce that neighboring inputs change the released
    vector by at most this amount. Noise is sampled with standard deviation
    ``noise_multiplier * l2_sensitivity``.

    This backend is intentionally decoupled from default VertiMosaic training so
    enabling it is an explicit research choice and cannot silently upgrade privacy
    claims for existing experiments.
    """

    l2_sensitivity: float
    noise_multiplier: float
    seed: int = 42
    accountant: GaussianZCDPAccountant = field(init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.l2_sensitivity) or self.l2_sensitivity <= 0:
            raise ValueError("l2_sensitivity must be finite and positive")
        self.accountant = GaussianZCDPAccountant(self.noise_multiplier)
        self._rng = np.random.default_rng(self.seed)

    @property
    def noise_std(self) -> float:
        return float(self.l2_sensitivity * self.noise_multiplier)

    def release(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("DP release values must be finite")
        noise = self._rng.normal(0.0, self.noise_std, size=array.shape)
        self.accountant.step()
        return array + noise

    def privacy_report(self, *, delta: float) -> dict[str, float | int | str]:
        report = self.accountant.summary(delta=delta)
        report["l2_sensitivity"] = float(self.l2_sensitivity)
        report["noise_std"] = self.noise_std
        report["scope"] = (
            "release-level accounting only; end-to-end VFL DP requires every sensitive "
            "message to satisfy the declared sensitivity bound"
        )
        return report


@dataclass(frozen=True)
class PrivacyBackendRegistry:
    """Discoverability metadata for optional privacy research backends."""

    available: tuple[str, ...] = ("gaussian-zcdp",)
    planned: tuple[str, ...] = ("psi", "secure-aggregation", "mpc", "homomorphic-encryption")

    def describe(self) -> dict[str, tuple[str, ...]]:
        return {"available": self.available, "planned": self.planned}
