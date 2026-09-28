# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt

import numpy as np


@dataclass
class GaussianZCDPAccountant:
    """Account repeated Gaussian-mechanism releases using zCDP composition."""

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
    """Gaussian release with a caller-enforced L2-sensitivity contract.

    This legacy research backend is retained for experiments where sensitivity is
    established by protocol-specific reasoning outside this class. Prefer
    :class:`ClippedGaussianDPBackend` when the release can be bounded by clipping.
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
            "release-level accounting only; caller must establish the sensitivity bound; "
            "end-to-end VFL DP requires every sensitive release to be covered"
        )
        return report


@dataclass
class ClippedGaussianDPBackend:
    """Sensitivity-enforcing Gaussian release for one bounded vector message.

    The input vector is clipped to ``clip_l2_norm`` before noise is added. Under
    ``replace_one`` message adjacency, two clipped vectors can differ by at most
    ``2 * clip_l2_norm`` in L2 norm. Under ``add_remove`` adjacency the bound is
    ``clip_l2_norm``. This makes the sensitivity contract executable instead of
    caller-supplied, while remaining a *message-level* mechanism. It does not turn
    a complete VFL protocol into end-to-end differential privacy unless all
    sensitive releases and the relevant neighboring-dataset relation are covered.
    """

    clip_l2_norm: float
    noise_multiplier: float
    adjacency: str = "replace_one"
    seed: int = 42
    accountant: GaussianZCDPAccountant = field(init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.clip_l2_norm) or self.clip_l2_norm <= 0:
            raise ValueError("clip_l2_norm must be finite and positive")
        if self.adjacency not in {"replace_one", "add_remove"}:
            raise ValueError("adjacency must be replace_one or add_remove")
        self.accountant = GaussianZCDPAccountant(self.noise_multiplier)
        self._rng = np.random.default_rng(self.seed)

    @property
    def l2_sensitivity(self) -> float:
        factor = 2.0 if self.adjacency == "replace_one" else 1.0
        return float(factor * self.clip_l2_norm)

    @property
    def noise_std(self) -> float:
        return float(self.noise_multiplier * self.l2_sensitivity)

    def clip(self, value: np.ndarray) -> tuple[np.ndarray, float]:
        array = np.asarray(value, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError("DP release values must be finite")
        norm = float(np.linalg.norm(array.reshape(-1), ord=2))
        if norm <= self.clip_l2_norm or norm == 0.0:
            return array.copy(), norm
        return array * (self.clip_l2_norm / norm), norm

    def release(self, value: np.ndarray) -> np.ndarray:
        clipped, _ = self.clip(value)
        noise = self._rng.normal(0.0, self.noise_std, size=clipped.shape)
        self.accountant.step()
        return clipped + noise

    def privacy_report(self, *, delta: float) -> dict[str, float | int | str]:
        report = self.accountant.summary(delta=delta)
        report.update(
            {
                "clip_l2_norm": float(self.clip_l2_norm),
                "adjacency": self.adjacency,
                "l2_sensitivity": self.l2_sensitivity,
                "noise_std": self.noise_std,
                "sensitivity_enforcement": "L2 clipping before every release",
                "scope": (
                    "message-level clipped Gaussian mechanism; end-to-end VFL DP requires "
                    "all sensitive protocol releases and the dataset adjacency to be covered"
                ),
            }
        )
        return report


@dataclass(frozen=True)
class PrivacyBackendRegistry:
    """Discoverability metadata for privacy research backends and their scope."""

    available: tuple[str, ...] = (
        "gaussian-zcdp",
        "bounded-gaussian-zcdp",
        "pairwise-mask-secagg",
        "additive-secret-sharing-sum",
    )
    optional: tuple[str, ...] = ("openmined-psi", "paillier-homomorphic-sum")
    planned: tuple[str, ...] = (
        "dropout-resilient-secagg",
        "general-purpose-mpc",
        "he-vfl-training",
    )

    def describe(self) -> dict[str, tuple[str, ...]]:
        return {
            "available": self.available,
            "optional": self.optional,
            "planned": self.planned,
        }
