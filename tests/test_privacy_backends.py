from __future__ import annotations

import numpy as np

from vertimosaic.privacy.backends import (
    GaussianDPBackend,
    GaussianZCDPAccountant,
    PrivacyBackendRegistry,
)


def test_gaussian_zcdp_accounting_composes_releases() -> None:
    accountant = GaussianZCDPAccountant(noise_multiplier=2.0)
    accountant.step(3)
    assert accountant.releases == 3
    assert np.isclose(accountant.rho, 3.0 / 8.0)
    epsilon = accountant.epsilon(1e-6)
    assert np.isfinite(epsilon)
    assert epsilon > 0.0


def test_gaussian_backend_is_seed_reproducible_and_reports_scope() -> None:
    values = np.array([1.0, -2.0, 3.0])
    first = GaussianDPBackend(l2_sensitivity=1.5, noise_multiplier=2.0, seed=7)
    second = GaussianDPBackend(l2_sensitivity=1.5, noise_multiplier=2.0, seed=7)
    released_first = first.release(values)
    released_second = second.release(values)
    assert np.array_equal(released_first, released_second)
    assert not np.array_equal(released_first, values)
    report = first.privacy_report(delta=1e-6)
    assert report["releases"] == 1
    assert report["epsilon"] > 0.0
    assert "end-to-end" in str(report["scope"])


def test_privacy_registry_does_not_claim_planned_backends_are_implemented() -> None:
    registry = PrivacyBackendRegistry()
    assert registry.available == ("gaussian-zcdp",)
    assert "psi" in registry.planned
