from __future__ import annotations

import numpy as np

from vertimosaic.privacy.backends import (
    ClippedGaussianDPBackend,
    GaussianDPBackend,
    GaussianZCDPAccountant,
    PrivacyBackendRegistry,
)


def test_gaussian_zcdp_accounting_composes_releases() -> None:
    accountant = GaussianZCDPAccountant(noise_multiplier=2.0)
    accountant.step(3)
    assert accountant.releases == 3
    assert np.isclose(accountant.rho, 3.0 / 8.0)
    epsilon = accountant.epsilon(delta=1e-6)
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


def test_clipped_gaussian_enforces_message_sensitivity() -> None:
    backend = ClippedGaussianDPBackend(
        clip_l2_norm=2.0,
        noise_multiplier=1.5,
        adjacency="replace_one",
        seed=13,
    )
    value = np.array([6.0, 8.0])
    clipped, original_norm = backend.clip(value)
    assert np.isclose(original_norm, 10.0)
    assert np.isclose(np.linalg.norm(clipped), 2.0)
    assert np.isclose(backend.l2_sensitivity, 4.0)
    released = backend.release(value)
    assert released.shape == value.shape
    report = backend.privacy_report(delta=1e-6)
    assert report["sensitivity_enforcement"] == "L2 clipping before every release"
    assert report["releases"] == 1
    assert "message-level" in str(report["scope"])


def test_privacy_registry_distinguishes_core_optional_and_larger_protocols() -> None:
    registry = PrivacyBackendRegistry()
    assert "bounded-gaussian-zcdp" in registry.available
    assert "pairwise-mask-secagg" in registry.available
    assert "additive-secret-sharing-sum" in registry.available
    assert "openmined-psi" in registry.optional
    assert "paillier-homomorphic-sum" in registry.optional
    assert "general-purpose-mpc" in registry.planned
