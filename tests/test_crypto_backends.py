from __future__ import annotations

import numpy as np
import pytest

from vertimosaic.privacy.crypto import (
    AdditiveSecretSharingSum,
    OpenMinedPSIBackend,
    PaillierHomomorphicSum,
    PairwiseMaskSecureAggregation,
)


def test_pairwise_masks_cancel_in_secure_aggregate() -> None:
    backend = PairwiseMaskSecureAggregation()
    parties = ("bank", "telecom", "retail")
    pairwise_secrets = {
        backend.pair_key("bank", "telecom"): b"a" * 32,
        backend.pair_key("bank", "retail"): b"b" * 32,
        backend.pair_key("telecom", "retail"): b"c" * 32,
    }
    updates = {
        "bank": np.array([1, -2, 4]),
        "telecom": np.array([5, 3, -1]),
        "retail": np.array([-4, 8, 2]),
    }
    masked = [
        backend.mask_update(
            party_id=party,
            update=updates[party],
            party_ids=parties,
            pairwise_secrets=pairwise_secrets,
            round_id="round-7",
        )
        for party in parties
    ]
    aggregate = np.asarray(backend.aggregate(masked), dtype=np.int64)
    assert np.array_equal(aggregate, np.array([2, 9, 5]))
    assert "no dropout recovery" in backend.security_scope


def test_additive_secret_sharing_reconstructs_signed_vector() -> None:
    backend = AdditiveSecretSharingSum(share_count=3)
    value = np.array([4, -7, 11], dtype=np.int64)
    shares = backend.split(value)
    assert len(shares) == 3
    reconstructed = np.asarray(backend.reconstruct(shares), dtype=np.int64)
    assert np.array_equal(reconstructed, value)
    assert "not a general MPC" in backend.security_scope


def test_optional_psi_backend_fails_closed_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("vertimosaic.privacy.crypto.import_module", missing)
    with pytest.raises(RuntimeError, match="privacy-crypto"):
        OpenMinedPSIBackend().intersection(["a"], ["a"])


def test_optional_paillier_backend_fails_closed_without_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr("vertimosaic.privacy.crypto.import_module", missing)
    with pytest.raises(RuntimeError, match="privacy-crypto"):
        PaillierHomomorphicSum().sum([np.array([1]), np.array([2])])
