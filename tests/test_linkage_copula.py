import numpy as np
import pytest

from vertimosaic.linkage import GaussianCopulaLinker, validate_target_blind_inputs


def test_copula_linkage_is_deterministic_and_in_range() -> None:
    rng = np.random.default_rng(9)
    anchor = rng.normal(size=(100, 3))
    donor = rng.normal(size=(40, 4))
    linker = GaussianCopulaLinker(cross_party_correlation=0.25, seed=11)
    first = linker.link(anchor, donor)
    second = linker.link(anchor, donor)
    assert np.array_equal(first.donor_indices, second.donor_indices)
    assert first.donor_indices.min() >= 0
    assert first.donor_indices.max() < len(donor)
    assert first.anchor_rows == 100


def test_target_blind_validator_rejects_target() -> None:
    with pytest.raises(ValueError):
        validate_target_blind_inputs(["age", "default_payment"], "default_payment")
