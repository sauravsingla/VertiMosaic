# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from vertimosaic.parties import ActiveParty, Coordinator, PassiveParty


def test_party_validation_and_local_operations() -> None:
    with pytest.raises(ValueError):
        PassiveParty("x", np.array([1.0, 2.0]))
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    p = PassiveParty("x", x)
    idx = np.array([0, 2])
    w = np.array([0.5, -0.25])
    assert np.allclose(p.local_logits(w, idx), x[idx] @ w)
    residual = np.array([1.0, -1.0])
    assert np.allclose(p.local_gradient(residual, idx), x[idx].T @ residual / 2)
    assert p.candidate_histograms(np.ones(4), np.ones(4), np.arange(4), 4, 3) == []


def test_active_party_rejects_bad_labels() -> None:
    x = np.ones((3, 1))
    with pytest.raises(ValueError):
        ActiveParty("bank", x, np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        ActiveParty("bank", x, np.array([0.0, 2.0, 1.0]))


def test_coordinator_validates_party_names_and_membership() -> None:
    with pytest.raises(ValueError, match="at least one party"):
        Coordinator(())
    with pytest.raises(ValueError, match="must be unique"):
        Coordinator(("bank", "bank"))

    coordinator = Coordinator(("bank", "telecom"))
    assert coordinator.has_party("bank")
    assert not coordinator.has_party("retail")
