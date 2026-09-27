import numpy as np

from vertimosaic.parties import PassiveParty


def test_party_routing_is_local_and_deterministic() -> None:
    x = np.array([[0.1], [0.7], [0.2], [0.9]])
    party = PassiveParty("telecom", x)
    left, right = party.route(np.arange(4), 0, 0.5)
    assert np.array_equal(left, np.array([0, 2]))
    assert np.array_equal(right, np.array([1, 3]))


def test_histogram_candidates_hide_numeric_threshold_and_route_by_opaque_reference() -> None:
    x = np.array([[0.1], [0.2], [0.7], [0.9]])
    party = PassiveParty("telecom", x)
    gradients = np.array([-1.0, -0.5, 0.5, 1.0])
    hessians = np.ones(4)
    candidates = party.candidate_histograms(
        gradients,
        hessians,
        np.arange(4),
        max_bins=4,
        min_samples_leaf=1,
    )
    assert candidates
    candidate = candidates[0]
    assert "threshold" not in candidate
    assert "feature" not in candidate
    assert "threshold_index" not in candidate
    split_ref = candidate["split_ref"]
    assert split_ref.feature_ref == 0
    assert isinstance(split_ref.bin_ref, int)
    assert "_threshold" not in repr(split_ref)
    left, right = party.route_split(np.arange(4), split_ref)
    assert len(left) + len(right) == 4
    assert set(left).isdisjoint(set(right))
