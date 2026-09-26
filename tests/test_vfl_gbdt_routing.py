import numpy as np

from vertimosaic.parties import PassiveParty


def test_party_routing_is_local_and_deterministic() -> None:
    x = np.array([[0.1], [0.7], [0.2], [0.9]])
    party = PassiveParty("telecom", x)
    left, right = party.route(np.arange(4), 0, 0.5)
    assert np.array_equal(left, np.array([0, 2]))
    assert np.array_equal(right, np.array([1, 3]))
