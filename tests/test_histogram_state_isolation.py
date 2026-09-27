# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from vertimosaic.parties import PassiveParty


def test_detached_same_name_party_cannot_resolve_training_routing_handle() -> None:
    train = PassiveParty("telecom", np.array([[0.0], [1.0], [2.0], [3.0]]))
    candidates = train.candidate_histograms(
        gradients=np.array([-1.0, -0.5, 0.5, 1.0]),
        hessians=np.ones(4),
        indices=np.arange(4),
        max_bins=2,
        min_samples_leaf=1,
    )
    split_ref = candidates[0]["split_ref"]
    handle = train.export_histogram_routing_state()

    detached = PassiveParty("telecom", np.array([[-10.0], [1.0], [2.0], [10.0]]))
    with pytest.raises(ValueError, match="local tree state"):
        detached.route_split(np.arange(4), split_ref, handle)

    train.share_histogram_routing_state_with(detached)
    left, right = detached.route_split(np.arange(4), split_ref, handle)
    assert np.array_equal(left, np.array([0, 1]))
    assert np.array_equal(right, np.array([2, 3]))


def test_party_module_has_no_hidden_histogram_registry() -> None:
    import vertimosaic.parties.core as core

    assert not hasattr(core, "_HISTOGRAM_THRESHOLD_REGISTRY")
