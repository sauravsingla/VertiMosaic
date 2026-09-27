# SPDX-License-Identifier: Apache-2.0
import numpy as np

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.models import VFLHistGBDT
from vertimosaic.parties import ActiveParty, PassiveParty


def test_vfl_gbdt_matches_controlled_one_tree_reference() -> None:
    """Check best split, owner, routing, leaves and prediction update exactly."""
    active = ActiveParty(
        "bank",
        np.zeros((4, 1)),
        np.array([0.0, 0.0, 1.0, 1.0]),
    )
    passive = PassiveParty(
        "telecom",
        np.array([[0.0], [0.0], [1.0], [1.0]]),
    )
    model = VFLHistGBDT(
        n_estimators=1,
        learning_rate=0.1,
        max_depth=1,
        min_samples_leaf=1,
        min_child_weight=0.0,
        l2_leaf_reg=1.0,
        max_bins=2,
        seed=5,
    )
    model.fit(active, [passive])

    root = model.trees_[0]
    assert not root.is_leaf
    assert root.party == "telecom"
    assert root.split_ref is not None
    assert root.split_ref.feature_ref == 0
    assert root.split_ref.bin_ref == 0
    assert np.isclose(root.gain, 2.0 / 3.0, atol=1e-12)

    assert root.left is not None
    assert root.right is not None
    assert np.array_equal(root.left.indices, np.array([0, 1]))
    assert np.array_equal(root.right.indices, np.array([2, 3]))
    assert np.isclose(root.left.value, -2.0 / 3.0, atol=1e-12)
    assert np.isclose(root.right.value, 2.0 / 3.0, atol=1e-12)

    expected_raw = np.array([-1.0 / 15.0, -1.0 / 15.0, 1.0 / 15.0, 1.0 / 15.0])
    actual_raw = model.decision_function([active, passive])
    assert np.allclose(actual_raw, expected_raw, atol=1e-12)

    expected_probability = 1.0 / (1.0 + np.exp(-expected_raw))
    actual_probability = model.predict_proba([active, passive])[:, 1]
    assert np.allclose(actual_probability, expected_probability, atol=1e-12)


def test_vfl_gbdt_fits_and_predicts_probabilities() -> None:
    active, passive = make_vertical_synthetic(300, seed=4)
    model = VFLHistGBDT(n_estimators=3, max_depth=2, min_samples_leaf=15, max_bins=8)
    model.fit(active, passive)
    p = model.predict_proba([active, *passive])[:, 1]
    assert p.shape == (300,)
    assert ((p >= 0.0) & (p <= 1.0)).all()
    assert len(model.trees_) == 3
