# SPDX-License-Identifier: Apache-2.0
import numpy as np

from vertimosaic.models import VFLHistGBDT


def test_split_gain_matches_closed_form_reference() -> None:
    model = VFLHistGBDT(l2_leaf_reg=1.0, min_child_weight=0.0)
    candidate = {
        "g_left": 1.0,
        "h_left": 0.5,
        "g_right": -1.0,
        "h_right": 0.5,
    }
    expected = 0.5 * ((1.0**2) / 1.5 + ((-1.0) ** 2) / 1.5 - 0.0)
    assert np.isclose(model._split_gain(candidate), 2.0 / 3.0, atol=1e-12)
    assert np.isclose(model._split_gain(candidate), expected, atol=1e-12)


def test_leaf_value_matches_newton_reference() -> None:
    model = VFLHistGBDT(l2_leaf_reg=1.0)
    gradients = np.array([0.5, 0.5, -0.5, -0.5])
    hessians = np.full(4, 0.25)

    left = model._leaf_value(gradients, hessians, np.array([0, 1]))
    right = model._leaf_value(gradients, hessians, np.array([2, 3]))

    assert np.isclose(left, -2.0 / 3.0, atol=1e-12)
    assert np.isclose(right, 2.0 / 3.0, atol=1e-12)


def test_split_gain_prefers_purer_partition() -> None:
    model = VFLHistGBDT(l2_leaf_reg=1.0)
    pure = {"g_left": -5.0, "h_left": 2.0, "g_right": 5.0, "h_right": 2.0}
    weak = {"g_left": -1.0, "h_left": 2.0, "g_right": 1.0, "h_right": 2.0}
    assert model._split_gain(pure) > model._split_gain(weak)
