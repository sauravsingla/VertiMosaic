import numpy as np

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.experiments import make_availability_masks, prepare_missing_party_method


def test_missing_party_methods_have_distinct_explicit_representations() -> None:
    active, passive = make_vertical_synthetic(200, seed=17)
    masks = make_availability_masks(
        200,
        seed=17,
        bank=1.0,
        telecom=0.75,
        insurance=0.65,
        retail=0.80,
    )
    intersection = prepare_missing_party_method(active, passive, masks, "intersection_only")
    zero = prepare_missing_party_method(active, passive, masks, "zero_contribution")
    availability = prepare_missing_party_method(active, passive, masks, "availability_indicator")
    bias = prepare_missing_party_method(active, passive, masks, "learned_party_bias")

    assert intersection.active.n_rows < active.n_rows
    assert zero.active.n_rows == active.n_rows
    assert availability.active.n_features == active.n_features + 3
    assert all(
        prepared.n_features == original.n_features + 1
        for prepared, original in zip(bias.passive, passive, strict=True)
    )
    telecom_available = masks.telecom
    telecom_zero = zero.passive[0]._x
    assert np.all(telecom_zero[~telecom_available] == 0.0)
