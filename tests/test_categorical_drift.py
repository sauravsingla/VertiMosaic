import numpy as np

from vertimosaic.experiments import apply_categorical_frequency_drift


def test_categorical_frequency_drift_is_deterministic_and_non_mutating() -> None:
    values = np.array(["a", "a", "a", "b", "b", "c", None], dtype=object)
    original = values.copy()
    first = apply_categorical_frequency_drift(values, strength=0.6, seed=12)
    second = apply_categorical_frequency_drift(values, strength=0.6, seed=12)
    assert np.array_equal(values, original)
    assert np.array_equal(first, second)
    assert first[-1] is None
    assert sum(value == "a" for value in first[:-1]) >= 3
