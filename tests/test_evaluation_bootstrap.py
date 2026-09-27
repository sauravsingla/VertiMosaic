import numpy as np

from vertimosaic.evaluation import (
    bootstrap_confidence_intervals,
    entity_level_split,
    paired_bootstrap_difference,
    select_f1_threshold,
)


def test_entity_split_is_disjoint_and_complete() -> None:
    y = np.array([0, 1] * 100)
    split = entity_level_split(y, seed=7)
    assert len(split.train) == 140
    assert len(split.validation) == 30
    assert len(split.test) == 30
    combined = np.concatenate([split.train, split.validation, split.test])
    assert len(np.unique(combined)) == len(y)


def test_bootstrap_is_deterministic_and_paired() -> None:
    y = np.array([0, 1] * 50)
    p = np.linspace(0.05, 0.95, 100)
    first = bootstrap_confidence_intervals(y, p, replicates=50, seed=3)
    second = bootstrap_confidence_intervals(y, p, replicates=50, seed=3)
    assert first == second
    paired = paired_bootstrap_difference(y, p, p, replicates=50, seed=3)
    assert paired["delta"] == 0.0
    assert paired["lower"] == 0.0
    assert paired["upper"] == 0.0


def test_threshold_selection_uses_validation_predictions() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    threshold = select_f1_threshold(y, p, grid_size=11)
    assert 0.4 < threshold <= 0.6
