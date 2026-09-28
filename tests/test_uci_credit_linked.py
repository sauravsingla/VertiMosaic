from __future__ import annotations

import numpy as np
import pandas as pd

from vertimosaic.experiments.linked_uci_credit import prepare_exact_row_vertical_partitions


def test_exact_row_partition_uses_disjoint_source_columns_and_same_entities() -> None:
    rng = np.random.default_rng(3)
    features = pd.DataFrame(
        rng.normal(size=(300, 6)),
        columns=[f"x{index}" for index in range(6)],
    )
    target = pd.Series((features["x0"] + features["x5"] > 0).astype(int))
    prepared = prepare_exact_row_vertical_partitions(features, target, seed=3)
    active_columns = set(prepared["active_columns"])
    passive_columns = set(prepared["passive_columns"])
    assert active_columns
    assert passive_columns
    assert active_columns.isdisjoint(passive_columns)
    assert active_columns | passive_columns == set(features.columns)
    assert prepared["train_active"].n_rows == prepared["train_passive"][0].n_rows
    assert prepared["test_active"].n_rows == prepared["test_passive"][0].n_rows
