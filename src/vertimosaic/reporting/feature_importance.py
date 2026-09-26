from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.models.vfl_hist_gbdt import TreeNode
from vertimosaic.parties import PassiveParty


def _feature_label(party: str, index: int, names: dict[str, list[str]] | None) -> str:
    if names is None or party not in names:
        return f"feature_{index}"
    party_names = names[party]
    if index >= len(party_names):
        raise ValueError(f"feature name list for {party} is shorter than model width")
    return party_names[index]


def logistic_local_feature_importance(
    model: VFLLogisticRegression,
    parties: Iterable[PassiveParty],
    *,
    feature_names: dict[str, list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return standardized coefficient magnitude separately for each party."""
    if not model.weights_:
        raise RuntimeError("logistic model is not fitted")
    output: dict[str, pd.DataFrame] = {}
    for party in parties:
        weights = model.weights_.get(party.name)
        if weights is None:
            raise ValueError(f"model has no coefficients for party: {party.name}")
        scale = np.std(party._x, axis=0)
        standardized = np.abs(weights * scale)
        output[party.name] = pd.DataFrame(
            {
                "party": party.name,
                "feature": [
                    _feature_label(party.name, index, feature_names)
                    for index in range(len(weights))
                ],
                "standardized_coefficient_importance": standardized,
            }
        ).sort_values("standardized_coefficient_importance", ascending=False, ignore_index=True)
    return output


def _walk_splits(node: TreeNode) -> Iterable[TreeNode]:
    if node.is_leaf:
        return
    yield node
    if node.left is not None:
        yield from _walk_splits(node.left)
    if node.right is not None:
        yield from _walk_splits(node.right)


def gbdt_local_feature_importance(
    model: VFLHistGBDT,
    *,
    feature_names: dict[str, list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Aggregate split count and gain by opaque party-local feature reference."""
    if not model.trees_:
        raise RuntimeError("GBDT model is not fitted")
    aggregates: dict[tuple[str, int], list[float]] = defaultdict(list)
    for tree in model.trees_:
        for node in _walk_splits(tree):
            if node.party is None or node.feature is None:
                continue
            aggregates[(node.party, node.feature)].append(node.gain)
    rows_by_party: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for (party, feature_index), gains in aggregates.items():
        gain_sum = float(np.sum(gains))
        rows_by_party[party].append(
            {
                "party": party,
                "feature": _feature_label(party, feature_index, feature_names),
                "split_count": len(gains),
                "gain_sum": gain_sum,
                "gain_mean": gain_sum / len(gains),
            }
        )
    return {
        party: pd.DataFrame(rows).sort_values("gain_sum", ascending=False, ignore_index=True)
        for party, rows in rows_by_party.items()
    }


def write_party_feature_importance(
    frames: dict[str, pd.DataFrame],
    directory: Path = Path("results"),
) -> list[Path]:
    """Write one aggregate feature-importance CSV per party."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for party, frame in frames.items():
        path = directory / f"{party}_feature_importance.csv"
        frame.to_csv(path, index=False)
        paths.append(path)
    return paths
