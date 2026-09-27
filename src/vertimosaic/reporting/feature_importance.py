from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.models.vfl_hist_gbdt import TreeNode
from vertimosaic.parties import PassiveParty
from vertimosaic.parties.core import OpaqueSplitReference


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


def _split_records_by_party(
    model: VFLHistGBDT,
) -> dict[str, list[tuple[OpaqueSplitReference, float]]]:
    records: dict[str, list[tuple[OpaqueSplitReference, float]]] = defaultdict(list)
    for tree in model.trees_:
        for node in _walk_splits(tree):
            if node.party is None or node.split_ref is None:
                continue
            records[node.party].append((node.split_ref, node.gain))
    return records


def gbdt_local_feature_importance(
    model: VFLHistGBDT,
    parties: Iterable[PassiveParty] | None = None,
    *,
    feature_names: dict[str, list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Return party-local aggregate split counts and gains.

    When party objects are supplied, each owning party resolves its own opaque split
    references and computes split_count/gain_sum/gain_mean locally. The no-party fallback
    retains aggregate diagnostics from opaque feature references already present in the
    fitted tree; numeric thresholds and raw feature values are never required.
    """
    if not model.trees_:
        raise RuntimeError("GBDT model is not fitted")

    rows_by_party: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    split_records = _split_records_by_party(model)
    if parties is not None:
        party_map = {party.name: party for party in parties}
        missing = set(split_records) - set(party_map)
        if missing:
            message = f"missing split-owning parties for local importance: {sorted(missing)}"
            raise ValueError(message)
        for party_name, records in split_records.items():
            local_aggregates = party_map[party_name].aggregate_local_split_importance(records)
            for feature_index, statistics in local_aggregates.items():
                rows_by_party[party_name].append(
                    {
                        "party": party_name,
                        "feature": _feature_label(party_name, feature_index, feature_names),
                        "split_count": int(statistics["split_count"]),
                        "gain_sum": float(statistics["gain_sum"]),
                        "gain_mean": float(statistics["gain_mean"]),
                    }
                )
    else:
        fallback_gains: dict[tuple[str, int], list[float]] = defaultdict(list)
        for tree in model.trees_:
            for node in _walk_splits(tree):
                if node.party is None or node.split_ref is None:
                    continue
                fallback_gains[(node.party, node.split_ref.feature_ref)].append(node.gain)
        for (party_name, feature_index), gains in fallback_gains.items():
            gain_sum = float(np.sum(gains))
            rows_by_party[party_name].append(
                {
                    "party": party_name,
                    "feature": _feature_label(party_name, feature_index, feature_names),
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
