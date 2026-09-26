from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import entity_level_split
from vertimosaic.experiments.contribution import exact_shapley_party_utility
from vertimosaic.experiments.pipeline import slice_parties
from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty

_PASSIVE_NAMES = ("telecom", "insurance", "retail")
_ALL_NAMES = ("bank", *_PASSIVE_NAMES)


def _passive_map(items: list[PassiveParty]) -> dict[str, PassiveParty]:
    return {item.name: item for item in items}


def _fit_logistic(
    active: ActiveParty,
    passive: list[PassiveParty],
    seed: int,
) -> VFLLogisticRegression:
    model = VFLLogisticRegression(
        learning_rate=0.08,
        max_iter=300,
        l2=1e-3,
        seed=seed,
    )
    model.fit(active, passive)
    return model


def run_contribution_study(
    *,
    rows: int = 800,
    seed: int = 42,
    output: Path = Path("results/party_contribution.csv"),
) -> pd.DataFrame:
    """Measure predictive party utility with retraining, permutation, and exact Shapley."""
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    test_active, test_passive = slice_parties(active, passive, split.test)
    train_map = _passive_map(train_passive)
    test_map = _passive_map(test_passive)
    cache: dict[tuple[str, ...], float] = {}

    def score(subset: tuple[str, ...]) -> float:
        key = tuple(sorted(subset))
        if key in cache:
            return cache[key]
        bank_included = "bank" in key
        train_x = train_active._x if bank_included else np.zeros((train_active.n_rows, 0))
        test_x = test_active._x if bank_included else np.zeros((test_active.n_rows, 0))
        local_train_active = ActiveParty("bank", train_x, train_active.labels)
        local_test_active = ActiveParty("bank", test_x, test_active.labels)
        selected = [name for name in _PASSIVE_NAMES if name in key]
        model = _fit_logistic(
            local_train_active,
            [train_map[name] for name in selected],
            seed,
        )
        probability = model.predict_proba(
            [local_test_active, *[test_map[name] for name in selected]]
        )[:, 1]
        value = float(roc_auc_score(local_test_active.labels, probability))
        cache[key] = value
        return value

    full_score = score(_ALL_NAMES)
    shapley = exact_shapley_party_utility(_ALL_NAMES, score)
    full_model = _fit_logistic(train_active, train_passive, seed)
    records: list[dict[str, Any]] = []

    for party_index, party in enumerate(_ALL_NAMES):
        without = tuple(name for name in _ALL_NAMES if name != party)
        ablated_score = score(without)
        records.append(
            {
                "model": "logistic",
                "party": party,
                "method": "leave_one_party_out",
                "metric": "roc_auc",
                "full_score": full_score,
                "ablated_score": ablated_score,
                "delta": full_score - ablated_score,
                "run_id": f"synthetic-{seed}",
            }
        )
        records.append(
            {
                "model": "logistic",
                "party": party,
                "method": "exact_shapley_predictive_utility",
                "metric": "roc_auc",
                "full_score": full_score,
                "ablated_score": np.nan,
                "delta": shapley[party],
                "run_id": f"synthetic-{seed}",
            }
        )

        rng = np.random.default_rng(seed + 100 + party_index)
        permutation = rng.permutation(test_active.n_rows)
        permuted_active = ActiveParty("bank", test_active._x.copy(), test_active.labels)
        permuted_passive = [PassiveParty(item.name, item._x.copy()) for item in test_passive]
        if party == "bank":
            permuted_active = ActiveParty(
                "bank",
                test_active._x[permutation],
                test_active.labels,
            )
        else:
            permuted_passive = [
                PassiveParty(
                    item.name,
                    item._x[permutation] if item.name == party else item._x,
                )
                for item in permuted_passive
            ]
        permuted_probability = full_model.predict_proba([permuted_active, *permuted_passive])[:, 1]
        permuted_score = float(roc_auc_score(test_active.labels, permuted_probability))
        records.append(
            {
                "model": "logistic",
                "party": party,
                "method": "representation_permutation_ablation",
                "metric": "roc_auc",
                "full_score": full_score,
                "ablated_score": permuted_score,
                "delta": full_score - permuted_score,
                "run_id": f"synthetic-{seed}",
            }
        )

    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
