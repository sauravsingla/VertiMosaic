from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from vertimosaic.alignment import bind_entity_ids, entity_ids_for
from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import entity_level_split
from vertimosaic.experiments.contribution import exact_shapley_party_utility
from vertimosaic.experiments.pipeline import slice_parties
from vertimosaic.experiments.study_artifacts import (
    model_communication_frame,
    model_training_frame,
    prediction_frame,
    write_synthetic_study_run,
)
from vertimosaic.models import VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty

_PASSIVE_NAMES = ("telecom", "insurance", "retail")
_ALL_NAMES = ("bank", *_PASSIVE_NAMES)


def _passive_map(items: list[PassiveParty]) -> dict[str, PassiveParty]:
    return {item.name: item for item in items}


def _bind_partition_ids(
    target: ActiveParty | PassiveParty,
    source: ActiveParty | PassiveParty,
) -> None:
    entity_ids = entity_ids_for(source)
    if entity_ids is None:
        raise RuntimeError(f"source party {source.name!r} is missing entity identifiers")
    bind_entity_ids(target, entity_ids)


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
    write_run: bool = True,
    runs_root: Path = Path("runs"),
) -> pd.DataFrame:
    """Measure predictive party utility with retraining, permutation, and exact Shapley."""
    active, passive = make_vertical_synthetic(rows, seed)
    split = entity_level_split(active.labels, seed=seed)
    train_active, train_passive = slice_parties(active, passive, split.train)
    test_active, test_passive = slice_parties(active, passive, split.test)
    train_map = _passive_map(train_passive)
    test_map = _passive_map(test_passive)
    score_cache: dict[tuple[str, ...], float] = {}
    prediction_cache: dict[tuple[str, ...], np.ndarray] = {}
    model_cache: dict[tuple[str, ...], VFLLogisticRegression] = {}
    histories: list[pd.DataFrame] = []
    communications: list[pd.DataFrame] = []

    def score(subset: tuple[str, ...]) -> float:
        key = tuple(sorted(subset))
        if key in score_cache:
            return score_cache[key]
        bank_included = "bank" in key
        train_x = train_active._x if bank_included else np.zeros((train_active.n_rows, 0))
        test_x = test_active._x if bank_included else np.zeros((test_active.n_rows, 0))
        local_train_active = ActiveParty("bank", train_x, train_active.labels)
        local_test_active = ActiveParty("bank", test_x, test_active.labels)
        _bind_partition_ids(local_train_active, train_active)
        _bind_partition_ids(local_test_active, test_active)
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
        score_cache[key] = value
        prediction_cache[key] = probability
        model_cache[key] = model
        condition = "subset:" + ("+".join(key) if key else "none")
        histories.append(model_training_frame(model, condition=condition))
        communications.append(model_communication_frame(model, condition=condition))
        return value

    full_score = score(_ALL_NAMES)
    shapley = exact_shapley_party_utility(_ALL_NAMES, score)
    full_key = tuple(sorted(_ALL_NAMES))
    full_model = model_cache[full_key]
    records: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = [
        prediction_frame(
            split.test,
            test_active.labels,
            probability,
            seed=seed,
            condition="subset:" + ("+".join(key) if key else "none"),
        )
        for key, probability in sorted(prediction_cache.items())
    ]

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
            }
        )

        rng = np.random.default_rng(seed + 100 + party_index)
        permutation = rng.permutation(test_active.n_rows)

        # The permutation ablation deliberately replaces feature representations inside
        # fixed entity slots. The entity IDs therefore remain in the original slot order
        # so protocol alignment is still explicit while feature-to-entity signal is broken.
        active_features = (
            test_active._x[permutation] if party == "bank" else test_active._x.copy()
        )
        permuted_active = ActiveParty("bank", active_features, test_active.labels)
        _bind_partition_ids(permuted_active, test_active)

        permuted_passive: list[PassiveParty] = []
        for item in test_passive:
            features = item._x[permutation] if item.name == party else item._x.copy()
            permuted_item = PassiveParty(item.name, features)
            _bind_partition_ids(permuted_item, item)
            permuted_passive.append(permuted_item)

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
            }
        )
        predictions.append(
            prediction_frame(
                split.test,
                test_active.labels,
                permuted_probability,
                seed=seed,
                condition=f"permutation:{party}",
            )
        )

    frame = pd.DataFrame(records)
    if write_run:
        run_id, directory = write_synthetic_study_run(
            study_name="party_contribution",
            seed=seed,
            active=active,
            passive=passive,
            config={
                "rows": rows,
                "methods": [
                    "leave_one_party_out",
                    "representation_permutation_ablation",
                    "exact_shapley_predictive_utility",
                ],
            },
            results=frame,
            predictions=pd.concat(predictions, ignore_index=True),
            training_history=pd.concat(histories, ignore_index=True),
            communication=pd.concat(communications, ignore_index=True),
            runs_root=runs_root,
        )
        frame["run_id"] = run_id
        frame["run_directory"] = str(directory)
    else:
        frame["run_id"] = None
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
