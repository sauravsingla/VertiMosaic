"""End-to-end synthetic research demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from vertimosaic.datasets import generate_synthetic
from vertimosaic.evaluation import bootstrap_metric_ci, evaluate_binary
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.parties import ActiveParty, PassiveParty, Party
from vertimosaic.preprocessing import LocalPreprocessor


def _build_parties(
    data: dict[str, Any], train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[list[Party], ActiveParty, list[Party], ActiveParty]:
    train_parties: list[Party] = []
    test_parties: list[Party] = []
    active_train: ActiveParty | None = None
    active_test: ActiveParty | None = None
    for name in ("bank", "telecom", "insurance", "retail"):
        frame = data[name]
        feature_cols = [c for c in frame.columns if c not in {"entity_id", "target"}]
        prep = LocalPreprocessor().fit(frame.iloc[train_idx][feature_cols])
        x_train = prep.transform(frame.iloc[train_idx][feature_cols])
        x_test = prep.transform(frame.iloc[test_idx][feature_cols])
        names = prep.feature_names()
        if name == "bank":
            y_train = frame.iloc[train_idx]["target"].to_numpy()
            y_test = frame.iloc[test_idx]["target"].to_numpy()
            active_train = ActiveParty(name, x_train, names, y_train)
            active_test = ActiveParty(name, x_test, names, y_test)
            train_parties.append(active_train)
            test_parties.append(active_test)
        else:
            train_parties.append(PassiveParty(name, x_train, names))
            test_parties.append(PassiveParty(name, x_test, names))
    assert active_train is not None and active_test is not None
    return train_parties, active_train, test_parties, active_test


def run_demo(
    *, rows: int = 5000, seed: int = 42, model: str = "logistic", output: Path | None = None
) -> dict[str, Any]:
    data = generate_synthetic(rows=rows, seed=seed)
    y = data["bank"]["target"].to_numpy()
    idx = np.arange(rows)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    train_parties, active_train, test_parties, active_test = _build_parties(
        data, train_idx, test_idx
    )
    if model == "logistic":
        estimator = VFLLogisticRegression(
            learning_rate=0.15, epochs=80, batch_size=512, l2=1e-4, seed=seed
        )
    elif model == "vfl-hist-gbdt":
        estimator = VFLHistGBDT(
            n_estimators=20,
            max_depth=3,
            max_bins=24,
            min_samples_leaf=max(10, rows // 500),
        )
    else:
        raise ValueError(f"unknown model: {model}")
    estimator.fit(train_parties, active_train)
    p = estimator.predict_proba(test_parties)[:, 1]
    metrics = evaluate_binary(active_test.y, p).to_dict()
    result: dict[str, Any] = {
        "model": model,
        "rows": rows,
        "seed": seed,
        "positive_rate": float(np.mean(y)),
        "metrics": metrics,
        "roc_auc_ci": bootstrap_metric_ci(
            active_test.y, p, metric="roc_auc", replicates=100, seed=seed
        ),
        "pr_auc_ci": bootstrap_metric_ci(
            active_test.y, p, metric="pr_auc", replicates=100, seed=seed
        ),
        "communication": estimator.transport.summary(),
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(result, indent=2))
    return result
