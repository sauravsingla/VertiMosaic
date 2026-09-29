# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from vertimosaic import __version__
from vertimosaic.datasets import make_vertical_synthetic
from vertimosaic.evaluation import binary_metrics, entity_level_split, select_f1_threshold
from vertimosaic.experiments.pipeline import slice_parties
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.models.vfl_hist_gbdt import TreeNode
from vertimosaic.parties import PassiveParty

ROWS = 800
SEED = 42
LOGISTIC_MAX_ITER = 150
GBDT_ESTIMATORS = 8


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float_list(values: np.ndarray | list[float]) -> list[float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    return [float(value) for value in array]


def _prediction_digest(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8").reshape(-1)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _tree_payload(node: TreeNode) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "depth": int(node.depth),
        "value": float(node.value),
        "gain": float(node.gain),
        "leaf": bool(node.is_leaf),
    }
    if node.is_leaf:
        return payload
    if node.party is None or node.split_ref is None:
        raise RuntimeError("fitted GBDT tree contains incomplete split state")
    if node.left is None or node.right is None:
        raise RuntimeError("fitted GBDT tree contains incomplete child state")
    payload.update(
        {
            "party": node.party,
            "split_ref": {
                "feature_ref": int(node.split_ref.feature_ref),
                "bin_ref": int(node.split_ref.bin_ref),
            },
            "left": _tree_payload(node.left),
            "right": _tree_payload(node.right),
        }
    )
    return payload


def _logistic_payload(model: VFLLogisticRegression) -> dict[str, Any]:
    return {
        "format": "vertimosaic-vfl-logistic-checkpoint-v1",
        "class": "VFLLogisticRegression",
        "hyperparameters": {
            "learning_rate": model.learning_rate,
            "max_iter": model.max_iter,
            "l2": model.l2,
            "l1": model.l1,
            "tolerance": model.tolerance,
            "gradient_clip": model.gradient_clip,
            "class_weight": model.class_weight,
            "batch_size": model.batch_size,
            "learning_rate_schedule": model.learning_rate_schedule,
            "early_stopping_rounds": model.early_stopping_rounds,
            "missing_party_policy": model.missing_party_policy,
            "seed": model.seed,
        },
        "state": {
            "intercept": float(model.intercept_),
            "weights": {
                name: _float_list(weights) for name, weights in sorted(model.weights_.items())
            },
            "trained_party_names": list(model.trained_party_names_),
            "active_party_name": model.active_party_name_,
            "n_iter": int(model.n_iter_),
            "converged": bool(model.converged_),
            "best_iteration": model.best_iteration_,
            "loss_history": [float(value) for value in model.loss_history_],
            "validation_loss_history": [float(value) for value in model.validation_loss_history_],
        },
    }


def _gbdt_payload(model: VFLHistGBDT) -> dict[str, Any]:
    return {
        "format": "vertimosaic-vfl-hist-gbdt-checkpoint-v1",
        "class": "VFLHistGBDT",
        "hyperparameters": {
            "n_estimators": model.n_estimators,
            "learning_rate": model.learning_rate,
            "max_depth": model.max_depth,
            "max_leaves": model.max_leaves,
            "min_samples_leaf": model.min_samples_leaf,
            "min_child_weight": model.min_child_weight,
            "l2_leaf_reg": model.l2_leaf_reg,
            "max_bins": model.max_bins,
            "subsample": model.subsample,
            "feature_subsample": model.feature_subsample,
            "early_stopping_rounds": model.early_stopping_rounds,
            "missing_party_policy": model.missing_party_policy,
            "seed": model.seed,
        },
        "state": {
            "base_score": float(model.base_score_),
            "party_names": list(model.party_names_),
            "tree_count": len(model.trees_),
            "best_iteration": model.best_iteration_,
            "training_loss_history": [float(value) for value in model.training_loss_history_],
            "validation_loss_history": [float(value) for value in model.validation_loss_history_],
            "trees": [_tree_payload(tree) for tree in model.trees_],
        },
    }


def _party_routing_payload(party: PassiveParty, max_bins: int) -> dict[str, Any]:
    thresholds = getattr(party, "_histogram_thresholds", None)
    fitted_max_bins = getattr(party, "_histogram_max_bins", None)
    if not isinstance(thresholds, dict) or fitted_max_bins != max_bins:
        raise RuntimeError(f"party {party.name} lacks fitted histogram routing thresholds")
    return {
        "format": "vertimosaic-vfl-hist-gbdt-party-routing-v1",
        "party": party.name,
        "n_features": int(party.n_features),
        "max_bins": int(max_bins),
        "thresholds": {
            str(index): _float_list(thresholds[index]) for index in range(party.n_features)
        },
        "boundary_note": (
            "This public synthetic checkpoint exposes train-derived thresholds for "
            "reproducibility. In a real VFL deployment, party-local routing state should "
            "remain with the owning organization."
        ),
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _portable_logistic_probability(
    payload: dict[str, Any], parties: list[PassiveParty]
) -> np.ndarray:
    state = payload["state"]
    by_name = {party.name: party for party in parties}
    logits = np.full(parties[0].n_rows, float(state["intercept"]), dtype=float)
    for name in state["trained_party_names"]:
        weights = np.asarray(state["weights"][name], dtype=float)
        logits += by_name[name]._x @ weights
    return _sigmoid(logits)


def _portable_gbdt_probability(
    payload: dict[str, Any],
    routing: dict[str, dict[str, Any]],
    parties: list[PassiveParty],
) -> np.ndarray:
    state = payload["state"]
    by_name = {party.name: party for party in parties}
    n_rows = parties[0].n_rows

    def tree_prediction(tree: dict[str, Any]) -> np.ndarray:
        output = np.zeros(n_rows, dtype=float)

        def walk(node: dict[str, Any], indices: np.ndarray) -> None:
            if bool(node["leaf"]):
                output[indices] = float(node["value"])
                return
            party_name = str(node["party"])
            feature_ref = int(node["split_ref"]["feature_ref"])
            bin_ref = int(node["split_ref"]["bin_ref"])
            party_thresholds = routing[party_name]["thresholds"]
            threshold = float(party_thresholds[str(feature_ref)][bin_ref])
            values = by_name[party_name]._x[indices, feature_ref]
            left = indices[values <= threshold]
            right = indices[values > threshold]
            walk(node["left"], left)
            walk(node["right"], right)

        walk(tree, np.arange(n_rows, dtype=int))
        return output

    raw = np.full(n_rows, float(state["base_score"]), dtype=float)
    learning_rate = float(payload["hyperparameters"]["learning_rate"])
    for tree in state["trees"]:
        raw += learning_rate * tree_prediction(tree)
    return _sigmoid(raw)


def _evaluation_record(
    *,
    model_name: str,
    validation_labels: np.ndarray,
    validation_probability: np.ndarray,
    test_labels: np.ndarray,
    test_probability: np.ndarray,
) -> dict[str, Any]:
    threshold = float(select_f1_threshold(validation_labels, validation_probability))
    metrics = binary_metrics(test_labels, test_probability, threshold=threshold)
    return {
        "model": model_name,
        "decision_threshold_selected_on_validation": threshold,
        "test_metrics": {key: float(value) for key, value in metrics.items()},
        "test_prediction_sha256_float64_le": _prediction_digest(test_probability),
    }


def _metric_row(label: str, record: dict[str, Any]) -> str:
    metrics = record["test_metrics"]
    return (
        f"| {label} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | "
        f"{metrics['f1']:.4f} | {metrics['brier']:.4f} |"
    )


def _model_card(evaluation: list[dict[str, Any]], source_commit: str) -> str:
    by_name = {record["model"]: record for record in evaluation}
    logistic_row = _metric_row("VFL Logistic", by_name["vfl_logistic"])
    gbdt_row = _metric_row("VFL HistGBDT", by_name["vfl_hist_gbdt"])
    return f"""---
license: apache-2.0
library_name: vertimosaic
pipeline_tag: tabular-classification
tags:
- federated-learning
- vertical-federated-learning
- tabular
- cpu
- reproducibility
- privacy-research
- logistic-regression
- gradient-boosting
datasets:
- sauravsingla08/VertiMosaic-VFL-Benchmark
---

# VertiMosaic VFL Reference Models

**Reproducible CPU reference checkpoints for VertiMosaic vertical federated learning.**

This repository contains two deterministic research checkpoints generated from the public
VertiMosaic source tree:

- `VFLLogisticRegression` — first-principles vertical logistic regression;
- `VFLHistGBDT` — vertical histogram gradient boosting with party-local routing state.

They are trained on VertiMosaic's deterministic synthetic four-party benchmark using the
formal v0.3.0 release configuration: **800 aligned entities**, **seed 42**, a 70/15/15
entity-level train/validation/test split, logistic `max_iter=150`, and GBDT
`n_estimators=8`.

## Files

- `logistic/model.json` — coordinator-visible logistic configuration and per-party weights.
- `gbdt/model.json` — GBDT ensemble topology, leaf values, opaque party/feature/bin references.
- `gbdt/parties/*.json` — synthetic benchmark party-local routing thresholds, separated by party.
- `evaluation.json` — held-out metrics, validation-selected thresholds, and prediction digests.
- `metadata.json` — source commit and exact benchmark-generation configuration.

## Reference evaluation

| Checkpoint | ROC-AUC | PR-AUC | F1 | Brier |
|---|---:|---:|---:|---:|
{logistic_row}
{gbdt_row}

These are single deterministic synthetic benchmark measurements, not a leaderboard and not
evidence that VFL generally outperforms centralized learning.

## Privacy and deployment boundary

The original VertiMosaic protocols keep raw party feature matrices local, but that does **not**
mean the reference protocols provide end-to-end cryptographic privacy. Gradient, Hessian,
residual, logit, routing, timing, and transport information can remain outside stronger
privacy guarantees depending on the path used.

For the public synthetic GBDT checkpoint, train-derived routing thresholds are deliberately
published as **separate party shards** so the checkpoint can be reproduced and independently
verified. Real organizations should not centralize or publicly release analogous private
party-local routing state merely because this synthetic research artifact does so.

These checkpoints are **not production models**, are not trained on real organizational data,
and should not be used to make real-world decisions.

## Reproduce

```bash
git clone https://github.com/sauravsingla/VertiMosaic.git
cd VertiMosaic
python -m pip install -e .
SOURCE_SHA=$(git rev-parse HEAD) python huggingface/build_model.py --output hf-model
```

The builder performs a prediction-level round-trip check for both checkpoint formats before
writing the final package.

## Related resources

- Source: https://github.com/sauravsingla/VertiMosaic
- Benchmark dataset: https://huggingface.co/datasets/sauravsingla08/VertiMosaic-VFL-Benchmark
- Interactive Space: https://huggingface.co/spaces/sauravsingla08/VertiMosaic
- PyPI: https://pypi.org/project/vertimosaic/

Generated from source commit `{source_commit}` with VertiMosaic `{__version__}`.
"""


def build(output: Path) -> None:
    source_commit = os.environ.get("SOURCE_SHA") or os.environ.get("GITHUB_SHA") or "local"
    active, passive = make_vertical_synthetic(ROWS, SEED)
    split = entity_level_split(active.labels, seed=SEED)
    train_active, train_passive = slice_parties(active, passive, split.train)
    validation_active, validation_passive = slice_parties(active, passive, split.validation)
    test_active, test_passive = slice_parties(active, passive, split.test)

    logistic = VFLLogisticRegression(
        learning_rate=0.08,
        max_iter=LOGISTIC_MAX_ITER,
        l2=1e-3,
        early_stopping_rounds=5,
        seed=SEED,
    )
    logistic.fit(train_active, train_passive, validation_active, validation_passive)
    logistic_payload = _logistic_payload(logistic)
    logistic_validation = logistic.predict_proba([validation_active, *validation_passive])[:, 1]
    logistic_test = logistic.predict_proba([test_active, *test_passive])[:, 1]
    portable_logistic = _portable_logistic_probability(
        logistic_payload, [test_active, *test_passive]
    )
    if not np.allclose(logistic_test, portable_logistic, rtol=0.0, atol=1e-12):
        raise RuntimeError("portable logistic checkpoint failed prediction round-trip validation")

    gbdt = VFLHistGBDT(
        n_estimators=GBDT_ESTIMATORS,
        max_depth=3,
        min_samples_leaf=10,
        early_stopping_rounds=3,
        seed=SEED,
    )
    gbdt.fit(train_active, train_passive, validation_active, validation_passive)
    training_parties: list[PassiveParty] = [train_active, *train_passive]
    for evaluation_parties in (
        [validation_active, *validation_passive],
        [test_active, *test_passive],
    ):
        for source_party, target_party in zip(training_parties, evaluation_parties, strict=True):
            source_party.share_histogram_routing_state_with(target_party)

    gbdt_payload = _gbdt_payload(gbdt)
    routing = {
        party.name: _party_routing_payload(party, gbdt.max_bins) for party in training_parties
    }
    gbdt_validation = gbdt.predict_proba([validation_active, *validation_passive])[:, 1]
    gbdt_test = gbdt.predict_proba([test_active, *test_passive])[:, 1]
    portable_gbdt = _portable_gbdt_probability(gbdt_payload, routing, [test_active, *test_passive])
    if not np.allclose(gbdt_test, portable_gbdt, rtol=0.0, atol=1e-12):
        raise RuntimeError("portable GBDT checkpoint failed prediction round-trip validation")

    evaluation = [
        _evaluation_record(
            model_name="vfl_logistic",
            validation_labels=validation_active.labels,
            validation_probability=logistic_validation,
            test_labels=test_active.labels,
            test_probability=logistic_test,
        ),
        _evaluation_record(
            model_name="vfl_hist_gbdt",
            validation_labels=validation_active.labels,
            validation_probability=gbdt_validation,
            test_labels=test_active.labels,
            test_probability=gbdt_test,
        ),
    ]

    metadata = {
        "format": "vertimosaic-huggingface-model-bundle-v1",
        "source_repository": "https://github.com/sauravsingla/VertiMosaic",
        "source_commit": source_commit,
        "vertimosaic_version": __version__,
        "benchmark": {
            "rows": ROWS,
            "seed": SEED,
            "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "train_rows": int(len(split.train)),
            "validation_rows": int(len(split.validation)),
            "test_rows": int(len(split.test)),
            "party_order": [party.name for party in training_parties],
            "features_per_party": {party.name: int(party.n_features) for party in training_parties},
            "release_alignment": {
                "logistic_max_iter": LOGISTIC_MAX_ITER,
                "gbdt_estimators": GBDT_ESTIMATORS,
            },
        },
        "checkpoint_files": {
            "logistic": "logistic/model.json",
            "gbdt": "gbdt/model.json",
            "gbdt_party_routing": [f"gbdt/parties/{party.name}.json" for party in training_parties],
            "evaluation": "evaluation.json",
        },
        "round_trip_validation": {
            "logistic_predictions_match": True,
            "gbdt_predictions_match": True,
            "absolute_tolerance": 1e-12,
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "logistic" / "model.json", logistic_payload)
    _write_json(output / "gbdt" / "model.json", gbdt_payload)
    for party_name, party_payload in routing.items():
        path = output / "gbdt" / "parties" / f"{party_name}.json"
        _write_json(path, party_payload)
    _write_json(output / "evaluation.json", evaluation)
    _write_json(output / "metadata.json", metadata)
    (output / "README.md").write_text(_model_card(evaluation, source_commit), encoding="utf-8")

    print(f"Wrote VertiMosaic Hugging Face model bundle to {output}")
    print(f"Source commit: {source_commit}")
    split_summary = f"{len(split.train)}/{len(split.validation)}/{len(split.test)}"
    print(f"Train/validation/test rows: {split_summary}")
    for record in evaluation:
        metrics = record["test_metrics"]
        print(
            f"{record['model']}: ROC-AUC={metrics['roc_auc']:.6f}, "
            f"PR-AUC={metrics['pr_auc']:.6f}, F1={metrics['f1']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VertiMosaic Hugging Face model bundle")
    parser.add_argument("--output", type=Path, default=Path("hf-model"))
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
