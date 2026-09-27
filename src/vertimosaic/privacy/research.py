# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from vertimosaic.datasets.synthetic import make_vertical_synthetic
from vertimosaic.models import VFLHistGBDT, VFLLogisticRegression
from vertimosaic.models.vfl_hist_gbdt import TreeNode
from vertimosaic.privacy.attacks import (
    confidence_membership_inference,
    residual_label_inference,
    routing_membership_exposure,
)


@dataclass(frozen=True)
class PrivacyResearchConfig:
    """Reproducible experiment grid for privacy/utility research."""

    seeds: tuple[int, ...]
    dataset_sizes: tuple[int, ...]
    party_counts: tuple[int, ...]
    logistic_rounds: tuple[int, ...]
    gbdt_rounds: tuple[int, ...]
    regularization: tuple[float, ...]
    residual_noise: tuple[float, ...]

    @classmethod
    def full(cls) -> PrivacyResearchConfig:
        return cls(
            seeds=(11, 23, 37, 53, 71),
            dataset_sizes=(400, 800, 1200),
            party_counts=(2, 3, 4),
            logistic_rounds=(20, 60, 120),
            gbdt_rounds=(2, 5, 10),
            regularization=(0.0, 1e-3, 1e-2, 1e-1),
            residual_noise=(0.0, 0.05, 0.15, 0.3),
        )

    @classmethod
    def smoke(cls) -> PrivacyResearchConfig:
        return cls(
            seeds=(11, 23),
            dataset_sizes=(240, 320),
            party_counts=(2, 4),
            logistic_rounds=(8, 16),
            gbdt_rounds=(1, 2),
            regularization=(0.0, 1e-2),
            residual_noise=(0.0, 0.15),
        )


_METRICS = (
    "membership_roc_auc",
    "membership_advantage",
    "label_inference_accuracy",
    "routing_exposure_fraction",
    "routing_events_per_entity",
    "holdout_roc_auc",
)


def _confidence_interval(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    half_width = 1.96 * std / sqrt(float(finite.size))
    return {
        "n": int(finite.size),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _summarize(
    records: list[dict[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record[field] for field in group_fields)
        groups.setdefault(key, []).append(record)
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        members = groups[key]
        row: dict[str, Any] = dict(zip(group_fields, key, strict=True))
        row["seed_count"] = len(members)
        for metric in _METRICS:
            values = [float(member[metric]) for member in members if metric in member]
            if values:
                row[metric] = _confidence_interval(values)
        output.append(row)
    return output


def _split_parties(
    rows: int,
    seed: int,
    party_count: int,
) -> tuple[Any, list[Any], Any, list[Any]]:
    if rows < 200:
        raise ValueError("privacy research requires at least 200 total rows")
    if party_count < 2 or party_count > 4:
        raise ValueError("party_count must be between 2 and 4 for the synthetic benchmark")
    train_rows = rows // 2
    holdout_rows = rows - train_rows
    train_active, train_passive_all = make_vertical_synthetic(train_rows, seed=seed)
    holdout_active, holdout_passive_all = make_vertical_synthetic(holdout_rows, seed=seed + 1000)
    passive_count = party_count - 1
    return (
        train_active,
        train_passive_all[:passive_count],
        holdout_active,
        holdout_passive_all[:passive_count],
    )


def _passive_routing_messages(node: TreeNode, active_name: str) -> list[np.ndarray]:
    messages: list[np.ndarray] = []
    if not node.is_leaf and node.party is not None and node.party != active_name:
        messages.append(node.indices.copy())
    if node.left is not None:
        messages.extend(_passive_routing_messages(node.left, active_name))
    if node.right is not None:
        messages.extend(_passive_routing_messages(node.right, active_name))
    return messages


def _logistic_measurement(
    *,
    rows: int,
    seed: int,
    party_count: int = 4,
    rounds: int = 60,
    l2: float = 1e-3,
    residual_noise_std: float = 0.0,
) -> dict[str, float]:
    train_active, train_passive, holdout_active, holdout_passive = _split_parties(
        rows,
        seed,
        party_count,
    )
    model = VFLLogisticRegression(
        learning_rate=0.05,
        max_iter=rounds,
        l2=l2,
        tolerance=0.0,
        gradient_clip=None,
        residual_noise_std=residual_noise_std,
        seed=seed,
    )
    model.fit(train_active, train_passive)
    train_probability = model.predict_proba([train_active, *train_passive])[:, 1]
    holdout_probability = model.predict_proba([holdout_active, *holdout_passive])[:, 1]
    membership = confidence_membership_inference(train_probability, holdout_probability)

    observed_residual = train_probability - train_active.labels
    if residual_noise_std > 0.0:
        privacy_rng = np.random.default_rng(seed + 104729)
        observed_residual = observed_residual + privacy_rng.normal(
            scale=residual_noise_std,
            size=observed_residual.shape,
        )
    label_attack = residual_label_inference(observed_residual, train_active.labels)
    return {
        "membership_roc_auc": membership.roc_auc,
        "membership_advantage": membership.attack_advantage,
        "label_inference_accuracy": label_attack.accuracy,
        "holdout_roc_auc": float(roc_auc_score(holdout_active.labels, holdout_probability)),
    }


def _gbdt_measurement(
    *,
    rows: int,
    seed: int,
    party_count: int = 4,
    rounds: int = 5,
    l2_leaf_reg: float = 1.0,
    min_samples_leaf: int | None = None,
    max_depth: int = 2,
) -> dict[str, float]:
    train_active, train_passive, holdout_active, holdout_passive = _split_parties(
        rows,
        seed,
        party_count,
    )
    train_rows = train_active.n_rows
    minimum_leaf = min_samples_leaf or max(8, train_rows // 40)
    model = VFLHistGBDT(
        n_estimators=rounds,
        learning_rate=0.1,
        max_depth=max_depth,
        min_samples_leaf=minimum_leaf,
        l2_leaf_reg=l2_leaf_reg,
        max_bins=12,
        seed=seed,
    )
    model.fit(train_active, train_passive)
    train_probability = model.predict_proba([train_active, *train_passive])[:, 1]

    train_active.share_histogram_routing_state_with(holdout_active)
    for source, target in zip(train_passive, holdout_passive, strict=True):
        source.share_histogram_routing_state_with(target)
    holdout_probability = model.predict_proba([holdout_active, *holdout_passive])[:, 1]
    membership = confidence_membership_inference(train_probability, holdout_probability)

    routing_messages: list[np.ndarray] = []
    for tree in model.trees_:
        routing_messages.extend(_passive_routing_messages(tree, active_name=train_active.name))
    routing = routing_membership_exposure(train_rows, routing_messages)
    return {
        "membership_roc_auc": membership.roc_auc,
        "membership_advantage": membership.attack_advantage,
        "routing_exposure_fraction": routing.unique_exposure_fraction,
        "routing_events_per_entity": float(routing.routed_entity_events / train_rows),
        "holdout_roc_auc": float(roc_auc_score(holdout_active.labels, holdout_probability)),
    }


def _append_measurement(
    records: list[dict[str, Any]],
    *,
    seed: int,
    model: str,
    context: dict[str, Any],
    measurement: dict[str, float],
) -> None:
    records.append({"seed": seed, "model": model, **context, **measurement})


def run_privacy_research(
    *,
    config: PrivacyResearchConfig | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run the complete multi-factor privacy/utility experiment suite.

    Every sweep retains per-seed raw measurements and reports 95% normal-approximation
    confidence intervals across independent deterministic seeds. The suite measures
    attack behavior against dataset size, party count, training rounds, regularization,
    model family, and mitigation strength. These are empirical measurements, not formal
    privacy guarantees.
    """

    cfg = config or (PrivacyResearchConfig.smoke() if smoke else PrivacyResearchConfig.full())
    if len(cfg.seeds) < 2:
        raise ValueError("privacy research requires at least two seeds for uncertainty estimates")

    baseline_rows = cfg.dataset_sizes[-1]
    baseline_logistic_rounds = cfg.logistic_rounds[len(cfg.logistic_rounds) // 2]
    baseline_gbdt_rounds = cfg.gbdt_rounds[len(cfg.gbdt_rounds) // 2]

    size_records: list[dict[str, Any]] = []
    for rows in cfg.dataset_sizes:
        for seed in cfg.seeds:
            _append_measurement(
                size_records,
                seed=seed,
                model="logistic",
                context={"dataset_size": rows},
                measurement=_logistic_measurement(
                    rows=rows,
                    seed=seed,
                    rounds=baseline_logistic_rounds,
                ),
            )
            _append_measurement(
                size_records,
                seed=seed,
                model="gbdt",
                context={"dataset_size": rows},
                measurement=_gbdt_measurement(
                    rows=rows,
                    seed=seed,
                    rounds=baseline_gbdt_rounds,
                ),
            )

    party_records: list[dict[str, Any]] = []
    for party_count in cfg.party_counts:
        for seed in cfg.seeds:
            for model_name in ("logistic", "gbdt"):
                measurement = (
                    _logistic_measurement(
                        rows=baseline_rows,
                        seed=seed,
                        party_count=party_count,
                        rounds=baseline_logistic_rounds,
                    )
                    if model_name == "logistic"
                    else _gbdt_measurement(
                        rows=baseline_rows,
                        seed=seed,
                        party_count=party_count,
                        rounds=baseline_gbdt_rounds,
                    )
                )
                _append_measurement(
                    party_records,
                    seed=seed,
                    model=model_name,
                    context={"party_count": party_count},
                    measurement=measurement,
                )

    round_records: list[dict[str, Any]] = []
    for rounds in cfg.logistic_rounds:
        for seed in cfg.seeds:
            _append_measurement(
                round_records,
                seed=seed,
                model="logistic",
                context={"training_rounds": rounds},
                measurement=_logistic_measurement(
                    rows=baseline_rows,
                    seed=seed,
                    rounds=rounds,
                ),
            )
    for rounds in cfg.gbdt_rounds:
        for seed in cfg.seeds:
            _append_measurement(
                round_records,
                seed=seed,
                model="gbdt",
                context={"training_rounds": rounds},
                measurement=_gbdt_measurement(
                    rows=baseline_rows,
                    seed=seed,
                    rounds=rounds,
                ),
            )

    regularization_records: list[dict[str, Any]] = []
    for regularization in cfg.regularization:
        for seed in cfg.seeds:
            _append_measurement(
                regularization_records,
                seed=seed,
                model="logistic",
                context={"regularization": regularization},
                measurement=_logistic_measurement(
                    rows=baseline_rows,
                    seed=seed,
                    rounds=baseline_logistic_rounds,
                    l2=regularization,
                ),
            )
            _append_measurement(
                regularization_records,
                seed=seed,
                model="gbdt",
                context={"regularization": regularization},
                measurement=_gbdt_measurement(
                    rows=baseline_rows,
                    seed=seed,
                    rounds=baseline_gbdt_rounds,
                    l2_leaf_reg=regularization,
                ),
            )

    comparison_records: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        _append_measurement(
            comparison_records,
            seed=seed,
            model="logistic",
            context={},
            measurement=_logistic_measurement(
                rows=baseline_rows,
                seed=seed,
                rounds=baseline_logistic_rounds,
            ),
        )
        _append_measurement(
            comparison_records,
            seed=seed,
            model="gbdt",
            context={},
            measurement=_gbdt_measurement(
                rows=baseline_rows,
                seed=seed,
                rounds=baseline_gbdt_rounds,
            ),
        )

    mitigation_records: list[dict[str, Any]] = []
    for noise in cfg.residual_noise:
        for seed in cfg.seeds:
            _append_measurement(
                mitigation_records,
                seed=seed,
                model="logistic",
                context={
                    "mitigation": "passive_residual_gaussian_noise",
                    "strength": noise,
                },
                measurement=_logistic_measurement(
                    rows=baseline_rows,
                    seed=seed,
                    rounds=baseline_logistic_rounds,
                    residual_noise_std=noise,
                ),
            )

    gbdt_mitigations = (
        ("baseline", 2, max(8, baseline_rows // 80), 1.0),
        ("shallow_tree", 1, max(8, baseline_rows // 80), 1.0),
        ("large_leaf", 2, max(16, baseline_rows // 20), 1.0),
        ("high_leaf_regularization", 2, max(8, baseline_rows // 80), 10.0),
    )
    if smoke:
        gbdt_mitigations = gbdt_mitigations[:2]
    for mitigation, depth, min_leaf, leaf_reg in gbdt_mitigations:
        for seed in cfg.seeds:
            _append_measurement(
                mitigation_records,
                seed=seed,
                model="gbdt",
                context={"mitigation": mitigation, "strength": float(min_leaf)},
                measurement=_gbdt_measurement(
                    rows=baseline_rows,
                    seed=seed,
                    rounds=baseline_gbdt_rounds,
                    l2_leaf_reg=leaf_reg,
                    min_samples_leaf=min_leaf,
                    max_depth=depth,
                ),
            )

    return {
        "schema_version": 2,
        "formal_privacy_guarantee": False,
        "confidence_interval": "mean +/- 1.96 * sample_std / sqrt(n)",
        "config": asdict(cfg),
        "experiments": {
            "attack_vs_dataset_size": {
                "raw": size_records,
                "summary": _summarize(size_records, ("model", "dataset_size")),
            },
            "attack_vs_party_count": {
                "raw": party_records,
                "summary": _summarize(party_records, ("model", "party_count")),
            },
            "attack_vs_training_rounds": {
                "raw": round_records,
                "summary": _summarize(round_records, ("model", "training_rounds")),
            },
            "attack_vs_regularization": {
                "raw": regularization_records,
                "summary": _summarize(
                    regularization_records,
                    ("model", "regularization"),
                ),
            },
            "logistic_vs_gbdt": {
                "raw": comparison_records,
                "summary": _summarize(comparison_records, ("model",)),
            },
            "mitigation_utility_tradeoff": {
                "raw": mitigation_records,
                "summary": _summarize(
                    mitigation_records,
                    ("model", "mitigation", "strength"),
                ),
            },
        },
        "interpretation": {
            "membership_roc_auc": "Confidence-based membership attack; 0.5 is chance.",
            "membership_advantage": "Maximum measured TPR-FPR over confidence thresholds.",
            "label_inference_accuracy": "Binary target inference from passive residual signals.",
            "routing_exposure_fraction": "Fraction of training entities exposed at passive splits.",
            "holdout_roc_auc": "Predictive utility measured on independently generated holdout rows.",
        },
    }


def privacy_research_summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten confidence-interval summaries for CSV/reporting pipelines."""

    rows: list[dict[str, Any]] = []
    experiments = result.get("experiments", {})
    if not isinstance(experiments, dict):
        return rows
    for experiment_name, experiment in experiments.items():
        if not isinstance(experiment, dict):
            continue
        summary = experiment.get("summary", [])
        if not isinstance(summary, list):
            continue
        for record in summary:
            if not isinstance(record, dict):
                continue
            base: dict[str, Any] = {"experiment": experiment_name}
            for key, value in record.items():
                if isinstance(value, dict):
                    for stat, stat_value in value.items():
                        base[f"{key}_{stat}"] = stat_value
                else:
                    base[key] = value
            rows.append(base)
    return rows
